from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .common import (
    atomic_json,
    distributed_context,
    optimizer_schedule,
    set_seed,
    validate_resume_plan,
)
from .config import ensure_run_dirs, load_config
from .data import make_student_loader
from .metrics import compute_value_metrics
from .student import (
    QwenValueModel,
    architecture_metadata,
    load_latent_lookup,
    load_processor,
    process_student_batch,
    student_loss,
    trainable_state,
)


def _cosine_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max(step / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    processor: Any,
    loader: Any,
    latent_lookup: dict[str, np.ndarray] | None,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    rank: int,
    world_size: int,
    lambda_latent: float,
    progress_interval: int = 20,
) -> dict[str, float]:
    model.eval()
    rows_out = []
    loss_sum = 0.0
    count = 0
    for batch_index, rows in enumerate(loader, start=1):
        inputs, targets = process_student_batch(
            processor, rows, latent_lookup if lambda_latent > 0 else None, device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
            loss, _ = student_loss(output, targets, lambda_latent)
        predictions = output["value"].float().cpu().tolist()
        loss_sum += float(loss) * len(rows)
        count += len(rows)
        for row, prediction in zip(rows, predictions, strict=True):
            metadata = row["metadata"]
            rows_out.append(
                {
                    "sample_key": row["key"],
                    "uuid": metadata["uuid"],
                    "anchor": int(metadata["anchor"]),
                    "length": int(metadata["length"]),
                    "outcome": metadata["outcome"],
                    "task_family": metadata["task_family"],
                    "target": float(metadata["value"]),
                    "prediction": float(prediction),
                }
            )
        if batch_index == 1 or batch_index % progress_interval == 0:
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(
                f"validation epoch={epoch} rank={rank} batch={batch_index} "
                f"examples={count} peak_gib={peak_gib:.3f}",
                flush=True,
            )
    prediction_root = output_dir / "validation_predictions" / f"epoch-{epoch:03d}"
    prediction_root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows_out), prediction_root / f"rank-{rank:03d}.parquet", compression="zstd"
    )
    if world_size > 1:
        dist.barrier()
    metrics: dict[str, float] = {}
    if rank == 0:
        frames = [pq.read_table(path).to_pandas() for path in sorted(prediction_root.glob("rank-*.parquet"))]
        frame = pd.concat(frames, ignore_index=True).drop_duplicates("sample_key")
        metrics = compute_value_metrics(frame)
        loss_tensor = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    else:
        loss_tensor = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        metrics["loss"] = float(loss_tensor[0].item() / max(loss_tensor[1].item(), 1))
    payload = [metrics]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def train(
    config_path: str | Path,
    *,
    run_name: str,
    latent_mode: str,
    lambda_latent_override: float | None,
    max_optimizer_steps: int | None,
    micro_batch_size_override: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_run_dirs(config)
    student_cfg = config["student"]
    rank, local_rank, world_size, device = distributed_context()
    if device.type != "cuda":
        raise RuntimeError("Student training requires GPU")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    set_seed(int(config["seed"]), rank)
    lambda_latent = (
        float(student_cfg["lambda_latent"])
        if lambda_latent_override is None
        else float(lambda_latent_override)
    )
    if latent_mode == "none":
        lambda_latent = 0.0
    output_dir = Path(config["output_root"]) / "student" / run_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    processor = load_processor(config)
    splits_needed = ("train", "val") if lambda_latent > 0 else ("val",)
    latent_lookup = load_latent_lookup(config["artifact_root"], splits_needed)
    if lambda_latent > 0 and not latent_lookup:
        raise RuntimeError("Latent lookup is empty; run generate_latents first")
    model: torch.nn.Module = QwenValueModel(config, trainable=True).to(device)
    architecture = architecture_metadata(model)
    last_path = output_dir / "last.pt"
    checkpoint = None
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["trainable_model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")

    adapter_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("value_head") or name.startswith("latent_head"):
            head_parameters.append(parameter)
        else:
            adapter_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(student_cfg["adapter_learning_rate"])},
            {"params": head_parameters, "lr": float(student_cfg["head_learning_rate"])},
        ],
        weight_decay=float(student_cfg["weight_decay"]),
    )
    manifest_stats = json.loads((Path(config["artifact_root"]) / "manifest_stats.json").read_text())
    train_samples = int(manifest_stats["samples"]["train"])
    micro_batch = (
        int(student_cfg["micro_batch_size"])
        if micro_batch_size_override is None
        else int(micro_batch_size_override)
    )
    if micro_batch < 1:
        raise ValueError("micro batch size must be positive")
    validation_micro_batch = int(student_cfg.get("validation_micro_batch_size", micro_batch))
    if validation_micro_batch < 1:
        raise ValueError("validation micro batch size must be positive")
    accumulation = int(student_cfg["grad_accum_steps"])
    schedule = optimizer_schedule(
        train_samples,
        world_size,
        micro_batch,
        accumulation,
        int(student_cfg["epochs"]),
        max_optimizer_steps,
    )
    micro_steps_per_epoch = schedule["micro_steps_per_epoch"]
    epochs = schedule["epochs"]
    total_optimizer_steps = schedule["total_optimizer_steps"]
    training_plan = {
        **schedule,
        "world_size": world_size,
        "micro_batch_size": micro_batch,
        "validation_micro_batch_size": validation_micro_batch,
        "grad_accum_steps": accumulation,
        "adapter_learning_rate": float(student_cfg["adapter_learning_rate"]),
        "head_learning_rate": float(student_cfg["head_learning_rate"]),
        "latent_mode": latent_mode,
        "lambda_latent": lambda_latent,
        "run_name": run_name,
    }
    warmup = round(total_optimizer_steps * float(student_cfg["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_lambda(step, total_optimizer_steps, warmup)
    )
    start_epoch = 0
    global_step = 0
    best_spearman = -float("inf")
    stale_epochs = 0
    if checkpoint is not None:
        validate_resume_plan(checkpoint, training_plan, last_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_spearman = float(checkpoint["best_spearman"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=lambda_latent == 0,
        )
    train_dataset, train_loader = make_student_loader(
        Path(config["artifact_root"]) / "shards",
        "train",
        batch_size=micro_batch,
        num_workers=int(student_cfg["num_workers"]),
        seed=int(config["seed"]),
        train=True,
    )
    _, val_loader = make_student_loader(
        Path(config["artifact_root"]) / "shards",
        "val",
        batch_size=validation_micro_batch,
        num_workers=max(0, int(student_cfg["num_workers"]) // 2),
        seed=int(config["seed"]),
        train=False,
    )
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(train_loader)
        running = defaultdict(float)
        for micro_step in range(micro_steps_per_epoch):
            rows = next(iterator)
            inputs, targets = process_student_batch(
                processor, rows, latent_lookup if lambda_latent > 0 else None, device
            )
            if lambda_latent > 0 and latent_mode == "random":
                targets["latent"] = torch.randn_like(targets["latent"])
            elif lambda_latent > 0 and latent_mode == "shuffled":
                targets["latent"] = targets["latent"].roll(1, dims=0)
            update = (micro_step + 1) % accumulation == 0
            sync_context = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not update:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**inputs)
                    loss, loss_metrics = student_loss(output, targets, lambda_latent)
                    scaled = loss / accumulation
                if not torch.isfinite(scaled):
                    details = {name: float(value) for name, value in loss_metrics.items()}
                    raise FloatingPointError(f"Non-finite student loss before backward: {details}")
                scaled.backward()
            for name, value in loss_metrics.items():
                running[name] += float(value)
            if update:
                if global_step < 5:
                    bad_gradients = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    ]
                    if bad_gradients:
                        raise FloatingPointError(
                            "Non-finite student gradients: " + ", ".join(bad_gradients[:16])
                        )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (global_step <= 5 or global_step % 20 == 0):
                    denominator = micro_step + 1
                    message = " ".join(f"{k}={v/denominator:.5f}" for k, v in running.items())
                    learning_rates = ",".join(f"{value:.3e}" for value in scheduler.get_last_lr())
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
                    print(
                        f"epoch={epoch} step={global_step}/{total_optimizer_steps} "
                        f"grad_norm={float(grad_norm):.5f} peak_gib={peak_gib:.3f} "
                        f"lr={learning_rates} {message}",
                        flush=True,
                    )
                    for name, value in running.items():
                        writer.add_scalar(f"train/{name}", value / denominator, global_step)
        metrics = validate(
            model,
            processor,
            val_loader,
            latent_lookup,
            device,
            output_dir,
            epoch,
            rank,
            world_size,
            lambda_latent,
        )
        improved = metrics["macro_spearman"] > best_spearman
        if improved:
            best_spearman = metrics["macro_spearman"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if rank == 0:
            print(f"validation epoch={epoch}: {json.dumps(metrics)}", flush=True)
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    writer.add_scalar(f"val/{name}", value, global_step)
            state = {
                "trainable_model": trainable_state(_unwrap(model)),
                "architecture": architecture,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_spearman": best_spearman,
                "stale_epochs": stale_epochs,
                "metrics": metrics,
                "latent_mode": latent_mode,
                "lambda_latent": lambda_latent,
                "micro_batch_size": micro_batch,
                "config": config,
                "training_plan": training_plan,
            }
            torch.save(state, last_path)
            if improved:
                torch.save(state, output_dir / "best.pt")
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "step": global_step, **metrics}) + "\n")
        stop = stale_epochs >= int(student_cfg["patience"])
        if world_size > 1:
            stop_tensor = torch.tensor([stop], device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
        if stop:
            break
    if writer:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="proposed")
    parser.add_argument("--latent-mode", choices=("true", "none", "random", "shuffled"), default="true")
    parser.add_argument("--lambda-latent", type=float)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    args = parser.parse_args()
    train(
        args.config,
        run_name=args.run_name,
        latent_mode=args.latent_mode,
        lambda_latent_override=args.lambda_latent,
        max_optimizer_steps=args.max_optimizer_steps,
        micro_batch_size_override=args.micro_batch_size,
    )


if __name__ == "__main__":
    main()
