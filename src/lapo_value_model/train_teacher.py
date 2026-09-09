from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from skimage.metrics import structural_similarity
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from .common import distributed_context, optimizer_schedule, set_seed, validate_resume_plan
from .config import ensure_run_dirs, load_config
from .data import make_teacher_loader
from .teacher import LAPOTeacher, teacher_loss


def _init_distributed() -> tuple[int, int, int, torch.device]:
    rank, local_rank, world_size, device = distributed_context()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, device


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return (model.module if isinstance(model, DistributedDataParallel) else model).state_dict()


def _cosine_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max(step / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    world_size: int,
    max_ssim: int = 32,
    preview_path: Path | None = None,
    rank: int = 0,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    ssim_sum = 0.0
    ssim_count = 0
    for batch in loader:
        current = batch["current"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(current, future, deterministic=True)
        reconstruction = output["reconstruction"].float()
        if preview_path is not None and rank == 0 and totals[3].item() == 0:
            limit = min(8, current.shape[0])
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview = torch.cat(
                (current[:limit].float(), future[:limit].float(), reconstruction[:limit]), dim=0
            ).clamp(0, 1)
            save_image(preview, preview_path, nrow=limit)
        diff = (reconstruction - future.float()).square()
        copy_diff = (current.float() - future.float()).square()
        totals[0] += diff.sum(dtype=torch.float64)
        totals[1] += copy_diff.sum(dtype=torch.float64)
        totals[2] += diff.numel()
        totals[3] += current.shape[0]
        if ssim_count < max_ssim:
            limit = min(current.shape[0], max_ssim - ssim_count)
            predicted = reconstruction[:limit].permute(0, 2, 3, 1).cpu().numpy()
            target = future[:limit].permute(0, 2, 3, 1).cpu().numpy()
            for left, right in zip(predicted, target, strict=True):
                ssim_sum += structural_similarity(left, right, channel_axis=2, data_range=1.0)
                ssim_count += 1
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        ssim_tensor = torch.tensor([ssim_sum, ssim_count], dtype=torch.float64, device=device)
        dist.all_reduce(ssim_tensor, op=dist.ReduceOp.SUM)
        ssim_sum, ssim_count = ssim_tensor.tolist()
    mse = float((totals[0] / totals[2]).item())
    copy_mse = float((totals[1] / totals[2]).item())
    return {
        "mse": mse,
        "copy_mse": copy_mse,
        "reconstruction_gain": copy_mse - mse,
        "ssim": float(ssim_sum / max(ssim_count, 1)),
        "examples": int(totals[3].item()),
    }


def train(
    config_path: str | Path,
    *,
    output_name: str = "teacher",
    micro_batch_size_override: int | None = None,
    max_optimizer_steps: int | None = None,
) -> None:
    config = load_config(config_path)
    ensure_run_dirs(config)
    teacher_cfg = config["teacher"]
    rank, local_rank, world_size, device = _init_distributed()
    if device.type != "cuda":
        raise RuntimeError("LAPO teacher training requires GPU")
    set_seed(int(config["seed"]), rank)
    output_dir = Path(config["output_root"]) / output_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    manifest_stats = json.loads((Path(config["artifact_root"]) / "manifest_stats.json").read_text())
    train_samples = int(manifest_stats["samples"]["train"])
    micro_batch = (
        int(teacher_cfg["micro_batch_size"])
        if micro_batch_size_override is None
        else int(micro_batch_size_override)
    )
    if micro_batch < 1:
        raise ValueError("micro batch size must be positive")
    accumulation = int(teacher_cfg["grad_accum_steps"])
    schedule = optimizer_schedule(
        train_samples,
        world_size,
        micro_batch,
        accumulation,
        int(teacher_cfg["epochs"]),
        max_optimizer_steps,
    )
    micro_steps_per_epoch = schedule["micro_steps_per_epoch"]
    epochs = schedule["epochs"]
    total_optimizer_steps = schedule["total_optimizer_steps"]
    training_plan = {
        **schedule,
        "world_size": world_size,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": accumulation,
        "learning_rate": float(teacher_cfg["learning_rate"]),
        "output_name": output_name,
    }

    train_dataset, train_loader = make_teacher_loader(
        Path(config["artifact_root"]) / "shards",
        "train",
        batch_size=micro_batch,
        num_workers=int(teacher_cfg["num_workers"]),
        image_size=int(teacher_cfg["image_size"]),
        seed=int(config["seed"]),
        train=True,
    )
    _, val_loader = make_teacher_loader(
        Path(config["artifact_root"]) / "shards",
        "val",
        batch_size=micro_batch,
        num_workers=max(0, int(teacher_cfg["num_workers"]) // 2),
        image_size=int(teacher_cfg["image_size"]),
        seed=int(config["seed"]),
        train=False,
    )
    model: torch.nn.Module = LAPOTeacher(int(teacher_cfg["latent_dim"])).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(teacher_cfg["learning_rate"]),
        weight_decay=float(teacher_cfg["weight_decay"]),
    )
    warmup_steps = round(total_optimizer_steps * float(teacher_cfg["kl_warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_lambda(step, total_optimizer_steps, warmup_steps)
    )
    start_epoch = 0
    global_step = 0
    best_gain = -float("inf")
    stale_epochs = 0
    last_path = output_dir / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        validate_resume_plan(checkpoint, training_plan, last_path)
        (model.module if isinstance(model, DistributedDataParallel) else model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_gain = float(checkpoint["best_gain"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    beta_target = float(teacher_cfg["beta_kl"])
    kl_warmup_micro = max(1, round(micro_steps_per_epoch * float(teacher_cfg["kl_warmup_fraction"])))
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = torch.zeros(3, dtype=torch.float64, device=device)
        iterator = iter(train_loader)
        for micro_step in range(micro_steps_per_epoch):
            batch = next(iterator)
            current = batch["current"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            update = (micro_step + 1) % accumulation == 0
            sync_context = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not update:
                sync_context = model.no_sync()
            beta = beta_target * min(1.0, (epoch * micro_steps_per_epoch + micro_step + 1) / kl_warmup_micro)
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(current, future)
                    loss, metrics = teacher_loss(output, future, beta)
                    scaled = loss / accumulation
                if not torch.isfinite(scaled):
                    raise FloatingPointError(
                        f"Non-finite teacher loss: loss={float(metrics['loss'])}, "
                        f"reconstruction={float(metrics['reconstruction'])}, kl={float(metrics['kl'])}"
                    )
                scaled.backward()
            running += torch.tensor(
                [float(metrics["loss"]), float(metrics["reconstruction"]), float(metrics["kl"])],
                dtype=torch.float64,
                device=device,
            )
            if update:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (global_step <= 3 or global_step % 50 == 0):
                    denom = micro_step + 1
                    values = (running / denom).tolist()
                    print(
                        f"epoch={epoch} step={global_step} loss={values[0]:.6f} "
                        f"recon={values[1]:.6f} kl={values[2]:.6f} "
                        f"grad_norm={float(grad_norm):.5f} "
                        f"peak_gib={torch.cuda.max_memory_allocated(device)/(1024**3):.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                    writer.add_scalar("train/loss", values[0], global_step)
                    writer.add_scalar("train/reconstruction", values[1], global_step)
                    writer.add_scalar("train/kl", values[2], global_step)
        metrics = validate(
            model,
            val_loader,
            device,
            world_size,
            preview_path=output_dir / "reconstruction_previews" / f"epoch-{epoch:03d}.jpg",
            rank=rank,
        )
        improved = metrics["reconstruction_gain"] > best_gain
        if improved:
            best_gain = metrics["reconstruction_gain"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if rank == 0:
            print(f"validation epoch={epoch}: {json.dumps(metrics)}", flush=True)
            for name, value in metrics.items():
                if name != "examples":
                    writer.add_scalar(f"val/{name}", value, global_step)
            state = {
                "model": _model_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_gain": best_gain,
                "stale_epochs": stale_epochs,
                "metrics": metrics,
                "config": config,
                "micro_batch_size": micro_batch,
                "training_plan": training_plan,
            }
            torch.save(state, last_path)
            if improved:
                torch.save(state, output_dir / "best.pt")
            (output_dir / "metrics.jsonl").open("a", encoding="utf-8").write(
                json.dumps({"epoch": epoch, "global_step": global_step, **metrics}) + "\n"
            )
        if world_size > 1:
            stop = torch.tensor([stale_epochs >= int(teacher_cfg["patience"])], device=device)
            dist.broadcast(stop, src=0)
            if bool(stop.item()):
                break
        elif stale_epochs >= int(teacher_cfg["patience"]):
            break
    if writer:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-name", default="teacher")
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    args = parser.parse_args()
    train(
        args.config,
        output_name=args.output_name,
        micro_batch_size_override=args.micro_batch_size,
        max_optimizer_steps=args.max_optimizer_steps,
    )


if __name__ == "__main__":
    main()
