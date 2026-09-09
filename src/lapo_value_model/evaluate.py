from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from .common import atomic_json, distributed_context, json_ready, set_seed
from .config import load_config
from .data import make_student_loader
from .metrics import compute_group_metrics, compute_stage_auc, compute_value_metrics
from .student import (
    QwenValueModel,
    VALUE_PATH_LATENT_HIDDEN,
    checkpoint_value_path,
    load_processor,
    process_student_batch,
)


def _load_checkpoint(model: QwenValueModel, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "trainable_model" not in checkpoint:
        raise RuntimeError(f"Not a student checkpoint: {path}")
    deployment_state = checkpoint["trainable_model"]
    if model.latent_head is None:
        deployment_state = {
            name: tensor
            for name, tensor in deployment_state.items()
            if not name.startswith("latent_head.")
        }
    _, unexpected = model.load_state_dict(deployment_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected student checkpoint keys: {unexpected}")
    return checkpoint


@torch.inference_mode()
def evaluate(
    config_path: str | Path,
    *,
    run_name: str = "proposed",
    checkpoint_name: str = "best.pt",
) -> dict[str, Any]:
    config = load_config(config_path)
    rank, _, world_size, device = distributed_context()
    if device.type != "cuda":
        raise RuntimeError("Qwen3-VL evaluation requires a GPU")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    set_seed(int(config["seed"]), rank)

    run_dir = Path(config["output_root"]) / "student" / run_name
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    evaluation_dir = run_dir / "evaluation"
    if rank == 0:
        evaluation_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    processor = load_processor(config)
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    value_path = checkpoint_value_path(checkpoint_header)
    model = QwenValueModel(
        config,
        trainable=True,
        include_latent_head=value_path == VALUE_PATH_LATENT_HIDDEN,
        value_path=value_path,
    ).to(device)
    checkpoint = _load_checkpoint(model, checkpoint_path)
    model.requires_grad_(False).eval()
    batch_size = int(config["student"].get("evaluation_batch_size", config["student"]["micro_batch_size"]))
    _, loader = make_student_loader(
        Path(config["artifact_root"]) / "shards",
        "test",
        batch_size=batch_size,
        num_workers=max(0, int(config["student"]["num_workers"]) // 2),
        seed=int(config["seed"]),
        train=False,
    )

    rows_out: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    for rows in loader:
        inputs, _ = process_student_batch(processor, rows, None, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        logits = output["value_logits"].float()
        probabilities = logits.softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = probabilities.max(dim=-1).values
        for row, prediction, entropy_value, confidence_value in zip(
            rows,
            output["value"].float().cpu().tolist(),
            entropy.cpu().tolist(),
            confidence.cpu().tolist(),
            strict=True,
        ):
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
                    "target_bin": int(metadata["value_bin"]),
                    "prediction": float(prediction),
                    "entropy": float(entropy_value),
                    "confidence": float(confidence_value),
                }
            )

    rank_path = evaluation_dir / f"test_predictions.rank-{rank:03d}.parquet"
    temporary = rank_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows_out), temporary, compression="zstd")
    temporary.replace(rank_path)
    if world_size > 1:
        dist.barrier()

    report: dict[str, Any] = {}
    if rank == 0:
        expected_paths = [
            evaluation_dir / f"test_predictions.rank-{index:03d}.parquet"
            for index in range(world_size)
        ]
        frames = [pq.read_table(path).to_pandas() for path in expected_paths]
        frame = pd.concat(frames, ignore_index=True).drop_duplicates("sample_key")
        frame = frame.sort_values("sample_key").reset_index(drop=True)
        metrics = compute_value_metrics(frame)
        metrics["mean_entropy"] = float(frame.entropy.mean())
        metrics["mean_confidence"] = float(frame.confidence.mean())
        manifest_stats = json.loads(
            (Path(config["artifact_root"]) / "manifest_stats.json").read_text(encoding="utf-8")
        )
        expected_examples = int(manifest_stats["samples"]["test"])
        if len(frame) != expected_examples:
            raise RuntimeError(
                f"Test prediction coverage mismatch: got {len(frame)}, expected {expected_examples}"
            )
        combined_path = evaluation_dir / "test_predictions.parquet"
        combined_tmp = combined_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), combined_tmp, compression="zstd")
        combined_tmp.replace(combined_path)
        report = {
            "run_name": run_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_step": int(checkpoint.get("global_step", -1)),
            "latent_mode": checkpoint.get("latent_mode"),
            "lambda_latent": checkpoint.get("lambda_latent"),
            "metrics": metrics,
            "stage_matched_separation": compute_stage_auc(frame),
            "by_task_family": compute_group_metrics(frame),
            "world_size": world_size,
            "per_rank_batch_size": batch_size,
            "peak_memory_gib_rank0": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        atomic_json(evaluation_dir / "report.json", report)
        print(json.dumps(json_ready(report), ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    payload = [report]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
        dist.destroy_process_group()
    return payload[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="proposed")
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()
    evaluate(args.config, run_name=args.run_name, checkpoint_name=args.checkpoint)


if __name__ == "__main__":
    main()
