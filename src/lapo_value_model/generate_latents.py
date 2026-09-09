from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from .common import atomic_json, distributed_context, set_seed
from .config import load_config
from .data import make_teacher_loader
from .teacher import LAPOTeacher


def generate(config_path: str | Path) -> None:
    config = load_config(config_path)
    rank, local_rank, world_size, device = distributed_context()
    if device.type != "cuda":
        raise RuntimeError("Latent generation requires GPU")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    set_seed(int(config["seed"]), rank)
    teacher_cfg = config["teacher"]
    checkpoint_path = Path(config["output_root"]) / "teacher/best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("metrics", {}).get("reconstruction_gain", -1.0) <= 0
        and "smoke" not in str(config["artifact_root"])
    ):
        raise RuntimeError("Teacher reconstruction gain is not positive; refusing to cache bad latents")
    model = LAPOTeacher(int(teacher_cfg["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    latent_root = Path(config["artifact_root"]) / "latents"
    latent_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_root = latent_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        final_path = split_root / f"rank-{rank:03d}.parquet"
        temporary = final_path.with_suffix(".parquet.tmp")
        _, loader = make_teacher_loader(
            Path(config["artifact_root"]) / "shards",
            split,
            batch_size=int(teacher_cfg["micro_batch_size"]),
            num_workers=int(teacher_cfg["num_workers"]),
            image_size=int(teacher_cfg["image_size"]),
            seed=int(config["seed"]),
            train=False,
        )
        schema = pa.schema(
            [
                ("sample_key", pa.string()),
                ("split", pa.string()),
                ("latent", pa.list_(pa.float32(), int(teacher_cfg["latent_dim"]))),
            ]
        )
        writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
        count = 0
        with torch.inference_mode():
            for batch in loader:
                current = batch["current"].to(device, non_blocking=True)
                future = batch["future"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mu, _ = model.encode(current, future)
                rows = [
                    {"sample_key": key, "split": split, "latent": vector}
                    for key, vector in zip(batch["keys"], mu.float().cpu().tolist(), strict=True)
                ]
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
        writer.close()
        temporary.replace(final_path)
        summary[split] = count
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        manifest_stats = json.loads(
            (Path(config["artifact_root"]) / "manifest_stats.json").read_text(encoding="utf-8")
        )
        counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            keys: set[str] = set()
            total_rows = 0
            for path in sorted((latent_root / split).glob("rank-*.parquet")):
                table = pq.read_table(path, columns=["sample_key"])
                total_rows += table.num_rows
                keys.update(table["sample_key"].to_pylist())
            counts[f"{split}_rows"] = total_rows
            counts[f"{split}_unique"] = len(keys)
            expected = int(manifest_stats["samples"][split])
            if len(keys) != expected:
                raise RuntimeError(
                    f"Latent coverage mismatch for {split}: got {len(keys)} unique keys, expected {expected}"
                )
            if total_rows != expected and "smoke" not in str(config["artifact_root"]):
                raise RuntimeError(
                    f"Duplicate full latents for {split}: got {total_rows} rows for {expected} samples"
                )
        atomic_json(
            latent_root / "summary.json",
            {
                "counts": counts,
                "teacher_checkpoint": str(checkpoint_path),
                "teacher_metrics": checkpoint.get("metrics", {}),
                "world_size": world_size,
            },
        )
        print(json.dumps(counts, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
