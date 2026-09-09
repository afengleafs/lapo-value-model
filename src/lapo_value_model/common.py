from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    """Recursively convert non-finite metric values to JSON null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def optimizer_schedule(
    train_samples: int,
    world_size: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    max_optimizer_steps: int | None = None,
) -> dict[str, int]:
    """Return the exact padded DDP optimizer-step budget used by both trainers."""
    if min(train_samples, world_size, micro_batch_size, grad_accum_steps, epochs) < 1:
        raise ValueError("optimizer schedule inputs must all be positive")
    micro_steps = math.ceil(train_samples / (world_size * micro_batch_size))
    micro_steps = math.ceil(micro_steps / grad_accum_steps) * grad_accum_steps
    optimizer_steps = micro_steps // grad_accum_steps
    effective_epochs = epochs
    total = optimizer_steps * epochs
    if max_optimizer_steps is not None:
        if max_optimizer_steps < 1:
            raise ValueError("max_optimizer_steps must be positive")
        total = min(total, int(max_optimizer_steps))
        micro_steps = total * grad_accum_steps
        effective_epochs = 1
    return {
        "micro_steps_per_epoch": micro_steps,
        "optimizer_steps_per_epoch": optimizer_steps,
        "epochs": effective_epochs,
        "total_optimizer_steps": total,
        "global_batch_size": world_size * micro_batch_size * grad_accum_steps,
    }


def validate_resume_plan(
    checkpoint: dict[str, Any], expected: dict[str, Any], checkpoint_path: str | Path
) -> None:
    """Reject a checkpoint created under a different optimizer/data plan."""
    saved = checkpoint.get("training_plan")
    if saved is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} predates resume-plan validation; "
            "use a new output name instead of silently resuming it"
        )
    mismatches = {
        key: {"checkpoint": saved.get(key), "current": value}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint plan mismatch for {checkpoint_path}: "
            + json.dumps(mismatches, sort_keys=True)
        )
