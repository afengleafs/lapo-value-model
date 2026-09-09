from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from .common import atomic_json, set_seed
from .config import load_config
from .student import (
    QwenValueModel,
    VALUE_PATH_LATENT_HIDDEN,
    architecture_metadata,
    trainable_state,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def transplant_non_value_state(
    model: QwenValueModel, source_state: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Copy every trainable source parameter except value_head and validate coverage."""
    target_trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    reset_names = sorted(name for name in target_trainable if name.startswith("value_head."))
    copied_state: dict[str, torch.Tensor] = {}
    for name, tensor in source_state.items():
        if name.startswith("value_head."):
            continue
        if name not in target_trainable:
            raise RuntimeError(f"Unexpected non-value source parameter: {name}")
        if tuple(tensor.shape) != tuple(target_trainable[name].shape):
            raise RuntimeError(
                f"Shape mismatch for transplanted parameter {name}: "
                f"{tuple(tensor.shape)} != {tuple(target_trainable[name].shape)}"
            )
        copied_state[name] = tensor
    missing, unexpected = model.load_state_dict(copied_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected transplanted keys: {unexpected}")
    missing_trainable = sorted(set(missing).intersection(target_trainable))
    if missing_trainable != reset_names:
        raise RuntimeError(
            "Only value_head parameters may be reset; "
            f"missing={missing_trainable[:16]}, expected={reset_names[:16]}"
        )
    return sorted(copied_state), reset_names


def create_cascaded_init(
    config_path: str | Path,
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Transplant LoRA/latent weights and deterministically reset the value head."""
    config = load_config(config_path)
    source_path = Path(source_checkpoint).expanduser().resolve()
    output_path = Path(output_checkpoint).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == output_path:
        raise ValueError("Source and output checkpoints must differ")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = source.get("trainable_model")
    if not isinstance(source_state, dict):
        raise RuntimeError(f"Not a Student checkpoint: {source_path}")

    set_seed(int(seed))
    model = QwenValueModel(
        config,
        trainable=True,
        include_latent_head=True,
        value_path=VALUE_PATH_LATENT_HIDDEN,
    )
    copied_names, reset_names = transplant_non_value_state(model, source_state)

    source_hash = _sha256(source_path)
    architecture = architecture_metadata(model)
    initialization = {
        "policy": "copy_lora_and_latent_reset_value_head",
        "seed": int(seed),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "copied_parameter_names": copied_names,
        "reset_parameter_names": reset_names,
    }
    payload = {
        "trainable_model": trainable_state(model),
        "architecture": architecture,
        "initialization": initialization,
        "config": config,
    }
    _atomic_torch_save(payload, output_path)
    output_hash = _sha256(output_path)
    report = {
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": output_hash,
        "architecture": architecture,
        "initialization": initialization,
        "copied_parameters": len(copied_names),
        "reset_parameters": len(reset_names),
    }
    atomic_json(output_path.parent / "initialization_report.json", report)
    print(
        json.dumps(
            {
                "output_checkpoint": report["output_checkpoint"],
                "output_checkpoint_sha256": output_hash,
                "architecture": architecture,
                "copied_parameters": len(copied_names),
                "reset_parameters": len(reset_names),
            },
            indent=2,
        ),
        flush=True,
    )
    return report
