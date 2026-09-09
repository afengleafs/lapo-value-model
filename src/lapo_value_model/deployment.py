from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

from .student import QwenValueModel, VALUE_PATH_DIRECT, VALUE_PATH_LATENT_HIDDEN


def load_exported_model(
    export_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    base_model_source: str | Path | None = None,
) -> tuple[QwenValueModel, Any]:
    """Load the causal deployment state without LAPO Teacher or future frames."""
    export_dir = Path(export_dir).expanduser().resolve()
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    student_config = dict(manifest["student"])
    if base_model_source is not None:
        student_config["model_source"] = str(Path(base_model_source).expanduser().resolve())
    config = {"student": student_config}
    architecture = manifest.get("architecture")
    value_path = (
        str(architecture.get("value_path", VALUE_PATH_DIRECT))
        if isinstance(architecture, dict)
        else VALUE_PATH_DIRECT
    )
    keep_latent_head = value_path == VALUE_PATH_LATENT_HIDDEN
    model = QwenValueModel(
        config,
        trainable=True,
        include_latent_head=keep_latent_head,
        value_path=value_path,
    )
    state = load_file(export_dir / manifest["files"]["deployment_state"], device="cpu")
    bins = state.pop("value_bin_centers")
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected exported keys: {unexpected}")
    model.value_bin_centers.copy_(bins)
    model.requires_grad_(False).to(device).eval()
    processor = AutoProcessor.from_pretrained(export_dir / manifest["files"]["processor"], local_files_only=True)
    return model, processor
