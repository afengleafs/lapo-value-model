from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .common import atomic_json
from .config import load_config
from .student import (
    QwenValueModel,
    VALUE_PATH_LATENT_HIDDEN,
    architecture_metadata,
    checkpoint_value_path,
    load_processor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(
    config_path: str | Path,
    *,
    run_name: str = "proposed",
    checkpoint_name: str = "best.pt",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export the causal Qwen/value path without the LAPO Teacher or future input."""
    config = load_config(config_path)
    run_dir = Path(config["output_root"]) / "student" / run_name
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "trainable_model" not in checkpoint:
        raise RuntimeError(f"Not a student checkpoint: {checkpoint_path}")

    value_path = checkpoint_value_path(checkpoint)
    keep_latent_head = value_path == VALUE_PATH_LATENT_HIDDEN
    model = QwenValueModel(
        config,
        trainable=True,
        include_latent_head=keep_latent_head,
        value_path=value_path,
    )
    deployment_checkpoint = checkpoint["trainable_model"]
    if not keep_latent_head:
        deployment_checkpoint = {
            name: tensor
            for name, tensor in deployment_checkpoint.items()
            if not name.startswith("latent_head.")
        }
    _, unexpected = model.load_state_dict(deployment_checkpoint, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected deployment checkpoint keys: {unexpected}")
    model.eval()

    export_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir / "export"
    )
    adapter_dir = export_dir / "adapter"
    processor_dir = export_dir / "processor"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    processor_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    load_processor(config).save_pretrained(processor_dir)

    deployment_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    deployment_state["value_bin_centers"] = model.value_bin_centers.detach().cpu().contiguous()
    save_file(deployment_state, export_dir / "deployment_state.safetensors")
    value_head_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.value_head.state_dict().items()
    }
    save_file(value_head_state, export_dir / "value_head.safetensors")

    student_config = dict(config["student"])
    manifest = {
        "format_version": 2,
        "architecture": architecture_metadata(model),
        "base_model_source": student_config["model_source"],
        "student": student_config,
        "history_frames": 3,
        "camera_key": config["data"]["camera_key"],
        "causal_inputs_only": True,
        "contains_teacher": False,
        "contains_latent_head": keep_latent_head,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "training_latent_mode": checkpoint.get("latent_mode"),
        "training_lambda_latent": checkpoint.get("lambda_latent"),
        "files": {
            "deployment_state": "deployment_state.safetensors",
            "standard_peft_adapter": "adapter",
            "value_head": "value_head.safetensors",
            "processor": "processor",
        },
    }
    atomic_json(export_dir / "manifest.json", manifest)
    files = sorted(path for path in export_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    checksums = {str(path.relative_to(export_dir)): _sha256(path) for path in files}
    checksum_path = export_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="utf-8"
    )
    report = {
        "export_dir": str(export_dir),
        "files": len(files) + 1,
        "bytes": sum(path.stat().st_size for path in files) + checksum_path.stat().st_size,
        "sha256": checksums,
    }
    atomic_json(export_dir / "export_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="proposed")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    export_model(
        args.config,
        run_name=args.run_name,
        checkpoint_name=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
