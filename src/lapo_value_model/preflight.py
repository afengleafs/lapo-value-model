from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .common import atomic_json
from .config import ensure_run_dirs, load_config


def run_preflight(config_path: str | Path, require_gpu: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_run_dirs(config)
    report: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if require_gpu and (not torch.cuda.is_available() or torch.cuda.device_count() != 2):
        raise RuntimeError(
            "GPU gate failed: run with HIP_VISIBLE_DEVICES=6,7 and require exactly two visible devices"
        )
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {"index": index, "name": props.name, "total_memory": int(props.total_memory)}
        )
    report["devices"] = devices
    if torch.cuda.is_available():
        left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
        loss = (left @ right).float().square().mean()
        loss.backward()
        report["bf16_matmul_loss"] = float(loss.detach().cpu())
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    )
    report["pip_freeze"] = packages.stdout.splitlines()
    atomic_json(Path(config["artifact_root"]) / "preflight.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    run_preflight(args.config, require_gpu=not args.allow_cpu)

