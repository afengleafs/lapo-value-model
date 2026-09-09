from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .common import atomic_json
from .config import load_config
from .export import export_model


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {})
    return {
        "path": str(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "step": int(checkpoint.get("global_step", -1)),
        "validation_macro_spearman": float(metrics.get("macro_spearman", float("-inf"))),
        "metrics": metrics,
    }


def select_and_export(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    student_root = Path(config["output_root"]) / "student"
    summaries = {}
    for run_name in ("baseline", "proposed"):
        path = student_root / run_name / "best.pt"
        if path.is_file():
            summaries[run_name] = _checkpoint_summary(path)
    if "baseline" not in summaries:
        raise FileNotFoundError(student_root / "baseline/best.pt")
    selected = "baseline"
    reason = "baseline is the only valid full run"
    if "proposed" in summaries:
        baseline_value = summaries["baseline"]["validation_macro_spearman"]
        proposed_value = summaries["proposed"]["validation_macro_spearman"]
        if proposed_value > baseline_value:
            selected = "proposed"
            reason = "proposed has strictly higher DROID validation macro_spearman"
        else:
            reason = "baseline wins ties or has higher DROID validation macro_spearman"
    selection = {
        "selected_run": selected,
        "selection_metric": "DROID validation macro_spearman",
        "openpi_used_for_selection": False,
        "reason": reason,
        "candidates": summaries,
    }
    selection_path = Path(config["output_root"]) / "final_selection.json"
    atomic_json(selection_path, selection)
    export_report = export_model(
        config_path,
        run_name=selected,
        checkpoint_name="best.pt",
        output_dir=Path(config["output_root"]) / "final",
    )
    result = {**selection, "selection_path": str(selection_path), "export": export_report}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    select_and_export(args.config)


if __name__ == "__main__":
    main()
