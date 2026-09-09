from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lapo_value_model.curve_report import (
    episode_curve_summary,
    generate_openpi_three_group_value_report,
    render_advantage_distribution_svg,
    render_advantage_svg,
    render_three_group_value_svg,
    render_value_svg,
    three_group_terminal_values,
)


def _curve_frame() -> pd.DataFrame:
    rows = []
    for outcome, offset in (("success", 0.1), ("failure", -0.1)):
        for episode in range(3):
            for progress in (0.1, 0.5, 0.9):
                prediction = -0.9 + progress * (0.8 if outcome == "success" else 0.2)
                rows.append(
                    {
                        "sample_key": f"{outcome}-{episode}-{progress}",
                        "uuid": f"{outcome}-{episode}",
                        "progress": progress,
                        "outcome": outcome,
                        "prediction": prediction + episode * 0.01,
                        "target": -1.0 + progress if outcome == "success" else -1.0,
                        "advantage": offset + (progress - 0.5) * 0.05 + episode * 0.005,
                        "advantage_scale": 0.055,
                    }
                )
    return pd.DataFrame(rows)


def _three_group_frame() -> pd.DataFrame:
    rows = []
    groups = (
        ("demonstration", "success", 0.18),
        ("rollout", "success", 0.04),
        ("rollout", "failure", -0.16),
    )
    for collection, outcome, offset in groups:
        for episode in range(5):
            uuid = f"{collection}-{outcome}-{episode}"
            for anchor, progress in enumerate((0.1, 0.5, 0.9)):
                rows.append(
                    {
                        "sample_key": f"{uuid}-{anchor}",
                        "uuid": uuid,
                        "collection": collection,
                        "outcome": outcome,
                        "heldout_fold": episode,
                        "progress": progress,
                        "target": -0.85 + 0.55 * progress,
                        "prediction": -0.8 + 0.6 * progress + offset + episode * 0.005,
                    }
                )
    return pd.DataFrame(rows)


def test_episode_curve_summary_is_episode_equal_and_does_not_extrapolate() -> None:
    frame = pd.DataFrame(
        [
            {"uuid": "short", "progress": 0.25, "value": 0.0},
            {"uuid": "short", "progress": 0.75, "value": 1.0},
            {"uuid": "long", "progress": 0.25, "value": 2.0},
            {"uuid": "long", "progress": 0.50, "value": 2.0},
            {"uuid": "long", "progress": 0.75, "value": 2.0},
        ]
    )
    summary = episode_curve_summary(frame, "value", grid=np.array([0.0, 0.5, 1.0]))
    assert summary.count.tolist() == [0, 2, 0]
    assert np.isnan(summary.mean[0]) and np.isnan(summary.mean[2])
    # Episode equal: mean of interpolated 0.5 and 2.0, not an anchor-weighted mean.
    assert summary.mean[1] == pytest.approx(1.25)


def test_svg_renderers_write_two_model_panels(tmp_path: Path) -> None:
    baseline = _curve_frame()
    proposed = baseline.copy()
    proposed["prediction"] = proposed.prediction + 0.02
    proposed["advantage"] = proposed.advantage * 0.8
    paths = {
        "value": tmp_path / "value.svg",
        "advantage": tmp_path / "advantage.svg",
        "distribution": tmp_path / "distribution.svg",
    }
    render_value_svg((baseline, proposed), paths["value"])
    render_advantage_svg(
        (baseline, proposed), paths["advantage"], y_limit=0.25, effect_scale=0.055
    )
    render_advantage_distribution_svg(
        (baseline, proposed), (0.1, 0.08), paths["distribution"]
    )
    for path in paths.values():
        source = path.read_text(encoding="utf-8")
        assert "<svg" in source
        assert "Baseline" in source
        assert "Proposed" in source
        assert "nan" not in source.lower()


def test_three_group_value_renderer_writes_all_groups(tmp_path: Path) -> None:
    baseline = _three_group_frame()
    proposed = baseline.copy()
    proposed["prediction"] = proposed.prediction + 0.02
    path = tmp_path / "three-group-value.svg"
    render_three_group_value_svg((baseline, proposed), path, step=2000)
    source = path.read_text(encoding="utf-8")
    assert "<svg" in source
    assert "Baseline" in source and "Proposed" in source
    assert "demo success n=5" in source
    assert "rollout success n=5" in source
    assert "rollout failure n=5" in source
    assert "clairvoyant" in source
    assert "nan" not in source.lower()
    terminal = three_group_terminal_values(baseline)
    assert set(terminal) == {
        "demo_success",
        "rollout_success",
        "rollout_failure",
        "clairvoyant_target",
    }


def test_generate_three_group_value_report_validates_and_records_coverage(
    tmp_path: Path,
) -> None:
    config = {
        "project_root": str(tmp_path),
        "data_root": str(tmp_path / "data"),
        "artifact_root": str(tmp_path / "artifacts"),
        "output_root": str(tmp_path / "outputs"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    baseline = _three_group_frame()
    proposed = baseline.copy()
    proposed["prediction"] = proposed.prediction + 0.02
    report = {
        "steps": [0, 1000, 2000],
        "episodes_per_step": 15,
        "anchors_per_step": len(baseline),
    }
    for name, frame in (("baseline", baseline), ("proposed", proposed)):
        root = tmp_path / "outputs" / name
        (root / "oof_predictions").mkdir(parents=True)
        (root / "oof_report.json").write_text(json.dumps(report), encoding="utf-8")
        frame.to_parquet(root / "oof_predictions" / "step-02000.parquet", index=False)

    html_path = generate_openpi_three_group_value_report(
        config_path,
        baseline_output_name="baseline",
        proposed_output_name="proposed",
        output_name="three-group-report",
        step=2000,
    )
    output_root = tmp_path / "outputs" / "three-group-report"
    assert html_path == output_root / "openpi_oof_three_group_value_curves.html"
    assert html_path.is_file()
    assert (output_root / "step-02000-three-group-value.svg").is_file()
    manifest = json.loads((output_root / "report_manifest.json").read_text())
    assert manifest["episodes"] == 15
    assert manifest["anchors"] == 45
    assert manifest["group_episode_counts"] == {
        "demo_success": 5,
        "rollout_failure": 5,
        "rollout_success": 5,
    }
    assert manifest["interpolation"] == {
        "episode_equal": True,
        "extrapolation": False,
        "grid_points": 101,
        "minimum_coverage_fraction": 0.1,
    }
