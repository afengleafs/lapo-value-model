from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import atomic_json
from .config import load_config
from .openpi_finetune import finetune_output_root


SURFACE = "#faf7f0"
INK = "#333b49"
MUTED = "#6d7480"
GRID = "#e6e0d2"
SUCCESS = "#0a8f85"
FAILURE = "#b23a55"
SUCCESS_FILL = "#d4e6df"
FAILURE_FILL = "#eed9d7"
MODEL_NAMES = ("Baseline", "Proposed")
GRID_X = np.linspace(0.0, 1.0, 101)
DEMO_SUCCESS = "#4c78d8"
ROLLOUT_SUCCESS = "#0a8f85"
ROLLOUT_FAILURE = "#ee6a3b"
THREE_VALUE_GROUPS = (
    ("demo_success", "demonstration", "success", "demo success", DEMO_SUCCESS, "-"),
    ("rollout_success", "rollout", "success", "rollout success", ROLLOUT_SUCCESS, "-"),
    ("rollout_failure", "rollout", "failure", "rollout failure", ROLLOUT_FAILURE, "--"),
)


@dataclass(frozen=True)
class CurveSummary:
    grid: np.ndarray
    mean: np.ndarray
    se: np.ndarray
    count: np.ndarray
    visible: np.ndarray


def _interpolate_episode_curve(
    episode: pd.DataFrame,
    field: str,
    *,
    grid: np.ndarray = GRID_X,
) -> np.ndarray:
    """Interpolate one episode on relative progress without endpoint extrapolation."""
    ordered = episode.sort_values("progress")
    x = ordered.progress.to_numpy(dtype=np.float64)
    y = ordered[field].to_numpy(dtype=np.float64)
    unique_x, positions = np.unique(x, return_index=True)
    unique_y = y[positions]
    curve = np.interp(grid, unique_x, unique_y)
    curve[(grid < unique_x[0]) | (grid > unique_x[-1])] = np.nan
    return curve


def episode_curve_summary(
    frame: pd.DataFrame,
    field: str,
    *,
    grid: np.ndarray = GRID_X,
    minimum_coverage_fraction: float = 0.10,
) -> CurveSummary:
    """Episode-equal interpolation without extrapolating outside observed progress."""
    episode_curves: list[np.ndarray] = []
    for _, episode in frame.groupby("uuid", sort=True):
        episode_curves.append(_interpolate_episode_curve(episode, field, grid=grid))
    if not episode_curves:
        raise ValueError("Cannot summarize an empty episode group")
    values = np.asarray(episode_curves, dtype=np.float64)
    count = np.isfinite(values).sum(axis=0)
    mean = np.full(len(grid), np.nan, dtype=np.float64)
    se = np.full(len(grid), np.nan, dtype=np.float64)
    for index, n_valid in enumerate(count):
        if n_valid:
            column = values[:, index]
            column = column[np.isfinite(column)]
            mean[index] = float(column.mean())
            se[index] = (
                float(column.std(ddof=1) / math.sqrt(n_valid)) if n_valid > 1 else 0.0
            )
    minimum = max(1, math.ceil(len(episode_curves) * minimum_coverage_fraction))
    visible = count >= minimum
    return CurveSummary(np.asarray(grid), mean, se, count, visible)


def _summary_by_outcome(frame: pd.DataFrame, field: str) -> dict[str, CurveSummary]:
    return {
        outcome: episode_curve_summary(frame.loc[frame.outcome == outcome], field)
        for outcome in ("success", "failure")
    }


def _style_axes(axis: plt.Axes, *, title: str, ylabel: str | None = None) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=10.5, fontweight="medium", color=INK)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks((0.0, 0.2, 0.5, 0.8, 1.0))
    axis.set_xlabel("Episode relative progress  t / L", fontsize=8.5, color=MUTED)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#d8d3c8")


def _plot_summary(
    axis: plt.Axes,
    summary: CurveSummary,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    band: bool = False,
) -> None:
    x = summary.grid[summary.visible]
    mean = summary.mean[summary.visible]
    se = summary.se[summary.visible]
    if band:
        axis.fill_between(x, mean - se, mean + se, color=color, alpha=0.18, linewidth=0)
    axis.plot(x, mean, color=SURFACE, linewidth=4.8, linestyle=linestyle, zorder=5)
    axis.plot(x, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label, zorder=6)


def _atomic_save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    matplotlib.rcParams["svg.hashsalt"] = path.name
    fig.savefig(
        temporary,
        format="svg",
        facecolor=SURFACE,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    temporary.replace(path)
    plt.close(fig)


def render_value_svg(frames: tuple[pd.DataFrame, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.56), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for axis, frame, model_name in zip(axes, frames, MODEL_NAMES, strict=True):
        _style_axes(axis, title=f"{model_name} · 5-fold held-out OOF", ylabel="Value V")
        axis.set_ylim(-1.02, 0.02)
        for outcome, color, linestyle in (
            ("success", SUCCESS, "-"),
            ("failure", FAILURE, "--"),
        ):
            subset = frame.loc[frame.outcome == outcome]
            for _, episode in subset.groupby("uuid", sort=False):
                ordered = episode.sort_values("progress")
                axis.plot(
                    ordered.progress,
                    ordered.prediction,
                    color=color,
                    alpha=0.055,
                    linewidth=0.45,
                    rasterized=False,
                )
            prediction = episode_curve_summary(subset, "prediction")
            _plot_summary(
                axis,
                prediction,
                color=color,
                label=f"{outcome} prediction (n={subset.uuid.nunique()})",
                linestyle=linestyle,
            )
            target = episode_curve_summary(subset, "target")
            valid = target.visible
            axis.plot(
                target.grid[valid],
                target.mean[valid],
                color=INK,
                linewidth=1.25,
                linestyle=(0, (1.5, 2.5)),
                label="V* target" if outcome == "success" else None,
                zorder=7,
            )
        axis.axhline(0.0, color="#b9b3a6", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.07, right=0.985, wspace=0.12)
    _atomic_save_figure(fig, path)


def _three_value_subsets(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subsets: dict[str, pd.DataFrame] = {}
    assigned = np.zeros(len(frame), dtype=bool)
    for key, collection, outcome, _, _, _ in THREE_VALUE_GROUPS:
        mask = (frame.collection == collection) & (frame.outcome == outcome)
        subset = frame.loc[mask]
        if subset.empty:
            raise ValueError(f"Three-group value plot has no rows for {key}")
        subsets[key] = subset
        assigned |= mask.to_numpy(dtype=bool)
    if not assigned.all():
        combinations = sorted(
            {
                (str(row.collection), str(row.outcome))
                for row in frame.loc[~assigned, ["collection", "outcome"]].itertuples()
            }
        )
        raise ValueError(f"Unexpected collection/outcome groups: {combinations}")
    return subsets


def _terminal_point(summary: CurveSummary) -> tuple[float, float]:
    positions = np.flatnonzero(summary.visible & np.isfinite(summary.mean))
    if not len(positions):
        raise RuntimeError("Curve summary has no visible terminal point")
    index = int(positions[-1])
    return float(summary.grid[index]), float(summary.mean[index])


def _separate_label_positions(
    values: list[float],
    *,
    lower: float = -0.98,
    upper: float = -0.02,
    minimum_gap: float = 0.065,
) -> list[float]:
    """Resolve right-edge label collisions while preserving vertical ordering."""
    order = np.argsort(np.asarray(values, dtype=float))
    positions = np.clip(np.asarray(values, dtype=float)[order], lower, upper)
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)
    if len(positions) and positions[-1] > upper:
        positions -= positions[-1] - upper
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - minimum_gap)
    if len(positions) and positions[0] < lower:
        positions += lower - positions[0]
    resolved = np.empty(len(values), dtype=float)
    resolved[order] = positions
    return resolved.tolist()


def three_group_terminal_values(frame: pd.DataFrame) -> dict[str, float]:
    subsets = _three_value_subsets(frame)
    values = {
        key: _terminal_point(episode_curve_summary(subset, "prediction"))[1]
        for key, subset in subsets.items()
    }
    values["clairvoyant_target"] = _terminal_point(
        episode_curve_summary(frame, "target")
    )[1]
    return values


def render_three_group_value_svg(
    frames: tuple[pd.DataFrame, pd.DataFrame],
    path: Path,
    *,
    step: int,
) -> None:
    """Render demo-success/rollout-success/rollout-failure OOF value curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for panel, (axis, frame, model_name) in enumerate(
        zip(axes, frames, MODEL_NAMES, strict=True)
    ):
        _style_axes(
            axis,
            title=f"{model_name} · 5-fold held-out OOF",
            ylabel="Value V" if panel == 0 else None,
        )
        axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
        axis.set_ylim(-1.02, 0.02)
        subsets = _three_value_subsets(frame)
        endpoints: list[tuple[str, str, float, float]] = []
        for key, _, _, label, color, linestyle in THREE_VALUE_GROUPS:
            subset = subsets[key]
            for _, episode in subset.groupby("uuid", sort=True):
                curve = _interpolate_episode_curve(episode, "prediction")
                valid = np.isfinite(curve)
                axis.plot(
                    GRID_X[valid],
                    curve[valid],
                    color=color,
                    alpha=0.055,
                    linewidth=0.45,
                    linestyle=linestyle,
                    rasterized=False,
                )
            summary = episode_curve_summary(subset, "prediction")
            count = int(subset.uuid.nunique())
            _plot_summary(
                axis,
                summary,
                color=color,
                label=f"{label} (n={count})",
                linestyle=linestyle,
            )
            x_terminal, y_terminal = _terminal_point(summary)
            endpoints.append((f"{label} n={count}", color, x_terminal, y_terminal))

        target = episode_curve_summary(frame, "target")
        valid = target.visible
        axis.plot(
            target.grid[valid],
            target.mean[valid],
            color=INK,
            linewidth=1.35,
            linestyle=(0, (1.5, 2.5)),
            label="V* (clairvoyant target)",
            zorder=7,
        )
        target_x, target_y = _terminal_point(target)
        endpoints.append(("V* clairvoyant", INK, target_x, target_y))
        label_y = _separate_label_positions([item[3] for item in endpoints])
        for (label, color, x_terminal, y_terminal), y_text in zip(
            endpoints, label_y, strict=True
        ):
            axis.plot(
                [x_terminal, 1.0],
                [y_terminal, y_text],
                color=color,
                linewidth=0.55,
                alpha=0.75,
                clip_on=False,
            )
            axis.text(
                1.015,
                y_text,
                f"{label}  {y_terminal:+.3f}",
                transform=axis.get_yaxis_transform(),
                color=color,
                fontsize=7.8,
                ha="left",
                va="center",
                clip_on=False,
            )
        axis.axhline(0.0, color="#b9b3a6", linewidth=0.8)
    fig.suptitle(
        f"OpenPI · step {step:,} · three-group OOF value curves",
        fontsize=12,
        fontweight="medium",
        color=INK,
        y=0.985,
    )
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.055, right=0.84, wspace=0.43)
    _atomic_save_figure(fig, path)


def render_advantage_svg(
    frames: tuple[pd.DataFrame, pd.DataFrame],
    path: Path,
    *,
    y_limit: float,
    effect_scale: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.56), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for axis, frame, model_name in zip(axes, frames, MODEL_NAMES, strict=True):
        _style_axes(axis, title=f"{model_name} · exact A50", ylabel="Advantage A50")
        axis.set_ylim(-y_limit, y_limit)
        for outcome, color, linestyle in (
            ("success", SUCCESS, "-"),
            ("failure", FAILURE, "--"),
        ):
            subset = frame.loc[frame.outcome == outcome]
            summary = episode_curve_summary(subset, "advantage")
            _plot_summary(
                axis,
                summary,
                color=color,
                label=f"{outcome} mean ± 1 SE (n={subset.uuid.nunique()})",
                linestyle=linestyle,
                band=True,
            )
        axis.axhline(0.0, color=INK, linewidth=0.9)
        axis.axhline(effect_scale, color="#c07a1a", linewidth=0.9, linestyle=(0, (2, 3)))
        axis.axhline(-effect_scale, color="#c07a1a", linewidth=0.9, linestyle=(0, (2, 3)))
        axis.text(
            0.985,
            effect_scale,
            f"physical scale ±{effect_scale:.5f}",
            ha="right",
            va="bottom",
            color="#c07a1a",
            fontsize=7.5,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.07, right=0.985, wspace=0.12)
    _atomic_save_figure(fig, path)


def render_advantage_distribution_svg(
    frames: tuple[pd.DataFrame, pd.DataFrame],
    epsilons: tuple[float, float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.56), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    edges = np.linspace(-0.25, 0.25, 81)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0]) * 0.92
    for axis, frame, epsilon, model_name in zip(
        axes, frames, epsilons, MODEL_NAMES, strict=True
    ):
        axis.set_facecolor(SURFACE)
        axis.set_title(model_name, loc="left", fontsize=10.5, fontweight="medium", color=INK)
        annotations = []
        for outcome, color in (("success", SUCCESS), ("failure", FAILURE)):
            values = frame.loc[frame.outcome == outcome, "advantage"].to_numpy(dtype=float)
            counts, _ = np.histogram(values, bins=edges)
            proportions = counts.astype(float) / len(values)
            if outcome == "success":
                axis.bar(
                    centers,
                    np.maximum(proportions, 1e-5),
                    width=width,
                    color=SUCCESS_FILL,
                    edgecolor=SUCCESS,
                    linewidth=0.35,
                    label="success windows",
                )
            else:
                axis.stairs(
                    np.maximum(proportions, 1e-5),
                    edges,
                    color=FAILURE,
                    linewidth=1.35,
                    label="failure windows",
                )
            outside = float(np.mean((values < edges[0]) | (values > edges[-1])))
            annotations.append(f"{outcome} outside ±0.25: {outside:.1%}")
        axis.axvline(
            epsilon,
            color=INK,
            linewidth=1.1,
            linestyle=(0, (3, 3)),
            label=f"top-30% ε={epsilon:+.4f}",
        )
        axis.set_xlim(-0.25, 0.25)
        axis.set_yscale("log")
        axis.set_ylim(1e-4, 1.0)
        axis.set_xlabel("Exact A50 advantage", fontsize=8.5, color=MUTED)
        axis.grid(True, axis="y", color=GRID, linewidth=0.6)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.text(
            0.02,
            0.03,
            "\n".join(annotations),
            transform=axis.transAxes,
            fontsize=7.5,
            color=MUTED,
            va="bottom",
        )
        for spine in axis.spines.values():
            spine.set_color("#d8d3c8")
    axes[0].set_ylabel("Fraction of outcome windows (log)", fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.07, right=0.985, wspace=0.12)
    _atomic_save_figure(fig, path)


def _load_report(root: Path) -> dict[str, Any]:
    path = root / "oof_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_pair(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    expected_rows: int,
    required: set[str],
    label: str,
    expected_episodes: int,
) -> None:
    for model, frame in (("baseline", baseline), ("proposed", proposed)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} {model} missing columns: {sorted(missing)}")
        if len(frame) != expected_rows or frame.sample_key.nunique() != expected_rows:
            raise RuntimeError(f"{label} {model} coverage mismatch: {len(frame)}")
        if frame.uuid.nunique() != expected_episodes:
            raise RuntimeError(
                f"{label} {model} episode coverage is not {expected_episodes}"
            )
        numeric_fields = [field for field in required if field in {"progress", "target", "prediction", "advantage"}]
        if not np.isfinite(frame[numeric_fields].to_numpy(dtype=float)).all():
            raise FloatingPointError(f"{label} {model} contains non-finite values")
    if set(baseline.sample_key) != set(proposed.sample_key):
        raise RuntimeError(f"{label} baseline/proposed sample keys differ")


def _validate_three_group_pair(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    expected_rows: int,
    expected_episodes: int,
    step: int,
) -> dict[str, int]:
    required = {
        "sample_key",
        "uuid",
        "collection",
        "outcome",
        "heldout_fold",
        "progress",
        "target",
        "prediction",
    }
    _validate_pair(
        baseline,
        proposed,
        expected_rows=expected_rows,
        required=required,
        label=f"step {step} three-group value",
        expected_episodes=expected_episodes,
    )
    for model, frame in (("baseline", baseline), ("proposed", proposed)):
        episode_metadata = frame.groupby("uuid", sort=False)[
            ["collection", "outcome", "heldout_fold"]
        ].nunique()
        if (episode_metadata != 1).any().any():
            raise RuntimeError(f"{model} episode metadata changes within an episode")
        if frame.heldout_fold.nunique() != 5:
            raise RuntimeError(f"{model} three-group value data does not cover 5 folds")
        _three_value_subsets(frame)

    baseline_ordered = baseline.sort_values("sample_key").reset_index(drop=True)
    proposed_ordered = proposed.sort_values("sample_key").reset_index(drop=True)
    for field in ("uuid", "collection", "outcome", "heldout_fold"):
        if not np.array_equal(
            baseline_ordered[field].to_numpy(), proposed_ordered[field].to_numpy()
        ):
            raise RuntimeError(f"Baseline/proposed {field} differs at step {step}")
    for field in ("progress", "target"):
        if not np.array_equal(
            baseline_ordered[field].to_numpy(dtype=np.float64),
            proposed_ordered[field].to_numpy(dtype=np.float64),
        ):
            raise RuntimeError(f"Baseline/proposed {field} differs at step {step}")

    subsets = _three_value_subsets(baseline)
    group_counts = {key: int(subset.uuid.nunique()) for key, subset in subsets.items()}
    if sum(group_counts.values()) != expected_episodes:
        raise RuntimeError(
            f"Three-group episode count {sum(group_counts.values())} != {expected_episodes}"
        )
    return group_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inline_svg(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    match = re.search(r"<svg\b.*</svg>", source, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"No SVG element in {path}")
    return match.group(0)


def _metric_rows(
    baseline: dict[str, Any], proposed: dict[str, Any]
) -> list[tuple[str, float, float]]:
    specs = (
        ("V MAE ↓", ("value", "mae")),
        ("V RMSE ↓", ("value", "rmse")),
        ("Success macro-Spearman ↑", ("value", "success_episode_macro_spearman")),
        ("Temporal monotonicity ↑", ("value", "success_temporal_monotonicity")),
        ("Mean-value AUC ↑", ("episode_auc", "mean_value")),
        ("A50 σ ↓", ("advantage_a50", "sigma")),
        ("A50 mean · success ↑", ("advantage_a50", "mean_success")),
        ("A50 mean · failure ↓", ("advantage_a50", "mean_failure")),
        ("Top-30% ε", ("advantage_a50", "epsilon_q70")),
        ("Positive · success ↑", ("advantage_a50", "positive_rate_success")),
        ("Positive · failure ↓", ("advantage_a50", "positive_rate_failure")),
        ("Positive · failure tail20 ↓", ("advantage_a50", "positive_rate_failure_tail20")),
    )
    rows = []
    for name, path in specs:
        left: Any = baseline
        right: Any = proposed
        for key in path:
            left = left[key]
            right = right[key]
        rows.append((name, float(left), float(right)))
    return rows


def generate_openpi_three_group_value_report(
    config_path: str | Path,
    *,
    baseline_output_name: str = "openpi_finetune_oof",
    proposed_output_name: str = "openpi_finetune_oof_proposed",
    output_name: str = "openpi_oof_three_group_value_curves",
    step: int = 2000,
) -> Path:
    config = load_config(config_path)
    baseline_root = finetune_output_root(config, baseline_output_name)
    proposed_root = finetune_output_root(config, proposed_output_name)
    output_root = finetune_output_root(config, output_name)
    if len({baseline_root, proposed_root, output_root}) != 3:
        raise ValueError("Baseline, proposed, and three-group output roots must differ")
    reports = (_load_report(baseline_root), _load_report(proposed_root))
    report_steps = tuple([int(value) for value in report["steps"]] for report in reports)
    if report_steps[0] != report_steps[1]:
        raise RuntimeError("Baseline/proposed OOF report steps differ")
    if int(step) not in report_steps[0]:
        raise ValueError(f"Step {step} is not available; found {report_steps[0]}")

    expected_anchors = int(reports[0]["anchors_per_step"])
    expected_episodes = int(reports[0]["episodes_per_step"])
    for report in reports[1:]:
        if (
            int(report["anchors_per_step"]) != expected_anchors
            or int(report["episodes_per_step"]) != expected_episodes
        ):
            raise RuntimeError("Baseline/proposed OpenPI value coverage differs")

    prediction_paths = tuple(
        root / "oof_predictions" / f"step-{int(step):05d}.parquet"
        for root in (baseline_root, proposed_root)
    )
    frames = tuple(pq.read_table(path).to_pandas() for path in prediction_paths)
    group_counts = _validate_three_group_pair(
        frames[0],
        frames[1],
        expected_rows=expected_anchors,
        expected_episodes=expected_episodes,
        step=int(step),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    svg_path = output_root / f"step-{int(step):05d}-three-group-value.svg"
    render_three_group_value_svg(frames, svg_path, step=int(step))
    terminal_values = {
        model_name.lower(): three_group_terminal_values(frame)
        for model_name, frame in zip(MODEL_NAMES, frames, strict=True)
    }
    labels = {spec[0]: spec[3] for spec in THREE_VALUE_GROUPS}
    rows = []
    for key, _, _, _, _, _ in THREE_VALUE_GROUPS:
        left = terminal_values["baseline"][key]
        right = terminal_values["proposed"][key]
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[key])}</td><td>{group_counts[key]}</td>"
            f"<td>{left:+.5f}</td><td>{right:+.5f}</td><td>{right-left:+.5f}</td>"
            "</tr>"
        )
    target_left = terminal_values["baseline"]["clairvoyant_target"]
    target_right = terminal_values["proposed"]["clairvoyant_target"]
    rows.append(
        "<tr><td>V* clairvoyant target</td><td>"
        f"{expected_episodes}</td><td>{target_left:+.5f}</td>"
        f"<td>{target_right:+.5f}</td><td>{target_right-target_left:+.5f}</td></tr>"
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenPI 三分组 OOF 价值曲线</title>
<style>
body{{background:{SURFACE};color:{INK};margin:0;font-family:'Noto Sans SC',system-ui,sans-serif}}
.shell{{max-width:1440px;margin:auto;padding:36px 20px 80px}}h1{{font-weight:400;font-size:32px;margin:0 0 8px}}
.sub,figcaption{{color:{MUTED};font-size:13px;line-height:1.65}}.note{{background:#f6f1e4;border-left:3px solid #c07a1a;padding:10px 16px;margin:18px 0}}
figure{{margin:20px 0 28px}}svg{{width:100%;height:auto;display:block}}
table{{border-collapse:collapse;font:12px ui-monospace,monospace;margin:18px 0 24px}}td,th{{border:1px solid #d8d3c8;padding:6px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f1ecdf}}
</style></head><body><div class="shell"><h1>OpenPI · Baseline vs Proposed · 三分组 OOF 价值曲线</h1>
<div class="sub">step {int(step):,} · 5-fold held-out out-of-fold · {expected_episodes} episodes / {expected_anchors:,} anchors。<br>
demo success n={group_counts['demo_success']}；rollout success n={group_counts['rollout_success']}；rollout failure n={group_counts['rollout_failure']}。</div>
<div class="note"><b>口径：</b>每个 episode 只由没有见过该 episode 的 fold 模型打分。细线为单 episode，粗线为 episode-equal 组均值；灰色点线为全部 episode 的 V* clairvoyant 目标均值。横轴统一插值到 101 个相对进度点，观测范围外不外推。</div>
<figure><figcaption>两个模型共用 x/y 坐标轴；右侧数字为各组最后一个可见进度点的均值。</figcaption>{_inline_svg(svg_path)}</figure>
<table><thead><tr><th>Curve</th><th>Episodes</th><th>Baseline terminal</th><th>Proposed terminal</th><th>Proposed − Baseline</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</div></body></html>"""
    html_path = output_root / "openpi_oof_three_group_value_curves.html"
    temporary = html_path.with_name(f".{html_path.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(html_path)

    manifest = {
        "protocol": "OpenPI paired 5-fold held-out OOF three-group value curves",
        "step": int(step),
        "baseline_root": str(baseline_root),
        "proposed_root": str(proposed_root),
        "episodes": expected_episodes,
        "anchors": expected_anchors,
        "group_episode_counts": group_counts,
        "interpolation": {
            "grid_points": len(GRID_X),
            "episode_equal": True,
            "extrapolation": False,
            "minimum_coverage_fraction": 0.10,
        },
        "value_axes": {"x": [0.0, 1.0], "y": [-1.02, 0.02]},
        "terminal_values": terminal_values,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in prediction_paths
        ],
        "svg": {"path": svg_path.name, "sha256": _sha256(svg_path)},
        "html": {"path": html_path.name, "sha256": _sha256(html_path)},
    }
    atomic_json(output_root / "report_manifest.json", manifest)
    return html_path


def generate_openpi_curve_report(
    config_path: str | Path,
    *,
    baseline_output_name: str = "openpi_finetune_oof",
    proposed_output_name: str = "openpi_finetune_oof_proposed",
    output_name: str = "openpi_finetune_oof_curves",
) -> Path:
    config = load_config(config_path)
    baseline_root = finetune_output_root(config, baseline_output_name)
    proposed_root = finetune_output_root(config, proposed_output_name)
    output_root = finetune_output_root(config, output_name)
    if len({baseline_root, proposed_root, output_root}) != 3:
        raise ValueError("Baseline, proposed, and curve output roots must differ")
    reports = (_load_report(baseline_root), _load_report(proposed_root))
    steps = [int(step) for step in reports[0]["steps"]]
    if steps != [int(step) for step in reports[1]["steps"]]:
        raise RuntimeError("Baseline/proposed OOF report steps differ")
    if not steps or steps[0] != 0 or steps[-1] != 2000:
        raise RuntimeError(f"Expected a step-0 to step-2000 OOF curve; got {steps}")
    expected_anchors = int(reports[0]["anchors_per_step"])
    expected_windows = int(reports[0]["advantage_windows_per_step"])
    expected_episodes = int(reports[0]["episodes_per_step"])
    if any(
        int(report[field]) != expected
        for report in reports
        for field, expected in (
            ("anchors_per_step", expected_anchors),
            ("advantage_windows_per_step", expected_windows),
            ("episodes_per_step", expected_episodes),
        )
    ):
        raise RuntimeError("Baseline/proposed OpenPI coverage differs")

    prediction_required = {
        "sample_key", "uuid", "progress", "outcome", "target", "prediction"
    }
    advantage_required = {"sample_key", "uuid", "progress", "outcome", "advantage"}
    shared_advantage_extent = 0.0
    effect_scales: set[float] = set()
    expected_advantage_episodes: int | None = None
    for step in steps:
        frames = tuple(
            pq.read_table(root / "oof_advantages" / f"step-{step:05d}.parquet").to_pandas()
            for root in (baseline_root, proposed_root)
        )
        advantage_episodes = int(frames[0].uuid.nunique())
        if expected_advantage_episodes is None:
            expected_advantage_episodes = advantage_episodes
        elif advantage_episodes != expected_advantage_episodes:
            raise RuntimeError("Advantage episode coverage changes across checkpoints")
        _validate_pair(
            frames[0], frames[1], expected_rows=expected_windows,
            required=advantage_required, label=f"step {step} advantage",
            expected_episodes=advantage_episodes,
        )
        for frame in frames:
            for outcome in ("success", "failure"):
                summary = episode_curve_summary(frame.loc[frame.outcome == outcome], "advantage")
                valid = summary.visible
                shared_advantage_extent = max(
                    shared_advantage_extent,
                    float(np.nanmax(np.abs(summary.mean[valid] - summary.se[valid]))),
                    float(np.nanmax(np.abs(summary.mean[valid] + summary.se[valid]))),
                )
            if "advantage_scale" in frame:
                effect_scales.update(frame.advantage_scale.astype(float).unique().tolist())
    if len(effect_scales) != 1:
        raise RuntimeError(f"Expected one physical A50 scale, got {effect_scales}")
    effect_scale = next(iter(effect_scales))
    y_limit = math.ceil(shared_advantage_extent * 1.05 / 0.05) * 0.05

    output_root.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    sections: list[str] = []
    for step in steps:
        prediction_frames = tuple(
            pq.read_table(root / "oof_predictions" / f"step-{step:05d}.parquet").to_pandas()
            for root in (baseline_root, proposed_root)
        )
        advantage_frames = tuple(
            pq.read_table(root / "oof_advantages" / f"step-{step:05d}.parquet").to_pandas()
            for root in (baseline_root, proposed_root)
        )
        _validate_pair(
            prediction_frames[0], prediction_frames[1], expected_rows=expected_anchors,
            required=prediction_required, label=f"step {step} value",
            expected_episodes=expected_episodes,
        )
        _validate_pair(
            advantage_frames[0], advantage_frames[1], expected_rows=expected_windows,
            required=advantage_required, label=f"step {step} advantage",
            expected_episodes=int(expected_advantage_episodes),
        )
        metrics = tuple(report["metrics_by_step"][str(step)] for report in reports)
        epsilons = tuple(float(metric["advantage_a50"]["epsilon_q70"]) for metric in metrics)
        paths = {
            "value": output_root / f"step-{step:05d}-value.svg",
            "advantage": output_root / f"step-{step:05d}-advantage.svg",
            "distribution": output_root / f"step-{step:05d}-advantage-distribution.svg",
        }
        render_value_svg(prediction_frames, paths["value"])
        render_advantage_svg(
            advantage_frames, paths["advantage"], y_limit=y_limit, effect_scale=effect_scale
        )
        render_advantage_distribution_svg(advantage_frames, epsilons, paths["distribution"])
        for kind, path in paths.items():
            assets.append(
                {"step": step, "kind": kind, "path": path.name, "sha256": _sha256(path)}
            )
        table = "".join(
            "<tr>"
            f"<td>{html.escape(name)}</td><td>{left:.5f}</td><td>{right:.5f}</td>"
            f"<td>{right-left:+.5f}</td></tr>"
            for name, left, right in _metric_rows(metrics[0], metrics[1])
        )
        primary_step = steps[-1]
        open_attr = " open" if step == primary_step else ""
        sections.append(
            f"<details id='step-{step:05d}'{open_attr}><summary>step {step:05d}"
            f"{' · preregistered primary' if step == primary_step else ''}</summary>"
            "<table><thead><tr><th>Metric</th><th>Baseline</th><th>Proposed</th>"
            f"<th>Proposed − Baseline</th></tr></thead><tbody>{table}</tbody></table>"
            f"<figure><figcaption><b>Value curves V(t).</b> Thin lines are all {expected_episodes} OOF episode "
            "trajectories; thick lines are episode-equal outcome means; dotted ink lines are V* targets.</figcaption>"
            f"{_inline_svg(paths['value'])}</figure>"
            "<figure><figcaption><b>Exact A50 curves.</b> Outcome mean ± 1 SE; amber dotted lines "
            "mark one physical 50-frame progress scale.</figcaption>"
            f"{_inline_svg(paths['advantage'])}</figure>"
            "<figure><figcaption><b>Exact A50 distributions.</b> Outcome-normalized log fractions; "
            "the dotted threshold selects the global top 30% at this step.</figcaption>"
            f"{_inline_svg(paths['distribution'])}</figure></details>"
        )
        print(f"Curve assets complete for step {step}/{steps[-1]}", flush=True)

    navigation = " ".join(
        f"<a href='#step-{step:05d}'>{step}</a>" for step in steps
    )
    outcomes = reports[0]["metrics_by_step"][str(steps[0])]["episode_outcomes"]
    collections = ", ".join(reports[0].get("training_collections", ["rollout"]))
    checkpoint_text = "、".join(str(step) for step in steps[1:])
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenPI OOF 价值/优势曲线</title>
<style>
body{{background:{SURFACE};color:{INK};margin:0;font-family:'Noto Sans SC',system-ui,sans-serif}}
.shell{{max-width:1280px;margin:auto;padding:36px 20px 80px}}h1{{font-weight:400;font-size:32px;margin:0 0 8px}}
.sub,figcaption{{color:{MUTED};font-size:13px;line-height:1.65}}.note{{background:#f6f1e4;border-left:3px solid #c07a1a;padding:10px 16px;margin:18px 0}}
.nav{{position:sticky;top:0;background:{SURFACE};padding:10px 0;z-index:10;border-bottom:1px solid #d8d3c8}}
.nav a{{display:inline-block;margin:2px 3px;padding:3px 7px;color:{INK};text-decoration:none;border:1px solid #d8d3c8;border-radius:3px;font:12px ui-monospace,monospace}}
details{{margin:22px 0 30px}}summary{{cursor:pointer;font-size:19px;font-weight:500;padding:8px 0}}figure{{margin:12px 0 24px}}svg{{width:100%;height:auto;display:block}}
table{{border-collapse:collapse;font:12px ui-monospace,monospace;margin:10px 0 20px}}td,th{{border:1px solid #d8d3c8;padding:5px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f1ecdf}}
</style></head><body><div class="shell"><h1>OpenPI · Baseline vs Proposed · 价值 / 优势曲线</h1>
<div class="sub">{expected_episodes} episodes（{int(outcomes.get('success', 0))} success / {int(outcomes.get('failure', 0))} failure；{html.escape(collections)}）· 5-fold held-out OOF · batch 16 · 2,000 optimizer steps · checkpoint {checkpoint_text}。<br>
Value: {expected_episodes} episodes / {expected_anchors:,} anchors/step；exact A50: {int(expected_advantage_episodes)} episodes / {expected_windows:,} windows/step；A50 = 50 × 15 Hz model frames = 100 × 30 Hz source frames。</div>
<div class="note"><b>读法：</b>step {steps[-1]} 是预注册主结果；其余 step 只展示学习过程。所有面板均为 episode-level held-out OOF，不含训练内预测。优势曲线跨两个模型和全部 step 共用 ±{y_limit:.2f} y 轴。</div>
<div class="nav">{navigation}</div>{''.join(sections)}</div></body></html>"""
    html_path = output_root / "openpi_oof_value_advantage_curves.html"
    temporary = html_path.with_name(f".{html_path.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(html_path)
    manifest = {
        "protocol": "OpenPI paired 5-fold held-out OOF value/advantage curve report",
        "baseline_root": str(baseline_root),
        "proposed_root": str(proposed_root),
        "steps": steps,
        "episodes_per_step": expected_episodes,
        "success_episodes": int(outcomes.get("success", 0)),
        "failure_episodes": int(outcomes.get("failure", 0)),
        "anchors_per_step": expected_anchors,
        "advantage_windows_per_step": expected_windows,
        "advantage_episodes_per_step": int(expected_advantage_episodes),
        "value_y_axis": [-1.02, 0.02],
        "advantage_y_axis": [-y_limit, y_limit],
        "advantage_distribution_x_axis": [-0.25, 0.25],
        "effect_scale": effect_scale,
        "html": {"path": html_path.name, "sha256": _sha256(html_path)},
        "assets": assets,
    }
    atomic_json(output_root / "curve_report_manifest.json", manifest)
    return html_path
