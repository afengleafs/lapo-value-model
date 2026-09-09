from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from .common import atomic_json
from .config import load_config
from .curve_report import CurveSummary, episode_curve_summary
from .openpi_eval import _atomic_parquet
from .openpi_finetune import finetune_output_root, sha256_file


DEMONSTRATION_DATASET = "fr3_plug_rgb_tactile_merge_20260812"
THREE_GROUP_EXPECTED = {
    "rollout_fr3_plug_pi05_base5k_20260820_143932": (8, 559, 479),
    "rollout_fr3_plug_pi05_base5k_20260820_145452": (9, 589, 499),
    "rollout_fr3_plug_pi05_base5k_20260820_150627": (4, 247, 207),
}
MODEL_COLORS = {"baseline": "#3568a8", "proposed": "#d4771f"}
SURFACE = "#faf7f0"
INK = "#333b49"
MUTED = "#6d7480"
GRID = "#e6e0d2"


def select_rollout_success(
    frame: pd.DataFrame, episode_inventory: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    required = {"uuid", "dataset_name", "outcome", "sample_key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF frame missing columns: {sorted(missing)}")
    inventory_required = {"uuid", "dataset_name", "collection", "outcome"}
    inventory_missing = inventory_required - set(episode_inventory.columns)
    if inventory_missing:
        raise ValueError(f"Episode inventory missing columns: {sorted(inventory_missing)}")
    episode_map = episode_inventory[list(inventory_required)].drop_duplicates("uuid")
    merged = frame.merge(
        episode_map,
        on="uuid",
        how="left",
        suffixes=("", "_inventory"),
        validate="many_to_one",
    )
    if merged.collection.isna().any():
        raise RuntimeError("OOF frame contains UUIDs absent from the OpenPI inventory")
    if not (
        (merged.dataset_name == merged.dataset_name_inventory)
        & (merged.outcome == merged.outcome_inventory)
    ).all():
        raise RuntimeError("OOF metadata disagrees with the OpenPI episode inventory")
    selected = merged.loc[
        (merged.collection == "rollout")
        & (merged.outcome == "success")
        & (merged.dataset_name != DEMONSTRATION_DATASET)
    ].copy()
    selected = selected.drop(columns=["dataset_name_inventory", "outcome_inventory"])
    all_rollouts = episode_inventory.loc[
        episode_inventory.collection == "rollout", "dataset_name"
    ].drop_duplicates()
    successful_groups = set(selected.dataset_name.astype(str))
    omitted = sorted(set(all_rollouts.astype(str)) - successful_groups)
    return selected.sort_values("sample_key").reset_index(drop=True), omitted


def resolve_selected_groups(
    available_groups: list[str], requested_groups: list[str] | None
) -> list[str]:
    available = sorted(set(str(value) for value in available_groups))
    if not requested_groups:
        return available
    requested = [str(value) for value in requested_groups]
    if len(requested) != len(set(requested)):
        raise ValueError("dataset-name values must be unique")
    if any(not value or Path(value).name != value for value in requested):
        raise ValueError("dataset-name must be one rollout directory name")
    if DEMONSTRATION_DATASET in requested:
        raise ValueError("The demonstration dataset is excluded from this report")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Requested rollout groups have no successful OOF data: {unknown}")
    return requested


def _validate_model_pair(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    expected_rows: int,
    expected_episodes: int,
    fields: set[str],
    label: str,
) -> None:
    for name, frame in (("baseline", baseline), ("proposed", proposed)):
        missing = fields - set(frame.columns)
        if missing:
            raise ValueError(f"{label} {name} missing columns: {sorted(missing)}")
        if len(frame) != expected_rows or frame.sample_key.nunique() != expected_rows:
            raise RuntimeError(f"{label} {name} expected {expected_rows} unique rows")
        if frame.uuid.nunique() != expected_episodes:
            raise RuntimeError(f"{label} {name} expected {expected_episodes} episodes")
        numeric = [field for field in fields if field in {"progress", "target", "prediction", "advantage"}]
        if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
            raise FloatingPointError(f"{label} {name} contains non-finite values")
        if set(frame.outcome.astype(str)) != {"success"}:
            raise RuntimeError(f"{label} {name} contains a non-success row")
        if DEMONSTRATION_DATASET in set(frame.dataset_name.astype(str)):
            raise RuntimeError(f"{label} {name} contains demonstration data")
    if baseline.sample_key.tolist() != proposed.sample_key.tolist():
        raise RuntimeError(f"{label} baseline/proposed sample keys differ")


def _curve_rows(
    group: str,
    model: str,
    prediction: pd.DataFrame,
    advantage: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, CurveSummary]]:
    summaries = {
        "prediction": episode_curve_summary(prediction, "prediction"),
        "target": episode_curve_summary(prediction, "target"),
        "advantage": episode_curve_summary(advantage, "advantage"),
    }
    rows: list[dict[str, Any]] = []
    for index, progress in enumerate(summaries["prediction"].grid):
        row: dict[str, Any] = {
            "dataset_name": group,
            "model": model,
            "progress": float(progress),
        }
        for field, summary in summaries.items():
            row[f"{field}_mean"] = float(summary.mean[index])
            row[f"{field}_se"] = float(summary.se[index])
            row[f"{field}_episode_count"] = int(summary.count[index])
            row[f"{field}_visible"] = bool(summary.visible[index])
        rows.append(row)
    return rows, summaries


def _safe_spearman(frame: pd.DataFrame) -> float:
    values: list[float] = []
    for _, episode in frame.groupby("uuid", sort=True):
        ordered = episode.sort_values("anchor")
        if ordered.prediction.nunique() < 2 or ordered.target.nunique() < 2:
            continue
        value = float(spearmanr(ordered.prediction, ordered.target).statistic)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _summary_row(
    group: str, model: str, prediction: pd.DataFrame, advantage: pd.DataFrame
) -> dict[str, Any]:
    monotonic: list[bool] = []
    episode_mae: list[float] = []
    for _, episode in prediction.groupby("uuid", sort=True):
        ordered = episode.sort_values("anchor")
        values = ordered.prediction.to_numpy(dtype=np.float64)
        monotonic.extend((np.diff(values) >= 0.0).tolist())
        episode_mae.append(float(np.abs(ordered.prediction - ordered.target).mean()))
    advantage_values = advantage.advantage.to_numpy(dtype=np.float64)
    return {
        "dataset_name": group,
        "model": model,
        "episodes": int(prediction.uuid.nunique()),
        "value_anchors": int(len(prediction)),
        "a50_windows": int(len(advantage)),
        "value_episode_macro_mae": float(np.mean(episode_mae)),
        "value_macro_spearman": _safe_spearman(prediction),
        "temporal_monotonicity": float(np.mean(monotonic)) if monotonic else 0.0,
        "a50_mean": float(advantage_values.mean()),
        "a50_std": float(advantage_values.std(ddof=0)),
        "a50_positive_rate": float(np.mean(advantage_values > 0.0)),
    }


def _style_axis(axis: plt.Axes, *, ylabel: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks((0.0, 0.2, 0.5, 0.8, 1.0))
    axis.set_xlabel("Episode relative progress  t / L", color=MUTED, fontsize=8.5)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=8.5)
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
    band: bool = True,
) -> None:
    visible = summary.visible & np.isfinite(summary.mean) & np.isfinite(summary.se)
    x = summary.grid[visible]
    mean = summary.mean[visible]
    se = summary.se[visible]
    if band:
        axis.fill_between(x, mean - se, mean + se, color=color, alpha=0.17, linewidth=0)
    axis.plot(x, mean, color=SURFACE, linewidth=4.5, linestyle=linestyle, zorder=4)
    axis.plot(x, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label, zorder=5)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    matplotlib.rcParams["svg.hashsalt"] = path.name
    figure.savefig(
        temporary,
        format="svg",
        facecolor=SURFACE,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    temporary.replace(path)
    plt.close(figure)


def _render_value(
    group: str,
    summaries: dict[str, dict[str, CurveSummary]],
    episodes: int,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 3.6))
    figure.patch.set_facecolor(SURFACE)
    _style_axis(axis, ylabel="Value V")
    axis.set_ylim(-1.02, 0.02)
    axis.set_title(f"{group} · success episodes n={episodes}", loc="left", color=INK, fontsize=10)
    for model in ("baseline", "proposed"):
        _plot_summary(
            axis,
            summaries[model]["prediction"],
            color=MODEL_COLORS[model],
            label=f"{model.title()} mean ± 1 SE",
        )
    _plot_summary(
        axis,
        summaries["baseline"]["target"],
        color=INK,
        label="V* target",
        linestyle=(0, (1.5, 2.5)),
        band=False,
    )
    axis.axhline(0.0, color="#b9b3a6", linewidth=0.8)
    axis.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.18)
    _save_figure(figure, path)


def _render_advantage(
    group: str,
    summaries: dict[str, dict[str, CurveSummary]],
    episodes: int,
    effect_scale: float,
    y_limit: float,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 3.6))
    figure.patch.set_facecolor(SURFACE)
    _style_axis(axis, ylabel="Exact A50")
    axis.set_ylim(-y_limit, y_limit)
    axis.set_title(f"{group} · exact A50 · success episodes n={episodes}", loc="left", color=INK, fontsize=10)
    for model in ("baseline", "proposed"):
        _plot_summary(
            axis,
            summaries[model]["advantage"],
            color=MODEL_COLORS[model],
            label=f"{model.title()} mean ± 1 SE",
        )
    axis.axhline(0.0, color=INK, linewidth=0.9, label="zero")
    axis.axhline(effect_scale, color="#a46b19", linewidth=0.8, linestyle=(0, (2, 3)))
    axis.axhline(-effect_scale, color="#a46b19", linewidth=0.8, linestyle=(0, (2, 3)))
    axis.text(
        0.99,
        effect_scale,
        f"physical scale ±{effect_scale:.5f}",
        ha="right",
        va="bottom",
        color="#a46b19",
        fontsize=7.5,
    )
    axis.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.18)
    _save_figure(figure, path)


def _group_colors(groups: list[str]) -> dict[str, Any]:
    color_map = plt.get_cmap("turbo")
    return {
        group: color_map(position)
        for group, position in zip(groups, np.linspace(0.04, 0.96, len(groups)), strict=True)
    }


def _combined_legends(axis: plt.Axes, groups: list[str], colors: dict[str, Any]) -> None:
    group_handles = [
        Line2D([0], [0], color=colors[group], linewidth=2.2, label=group.removeprefix("rollout_fr3_plug_pi05_base5k_"))
        for group in groups
    ]
    group_legend = axis.legend(
        handles=group_handles,
        title="Rollout group",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
    )
    axis.add_artist(group_legend)
    model_handles = [
        Line2D([0], [0], color=INK, linewidth=2.0, linestyle="--", label="Baseline"),
        Line2D([0], [0], color=INK, linewidth=2.0, linestyle="-", label="Proposed"),
    ]
    axis.legend(handles=model_handles, loc="upper left", ncol=2, frameon=False, fontsize=8)


def _render_combined_value(
    groups: list[str],
    summaries: dict[str, dict[str, dict[str, CurveSummary]]],
    pooled_target: CurveSummary,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14.2, 5.5))
    figure.patch.set_facecolor(SURFACE)
    _style_axis(axis, ylabel="Value V")
    axis.set_ylim(-1.02, 0.02)
    axis.set_title(
        f"{len(groups)} rollout success groups · episode-equal mean value curves · step 2000 OOF",
        loc="left",
        color=INK,
        fontsize=10.5,
    )
    colors = _group_colors(groups)
    for group in groups:
        for model, linestyle in (("baseline", "--"), ("proposed", "-")):
            summary = summaries[group][model]["prediction"]
            visible = summary.visible & np.isfinite(summary.mean)
            axis.plot(
                summary.grid[visible],
                summary.mean[visible],
                color=colors[group],
                linestyle=linestyle,
                linewidth=1.35,
                alpha=0.88,
            )
    target_visible = pooled_target.visible & np.isfinite(pooled_target.mean)
    axis.plot(
        pooled_target.grid[target_visible],
        pooled_target.mean[target_visible],
        color="#111111",
        linewidth=2.2,
        linestyle=(0, (1.5, 2.5)),
        label="Pooled V* target",
        zorder=6,
    )
    axis.axhline(0.0, color="#b9b3a6", linewidth=0.8)
    _combined_legends(axis, groups, colors)
    axis.text(
        0.01,
        0.04,
        "Color = rollout group; dashed = Baseline; solid = Proposed; dotted black = pooled V* target",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.065, right=0.79, top=0.89, bottom=0.14)
    _save_figure(figure, path)


def _render_combined_advantage(
    groups: list[str],
    summaries: dict[str, dict[str, dict[str, CurveSummary]]],
    *,
    effect_scale: float,
    y_limit: float,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14.2, 5.5))
    figure.patch.set_facecolor(SURFACE)
    _style_axis(axis, ylabel="Exact A50")
    axis.set_ylim(-y_limit, y_limit)
    axis.set_title(
        f"{len(groups)} rollout success groups · episode-equal mean exact A50 curves · step 2000 OOF",
        loc="left",
        color=INK,
        fontsize=10.5,
    )
    colors = _group_colors(groups)
    for group in groups:
        for model, linestyle in (("baseline", "--"), ("proposed", "-")):
            summary = summaries[group][model]["advantage"]
            visible = summary.visible & np.isfinite(summary.mean)
            axis.plot(
                summary.grid[visible],
                summary.mean[visible],
                color=colors[group],
                linestyle=linestyle,
                linewidth=1.35,
                alpha=0.88,
            )
    axis.axhline(0.0, color=INK, linewidth=1.1)
    axis.axhline(effect_scale, color="#a46b19", linewidth=0.9, linestyle=(0, (2, 3)))
    axis.axhline(-effect_scale, color="#a46b19", linewidth=0.9, linestyle=(0, (2, 3)))
    axis.text(
        0.99,
        effect_scale,
        f"physical scale ±{effect_scale:.5f}",
        ha="right",
        va="bottom",
        color="#a46b19",
        fontsize=7.5,
    )
    _combined_legends(axis, groups, colors)
    axis.text(
        0.01,
        0.04,
        "Color = rollout group; dashed = Baseline; solid = Proposed",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.065, right=0.79, top=0.89, bottom=0.14)
    _save_figure(figure, path)


def _inline_svg(path: Path) -> str:
    match = re.search(r"<svg\b.*</svg>", path.read_text(encoding="utf-8"), re.DOTALL)
    if not match:
        raise RuntimeError(f"No SVG element found in {path}")
    return match.group(0)


def _fmt(value: Any, digits: int = 5) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def report_openpi_rollout_success_curves(
    config_path: str | Path,
    *,
    baseline_output_name: str = "openpi_finetune_oof",
    proposed_output_name: str = "openpi_finetune_oof_proposed",
    output_name: str = "openpi_rollout_success_group_curves",
    step: int = 2000,
    dataset_names: list[str] | None = None,
) -> Path:
    if int(step) != 2000:
        raise ValueError("The preregistered success-group report uses step 2000")
    config = load_config(config_path)
    roots = {
        "baseline": finetune_output_root(config, baseline_output_name),
        "proposed": finetune_output_root(config, proposed_output_name),
    }
    output_root = finetune_output_root(config, output_name)
    if output_root in roots.values():
        raise ValueError("Curve output must not overwrite an OOF source directory")
    assets_root = output_root / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    external_artifact = Path(config["external_eval"]["openpi"]["artifact_root"])
    episode_inventory = pq.read_table(external_artifact / "episodes.parquet").to_pandas()
    finetune_episodes = pq.read_table(external_artifact / "finetune" / "episodes.parquet").to_pandas()
    predictions: dict[str, pd.DataFrame] = {}
    advantages: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    omitted_by_model: dict[str, list[str]] = {}
    for model, root in roots.items():
        prediction_path = root / "oof_predictions" / f"step-{step:05d}.parquet"
        advantage_path = root / "oof_advantages" / f"step-{step:05d}.parquet"
        source_paths[f"{model}_predictions"] = prediction_path
        source_paths[f"{model}_advantages"] = advantage_path
        prediction, omitted = select_rollout_success(
            pq.read_table(prediction_path).to_pandas(), episode_inventory
        )
        advantage, omitted_advantage = select_rollout_success(
            pq.read_table(advantage_path).to_pandas(), episode_inventory
        )
        if omitted != omitted_advantage:
            raise RuntimeError(f"{model} value/A50 omitted groups differ")
        predictions[model] = prediction
        advantages[model] = advantage
        omitted_by_model[model] = omitted
    if omitted_by_model["baseline"] != omitted_by_model["proposed"]:
        raise RuntimeError("Baseline/proposed omitted groups differ")

    prediction_fields = {
        "sample_key", "uuid", "dataset_name", "anchor", "progress", "outcome",
        "target", "prediction", "heldout_fold",
    }
    advantage_fields = {
        "sample_key", "uuid", "dataset_name", "anchor", "progress", "outcome",
        "advantage", "advantage_scale", "heldout_fold",
    }
    _validate_model_pair(
        predictions["baseline"], predictions["proposed"],
        expected_rows=5616, expected_episodes=88, fields=prediction_fields, label="value",
    )
    _validate_model_pair(
        advantages["baseline"], advantages["proposed"],
        expected_rows=4736, expected_episodes=88, fields=advantage_fields, label="A50",
    )
    fold_map = finetune_episodes.set_index("uuid").heldout_fold.astype(int)
    for label, frames in (("value", predictions), ("A50", advantages)):
        for model, frame in frames.items():
            expected_fold = frame.uuid.map(fold_map)
            if expected_fold.isna().any() or not np.array_equal(
                expected_fold.to_numpy(dtype=np.int64), frame.heldout_fold.to_numpy(dtype=np.int64)
            ):
                raise RuntimeError(f"{label} {model} held-out fold provenance failed")

    available_groups = sorted(predictions["baseline"].dataset_name.astype(str).unique())
    if len(available_groups) != 15:
        raise RuntimeError(f"Expected 15 successful rollout groups, got {len(available_groups)}")
    omitted = omitted_by_model["baseline"]
    if omitted != ["rollout_fr3_plug_pi05_base5k_20260820_162431"]:
        raise RuntimeError(f"Unexpected zero-success groups: {omitted}")
    groups = resolve_selected_groups(available_groups, dataset_names)
    excluded_by_selection = sorted(set(available_groups) - set(groups))
    for model in ("baseline", "proposed"):
        predictions[model] = predictions[model].loc[
            predictions[model].dataset_name.isin(groups)
        ].copy()
        advantages[model] = advantages[model].loc[
            advantages[model].dataset_name.isin(groups)
        ].copy()

    if set(groups) == set(THREE_GROUP_EXPECTED):
        for group, (expected_episodes, expected_anchors, expected_windows) in THREE_GROUP_EXPECTED.items():
            prediction = predictions["baseline"].loc[
                predictions["baseline"].dataset_name == group
            ]
            advantage = advantages["baseline"].loc[
                advantages["baseline"].dataset_name == group
            ]
            actual = (prediction.uuid.nunique(), len(prediction), len(advantage))
            expected = (expected_episodes, expected_anchors, expected_windows)
            if actual != expected:
                raise RuntimeError(f"{group} selected coverage {actual} != {expected}")

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_summaries: dict[str, dict[str, dict[str, CurveSummary]]] = {}
    effect_scales: set[float] = set()
    advantage_extent = 0.0
    for group in groups:
        group_summaries[group] = {}
        for model in ("baseline", "proposed"):
            prediction = predictions[model].loc[predictions[model].dataset_name == group]
            advantage = advantages[model].loc[advantages[model].dataset_name == group]
            if prediction.empty or advantage.empty:
                raise RuntimeError(f"{group} {model} has no success value/A50 data")
            rows, summaries = _curve_rows(group, model, prediction, advantage)
            curve_rows.extend(rows)
            summary_rows.append(_summary_row(group, model, prediction, advantage))
            group_summaries[group][model] = summaries
            effect_scales.update(advantage.advantage_scale.astype(float).unique())
            summary = summaries["advantage"]
            visible = summary.visible
            advantage_extent = max(
                advantage_extent,
                float(np.nanmax(np.abs(summary.mean[visible] - summary.se[visible]))),
                float(np.nanmax(np.abs(summary.mean[visible] + summary.se[visible]))),
            )
        left = predictions["baseline"].loc[predictions["baseline"].dataset_name == group]
        right = predictions["proposed"].loc[predictions["proposed"].dataset_name == group]
        if not np.allclose(left.target.to_numpy(), right.target.to_numpy(), rtol=0.0, atol=0.0):
            raise RuntimeError(f"{group} baseline/proposed targets differ")
    if len(effect_scales) != 1:
        raise RuntimeError(f"Expected one physical A50 scale, got {effect_scales}")
    effect_scale = next(iter(effect_scales))
    y_limit = max(0.05, math.ceil(advantage_extent * 1.08 / 0.05) * 0.05)

    curve_frame = pd.DataFrame(curve_rows)
    summary_frame = pd.DataFrame(summary_rows)
    totals_by_model = {
        str(model): {
            "episodes": int(frame.episodes.sum()),
            "value_anchors": int(frame.value_anchors.sum()),
            "a50_windows": int(frame.a50_windows.sum()),
        }
        for model, frame in summary_frame.groupby("model", sort=True)
    }
    if totals_by_model.get("baseline") != totals_by_model.get("proposed"):
        raise RuntimeError("Baseline/proposed selected totals differ")
    totals = totals_by_model["baseline"]
    _atomic_parquet(
        pa.Table.from_pandas(curve_frame, preserve_index=False), output_root / "group_curve_data.parquet"
    )
    _atomic_parquet(
        pa.Table.from_pandas(summary_frame, preserve_index=False), output_root / "group_summary.parquet"
    )

    combined_value_path = assets_root / "all-rollout-groups-value.svg"
    combined_advantage_path = assets_root / "all-rollout-groups-advantage.svg"
    pooled_target = episode_curve_summary(predictions["baseline"], "target")
    _render_combined_value(groups, group_summaries, pooled_target, combined_value_path)
    _render_combined_advantage(
        groups,
        group_summaries,
        effect_scale=effect_scale,
        y_limit=y_limit,
        path=combined_advantage_path,
    )
    assets: list[dict[str, Any]] = [
        {
            "kind": kind,
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
        }
        for kind, path in (
            ("combined_value", combined_value_path),
            ("combined_advantage", combined_advantage_path),
        )
    ]
    sections: list[str] = []
    for group in groups:
        episodes = int(
            predictions["baseline"].loc[predictions["baseline"].dataset_name == group, "uuid"].nunique()
        )
        metrics = summary_frame.loc[summary_frame.dataset_name == group]
        metric_columns = [
            "model", "episodes", "value_anchors", "a50_windows", "value_episode_macro_mae",
            "value_macro_spearman", "temporal_monotonicity", "a50_mean", "a50_std",
            "a50_positive_rate",
        ]
        table_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in row) + "</tr>"
            for row in metrics[metric_columns].itertuples(index=False, name=None)
        )
        sections.append(
            f'<section id="{html.escape(group)}"><h2>{html.escape(group)}</h2>'
            f'<p>{episodes} success episodes；五折 held-out OOF，step {step}。</p>'
            f'<table><thead><tr>{"".join(f"<th>{html.escape(column)}</th>" for column in metric_columns)}</tr></thead>'
            f'<tbody>{table_rows}</tbody></table></section>'
        )

    overview_columns = [
        "dataset_name", "model", "episodes", "value_anchors", "a50_windows",
        "value_episode_macro_mae", "value_macro_spearman", "temporal_monotonicity",
        "a50_mean", "a50_std", "a50_positive_rate",
    ]
    overview_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in row) + "</tr>"
        for row in summary_frame[overview_columns].itertuples(index=False, name=None)
    )
    navigation = "".join(
        f'<li><a href="#{html.escape(group)}">{html.escape(group)}</a></li>' for group in groups
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OpenPI rollout success value and A50 curves</title>
<style>body{{font-family:system-ui;margin:30px;max-width:1450px;color:#333b49}}h1,h2{{scroll-margin-top:16px}}nav ul{{columns:2}}table{{border-collapse:collapse;font-size:12px;margin:14px 0 30px}}th,td{{border:1px solid #d3cec2;padding:6px 8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto;display:block;margin:10px 0 18px}}section{{border-top:2px solid #e6e0d2;padding-top:12px;margin-top:34px}}code{{background:#f4f1e9;padding:2px 5px}}</style></head><body>
<h1>OpenPI 各 Rollout 成功组：平均价值曲线与精确 A50 曲线</h1>
<p>仅使用 step 2000 五折 held-out OOF。当前选择 {len(groups)} 个 rollout 组、{totals['episodes']} 个成功 episodes；每模型 {totals['value_anchors']:,} 个 value anchors 和 {totals['a50_windows']:,} 个精确 A50 windows。所有曲线均先按 episode 插值，再进行 episode-equal 平均。</p>
<p>已完全排除 <code>{DEMONSTRATION_DATASET}</code>、所有 failure episodes，以及未被显式选择的其他 rollout 组。</p>
<p>{len(groups)} 个 rollout 组叠加在同一个坐标轴中：颜色区分 rollout，虚线为 Baseline，实线为 Proposed。共享图只画 episode-equal 平均线；逐点 SE 和有效 episode 数保存在 <code>group_curve_data.parquet</code>。A50 严格匹配相隔 100 个 source frames 的 anchors。</p>
<h2>所有 Rollout 组的平均价值曲线（同一坐标轴）</h2>{_inline_svg(combined_value_path)}
<h2>所有 Rollout 组的平均优势曲线（同一坐标轴）</h2>{_inline_svg(combined_advantage_path)}
<nav><h2>Rollout 导航</h2><ul>{navigation}</ul></nav>
<h2>全组汇总</h2><table><thead><tr>{''.join(f'<th>{html.escape(column)}</th>' for column in overview_columns)}</tr></thead><tbody>{overview_rows}</tbody></table>
{''.join(sections)}</body></html>"""
    html_path = output_root / "openpi_rollout_success_group_value_advantage_curves.html"
    temporary = html_path.with_suffix(html_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(html_path)

    manifest = {
        "protocol": "step-2000 5-fold held-out OOF, rollout success groups only",
        "step": int(step),
        "groups": groups,
        "omitted_zero_success_groups": omitted,
        "excluded_by_selection": excluded_by_selection,
        "demonstration_dataset_excluded": DEMONSTRATION_DATASET,
        "totals_per_model": totals,
        "curve_protocol": f"{len(groups)} group means overlaid on one value axis and one A50 axis; episode-equal interpolation on 101 progress points, no extrapolation; SE retained in parquet",
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "assets": assets,
        "outputs": {
            "html": {"path": html_path.name, "sha256": sha256_file(html_path)},
            "curve_data": {"path": "group_curve_data.parquet", "sha256": sha256_file(output_root / "group_curve_data.parquet")},
            "summary": {"path": "group_summary.parquet", "sha256": sha256_file(output_root / "group_summary.parquet")},
        },
    }
    atomic_json(output_root / "report_manifest.json", manifest)
    print(f"Rollout success curve report: {html_path}", flush=True)
    return html_path
