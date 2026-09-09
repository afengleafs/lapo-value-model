from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .common import atomic_json, json_ready
from .config import load_config
from .metrics import compute_stage_auc
from .openpi_eval import _atomic_parquet
from .openpi_finetune import (
    _ft_cfg,
    checkpoint_steps,
    finetune_artifact_root,
    finetune_output_root,
)


def _auc(outcomes: pd.Series | np.ndarray, values: pd.Series | np.ndarray) -> float:
    labels = np.asarray(outcomes) == "success"
    if np.unique(labels).size != 2:
        return float("nan")
    return float(roc_auc_score(labels.astype(np.int64), np.asarray(values, dtype=np.float64)))


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def episode_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for uuid, group in frame.groupby("uuid", sort=True):
        ordered = group.sort_values("anchor")
        rows.append(
            {
                "uuid": uuid,
                "dataset_name": ordered.dataset_name.iloc[0],
                "outcome": ordered.outcome.iloc[0],
                "task_family": ordered.task_family.iloc[0],
                "first_prediction": float(ordered.prediction.iloc[0]),
                "mean_prediction": float(ordered.prediction.mean()),
                "last_prediction": float(ordered.prediction.iloc[-1]),
                "first_target": float(ordered.target.iloc[0]),
                "mean_target": float(ordered.target.mean()),
                "last_target": float(ordered.target.iloc[-1]),
                "anchors": len(ordered),
            }
        )
    return pd.DataFrame(rows)


def build_advantage_frame(
    frame: pd.DataFrame,
    *,
    tmax_by_family: dict[str, float],
    horizon_model_frames: int,
    model_fps: int,
) -> pd.DataFrame:
    source_fps_values = set(frame.source_fps.astype(int).tolist())
    if len(source_fps_values) != 1:
        raise ValueError(f"Mixed source FPS: {source_fps_values}")
    source_fps = next(iter(source_fps_values))
    horizon_source = round(horizon_model_frames * source_fps / model_fps)
    current = frame.copy()
    future = frame[["uuid", "anchor", "prediction"]].rename(
        columns={"prediction": "future_prediction"}
    )
    future["anchor"] = future.anchor.astype(np.int64) - horizon_source
    advantage = current.merge(future, on=["uuid", "anchor"], how="inner", validate="one_to_one")
    scales = advantage.task_family.map(tmax_by_family)
    if scales.isna().any():
        missing = sorted(set(advantage.loc[scales.isna(), "task_family"]))
        raise KeyError(f"Missing T_max for task families: {missing}")
    advantage["reward_sum"] = -float(horizon_model_frames) / scales.astype(float)
    advantage["advantage"] = (
        advantage.reward_sum + advantage.future_prediction - advantage.prediction
    )
    advantage["advantage_scale"] = -advantage.reward_sum
    advantage["horizon_model_frames"] = int(horizon_model_frames)
    advantage["horizon_source_frames"] = int(horizon_source)
    return advantage


def _gap(episodes: pd.DataFrame, field: str) -> float:
    success = episodes.loc[episodes.outcome == "success", field]
    failure = episodes.loc[episodes.outcome == "failure", field]
    return float(success.mean() - failure.mean())


def _rate(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values > threshold)) if len(values) else float("nan")


def _advantage_point_metrics(advantage: pd.DataFrame) -> dict[str, Any]:
    values = advantage.advantage.to_numpy(dtype=np.float64)
    epsilon = float(np.quantile(values, 0.70))
    success = advantage.loc[advantage.outcome == "success"]
    failure = advantage.loc[advantage.outcome == "failure"]
    failure_tail = failure.loc[failure.progress >= 0.8]
    scale_mean = float(advantage.advantage_scale.mean())
    sigma = float(np.std(values))
    by_family: dict[str, Any] = {}
    for family, group in advantage.groupby("task_family", sort=True):
        family_sigma = float(group.advantage.std(ddof=0))
        family_scale = float(group.advantage_scale.iloc[0])
        by_family[str(family)] = {
            "windows": len(group),
            "sigma": family_sigma,
            "scale": family_scale,
            "sigma_over_scale": family_sigma / family_scale,
        }
    return {
        "windows": len(advantage),
        "sigma": sigma,
        "scale_mean": scale_mean,
        "sigma_over_scale_mean": sigma / scale_mean,
        "mean_success": float(success.advantage.mean()),
        "mean_failure": float(failure.advantage.mean()),
        "epsilon_q70": epsilon,
        "positive_rate_success": _rate(success.advantage.to_numpy(), epsilon),
        "positive_rate_failure": _rate(failure.advantage.to_numpy(), epsilon),
        "positive_rate_failure_tail20": _rate(
            failure_tail.advantage.to_numpy(), epsilon
        ),
        "failure_tail_windows": len(failure_tail),
        "by_task_family": by_family,
    }


def point_metrics(
    frame: pd.DataFrame,
    *,
    tmax_by_family: dict[str, float],
    horizon_model_frames: int,
    model_fps: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {
        "sample_key",
        "uuid",
        "dataset_name",
        "anchor",
        "length",
        "source_fps",
        "progress",
        "outcome",
        "task_family",
        "target",
        "prediction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF predictions missing columns: {sorted(missing)}")
    if frame.empty or frame.sample_key.duplicated().any():
        raise RuntimeError("OOF prediction keys must be non-empty and unique")
    numeric = frame[["target", "prediction"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("OOF targets/predictions contain non-finite values")

    episodes = episode_summary(frame)
    advantage = build_advantage_frame(
        frame,
        tmax_by_family=tmax_by_family,
        horizon_model_frames=horizon_model_frames,
        model_fps=model_fps,
    )
    errors = frame.prediction.to_numpy() - frame.target.to_numpy()
    correlations: list[float] = []
    monotonic: list[bool] = []
    for _, group in frame.loc[frame.outcome == "success"].groupby("uuid", sort=False):
        ordered = group.sort_values("anchor")
        correlation = _safe_spearman(
            ordered.prediction.to_numpy(dtype=np.float64),
            ordered.target.to_numpy(dtype=np.float64),
        )
        if correlation is not None:
            correlations.append(correlation)
        monotonic.extend((np.diff(ordered.prediction.to_numpy(dtype=np.float64)) >= 0).tolist())
    metrics = {
        "anchors": len(frame),
        "episodes": len(episodes),
        "episode_outcomes": dict(Counter(episodes.outcome)),
        "value": {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "success_episode_macro_spearman": (
                float(np.mean(correlations)) if correlations else 0.0
            ),
            "success_temporal_monotonicity": (
                float(np.mean(monotonic)) if monotonic else 0.0
            ),
        },
        "episode_auc": {
            "first_valid_anchor": _auc(episodes.outcome, episodes.first_prediction),
            "mean_value": _auc(episodes.outcome, episodes.mean_prediction),
            "last_value": _auc(episodes.outcome, episodes.last_prediction),
        },
        "success_minus_failure_gap": {
            "first_valid_anchor": _gap(episodes, "first_prediction"),
            "mean_value": _gap(episodes, "mean_prediction"),
            "last_value": _gap(episodes, "last_prediction"),
        },
        "stage_matched_auc": compute_stage_auc(frame, stages=4),
        "advantage_a50": _advantage_point_metrics(advantage),
    }
    return metrics, episodes, advantage


def _bootstrap_ci(
    episodes: pd.DataFrame,
    advantage: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    episode_lookup = episodes.set_index("uuid", drop=False)
    success_ids = episodes.loc[episodes.outcome == "success", "uuid"].to_numpy()
    failure_ids = episodes.loc[episodes.outcome == "failure", "uuid"].to_numpy()
    advantage_lookup = {
        str(uuid): {
            "all": group.advantage.to_numpy(dtype=np.float64),
            "tail": group.loc[group.progress >= 0.8, "advantage"].to_numpy(dtype=np.float64),
            "outcome": str(group.outcome.iloc[0]),
        }
        for uuid, group in advantage.groupby("uuid", sort=False)
    }
    names = (
        "auc_first_valid_anchor",
        "auc_mean_value",
        "auc_last_value",
        "gap_first_valid_anchor",
        "gap_mean_value",
        "gap_last_value",
        "advantage_mean_success",
        "advantage_mean_failure",
        "positive_rate_success",
        "positive_rate_failure",
        "positive_rate_failure_tail20",
    )
    draws: dict[str, list[float]] = {name: [] for name in names}
    fields = ("first_prediction", "mean_prediction", "last_prediction")
    for _ in range(samples):
        selected_success = success_ids[rng.integers(0, len(success_ids), size=len(success_ids))]
        selected_failure = failure_ids[rng.integers(0, len(failure_ids), size=len(failure_ids))]
        success_rows = episode_lookup.loc[selected_success]
        failure_rows = episode_lookup.loc[selected_failure]
        labels = np.concatenate(
            (np.ones(len(success_rows), dtype=np.int64), np.zeros(len(failure_rows), dtype=np.int64))
        )
        for label, field in zip(("first_valid_anchor", "mean_value", "last_value"), fields):
            success_values = success_rows[field].to_numpy(dtype=np.float64)
            failure_values = failure_rows[field].to_numpy(dtype=np.float64)
            combined = np.concatenate((success_values, failure_values))
            draws[f"auc_{label}"].append(float(roc_auc_score(labels, combined)))
            draws[f"gap_{label}"].append(float(success_values.mean() - failure_values.mean()))

        success_adv = [
            advantage_lookup[str(uuid)]["all"]
            for uuid in selected_success
            if str(uuid) in advantage_lookup
        ]
        failure_adv = [
            advantage_lookup[str(uuid)]["all"]
            for uuid in selected_failure
            if str(uuid) in advantage_lookup
        ]
        failure_tail = [
            advantage_lookup[str(uuid)]["tail"]
            for uuid in selected_failure
            if str(uuid) in advantage_lookup and len(advantage_lookup[str(uuid)]["tail"])
        ]
        success_values = np.concatenate(success_adv)
        failure_values = np.concatenate(failure_adv)
        tail_values = np.concatenate(failure_tail) if failure_tail else np.empty(0)
        combined_advantage = np.concatenate((success_values, failure_values))
        epsilon = float(np.quantile(combined_advantage, 0.70))
        draws["advantage_mean_success"].append(float(success_values.mean()))
        draws["advantage_mean_failure"].append(float(failure_values.mean()))
        draws["positive_rate_success"].append(_rate(success_values, epsilon))
        draws["positive_rate_failure"].append(_rate(failure_values, epsilon))
        draws["positive_rate_failure_tail20"].append(_rate(tail_values, epsilon))
    return {
        name: {
            "ci95": [
                float(np.nanquantile(values, 0.025)),
                float(np.nanquantile(values, 0.975)),
            ],
            "bootstrap_samples": samples,
        }
        for name, values in draws.items()
    }


def compute_recap_metrics(
    frame: pd.DataFrame,
    *,
    tmax_by_family: dict[str, float],
    horizon_model_frames: int = 50,
    model_fps: int = 15,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics, episodes, advantage = point_metrics(
        frame,
        tmax_by_family=tmax_by_family,
        horizon_model_frames=horizon_model_frames,
        model_fps=model_fps,
    )
    metrics["bootstrap_ci95"] = _bootstrap_ci(
        episodes, advantage, samples=bootstrap_samples, seed=seed
    )
    metrics["bootstrap_protocol"] = "outcome-stratified episode resampling"
    return metrics, advantage


def source_success_calibration(frame: pd.DataFrame) -> dict[str, Any]:
    """Compare successful episode calibration across data collections."""
    if "collection" not in frame.columns:
        return {}
    success = frame.loc[frame.outcome == "success"].copy()
    if success.empty:
        return {}
    last = (
        success.sort_values(["uuid", "anchor"])
        .groupby("uuid", sort=True, as_index=False)
        .tail(1)
    )
    collections: dict[str, Any] = {}
    for collection, group in last.groupby("collection", sort=True):
        collections[str(collection)] = {
            "episodes": int(group.uuid.nunique()),
            "mean_last_prediction": float(group.prediction.mean()),
            "mean_last_target": float(group.target.mean()),
            "last_mae": float((group.prediction - group.target).abs().mean()),
        }
    demo = collections.get("demonstration")
    rollout = collections.get("rollout")
    gap = (
        float(demo["mean_last_prediction"] - rollout["mean_last_prediction"])
        if demo is not None and rollout is not None
        else float("nan")
    )
    return {
        "collections": collections,
        "demo_minus_rollout_success_last_prediction": gap,
        "absolute_endpoint_gap": abs(gap),
    }


def _load_tmax(config: dict[str, Any]) -> dict[str, float]:
    stats = json.loads(
        (Path(config["artifact_root"]) / "manifest_stats.json").read_text(encoding="utf-8")
    )
    return {str(key): float(value) for key, value in stats["tmax"].items()}


def evaluate_openpi_oof(
    config_path: str | Path,
    *,
    bootstrap_samples: int | None = None,
    output_name: str = "openpi_finetune_oof",
) -> dict[str, Any]:
    config = load_config(config_path)
    ft_cfg = _ft_cfg(config)
    output_root = finetune_output_root(config, output_name)
    artifact_root = finetune_artifact_root(config)
    manifest_stats = json.loads(
        (artifact_root / "manifest_stats.json").read_text(encoding="utf-8")
    )
    folds = int(ft_cfg["folds"])
    steps = [0] + checkpoint_steps(int(ft_cfg["steps"]), int(ft_cfg["save_every_steps"]))
    tmax = _load_tmax(config)
    bootstrap_count = (
        int(ft_cfg["bootstrap_samples"]) if bootstrap_samples is None else int(bootstrap_samples)
    )
    expected_keys = set(
        pq.read_table(artifact_root / "samples.parquet", columns=["sample_key"])[
            "sample_key"
        ].to_pylist()
    )
    expected_episodes = int(manifest_stats["episodes"])
    summaries: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []
    for step in steps:
        frames = []
        for fold in range(folds):
            path = output_root / f"fold-{fold}" / "predictions" / f"step-{step:05d}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            fold_frame = pq.read_table(path).to_pandas()
            if set(fold_frame.heldout_fold.astype(int)) != {fold}:
                raise RuntimeError(f"Fold {fold} prediction file contains another fold")
            frames.append(fold_frame)
        frame = pd.concat(frames, ignore_index=True).sort_values("sample_key").reset_index(drop=True)
        actual_keys = set(frame.sample_key.tolist())
        if len(frame) != len(expected_keys) or frame.sample_key.nunique() != len(expected_keys):
            raise RuntimeError(f"Step {step} does not have one prediction per dense anchor")
        if actual_keys != expected_keys or frame.uuid.nunique() != expected_episodes:
            raise RuntimeError(f"Step {step} OOF coverage/leakage gate failed")
        prediction_path = output_root / "oof_predictions" / f"step-{step:05d}.parquet"
        _atomic_parquet(pa.Table.from_pandas(frame, preserve_index=False), prediction_path)
        metrics, advantage = compute_recap_metrics(
            frame,
            tmax_by_family=tmax,
            horizon_model_frames=int(ft_cfg["advantage_horizon_model_frames"]),
            model_fps=int(config["data"]["fps"]),
            bootstrap_samples=bootstrap_count,
            seed=int(config["seed"]) + step,
        )
        metrics["source_success_calibration"] = source_success_calibration(frame)
        by_collection: dict[str, Any] = {}
        for collection, subset in frame.groupby("collection", sort=True):
            if subset.outcome.nunique() == 2:
                collection_metrics, _ = compute_recap_metrics(
                    subset,
                    tmax_by_family=tmax,
                    horizon_model_frames=int(ft_cfg["advantage_horizon_model_frames"]),
                    model_fps=int(config["data"]["fps"]),
                    bootstrap_samples=bootstrap_count,
                    seed=int(config["seed"]) + step + 100_000,
                )
            else:
                collection_metrics, _, _ = point_metrics(
                    subset,
                    tmax_by_family=tmax,
                    horizon_model_frames=int(ft_cfg["advantage_horizon_model_frames"]),
                    model_fps=int(config["data"]["fps"]),
                )
                collection_metrics["bootstrap_ci95"] = {}
                collection_metrics["bootstrap_protocol"] = (
                    "not run: collection contains only one outcome"
                )
            by_collection[str(collection)] = collection_metrics
        metrics["by_collection"] = by_collection
        expected_windows = int(manifest_stats["advantage_windows_exact"])
        if len(advantage) != expected_windows:
            raise RuntimeError(
                f"Step {step} exact A50 windows {len(advantage)} != {expected_windows}"
            )
        advantage_path = output_root / "oof_advantages" / f"step-{step:05d}.parquet"
        _atomic_parquet(pa.Table.from_pandas(advantage, preserve_index=False), advantage_path)
        metrics_path = output_root / "oof_metrics" / f"step-{step:05d}.json"
        atomic_json(metrics_path, metrics)
        summaries[str(step)] = metrics
        curve_rows.append(
            {
                "step": step,
                "mae": metrics["value"]["mae"],
                "rmse": metrics["value"]["rmse"],
                "macro_spearman": metrics["value"]["success_episode_macro_spearman"],
                "first_auc": metrics["episode_auc"]["first_valid_anchor"],
                "mean_auc": metrics["episode_auc"]["mean_value"],
                "last_auc": metrics["episode_auc"]["last_value"],
                "stage_auc": metrics["stage_matched_auc"]["macro_auc"],
                "advantage_sigma": metrics["advantage_a50"]["sigma"],
                "advantage_sigma_over_scale": metrics["advantage_a50"][
                    "sigma_over_scale_mean"
                ],
                "positive_rate_success": metrics["advantage_a50"][
                    "positive_rate_success"
                ],
                "positive_rate_failure": metrics["advantage_a50"][
                    "positive_rate_failure"
                ],
                "positive_rate_failure_tail20": metrics["advantage_a50"][
                    "positive_rate_failure_tail20"
                ],
            }
        )
        print(f"OOF metrics complete for step {step}/{steps[-1]}", flush=True)
    curve_frame = pd.DataFrame(curve_rows)
    _atomic_parquet(
        pa.Table.from_pandas(curve_frame, preserve_index=False), output_root / "curve_data.parquet"
    )
    report = {
        "protocol": "5-fold episode-level OOF fine-tuning",
        "selection_policy": (
            f"step {int(ft_cfg['steps'])} is the preregistered primary result; "
            "intermediate checkpoints are learning-curve diagnostics only"
        ),
        "openpi_status": "OOF fine-tuning evaluation; no longer an untouched external holdout",
        "training_collections": manifest_stats.get("training_collections", ["rollout"]),
        "demonstration_episodes_included": int(
            manifest_stats.get("demonstration_episodes_included", 0)
        ),
        "folds": folds,
        "steps": steps,
        "anchors_per_step": len(expected_keys),
        "episodes_per_step": expected_episodes,
        "advantage_windows_per_step": int(manifest_stats["advantage_windows_exact"]),
        "bootstrap_samples": bootstrap_count,
        "metrics_by_step": summaries,
    }
    atomic_json(output_root / "oof_report.json", report)
    return report


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "—"


def report_openpi_oof(
    config_path: str | Path, *, output_name: str = "openpi_finetune_oof"
) -> Path:
    config = load_config(config_path)
    root = finetune_output_root(config, output_name)
    report_path = root / "oof_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for step in report["steps"]:
        metrics = report["metrics_by_step"][str(step)]
        rows.append(
            "<tr>"
            f"<td>{step}</td>"
            f"<td>{_fmt(metrics['value']['mae'])}</td>"
            f"<td>{_fmt(metrics['value']['rmse'])}</td>"
            f"<td>{_fmt(metrics['value']['success_episode_macro_spearman'])}</td>"
            f"<td>{_fmt(metrics['episode_auc']['first_valid_anchor'])}</td>"
            f"<td>{_fmt(metrics['episode_auc']['mean_value'])}</td>"
            f"<td>{_fmt(metrics['episode_auc']['last_value'])}</td>"
            f"<td>{_fmt(metrics['stage_matched_auc']['macro_auc'])}</td>"
            f"<td>{_fmt(metrics['advantage_a50']['sigma'])}</td>"
            f"<td>{_fmt(metrics['advantage_a50']['sigma_over_scale_mean'])}</td>"
            f"<td>{_fmt(metrics['advantage_a50']['positive_rate_success'])}</td>"
            f"<td>{_fmt(metrics['advantage_a50']['positive_rate_failure'])}</td>"
            f"<td>{_fmt(metrics['advantage_a50']['positive_rate_failure_tail20'])}</td>"
            "</tr>"
        )
    primary_step = str(max(int(step) for step in report["steps"]))
    primary = report["metrics_by_step"][primary_step]
    primary_json = html.escape(json.dumps(json_ready(primary), ensure_ascii=False, indent=2))
    outcomes = report["metrics_by_step"][str(report["steps"][0])]["episode_outcomes"]
    collections = ", ".join(report.get("training_collections", ["rollout"]))
    diagnostic_steps = "、".join(str(step) for step in report["steps"][:-1]) or "无"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OpenPI 5-fold OOF 价值模型微调报告</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:32px;color:#27313d;background:#fbfaf7}}
h1{{font-size:28px}} .note{{padding:12px 16px;background:#fff2cf;border-left:4px solid #c77a00;line-height:1.6}}
table{{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin-top:18px}}
th,td{{border:1px solid #d6d3cb;padding:6px 9px;text-align:right;white-space:nowrap}} th{{background:#eeece6}}
pre{{background:#f0eee8;padding:16px;overflow:auto;font-size:11px}} code{{font-family:ui-monospace,monospace}}
</style></head><body>
<h1>OpenPI {report['episodes_per_step']} episodes · 5 折 OOF · 2k 微调</h1>
<p>{int(outcomes.get('success', 0))} success / {int(outcomes.get('failure', 0))} failure；训练集合：{html.escape(collections)}；每个 episode 只在未见过它的 fold 模型上打分。</p>
<div class="note"><b>主结果锁定 step {primary_step}</b>。step {diagnostic_steps} 仅用于学习曲线，不能据此挑 checkpoint。OpenPI 已参与微调，因此结果属于 OOF 微调评估，不再称为 untouched holdout。</div>
<p>A50 使用严格物理定义：50 个 15 Hz model frames = 100 个 30 Hz source frames；每步共有 {report['advantage_windows_per_step']:,} 个精确配对窗口。positive 阈值为当步全部 OOF A50 的第 70 百分位。</p>
<table><thead><tr><th>step</th><th>V MAE</th><th>V RMSE</th><th>成功 macro-ρ</th><th>首 anchor AUC</th><th>均值 AUC</th><th>末值 AUC</th><th>4-stage AUC</th><th>σ(A50)</th><th>σ/标尺</th><th>成功 positive</th><th>失败 positive</th><th>失败尾20%</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>step {primary_step} 完整指标（含 2,000 次 episode bootstrap CI）</h2><pre>{primary_json}</pre>
</body></html>"""
    output_path = root / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".html.tmp.{Path.cwd().stat().st_ino}")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


_COMPARISON_METRICS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("MAE", ("value", "mae"), "lower"),
    ("RMSE", ("value", "rmse"), "lower"),
    (
        "Success macro-Spearman",
        ("value", "success_episode_macro_spearman"),
        "higher",
    ),
    (
        "Temporal monotonicity",
        ("value", "success_temporal_monotonicity"),
        "higher",
    ),
    ("First-anchor AUC", ("episode_auc", "first_valid_anchor"), "higher"),
    ("Mean-value AUC", ("episode_auc", "mean_value"), "higher"),
    ("Last-value AUC", ("episode_auc", "last_value"), "higher"),
    ("4-stage AUC", ("stage_matched_auc", "macro_auc"), "higher"),
    ("A50 sigma", ("advantage_a50", "sigma"), "lower"),
    (
        "A50 sigma/scale",
        ("advantage_a50", "sigma_over_scale_mean"),
        "lower",
    ),
    (
        "A50 success positive rate",
        ("advantage_a50", "positive_rate_success"),
        "higher",
    ),
    (
        "A50 failure positive rate",
        ("advantage_a50", "positive_rate_failure"),
        "lower",
    ),
    (
        "A50 failure tail20 positive rate",
        ("advantage_a50", "positive_rate_failure_tail20"),
        "lower",
    ),
)


def _nested_metric(metrics: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = metrics
    for key in path:
        value = value[key]
    return float(value)


def _load_oof_training_plans(
    root: Path, *, folds: int, final_step: int
) -> list[dict[str, Any]]:
    plans = []
    for fold in range(folds):
        path = root / f"fold-{fold}" / "checkpoints" / f"step-{final_step:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        plan = checkpoint.get("training_plan")
        if not isinstance(plan, dict):
            raise RuntimeError(f"Missing training_plan in {path}")
        if int(plan.get("fold", -1)) != fold:
            raise RuntimeError(f"Fold identity mismatch in {path}")
        plans.append(plan)
    return plans


def _gate_oof_report(
    root: Path,
    report: dict[str, Any],
    *,
    expected_steps: list[int],
    expected_keys: set[str],
    expected_episodes: int,
) -> None:
    if [int(step) for step in report["steps"]] != expected_steps:
        raise RuntimeError(f"Unexpected checkpoint steps in {root}")
    if int(report["folds"]) != 5 or int(report["episodes_per_step"]) != expected_episodes:
        raise RuntimeError(f"Unexpected fold/episode count in {root}")
    if int(report["anchors_per_step"]) != len(expected_keys):
        raise RuntimeError(f"Unexpected anchor count in {root}")
    for step in expected_steps:
        path = root / "oof_predictions" / f"step-{step:05d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pq.read_table(path, columns=["sample_key", "uuid"] ).to_pandas()
        if (
            len(frame) != len(expected_keys)
            or frame.sample_key.nunique() != len(expected_keys)
            or set(frame.sample_key) != expected_keys
            or frame.uuid.nunique() != expected_episodes
        ):
            raise RuntimeError(f"Incomplete OOF coverage in {path}")


def compare_openpi_oof(
    config_path: str | Path,
    *,
    baseline_output_name: str = "openpi_finetune_oof",
    proposed_output_name: str = "openpi_finetune_oof_proposed",
    comparison_output_name: str = "openpi_finetune_oof_comparison",
) -> dict[str, Any]:
    """Gate and compare two complete OpenPI OOF learning curves."""
    config = load_config(config_path)
    ft_cfg = _ft_cfg(config)
    baseline_root = finetune_output_root(config, baseline_output_name)
    proposed_root = finetune_output_root(config, proposed_output_name)
    output_root = finetune_output_root(config, comparison_output_name)
    if len({baseline_root, proposed_root, output_root}) != 3:
        raise ValueError("Baseline, proposed, and comparison output names must differ")
    reports = {}
    for model, root in (("baseline", baseline_root), ("proposed", proposed_root)):
        path = root / "oof_report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[model] = json.loads(path.read_text(encoding="utf-8"))

    expected_steps = [0] + checkpoint_steps(
        int(ft_cfg["steps"]), int(ft_cfg["save_every_steps"])
    )
    artifact_root = finetune_artifact_root(config)
    expected_keys = set(
        pq.read_table(artifact_root / "samples.parquet", columns=["sample_key"])[
            "sample_key"
        ].to_pylist()
    )
    manifest_stats = json.loads(
        (artifact_root / "manifest_stats.json").read_text(encoding="utf-8")
    )
    expected_episodes = int(manifest_stats["episodes"])
    for model, root in (("baseline", baseline_root), ("proposed", proposed_root)):
        _gate_oof_report(
            root,
            reports[model],
            expected_steps=expected_steps,
            expected_keys=expected_keys,
            expected_episodes=expected_episodes,
        )
    report_gate_fields = (
        "folds",
        "steps",
        "anchors_per_step",
        "episodes_per_step",
        "advantage_windows_per_step",
    )
    for field in report_gate_fields:
        if reports["baseline"][field] != reports["proposed"][field]:
            raise RuntimeError(f"OOF report mismatch for {field}")

    folds = int(ft_cfg["folds"])
    final_step = int(ft_cfg["steps"])
    plans = {
        "baseline": _load_oof_training_plans(
            baseline_root, folds=folds, final_step=final_step
        ),
        "proposed": _load_oof_training_plans(
            proposed_root, folds=folds, final_step=final_step
        ),
    }
    plan_gate_fields = (
        "protocol",
        "folds",
        "steps",
        "save_every_steps",
        "checkpoint_steps",
        "world_size",
        "micro_batch_size",
        "global_batch_size",
        "grad_accum_steps",
        "sampler",
        "manifest_hash",
        "adapter_learning_rate",
        "head_learning_rate",
        "warmup_steps",
        "scheduler",
    )
    reference = plans["baseline"][0]
    for model, model_plans in plans.items():
        for plan in model_plans:
            for field in plan_gate_fields:
                if plan[field] != reference[field]:
                    raise RuntimeError(f"Training-plan mismatch for {field}")
            if float(plan["lambda_latent"]) != float(model_plans[0]["lambda_latent"]):
                raise RuntimeError(f"{model} folds disagree on lambda_latent")
            for field in (
                "architecture_version",
                "value_path",
                "value_feature_dim",
                "initialization_policy",
                "reset_modules",
            ):
                if plan.get(field) != model_plans[0].get(field):
                    raise RuntimeError(f"{model} folds disagree on {field}")

    rows: list[dict[str, Any]] = []
    for step in expected_steps:
        baseline_metrics = reports["baseline"]["metrics_by_step"][str(step)]
        proposed_metrics = reports["proposed"]["metrics_by_step"][str(step)]
        for metric_name, path, direction in _COMPARISON_METRICS:
            baseline_value = _nested_metric(baseline_metrics, path)
            proposed_value = _nested_metric(proposed_metrics, path)
            delta = proposed_value - baseline_value
            if np.isclose(delta, 0.0, equal_nan=True):
                winner = "tie"
            elif (direction == "higher" and delta > 0) or (
                direction == "lower" and delta < 0
            ):
                winner = "proposed"
            else:
                winner = "baseline"
            rows.append(
                {
                    "step": step,
                    "is_primary": step == final_step,
                    "metric": metric_name,
                    "direction": direction,
                    "baseline": baseline_value,
                    "proposed": proposed_value,
                    "proposed_minus_baseline": delta,
                    "winner": winner,
                }
            )
    frame = pd.DataFrame(rows)
    initializations = {
        model: [
            {
                "fold": fold,
                "checkpoint": plan["starting_checkpoint"],
                "sha256": plan["starting_checkpoint_sha256"],
                "value_path": plan.get("value_path", "direct"),
                "value_feature_dim": plan.get("value_feature_dim", 2048),
                "initialization_policy": plan.get(
                    "initialization_policy", "full_checkpoint"
                ),
                "reset_modules": plan.get("reset_modules", []),
            }
            for fold, plan in enumerate(model_plans)
        ]
        for model, model_plans in plans.items()
    }
    result = {
        "protocol": "paired OpenPI 5-fold episode-level OOF comparison",
        "primary_step": final_step,
        "baseline_output": str(baseline_root),
        "proposed_output": str(proposed_root),
        "comparison_caveat": (
            "Baseline uses its fully pretrained DROID value head while Proposed "
            "uses a newly initialized latent-hidden value head. The comparison "
            "therefore measures the complete architecture/initialization packages, "
            "not the isolated causal effect of architecture alone."
        ),
        "gates": {
            "passed": True,
            "folds": folds,
            "steps": expected_steps,
            "episodes_per_step": expected_episodes,
            "anchors_per_step": len(expected_keys),
            "advantage_windows_per_step": reports["baseline"][
                "advantage_windows_per_step"
            ],
            "manifest_hash": reference["manifest_hash"],
            "matched_training_plan_fields": list(plan_gate_fields),
            "lambda_latent": {
                model: float(model_plans[0]["lambda_latent"])
                for model, model_plans in plans.items()
            },
            "value_path": {
                model: model_plans[0].get("value_path", "direct")
                for model, model_plans in plans.items()
            },
            "initialization_policy": {
                model: model_plans[0].get(
                    "initialization_policy", "full_checkpoint"
                )
                for model, model_plans in plans.items()
            },
        },
        "initializations": initializations,
        "rows": frame.to_dict("records"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "openpi_oof_model_comparison.json", result)
    _atomic_parquet(
        pa.Table.from_pandas(frame, preserve_index=False),
        output_root / "openpi_oof_model_comparison.parquet",
    )
    primary_rows = frame.loc[frame.is_primary]
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row.metric)}</td><td>{row.direction}</td>"
        f"<td>{_fmt(row.baseline)}</td><td>{_fmt(row.proposed)}</td>"
        f"<td>{_fmt(row.proposed_minus_baseline)}</td><td>{row.winner}</td></tr>"
        for row in primary_rows.itertuples()
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OpenPI OOF baseline vs proposed</title><style>
body{{font-family:system-ui,sans-serif;margin:32px;color:#27313d;background:#fbfaf7}}
table{{border-collapse:collapse}}th,td{{border:1px solid #d6d3cb;padding:7px 10px;text-align:right}}
th{{background:#eeece6}}td:first-child{{text-align:left}}
</style></head><body><h1>OpenPI 五折 OOF：baseline vs proposed</h1>
<p>所有完整性与训练预算门禁已通过；预注册主结果为 step {final_step}，delta = proposed − baseline。</p>
<p><b>解释限制：</b>Baseline 完整加载 DROID Value Head；Proposed 使用新初始化的 128D 串联 Value Head。因此结果比较的是完整架构与初始化方案，不是架构的单一因果效应；step 0 不用于判断架构优劣。</p>
<table><thead><tr><th>指标</th><th>方向</th><th>baseline</th><th>proposed</th><th>delta</th><th>winner</th></tr></thead>
<tbody>{table_rows}</tbody></table></body></html>"""
    html_path = output_root / "openpi_oof_model_comparison.html"
    temporary = html_path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(html_path)
    return result
