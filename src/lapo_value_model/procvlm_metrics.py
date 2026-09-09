from __future__ import annotations

import html
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .common import atomic_json
from .config import load_config


TAU = 0.5
MCC_THRESHOLD = 0.95


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    pq.write_table(table, temporary)
    temporary.replace(path)


def _ft_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config["external_eval"]["openpi"]["finetune"]


def _output_root(config: dict[str, Any], output_name: str) -> Path:
    if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError(f"output_name must be one directory name, got {output_name!r}")
    return Path(config["output_root"]) / output_name


def progress_scores(values: np.ndarray | pd.Series) -> np.ndarray:
    """Map the value model's [-1, 0] range to ProcVLM progress [0, 1]."""
    result = np.clip(np.asarray(values, dtype=np.float64) + 1.0, 0.0, 1.0)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("Progress scores must be a finite one-dimensional array")
    return result


def _occupied_bins(sorted_scores: np.ndarray, k: int) -> int:
    bins = np.floor(sorted_scores * int(k)).astype(np.int64)
    np.minimum(bins, int(k) - 1, out=bins)
    return int(1 + np.count_nonzero(bins[1:] != bins[:-1]))


def epr_at_tau(
    scores: np.ndarray | pd.Series, *, tau: float = TAU, block_size: int = 256
) -> dict[str, float | int]:
    """Compute the exact EPR_tau by a descending, vectorized segmented search."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("EPR scores must be a non-empty finite vector")
    if not 0.0 < tau <= 1.0 or block_size < 1:
        raise ValueError("tau must be in (0, 1] and block_size must be positive")
    values = np.sort(np.clip(values, 0.0, 1.0))
    max_k = int(math.floor(len(values) / tau))
    for high in range(max_k, 0, -block_size):
        low = max(1, high - block_size + 1)
        ks = np.arange(high, low - 1, -1, dtype=np.int64)
        bins = np.floor(ks[:, None] * values[None, :]).astype(np.int64)
        np.minimum(bins, ks[:, None] - 1, out=bins)
        occupied = 1 + np.count_nonzero(bins[:, 1:] != bins[:, :-1], axis=1)
        feasible = occupied.astype(np.float64) / ks >= tau
        if feasible.any():
            index = int(np.flatnonzero(feasible)[0])
            k = int(ks[index])
            count = int(occupied[index])
            return {
                "epr": float(math.log2(k)),
                "k": k,
                "delta": float(1.0 / k),
                "occupied_bins": count,
                "coverage": float(count / k),
            }
    raise AssertionError("k=1 must always satisfy EPR coverage")


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("MCC labels must be aligned one-dimensional arrays")
    return np.asarray(
        [
            np.count_nonzero(truth & pred),
            np.count_nonzero(~truth & pred),
            np.count_nonzero(~truth & ~pred),
            np.count_nonzero(truth & ~pred),
        ],
        dtype=np.int64,
    )


def mcc_from_confusion(counts: np.ndarray) -> float:
    tp, fp, tn, fn = np.asarray(counts, dtype=np.float64)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def _ci(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return {"low": float(low), "high": float(high)}


def _stratified_draws(outcomes: np.ndarray, *, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    success = np.flatnonzero(outcomes == "success")
    failure = np.flatnonzero(outcomes == "failure")
    if not len(success) or not len(failure):
        raise ValueError("Bootstrap requires success and failure episodes")
    draws = np.empty((samples, len(outcomes)), dtype=np.int64)
    for row in range(samples):
        chosen = np.concatenate(
            (
                rng.choice(success, size=len(success), replace=True),
                rng.choice(failure, size=len(failure), replace=True),
            )
        )
        draws[row] = chosen
    return draws


def compute_procvlm_metrics(
    frame: pd.DataFrame, *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    required = {"sample_key", "uuid", "anchor", "outcome", "target", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF predictions missing columns: {sorted(missing)}")
    if frame.empty or frame.sample_key.duplicated().any():
        raise ValueError("OOF sample keys must be non-empty and unique")

    ordered = frame.sort_values(["uuid", "anchor", "sample_key"]).reset_index(drop=True)
    p_true = progress_scores(ordered.target)
    p_pred = progress_scores(ordered.prediction)
    pooled = epr_at_tau(p_pred)

    episode_rows: list[dict[str, Any]] = []
    frame_confusions: list[np.ndarray] = []
    for uuid, group in ordered.groupby("uuid", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        outcome_values = group.outcome.astype(str).unique()
        if len(outcome_values) != 1:
            raise ValueError(f"Episode {uuid} has inconsistent outcomes")
        episode_epr = epr_at_tau(p_pred[indices])
        frame_counts = _confusion(
            p_true[indices] >= MCC_THRESHOLD, p_pred[indices] >= MCC_THRESHOLD
        )
        frame_confusions.append(frame_counts)
        episode_rows.append(
            {
                "uuid": str(uuid),
                "outcome": str(outcome_values[0]),
                "epr": float(episode_epr["epr"]),
                "last_pred_positive": bool(p_pred[indices[-1]] >= MCC_THRESHOLD),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    outcomes = episodes.outcome.to_numpy()
    episode_truth = outcomes == "success"
    episode_pred = episodes.last_pred_positive.to_numpy(dtype=bool)
    frame_by_episode = np.stack(frame_confusions)
    frame_counts = frame_by_episode.sum(axis=0)
    episode_counts = _confusion(episode_truth, episode_pred)

    draws = _stratified_draws(outcomes, samples=bootstrap_samples, seed=seed)
    macro_samples = episodes.epr.to_numpy(dtype=np.float64)[draws].mean(axis=1)
    sampled_frame_counts = frame_by_episode[draws].sum(axis=1)
    episode_by_episode = np.stack(
        [_confusion(episode_truth[i : i + 1], episode_pred[i : i + 1]) for i in range(len(episodes))]
    )
    sampled_episode_counts = episode_by_episode[draws].sum(axis=1)
    frame_mcc_samples = np.asarray([mcc_from_confusion(row) for row in sampled_frame_counts])
    episode_mcc_samples = np.asarray([mcc_from_confusion(row) for row in sampled_episode_counts])

    def mcc_payload(
        counts: np.ndarray, bootstrap: np.ndarray, bootstrap_counts: np.ndarray
    ) -> dict[str, Any]:
        tp, fp, tn, fn = (int(value) for value in counts)
        btp, bfp, btn, bfn = np.asarray(bootstrap_counts, dtype=np.float64).T
        degenerate = (btp + bfp) * (btp + bfn) * (btn + bfp) * (btn + bfn) == 0
        return {
            "mcc": mcc_from_confusion(counts),
            "threshold": MCC_THRESHOLD,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "true_positive_rate": float((tp + fn) / counts.sum()),
            "predicted_positive_rate": float((tp + fp) / counts.sum()),
            "ci95": _ci(bootstrap),
            "degenerate_bootstrap_samples": int(np.count_nonzero(degenerate)),
        }

    return {
        "anchors": int(len(ordered)),
        "episodes": int(len(episodes)),
        "episode_outcomes": {
            "success": int(np.count_nonzero(outcomes == "success")),
            "failure": int(np.count_nonzero(outcomes == "failure")),
        },
        "epr50": {
            "pooled": pooled,
            "episode_macro": {
                "epr": float(episodes.epr.mean()),
                "ci95": _ci(macro_samples),
            },
        },
        "mcc95": {
            "frame": mcc_payload(frame_counts, frame_mcc_samples, sampled_frame_counts),
            "episode_last_anchor": mcc_payload(
                episode_counts, episode_mcc_samples, sampled_episode_counts
            ),
        },
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "protocol": "outcome-stratified episode resampling",
        },
    }


def _flatten_row(model: str, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
    frame = metrics["mcc95"]["frame"]
    episode = metrics["mcc95"]["episode_last_anchor"]
    macro = metrics["epr50"]["episode_macro"]
    return {
        "model": model,
        "step": int(step),
        "anchors": metrics["anchors"],
        "episodes": metrics["episodes"],
        "epr50_pooled": metrics["epr50"]["pooled"]["epr"],
        "epr50_pooled_k": metrics["epr50"]["pooled"]["k"],
        "epr50_episode_macro": macro["epr"],
        "epr50_episode_macro_ci_low": macro["ci95"]["low"],
        "epr50_episode_macro_ci_high": macro["ci95"]["high"],
        "mcc95_frame": frame["mcc"],
        "mcc95_frame_ci_low": frame["ci95"]["low"],
        "mcc95_frame_ci_high": frame["ci95"]["high"],
        "mcc95_frame_tp": frame["tp"],
        "mcc95_frame_fp": frame["fp"],
        "mcc95_frame_tn": frame["tn"],
        "mcc95_frame_fn": frame["fn"],
        "mcc95_episode": episode["mcc"],
        "mcc95_episode_ci_low": episode["ci95"]["low"],
        "mcc95_episode_ci_high": episode["ci95"]["high"],
        "mcc95_episode_tp": episode["tp"],
        "mcc95_episode_fp": episode["fp"],
        "mcc95_episode_tn": episode["tn"],
        "mcc95_episode_fn": episode["fn"],
    }


def _svg_curves(baseline: pd.DataFrame, proposed: pd.DataFrame) -> str:
    specs = [
        ("epr50_pooled", "Pooled EPR@50"),
        ("epr50_episode_macro", "Episode-macro EPR@50"),
        ("mcc95_frame", "Frame MCC@95"),
        ("mcc95_episode", "Episode MCC@95"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, specs, strict=True):
        axis.plot(baseline.step, baseline[column], marker="o", ms=3, label="Baseline")
        axis.plot(proposed.step, proposed[column], marker="o", ms=3, label="Proposed")
        axis.axvline(2000, color="#777", linestyle="--", linewidth=0.8)
        axis.set(title=title, xlabel="Optimizer step")
        axis.grid(alpha=0.25)
        axis.legend()
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg")
    plt.close(figure)
    return buffer.getvalue().split("<svg", 1)[1].join(("<svg", ""))


def generate_procvlm_report(output_root: Path) -> Path:
    baseline = pq.read_table(output_root / "baseline_metrics_by_step.parquet").to_pandas()
    proposed = pq.read_table(output_root / "proposed_metrics_by_step.parquet").to_pandas()
    primary = pd.concat(
        [baseline.loc[baseline.step == 2000], proposed.loc[proposed.step == 2000]],
        ignore_index=True,
    )
    columns = ["model", "epr50_pooled", "epr50_episode_macro", "mcc95_frame", "mcc95_episode"]
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in primary[columns].round(6).itertuples(index=False, name=None)
    )
    latency_path = output_root / "latency_report.json"
    if latency_path.is_file():
        latency = json.loads(latency_path.read_text(encoding="utf-8"))
        latency_html = f"<h2>Latency (fold 0 / step 2000)</h2><pre>{html.escape(json.dumps(latency, ensure_ascii=False, indent=2))}</pre>"
    else:
        latency_html = "<h2>Latency</h2><p>Latency benchmark 尚未运行。</p>"
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OpenPI ProcVLM metrics</title><style>body{{font-family:system-ui;margin:32px;max-width:1400px}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:7px 12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:16px}}svg{{max-width:100%;height:auto}}</style></head><body>
<h1>OpenPI 五折 OOF：ProcVLM 指标与延迟</h1>
<p>进度变换：p=clip(value+1,0,1)。EPR@50 与 MCC@95 越高越好；latency 越低越好。step 2000 为预注册主结果，其余 checkpoint 仅用于学习曲线。</p>
<h2>step 2000 主结果</h2><table><thead><tr>{''.join(f'<th>{c}</th>' for c in columns)}</tr></thead><tbody>{rows}</tbody></table>
<h2>21 个 checkpoint 曲线</h2>{_svg_curves(baseline, proposed)}
{latency_html}</body></html>"""
    path = output_root / "report.html"
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path


def evaluate_openpi_procvlm_metrics(
    config_path: str | Path,
    *,
    baseline_output_name: str = "openpi_finetune_oof",
    proposed_output_name: str = "openpi_finetune_oof_proposed",
    output_name: str = "openpi_procvlm_metrics",
) -> dict[str, Any]:
    config = load_config(config_path)
    ft_cfg = _ft_cfg(config)
    total_steps = int(ft_cfg["steps"])
    save_every = int(ft_cfg["save_every_steps"])
    if total_steps < 1 or save_every < 1 or total_steps % save_every:
        raise ValueError("Fine-tuning steps must be a positive multiple of save_every_steps")
    steps = [0] + list(range(save_every, total_steps + 1, save_every))
    bootstrap_samples = int(ft_cfg.get("bootstrap_samples", 2000))
    roots = {
        "baseline": _output_root(config, baseline_output_name),
        "proposed": _output_root(config, proposed_output_name),
    }
    output_root = _output_root(config, output_name)
    if output_root in roots.values():
        raise ValueError("ProcVLM output must not overwrite a source OOF directory")
    output_root.mkdir(parents=True, exist_ok=True)

    frames_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in roots}
    metrics_by_model: dict[str, dict[str, Any]] = {name: {} for name in roots}
    reference_keys: dict[int, list[str]] = {}
    for model_index, (model, root) in enumerate(roots.items()):
        for step in steps:
            path = root / "oof_predictions" / f"step-{step:05d}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pq.read_table(path).to_pandas().sort_values("sample_key").reset_index(drop=True)
            keys = frame.sample_key.astype(str).tolist()
            if len(frame) != 14_855 or frame.sample_key.nunique() != 14_855:
                raise RuntimeError(f"{model} step {step}: expected 14,855 unique anchors")
            if frame.uuid.nunique() != 302:
                raise RuntimeError(f"{model} step {step}: expected 302 UUID episodes")
            outcome_counts = frame.groupby("uuid").outcome.first().value_counts().to_dict()
            if outcome_counts != {"failure": 214, "success": 88}:
                raise RuntimeError(f"{model} step {step}: outcome gate failed: {outcome_counts}")
            if model_index == 0:
                reference_keys[step] = keys
            elif keys != reference_keys[step]:
                raise RuntimeError(f"Baseline/proposed keys differ at step {step}")
            metrics = compute_procvlm_metrics(
                frame, bootstrap_samples=bootstrap_samples, seed=int(config["seed"]) + step
            )
            metrics_by_model[model][str(step)] = metrics
            frames_by_model[model].append(_flatten_row(model, step, metrics))
            print(f"ProcVLM metrics: {model} step {step}/{steps[-1]}", flush=True)

    tables = {name: pd.DataFrame(rows) for name, rows in frames_by_model.items()}
    for model, table in tables.items():
        _atomic_parquet(
            pa.Table.from_pandas(table, preserve_index=False),
            output_root / f"{model}_metrics_by_step.parquet",
        )
    comparison = tables["baseline"].merge(
        tables["proposed"], on="step", suffixes=("_baseline", "_proposed"), validate="one_to_one"
    )
    for column in ("epr50_pooled", "epr50_episode_macro", "mcc95_frame", "mcc95_episode"):
        comparison[f"{column}_delta_proposed_minus_baseline"] = (
            comparison[f"{column}_proposed"] - comparison[f"{column}_baseline"]
        )
    _atomic_parquet(
        pa.Table.from_pandas(comparison, preserve_index=False), output_root / "comparison.parquet"
    )
    report = {
        "protocol": "ProcVLM metrics on 5-fold episode-level OpenPI OOF predictions",
        "progress_transform": "clip(value + 1, 0, 1)",
        "selection_policy": "step 2000 is primary; intermediate steps are diagnostics",
        "steps": steps,
        "anchors_per_step": 14_855,
        "episodes_per_step": 302,
        "bootstrap_samples": bootstrap_samples,
        "source_outputs": {name: str(path) for name, path in roots.items()},
        "metrics_by_model": metrics_by_model,
    }
    atomic_json(output_root / "metrics_report.json", report)
    generate_procvlm_report(output_root)
    return report
