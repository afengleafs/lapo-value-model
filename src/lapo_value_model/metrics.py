from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def compute_value_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute the RECAP-style offline metrics from one prediction per anchor."""
    required = {"sample_key", "uuid", "anchor", "outcome", "target", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("Predictions are empty")
    values = frame[["target", "prediction"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = frame.loc[~np.isfinite(values).all(axis=1), "sample_key"].head(10).tolist()
        raise FloatingPointError(f"Non-finite targets or predictions for samples: {bad}")

    target = frame.target.to_numpy(dtype=np.float64)
    prediction = frame.prediction.to_numpy(dtype=np.float64)
    absolute_error = np.abs(prediction - target)
    constant = float(np.median(target))
    episode_correlations: list[float] = []
    monotonic: list[bool] = []
    transition_predictions: list[float] = []
    transition_targets: list[float] = []
    advantage_signs: list[bool] = []

    for _, group in frame.groupby("uuid", sort=False):
        group = group.sort_values("anchor")
        if len(group) < 2:
            continue
        predicted = group.prediction.to_numpy(dtype=np.float64)
        expected = group.target.to_numpy(dtype=np.float64)
        if str(group.outcome.iloc[0]) == "success":
            correlation = _safe_spearman(predicted, expected)
            if correlation is not None:
                episode_correlations.append(correlation)
            monotonic.extend((np.diff(predicted) >= 0).tolist())
        predicted_delta = np.diff(predicted)
        expected_delta = np.diff(expected)
        informative = np.abs(expected_delta) > 1e-9
        if informative.any():
            transition_predictions.extend(predicted_delta[informative].tolist())
            transition_targets.extend(expected_delta[informative].tolist())
            advantage_signs.extend(
                (np.sign(predicted_delta[informative]) == np.sign(expected_delta[informative])).tolist()
            )

    labels = (frame.outcome == "success").astype(np.int64)
    auc = float(roc_auc_score(labels, prediction)) if labels.nunique() == 2 else float("nan")
    advantage_spearman = _safe_spearman(
        np.asarray(transition_predictions), np.asarray(transition_targets)
    )
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "median_absolute_error": float(np.median(absolute_error)),
        "constant_mae": float(np.mean(np.abs(target - constant))),
        "macro_spearman": float(np.mean(episode_correlations)) if episode_correlations else 0.0,
        "temporal_monotonicity": float(np.mean(monotonic)) if monotonic else 0.0,
        "success_failure_auc": auc,
        "advantage_sign_accuracy": float(np.mean(advantage_signs)) if advantage_signs else 0.0,
        "advantage_spearman": advantage_spearman if advantage_spearman is not None else 0.0,
        "transitions": int(len(transition_targets)),
        "examples": int(len(frame)),
    }


def compute_group_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for family, group in frame.groupby("task_family", sort=True):
        metrics = compute_value_metrics(group)
        groups[str(family)] = metrics
    return groups


def compute_stage_auc(frame: pd.DataFrame, stages: int = 4) -> dict[str, Any]:
    """Compare success/failure at roughly matched episode progress."""
    if "length" not in frame.columns:
        return {"macro_auc": float("nan"), "groups": {}, "group_count": 0}
    staged = frame.copy()
    denominator = np.maximum(staged.length.to_numpy(dtype=np.float64) - 1, 1)
    progress = staged.anchor.to_numpy(dtype=np.float64) / denominator
    staged["progress_stage"] = np.minimum((progress * stages).astype(np.int64), stages - 1)
    results: dict[str, float] = {}
    for (family, stage), group in staged.groupby(["task_family", "progress_stage"]):
        labels = (group.outcome == "success").astype(np.int64)
        if labels.nunique() != 2:
            continue
        results[f"{family}:stage-{int(stage)}"] = float(
            roc_auc_score(labels, group.prediction.to_numpy(dtype=np.float64))
        )
    return {
        "macro_auc": float(np.mean(list(results.values()))) if results else float("nan"),
        "groups": results,
        "group_count": len(results),
    }
