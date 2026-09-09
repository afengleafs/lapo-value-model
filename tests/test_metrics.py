from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lapo_value_model.metrics import compute_stage_auc, compute_value_metrics


def _perfect_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_key": "s0", "uuid": "success-a", "anchor": 10, "length": 100, "outcome": "success", "task_family": "lid", "target": -0.8, "prediction": -0.8},
            {"sample_key": "s1", "uuid": "success-a", "anchor": 80, "length": 100, "outcome": "success", "task_family": "lid", "target": -0.2, "prediction": -0.2},
            {"sample_key": "f0", "uuid": "failure-a", "anchor": 10, "length": 100, "outcome": "failure", "task_family": "lid", "target": -1.0, "prediction": -1.0},
            {"sample_key": "f1", "uuid": "failure-a", "anchor": 80, "length": 100, "outcome": "failure", "task_family": "lid", "target": -1.0, "prediction": -1.0},
        ]
    )


def test_perfect_value_metrics() -> None:
    metrics = compute_value_metrics(_perfect_predictions())
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["macro_spearman"] == pytest.approx(1.0)
    assert metrics["temporal_monotonicity"] == pytest.approx(1.0)
    assert metrics["success_failure_auc"] == pytest.approx(1.0)
    assert metrics["advantage_sign_accuracy"] == pytest.approx(1.0)


def test_stage_matched_auc() -> None:
    result = compute_stage_auc(_perfect_predictions(), stages=2)
    assert result["macro_auc"] == pytest.approx(1.0)
    assert result["group_count"] == 2


def test_nonfinite_predictions_fail_loudly() -> None:
    frame = _perfect_predictions()
    frame.loc[0, "prediction"] = np.nan
    with pytest.raises(FloatingPointError):
        compute_value_metrics(frame)
