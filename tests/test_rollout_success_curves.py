from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lapo_value_model.rollout_success_curves import (
    DEMONSTRATION_DATASET,
    _curve_rows,
    resolve_selected_groups,
    select_rollout_success,
)


def test_select_rollout_success_excludes_demo_and_failure() -> None:
    inventory = pd.DataFrame(
        [
            {"uuid": "demo", "dataset_name": DEMONSTRATION_DATASET, "collection": "demonstration", "outcome": "success"},
            {"uuid": "keep", "dataset_name": "rollout-a", "collection": "rollout", "outcome": "success"},
            {"uuid": "failure", "dataset_name": "rollout-b", "collection": "rollout", "outcome": "failure"},
            {"uuid": "zero-success", "dataset_name": "rollout-c", "collection": "rollout", "outcome": "failure"},
        ]
    )
    frame = pd.DataFrame(
        [
            {"sample_key": "d", "uuid": "demo", "dataset_name": DEMONSTRATION_DATASET, "outcome": "success"},
            {"sample_key": "k", "uuid": "keep", "dataset_name": "rollout-a", "outcome": "success"},
            {"sample_key": "f", "uuid": "failure", "dataset_name": "rollout-b", "outcome": "failure"},
        ]
    )
    selected, omitted = select_rollout_success(frame, inventory)
    assert selected.sample_key.tolist() == ["k"]
    assert omitted == ["rollout-b", "rollout-c"]


def test_group_curve_is_episode_equal_not_anchor_weighted() -> None:
    prediction = pd.DataFrame(
        [
            {"uuid": "short", "progress": 0.0, "prediction": 0.0, "target": 0.0},
            {"uuid": "short", "progress": 1.0, "prediction": 0.0, "target": 1.0},
            {"uuid": "long", "progress": 0.0, "prediction": 1.0, "target": 0.0},
            {"uuid": "long", "progress": 0.25, "prediction": 1.0, "target": 0.25},
            {"uuid": "long", "progress": 0.5, "prediction": 1.0, "target": 0.5},
            {"uuid": "long", "progress": 0.75, "prediction": 1.0, "target": 0.75},
            {"uuid": "long", "progress": 1.0, "prediction": 1.0, "target": 1.0},
        ]
    )
    advantage = pd.DataFrame(
        [
            {"uuid": "short", "progress": 0.0, "advantage": -0.2},
            {"uuid": "short", "progress": 1.0, "advantage": -0.2},
            {"uuid": "long", "progress": 0.0, "advantage": 0.2},
            {"uuid": "long", "progress": 0.5, "advantage": 0.2},
            {"uuid": "long", "progress": 1.0, "advantage": 0.2},
        ]
    )
    rows, summaries = _curve_rows("rollout-a", "baseline", prediction, advantage)
    assert len(rows) == 101
    assert summaries["prediction"].count[50] == 2
    assert np.isclose(summaries["prediction"].mean[50], 0.5)
    assert np.isclose(summaries["advantage"].mean[50], 0.0)


def test_selected_groups_preserve_requested_order_and_reject_invalid() -> None:
    available = ["rollout-c", "rollout-a", "rollout-b"]
    assert resolve_selected_groups(available, None) == ["rollout-a", "rollout-b", "rollout-c"]
    assert resolve_selected_groups(available, ["rollout-b", "rollout-a"]) == [
        "rollout-b",
        "rollout-a",
    ]
    with pytest.raises(ValueError, match="unique"):
        resolve_selected_groups(available, ["rollout-a", "rollout-a"])
    with pytest.raises(ValueError, match="no successful OOF data"):
        resolve_selected_groups(available, ["rollout-missing"])
    with pytest.raises(ValueError, match="demonstration"):
        resolve_selected_groups(available + [DEMONSTRATION_DATASET], [DEMONSTRATION_DATASET])
