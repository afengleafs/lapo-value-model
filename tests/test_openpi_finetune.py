from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from lapo_value_model.openpi_finetune import (
    BalancedEpisodeBatchSampler,
    _prediction_columns,
    assign_stratified_folds,
    checkpoint_steps,
    dense_anchor_indices,
    finetune_artifact_root,
    finetune_output_root,
)
from lapo_value_model.recap_metrics import _COMPARISON_METRICS, compute_recap_metrics
from lapo_value_model.student import process_student_batch


PROJECT_ROOT = Path("/home/tione/notebook/users/fhh/lapo_value_model")
OPENPI_ARTIFACT = PROJECT_ROOT / "artifacts/openpi_rollout"


def test_finetune_output_name_is_isolated_and_default_is_compatible(tmp_path) -> None:
    config = {"output_root": str(tmp_path)}
    assert finetune_output_root(config) == tmp_path / "openpi_finetune_oof"
    assert finetune_output_root(config, "openpi_finetune_oof_proposed") == (
        tmp_path / "openpi_finetune_oof_proposed"
    )
    for invalid in ("", "../escape", "/absolute", ".", ".."):
        with pytest.raises(ValueError):
            finetune_output_root(config, invalid)


def test_finetune_artifact_name_is_isolated_and_default_is_compatible(tmp_path) -> None:
    config = {
        "external_eval": {
            "openpi": {"artifact_root": str(tmp_path), "finetune": {}}
        }
    }
    assert finetune_artifact_root(config) == tmp_path / "finetune"
    config["external_eval"]["openpi"]["finetune"]["artifact_name"] = "all400"
    assert finetune_artifact_root(config) == tmp_path / "all400"
    for invalid in ("", "../escape", "/absolute", ".", ".."):
        config["external_eval"]["openpi"]["finetune"]["artifact_name"] = invalid
        with pytest.raises(ValueError):
            finetune_artifact_root(config)


def test_oof_comparison_metric_contract_includes_recap_axes() -> None:
    specs = {name: (path, direction) for name, path, direction in _COMPARISON_METRICS}
    assert specs["MAE"] == (("value", "mae"), "lower")
    assert specs["Success macro-Spearman"][1] == "higher"
    assert specs["Temporal monotonicity"] == (
        ("value", "success_temporal_monotonicity"),
        "higher",
    )
    assert specs["A50 failure tail20 positive rate"][1] == "lower"


def test_dense_rollout_inventory_and_exact_a50_count() -> None:
    episodes = pq.read_table(OPENPI_ARTIFACT / "episodes.parquet").to_pandas()
    rollout = episodes.loc[episodes.collection == "rollout"].reset_index(drop=True)
    assert len(rollout) == 302
    assert Counter(rollout.outcome) == Counter({"failure": 214, "success": 88})
    counts = []
    a50 = 0
    for row in rollout.itertuples():
        anchors = dense_anchor_indices(
            length=int(row.length),
            history_offsets=[-20, -10, 0],
            future_offset=10,
            stride=10,
        )
        counts.append(len(anchors))
        anchor_set = set(anchors)
        a50 += sum(anchor + 100 in anchor_set for anchor in anchors)
    assert sum(counts) == 14_855
    # 50 model frames at 15 Hz are exactly 100 source frames at 30 Hz.
    assert a50 == 11_835
    assert a50 == 14_855 - 302 * 10


def test_dense_all400_inventory_and_exact_a50_count() -> None:
    episodes = pq.read_table(OPENPI_ARTIFACT / "episodes.parquet").to_pandas()
    assert len(episodes) == 400
    assert Counter(episodes.outcome) == Counter({"failure": 214, "success": 186})
    assert Counter(episodes.collection) == Counter(
        {"rollout": 302, "demonstration": 98}
    )
    anchors_total = 0
    a50 = 0
    for row in episodes.itertuples():
        anchors = dense_anchor_indices(
            length=int(row.length),
            history_offsets=[-20, -10, 0],
            future_offset=10,
            stride=10,
        )
        anchors_total += len(anchors)
        anchor_set = set(anchors)
        a50 += sum(anchor + 100 in anchor_set for anchor in anchors)
    assert anchors_total == 18_071
    assert a50 == 14_071


def test_stratified_folds_have_complete_episode_oof_coverage() -> None:
    episodes = pq.read_table(OPENPI_ARTIFACT / "episodes.parquet").to_pandas()
    rollout = episodes.loc[episodes.collection == "rollout"].reset_index(drop=True)
    assigned = assign_stratified_folds(rollout, folds=5, seed=20260824)
    assert assigned.uuid.nunique() == 302
    assert set(assigned.heldout_fold) == set(range(5))
    assert assigned.groupby("uuid").heldout_fold.nunique().eq(1).all()
    for fold in range(5):
        heldout = set(assigned.loc[assigned.heldout_fold == fold, "uuid"])
        train = set(assigned.loc[assigned.heldout_fold != fold, "uuid"])
        assert not (heldout & train)
        assert 57 <= len(heldout) <= 65


def _sampler_rows() -> list[dict[str, object]]:
    rows = []
    for outcome in ("success", "failure"):
        for episode in range(3):
            for anchor in range(4):
                rows.append(
                    {
                        "outcome": outcome,
                        "uuid": f"{outcome}-{episode}",
                        "anchor": anchor,
                    }
                )
    return rows


def test_balanced_sampler_resume_is_step_exact() -> None:
    rows = _sampler_rows()
    kwargs = {
        "rows": rows,
        "eligible_indices": list(range(len(rows))),
        "global_batch_size": 8,
        "rank": 0,
        "world_size": 1,
        "seed": 17,
    }
    uninterrupted = list(
        BalancedEpisodeBatchSampler(**kwargs, start_step=0, end_step=20)
    )
    resumed = list(BalancedEpisodeBatchSampler(**kwargs, start_step=0, end_step=7)) + list(
        BalancedEpisodeBatchSampler(**kwargs, start_step=7, end_step=20)
    )
    assert resumed == uninterrupted
    for batch in uninterrupted:
        assert Counter(str(rows[index]["outcome"]) for index in batch) == Counter(
            {"success": 4, "failure": 4}
        )


def test_two_thousand_step_checkpoint_contract() -> None:
    assert checkpoint_steps(2000, 1000) == [1000, 2000]


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_count = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, *, text, images, padding, return_tensors):
        self.image_count = len(images)
        batch = len(text)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
        }


def test_heldout_student_input_excludes_future_and_latent() -> None:
    image = Image.new("RGB", (2, 2))
    rows = [
        {
            "key": f"key-{index}",
            "images": [image, image, image, image],
            "metadata": {"task_prompt": "task", "value_bin": 0, "value": -1.0},
        }
        for index in range(2)
    ]
    processor = _FakeProcessor()
    _, targets = process_student_batch(processor, rows, None, torch.device("cpu"))
    assert processor.image_count == 6
    assert "latent" not in targets


def test_prediction_columns_preserve_collection() -> None:
    row = {
        "key": "demo-key",
        "metadata": {
            "dataset_name": "demo",
            "collection": "demonstration",
            "episode_index": 0,
            "uuid": "demo:0",
            "anchor": 20,
            "length": 200,
            "source_fps": 30,
            "progress": 0.1,
            "outcome": "success",
            "task_family": "task",
            "value": -0.5,
            "value_bin": 100,
            "heldout_fold": 0,
        },
    }
    assert _prediction_columns(row, -0.4, 1.0)["collection"] == "demonstration"


def _synthetic_predictions(kind: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(9)
    for outcome in ("success", "failure"):
        for episode in range(4):
            uuid = f"{outcome}-{episode}"
            for position, anchor in enumerate((20, 120, 220)):
                target = -0.3 + position * 0.1 if outcome == "success" else -1.0
                if kind == "perfect":
                    prediction = target
                elif kind == "constant":
                    prediction = -0.5
                elif kind == "reversed":
                    prediction = -target
                elif kind == "random":
                    prediction = float(rng.normal())
                else:
                    raise AssertionError(kind)
                rows.append(
                    {
                        "sample_key": f"{uuid}-{anchor}",
                        "uuid": uuid,
                        "dataset_name": "synthetic",
                        "anchor": anchor,
                        "length": 300,
                        "source_fps": 30,
                        "progress": (0.1, 0.82, 0.95)[position],
                        "outcome": outcome,
                        "task_family": "task",
                        "target": target,
                        "prediction": prediction,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("kind", "auc", "spearman", "success_advantage", "overall_advantage"),
    [
        ("perfect", 1.0, 1.0, 0.0, -0.05),
        ("constant", 0.5, 0.0, -0.1, -0.1),
        ("reversed", 0.0, -1.0, -0.2, -0.15),
    ],
)
def test_synthetic_recap_metrics(
    kind, auc, spearman, success_advantage, overall_advantage
) -> None:
    metrics, advantage_frame = compute_recap_metrics(
        _synthetic_predictions(kind),
        tmax_by_family={"task": 500.0},
        horizon_model_frames=50,
        model_fps=15,
        bootstrap_samples=20,
        seed=1,
    )
    assert metrics["episode_auc"]["first_valid_anchor"] == pytest.approx(auc)
    assert metrics["value"]["success_episode_macro_spearman"] == pytest.approx(spearman)
    assert advantage_frame.advantage.mean() == pytest.approx(overall_advantage)
    assert metrics["advantage_a50"]["mean_success"] == pytest.approx(success_advantage)
    if kind == "constant":
        assert metrics["advantage_a50"]["positive_rate_success"] == 0.0
        assert metrics["advantage_a50"]["positive_rate_failure_tail20"] == 0.0


def test_random_metrics_are_finite_and_reproducible() -> None:
    first, _ = compute_recap_metrics(
        _synthetic_predictions("random"),
        tmax_by_family={"task": 500.0},
        bootstrap_samples=20,
        seed=7,
    )
    second, _ = compute_recap_metrics(
        _synthetic_predictions("random"),
        tmax_by_family={"task": 500.0},
        bootstrap_samples=20,
        seed=7,
    )
    assert first["episode_auc"] == second["episode_auc"]
    assert first["bootstrap_ci95"] == second["bootstrap_ci95"]
    assert np.isfinite(first["advantage_a50"]["epsilon_q70"])


def test_generated_latent_artifact_gate_when_present() -> None:
    summary_path = OPENPI_ARTIFACT / "finetune/latents/summary.json"
    if not summary_path.is_file():
        pytest.skip("Dense Teacher latents have not been generated yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows"] == 14_855
    assert summary["unique_keys"] == 14_855
    assert summary["latent_dim"] == 32
    assert summary["finite"] is True
