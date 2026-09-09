from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_full_manifest_is_episode_split_and_causal() -> None:
    frame = pq.read_table(PROJECT_ROOT / "artifacts/full/samples.parquet").to_pandas()
    assert len(frame) == 526_718
    assert frame.sample_key.is_unique
    assert frame.groupby("uuid").split.nunique().max() == 1
    assert np.all(frame.value.between(-1.0, 0.0))
    assert np.all(frame.value_bin.between(0, 200))
    assert np.all(frame.future_index > frame.anchor)
    history = np.stack(frame.history_indices.to_numpy())
    assert np.all(history[:, -1] == frame.anchor.to_numpy())
    assert np.all(history[:, :-1] <= frame.anchor.to_numpy()[:, None])
    assert np.allclose(frame.loc[frame.outcome == "failure", "value"], -1.0)


def test_success_and_failure_uuids_do_not_overlap() -> None:
    episodes = pq.read_table(PROJECT_ROOT / "artifacts/full/episodes.parquet").to_pandas()
    success = set(episodes.loc[episodes.outcome == "success", "uuid"])
    failure = set(episodes.loc[episodes.outcome == "failure", "uuid"])
    assert success.isdisjoint(failure)
