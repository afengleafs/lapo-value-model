from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from lapo_value_model.openpi_eval import external_success_value, scaled_temporal_offsets
from lapo_value_model.task_families import classify_task


OPENPI_ROOT = Path("/home/tione/notebook/users/fhh/datasets/openpi_rollout")


def test_openpi_source_episode_inventory() -> None:
    info_paths = sorted(OPENPI_ROOT.glob("*/meta/info.json"))
    episodes = []
    frames = 0
    for info_path in info_paths:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        assert info["codebase_version"] == "v3.0"
        assert info["fps"] == 30
        frames += int(info["total_frames"])
        for path in sorted((info_path.parent / "episodes").glob("**/*.parquet")):
            table = pq.read_table(path, columns=["episode_success"])
            collection = "demonstration" if info_path.parent.parent.name.startswith("fr3_plug_rgb_tactile") else "rollout"
            episodes.extend((collection, value) for value in table["episode_success"].to_pylist())
    assert len(info_paths) == 17
    assert len(episodes) == 400
    assert frames == 190_833
    rollout = Counter("success" if value else "failure" for group, value in episodes if group == "rollout")
    demonstration = Counter(
        "success" if value else "failure" for group, value in episodes if group == "demonstration"
    )
    assert rollout == Counter({"failure": 214, "success": 88})
    assert demonstration == Counter({"success": 98})


def test_openpi_temporal_and_task_mapping() -> None:
    assert scaled_temporal_offsets([-10, -5, 0], 15, 30) == [-20, -10, 0]
    task = "Pick up the white power adapter and plug it into the power strip"
    assert classify_task(task) == "container_transfer"
    value = external_success_value(
        remaining_source_frames=60,
        source_fps=30,
        model_fps=15,
        tmax=100,
    )
    assert value == pytest.approx(-0.3)
