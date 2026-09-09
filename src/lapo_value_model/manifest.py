from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import atomic_json, stable_fraction
from .config import ensure_run_dirs, load_config
from .task_families import classify_task, task_prompt


def _station(row: dict[str, Any]) -> str:
    return "/".join(
        str(row[name]) for name in ("left_serial", "wrist_serial", "right_serial")
    )


def _episode_table(dataset_root: Path) -> pa.Table:
    paths = sorted(glob.glob(str(dataset_root / "meta/episodes/**/*.parquet"), recursive=True))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one episode metadata parquet under {dataset_root}, got {len(paths)}")
    return pq.read_table(paths[0])


def _failure_audit(dataset_root: Path) -> dict[str, dict[str, Any]]:
    path = dataset_root / "meta/raw_task_prompts.parquet"
    if not path.exists():
        return {}
    return {str(row["uuid"]): row for row in pq.read_table(path).to_pylist()}


def _video_path(root: Path, camera_key: str, row: dict[str, Any]) -> Path:
    chunk = int(row[f"videos/{camera_key}/chunk_index"])
    file_index = int(row[f"videos/{camera_key}/file_index"])
    return root / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file_{file_index:03d}.mp4"


def _split(uuid: str, cell: str, ratios: list[float], seed: int) -> str:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split_ratios must contain three values summing to one: {ratios}")
    value = stable_fraction(f"{cell}:{uuid}", seed)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def build_manifest(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_run_dirs(config)
    seed = int(config["seed"])
    data_cfg = config["data"]
    data_root = Path(config["data_root"])
    artifact_root = Path(config["artifact_root"])
    camera_key = str(data_cfg["camera_key"])
    min_length = abs(min(data_cfg["history_offsets"])) + int(data_cfg["future_offset"]) + 1

    raw_rows: dict[str, list[dict[str, Any]]] = {}
    uuid_sets: dict[str, set[str]] = {}
    station_sets: dict[str, set[str]] = {}
    failure_audit = _failure_audit(data_root / "droid_failure")
    for outcome, dirname in (("success", "droid_success"), ("failure", "droid_failure")):
        rows = _episode_table(data_root / dirname).to_pylist()
        raw_rows[outcome] = rows
        uuid_sets[outcome] = {str(row["uuid"]) for row in rows}
        station_sets[outcome] = {_station(row) for row in rows}

    contradictory_uuids = uuid_sets["success"] & uuid_sets["failure"]
    shared_stations = station_sets["success"] & station_sets["failure"]
    filtered: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for outcome, dirname in (("success", "droid_success"), ("failure", "droid_failure")):
        root = data_root / dirname
        for row in raw_rows[outcome]:
            uuid = str(row["uuid"])
            station = _station(row)
            task = str(row.get("task") or "")
            if uuid in contradictory_uuids:
                dropped[f"{outcome}:duplicate_uuid"] += 1
                continue
            if int(row["length"]) < min_length:
                dropped[f"{outcome}:too_short"] += 1
                continue
            if outcome == "success" and "no action" in task.lower():
                dropped["success:no_action"] += 1
                continue
            if outcome == "failure":
                audit = failure_audit.get(uuid)
                if audit and audit.get("raw_success") is not False:
                    dropped["failure:raw_success_not_false"] += 1
                    continue
            if bool(data_cfg.get("require_shared_station", True)) and station not in shared_stations:
                dropped[f"{outcome}:unshared_station"] += 1
                continue
            video_path = _video_path(root, camera_key, row)
            if not video_path.is_file():
                dropped[f"{outcome}:missing_video"] += 1
                continue
            family = classify_task(task)
            task_counts[outcome][family] += 1
            lab = uuid.split("+", 1)[0]
            cell = f"{lab}:{station}:{family}:{outcome}"
            filtered.append(
                {
                    "source": dirname,
                    "outcome": outcome,
                    "episode_index": int(row["episode_index"]),
                    "uuid": uuid,
                    "length": int(row["length"]),
                    "fps": int(row["fps"]),
                    "task_text": task,
                    "task_family": family,
                    "task_prompt": task_prompt(family),
                    "lab": lab,
                    "station": station,
                    "split": _split(uuid, cell, list(data_cfg["split_ratios"]), seed),
                    "video_path": str(video_path),
                    "video_start_sec": float(row[f"videos/{camera_key}/from_timestamp"]),
                    "video_end_sec": float(row[f"videos/{camera_key}/to_timestamp"]),
                }
            )

    max_per_outcome = config.get("max_episodes_per_outcome")
    if max_per_outcome is not None:
        selected: list[dict[str, Any]] = []
        for outcome in ("success", "failure"):
            candidates = [row for row in filtered if row["outcome"] == outcome]
            # Keep smoke tests concentrated in as few packed MP4 files as possible.
            candidates.sort(key=lambda row: (row["video_path"], row["episode_index"]))
            selected.extend(candidates[: int(max_per_outcome)])
        filtered = selected

    outcome_totals = Counter(row["outcome"] for row in filtered)
    final_task_counts = {
        outcome: Counter(row["task_family"] for row in filtered if row["outcome"] == outcome)
        for outcome in ("success", "failure")
    }
    for outcome in ("success", "failure"):
        other_fraction = final_task_counts[outcome]["other"] / max(outcome_totals[outcome], 1)
        if other_fraction > float(data_cfg["task_other_max_fraction"]):
            raise RuntimeError(
                f"Task-family mapping coverage gate failed for {outcome}: "
                f"other={other_fraction:.2%}, maximum={data_cfg['task_other_max_fraction']:.2%}"
            )

    by_family_train: dict[str, list[int]] = defaultdict(list)
    by_family_success: dict[str, list[int]] = defaultdict(list)
    for row in filtered:
        if row["split"] == "train":
            by_family_train[row["task_family"]].append(row["length"])
            if row["outcome"] == "success":
                by_family_success[row["task_family"]].append(row["length"])
    tmax: dict[str, float] = {}
    quantile = float(data_cfg["tmax_quantile"])
    all_lengths = [row["length"] for row in filtered]
    global_tmax = max(float(np.quantile(all_lengths, quantile)), float(min_length))
    all_families = {row["task_family"] for row in filtered}
    for family in all_families:
        lengths = by_family_train.get(family, [])
        if not lengths:
            tmax[family] = global_tmax
            continue
        reference = by_family_success[family] if len(by_family_success[family]) >= 10 else lengths
        tmax[family] = max(float(np.quantile(reference, quantile)), float(min_length))

    offsets = [int(value) for value in data_cfg["history_offsets"]]
    future_offset = int(data_cfg["future_offset"])
    anchors_per_episode = int(data_cfg["anchors_per_episode"])
    value_bins = int(config["student"]["value_bins"])
    samples: list[dict[str, Any]] = []
    for episode in filtered:
        low = abs(min(offsets))
        high = episode["length"] - future_offset - 1
        count = min(anchors_per_episode, high - low + 1)
        anchors = np.unique(np.linspace(low, high, num=count, dtype=np.int64))
        for anchor in anchors.tolist():
            if episode["outcome"] == "success":
                value = float(np.clip(-(episode["length"] - 1 - anchor) / tmax[episode["task_family"]], -1, 0))
            else:
                value = -1.0
            value_bin = int(np.clip(round((value + 1.0) * (value_bins - 1)), 0, value_bins - 1))
            key = f"{episode['source']}-{episode['episode_index']:06d}-{anchor:05d}"
            samples.append(
                {
                    "sample_key": key,
                    "source": episode["source"],
                    "outcome": episode["outcome"],
                    "episode_index": episode["episode_index"],
                    "uuid": episode["uuid"],
                    "length": episode["length"],
                    "anchor": int(anchor),
                    "history_indices": [int(anchor + offset) for offset in offsets],
                    "future_index": int(anchor + future_offset),
                    "value": value,
                    "value_bin": value_bin,
                    "task_family": episode["task_family"],
                    "task_prompt": episode["task_prompt"],
                    "lab": episode["lab"],
                    "station": episode["station"],
                    "split": episode["split"],
                    "video_path": episode["video_path"],
                    "video_start_sec": episode["video_start_sec"],
                    "fps": episode["fps"],
                }
            )

    episode_path = artifact_root / "episodes.parquet"
    sample_path = artifact_root / "samples.parquet"
    _atomic_parquet(pa.Table.from_pylist(filtered), episode_path)
    _atomic_parquet(pa.Table.from_pylist(samples), sample_path)
    stats = {
        "raw_episodes": {outcome: len(rows) for outcome, rows in raw_rows.items()},
        "eligible_episodes": dict(outcome_totals),
        "samples": dict(Counter(row["split"] for row in samples)),
        "samples_by_outcome": dict(Counter(row["outcome"] for row in samples)),
        "dropped": dict(sorted(dropped.items())),
        "duplicate_uuids": len(contradictory_uuids),
        "shared_stations": len(shared_stations),
        "task_families": {
            outcome: dict(sorted(counts.items())) for outcome, counts in final_task_counts.items()
        },
        "tmax": dict(sorted(tmax.items())),
        "config": config,
    }
    atomic_json(artifact_root / "manifest_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_manifest(args.config)
