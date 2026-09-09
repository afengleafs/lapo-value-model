from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .common import atomic_json, distributed_context, json_ready, set_seed
from .config import load_config
from .evaluate import _load_checkpoint
from .extract import _add_bytes, _letterbox_jpeg
from .metrics import compute_stage_auc, compute_value_metrics
from .student import (
    QwenValueModel,
    VALUE_PATH_LATENT_HIDDEN,
    checkpoint_value_path,
    load_processor,
    process_student_batch,
)
from .task_families import classify_task, task_prompt


FRAME_NAMES = ("t0.jpg", "t1.jpg", "t2.jpg")


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return config["external_eval"]["openpi"]
    except KeyError as exc:
        raise KeyError("config.external_eval.openpi is required") from exc


def scaled_temporal_offsets(
    model_offsets: list[int], model_fps: int, source_fps: int
) -> list[int]:
    ratio = float(source_fps) / float(model_fps)
    return [int(round(offset * ratio)) for offset in model_offsets]


def external_success_value(
    *, remaining_source_frames: int, source_fps: int, model_fps: int, tmax: float
) -> float:
    remaining_model_frames = float(remaining_source_frames) * float(model_fps) / float(source_fps)
    return float(np.clip(-remaining_model_frames / float(tmax), -1.0, 0.0))


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames <= 0:
            return sum(1 for _ in container.decode(stream))
        return int(stream.frames)


def _episode_table(root: Path) -> pa.Table:
    paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(root / "meta/episodes")
    return pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")


def build_openpi_manifest(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    external_cfg = _cfg(config)
    data_root = Path(external_cfg["data_root"])
    artifact_root = Path(external_cfg["artifact_root"])
    camera_key = str(external_cfg["camera_key"])
    model_fps = int(config["data"]["fps"])
    model_offsets = [int(value) for value in config["data"]["history_offsets"]]
    model_future_margin = int(config["data"]["future_offset"])
    anchors_per_episode = int(external_cfg["anchors_per_episode"])
    value_bins = int(config["student"]["value_bins"])
    droid_stats = json.loads(
        (Path(config["artifact_root"]) / "manifest_stats.json").read_text(encoding="utf-8")
    )
    tmax_by_family = {key: float(value) for key, value in droid_stats["tmax"].items()}

    episodes: list[dict[str, Any]] = []
    video_checks: dict[str, Any] = {}
    info_paths = sorted(data_root.glob("*/meta/info.json"))
    for info_path in info_paths:
        root = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v3.0":
            raise RuntimeError(f"Expected LeRobot v3.0 at {root}, got {info.get('codebase_version')}")
        source_fps = int(info["fps"])
        table = _episode_table(root)
        rows = table.to_pylist()
        if len(rows) != int(info["total_episodes"]):
            raise RuntimeError(f"Episode count mismatch for {root}: {len(rows)} != {info['total_episodes']}")
        if sum(int(row["length"]) for row in rows) != int(info["total_frames"]):
            raise RuntimeError(f"Episode lengths do not sum to total_frames for {root}")

        per_camera: dict[str, Any] = {}
        for key in ("observation.images.camera_0", "observation.images.camera_1"):
            paths = sorted((root / "videos" / key).glob("**/*.mp4"))
            frames = [_video_frame_count(path) for path in paths]
            if sum(frames) != int(info["total_frames"]):
                raise RuntimeError(
                    f"Video frame count mismatch for {root.name}/{key}: "
                    f"{sum(frames)} != {info['total_frames']}"
                )
            per_camera[key] = {"files": len(paths), "frames": sum(frames)}
        video_checks[root.name] = per_camera

        source_offsets = scaled_temporal_offsets(model_offsets, model_fps, source_fps)
        source_future_margin = int(round(model_future_margin * source_fps / model_fps))
        collection = "demonstration" if root.name.startswith("fr3_plug_rgb_tactile") else "rollout"
        for row in rows:
            success = row.get("episode_success")
            if not isinstance(success, bool):
                raise RuntimeError(
                    f"Missing boolean episode_success for {root.name}:{row.get('episode_index')}"
                )
            tasks = row.get("tasks") or []
            if len(tasks) != 1:
                raise RuntimeError(f"Expected exactly one task for {root.name}:{row['episode_index']}")
            task_text = str(tasks[0])
            family = classify_task(task_text)
            chunk_index = int(row[f"videos/{camera_key}/chunk_index"])
            file_index = int(row[f"videos/{camera_key}/file_index"])
            relative_video = info["video_path"].format(
                video_key=camera_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            video_path = root / relative_video
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            episodes.append(
                {
                    "dataset_name": root.name,
                    "collection": collection,
                    "episode_index": int(row["episode_index"]),
                    "uuid": f"{root.name}:{int(row['episode_index']):06d}",
                    "length": int(row["length"]),
                    "source_fps": source_fps,
                    "outcome": "success" if success else "failure",
                    "episode_success": success,
                    "task_text": task_text,
                    "task_family": family,
                    "task_prompt": task_prompt(family),
                    "video_path": str(video_path),
                    "video_start_sec": float(row[f"videos/{camera_key}/from_timestamp"]),
                    "video_end_sec": float(row[f"videos/{camera_key}/to_timestamp"]),
                    "history_offsets": source_offsets,
                    "future_margin": source_future_margin,
                }
            )

    samples: list[dict[str, Any]] = []
    dropped = Counter()
    for episode in episodes:
        offsets = [int(value) for value in episode["history_offsets"]]
        low = abs(min(offsets))
        high = int(episode["length"]) - int(episode["future_margin"]) - 1
        available = high - low + 1
        if available <= 0:
            dropped["too_short"] += 1
            continue
        count = min(anchors_per_episode, available)
        anchors = np.unique(np.linspace(low, high, num=count, dtype=np.int64))
        for anchor_value in anchors.tolist():
            remaining = int(episode["length"]) - 1 - int(anchor_value)
            value = (
                external_success_value(
                    remaining_source_frames=remaining,
                    source_fps=int(episode["source_fps"]),
                    model_fps=model_fps,
                    tmax=tmax_by_family[episode["task_family"]],
                )
                if episode["outcome"] == "success"
                else -1.0
            )
            value_bin = int(np.clip(round((value + 1.0) * (value_bins - 1)), 0, value_bins - 1))
            sample_key = (
                f"{episode['dataset_name']}-{episode['episode_index']:06d}-{int(anchor_value):05d}"
            )
            samples.append(
                {
                    "sample_key": sample_key,
                    "dataset_name": episode["dataset_name"],
                    "collection": episode["collection"],
                    "episode_index": episode["episode_index"],
                    "uuid": episode["uuid"],
                    "length": episode["length"],
                    "source_fps": episode["source_fps"],
                    "outcome": episode["outcome"],
                    "anchor": int(anchor_value),
                    "progress": float(anchor_value / max(int(episode["length"]) - 1, 1)),
                    "history_indices": [int(anchor_value + offset) for offset in offsets],
                    "value": value,
                    "value_bin": value_bin,
                    "task_family": episode["task_family"],
                    "task_prompt": episode["task_prompt"],
                    "video_path": episode["video_path"],
                    "video_start_sec": episode["video_start_sec"],
                }
            )

    episode_counts = Counter(row["outcome"] for row in episodes)
    rollout_counts = Counter(row["outcome"] for row in episodes if row["collection"] == "rollout")
    demo_counts = Counter(row["outcome"] for row in episodes if row["collection"] == "demonstration")
    stats = {
        "datasets": len(info_paths),
        "episodes": len(episodes),
        "frames": sum(int(row["length"]) for row in episodes),
        "episode_outcomes": dict(episode_counts),
        "rollout_episode_outcomes": dict(rollout_counts),
        "demonstration_episode_outcomes": dict(demo_counts),
        "samples": len(samples),
        "samples_by_collection": dict(Counter(row["collection"] for row in samples)),
        "samples_by_outcome": dict(Counter(row["outcome"] for row in samples)),
        "dropped": dict(dropped),
        "camera_key": camera_key,
        "model_fps": model_fps,
        "video_checks": video_checks,
    }
    expected = {
        "datasets": 17,
        "episodes": 400,
        "frames": 190_833,
        "samples": 3_200,
    }
    mismatches = {key: (stats[key], value) for key, value in expected.items() if stats[key] != value}
    if mismatches:
        raise RuntimeError(f"OpenPI source gate failed: {mismatches}")
    if rollout_counts != Counter({"failure": 214, "success": 88}):
        raise RuntimeError(f"Unexpected rollout outcome counts: {dict(rollout_counts)}")
    if demo_counts != Counter({"success": 98}):
        raise RuntimeError(f"Unexpected demonstration outcome counts: {dict(demo_counts)}")

    _atomic_parquet(pa.Table.from_pylist(episodes), artifact_root / "episodes.parquet")
    _atomic_parquet(pa.Table.from_pylist(samples), artifact_root / "samples.parquet")
    atomic_json(artifact_root / "manifest_stats.json", stats)
    print(json.dumps(json_ready(stats), ensure_ascii=False, indent=2), flush=True)
    return stats


def _openpi_output_base(shard_root: Path, sample: dict[str, Any]) -> Path:
    video_path = Path(sample["video_path"])
    return shard_root / sample["dataset_name"] / video_path.parent.name / video_path.stem


def _extract_video_task(task: tuple[str, list[dict[str, Any]], str, int, int]) -> dict[str, Any]:
    video_raw, samples, shard_root_raw, size, quality = task
    video_path = Path(video_raw)
    shard_root = Path(shard_root_raw)
    base = _openpi_output_base(shard_root, samples[0])
    final_path = base.with_suffix(".tar")
    marker = base.with_suffix(".done.json")
    if marker.is_file() and final_path.is_file():
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        if int(marker_payload.get("samples", -1)) == len(samples):
            return {"video": str(video_path), "status": "skipped", "samples": len(samples)}

    base.parent.mkdir(parents=True, exist_ok=True)
    requests: dict[int, list[tuple[str, int]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for sample in samples:
        metadata[sample["sample_key"]] = sample
        for slot, frame_index in enumerate(sample["history_indices"]):
            target_us = round(
                (float(sample["video_start_sec"]) + frame_index / float(sample["source_fps"]))
                * 1_000_000
            )
            requests[target_us].append((sample["sample_key"], slot))
    wanted = sorted(requests)
    found: dict[str, dict[int, tuple[bytes, float]]] = defaultdict(dict)
    pointer = 0
    tolerance_us = round(0.51 / float(samples[0]["source_fps"]) * 1_000_000)
    started = time.time()
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if pointer >= len(wanted):
                break
            timestamp_us = round(float(frame.pts * stream.time_base) * 1_000_000)
            while pointer < len(wanted) and wanted[pointer] < timestamp_us - tolerance_us:
                raise RuntimeError(
                    f"Missed OpenPI frame in {video_path}: "
                    f"target_us={wanted[pointer]}, current_us={timestamp_us}"
                )
            if pointer < len(wanted) and abs(wanted[pointer] - timestamp_us) <= tolerance_us:
                encoded = _letterbox_jpeg(frame, size, quality)
                while pointer < len(wanted) and abs(wanted[pointer] - timestamp_us) <= tolerance_us:
                    for sample_key, slot in requests[wanted[pointer]]:
                        found[sample_key][slot] = (encoded, timestamp_us / 1_000_000.0)
                    pointer += 1
    if pointer != len(wanted):
        raise RuntimeError(f"Decoded {pointer}/{len(wanted)} OpenPI timestamps from {video_path}")

    temporary = final_path.with_suffix(f".tar.tmp.{os.getpid()}")
    with tarfile.open(temporary, mode="w") as archive:
        for key in sorted(metadata):
            sample = metadata[key]
            if len(found[key]) != len(FRAME_NAMES):
                raise RuntimeError(f"Incomplete OpenPI sample {key}: {sorted(found[key])}")
            actual_pts = []
            for slot, frame_name in enumerate(FRAME_NAMES):
                encoded, pts = found[key][slot]
                actual_pts.append(pts)
                _add_bytes(archive, f"{key}.{frame_name}", encoded)
            record = {name: value for name, value in sample.items() if name != "video_path"}
            record["actual_pts_sec"] = actual_pts
            _add_bytes(
                archive,
                f"{key}.json",
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )
    temporary.replace(final_path)
    atomic_json(
        marker,
        {
            "video": str(video_path),
            "samples": len(samples),
            "timestamps": len(wanted),
            "output": str(final_path),
            "elapsed_seconds": time.time() - started,
        },
    )
    return {
        "video": str(video_path),
        "status": "written",
        "samples": len(samples),
        "elapsed_seconds": time.time() - started,
    }


def prepare_openpi(config_path: str | Path, workers: int = 4) -> dict[str, Any]:
    build_openpi_manifest(config_path)
    config = load_config(config_path)
    external_cfg = _cfg(config)
    artifact_root = Path(external_cfg["artifact_root"])
    samples = pq.read_table(artifact_root / "samples.parquet").to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["video_path"])].append(sample)
    shard_root = artifact_root / "shards"
    tasks = [
        (
            video_path,
            rows,
            str(shard_root),
            int(config["data"]["image_size"]),
            int(config["data"]["jpeg_quality"]),
        )
        for video_path, rows in sorted(grouped.items())
    ]
    print(f"Extracting {len(samples)} OpenPI samples from {len(tasks)} videos", flush=True)
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_extract_video_task, task): task[0] for task in tasks}
        for index, future in enumerate(as_completed(future_map), 1):
            video = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(tasks)}] {result['status']} {video}", flush=True)
            except Exception as exc:
                failures.append({"video": video, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {video}: {exc}", flush=True)
    summary = {
        "videos": len(tasks),
        "samples": len(samples),
        "written": sum(row["status"] == "written" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failures": failures,
    }
    atomic_json(artifact_root / "extraction_summary.json", summary)
    if failures:
        raise RuntimeError(f"OpenPI extraction failed for {len(failures)} videos")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _iter_openpi_tar(path: Path) -> Iterator[dict[str, Any]]:
    current_key: str | None = None
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, suffix = member.name.split(".", 1)
            if current_key is not None and key != current_key:
                raise RuntimeError(f"Non-contiguous OpenPI sample {current_key} in {path}")
            current_key = key
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot extract {member.name} from {path}")
            payloads[suffix] = handle.read()
            if suffix == "json":
                record = json.loads(payloads.pop("json"))
                images = [
                    Image.open(io.BytesIO(payloads[name])).convert("RGB") for name in FRAME_NAMES
                ]
                yield {"key": key, "images": images, "metadata": record}
                current_key = None
                payloads = {}
    if current_key is not None or payloads:
        raise RuntimeError(f"Incomplete final OpenPI sample in {path}")


class OpenPIIterableDataset(IterableDataset):
    def __init__(self, shard_root: str | Path) -> None:
        super().__init__()
        self.shard_root = Path(shard_root)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        paths = sorted(self.shard_root.glob("**/*.tar"))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", "0"))
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        global_worker = rank * workers + worker_id
        global_workers = world * workers
        for path in paths[global_worker::global_workers]:
            yield from _iter_openpi_tar(path)


def make_openpi_loader(
    shard_root: str | Path, *, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        OpenPIIterableDataset(shard_root),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda rows: rows,
        persistent_workers=False,
    )


def _episode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uuid, group in frame.groupby("uuid", sort=True):
        ordered = group.sort_values("anchor")
        rows.append(
            {
                "uuid": uuid,
                "dataset_name": ordered.dataset_name.iloc[0],
                "collection": ordered.collection.iloc[0],
                "outcome": ordered.outcome.iloc[0],
                "mean_prediction": float(ordered.prediction.mean()),
                "last_prediction": float(ordered.prediction.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _auc(labels: pd.Series, values: pd.Series) -> float:
    return float(roc_auc_score((labels == "success").astype(np.int64), values)) if labels.nunique() == 2 else float("nan")


def _bootstrap_episode_auc(
    episodes: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    success = episodes.loc[episodes.outcome == "success"].reset_index(drop=True)
    failure = episodes.loc[episodes.outcome == "failure"].reset_index(drop=True)
    estimates = {
        "mean_value_auc": _auc(episodes.outcome, episodes.mean_prediction),
        "last_value_auc": _auc(episodes.outcome, episodes.last_prediction),
    }
    if success.empty or failure.empty:
        return {key: {"estimate": value, "ci95": [float("nan"), float("nan")]} for key, value in estimates.items()}
    draws: dict[str, list[float]] = {key: [] for key in estimates}
    for _ in range(samples):
        sampled = pd.concat(
            (
                success.iloc[rng.integers(0, len(success), size=len(success))],
                failure.iloc[rng.integers(0, len(failure), size=len(failure))],
            ),
            ignore_index=True,
        )
        draws["mean_value_auc"].append(_auc(sampled.outcome, sampled.mean_prediction))
        draws["last_value_auc"].append(_auc(sampled.outcome, sampled.last_prediction))
    return {
        key: {
            "estimate": estimates[key],
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
        for key, values in draws.items()
    }


def _subset_report(frame: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    episodes = _episode_frame(frame)
    return {
        "anchors": len(frame),
        "episodes": len(episodes),
        "episode_outcomes": dict(Counter(episodes.outcome)),
        "anchor_metrics": compute_value_metrics(frame),
        "stage_matched_separation": compute_stage_auc(frame),
        "episode_auc": _bootstrap_episode_auc(episodes, samples=bootstrap_samples, seed=seed),
    }


@torch.inference_mode()
def evaluate_openpi(
    config_path: str | Path, *, run_name: str, checkpoint_name: str = "best.pt"
) -> dict[str, Any]:
    config = load_config(config_path)
    external_cfg = _cfg(config)
    rank, _, world_size, device = distributed_context()
    if device.type != "cuda":
        raise RuntimeError("OpenPI evaluation requires a GPU")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    set_seed(int(config["seed"]), rank)

    run_dir = Path(config["output_root"]) / "student" / run_name
    checkpoint_path = run_dir / checkpoint_name
    evaluation_dir = run_dir / "external_evaluation" / "openpi_rollout"
    if rank == 0:
        evaluation_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    processor = load_processor(config)
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    value_path = checkpoint_value_path(checkpoint_header)
    model = QwenValueModel(
        config,
        trainable=True,
        include_latent_head=value_path == VALUE_PATH_LATENT_HIDDEN,
        value_path=value_path,
    ).to(device)
    checkpoint = _load_checkpoint(model, checkpoint_path)
    model.requires_grad_(False).eval()
    batch_size = int(config["student"].get("evaluation_batch_size", config["student"]["micro_batch_size"]))
    loader = make_openpi_loader(
        Path(external_cfg["artifact_root"]) / "shards",
        batch_size=batch_size,
        num_workers=max(0, int(config["student"]["num_workers"]) // 2),
    )
    rows_out = []
    torch.cuda.reset_peak_memory_stats(device)
    for rows in loader:
        inputs, _ = process_student_batch(processor, rows, None, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        probabilities = output["value_logits"].float().softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = probabilities.max(dim=-1).values
        for row, prediction, entropy_value, confidence_value in zip(
            rows,
            output["value"].float().cpu().tolist(),
            entropy.cpu().tolist(),
            confidence.cpu().tolist(),
            strict=True,
        ):
            metadata = row["metadata"]
            rows_out.append(
                {
                    "sample_key": row["key"],
                    "dataset_name": metadata["dataset_name"],
                    "collection": metadata["collection"],
                    "episode_index": int(metadata["episode_index"]),
                    "uuid": metadata["uuid"],
                    "anchor": int(metadata["anchor"]),
                    "length": int(metadata["length"]),
                    "progress": float(metadata["progress"]),
                    "outcome": metadata["outcome"],
                    "task_family": metadata["task_family"],
                    "target": float(metadata["value"]),
                    "target_bin": int(metadata["value_bin"]),
                    "prediction": float(prediction),
                    "entropy": float(entropy_value),
                    "confidence": float(confidence_value),
                }
            )
    rank_path = evaluation_dir / f"predictions.rank-{rank:03d}.parquet"
    _atomic_parquet(pa.Table.from_pylist(rows_out), rank_path)
    if world_size > 1:
        dist.barrier()

    report: dict[str, Any] = {}
    if rank == 0:
        paths = [evaluation_dir / f"predictions.rank-{index:03d}.parquet" for index in range(world_size)]
        frame = pd.concat([pq.read_table(path).to_pandas() for path in paths], ignore_index=True)
        if frame.sample_key.duplicated().any():
            duplicates = frame.loc[frame.sample_key.duplicated(), "sample_key"].head(10).tolist()
            raise RuntimeError(f"Duplicate OpenPI predictions: {duplicates}")
        expected = int(
            json.loads(
                (Path(external_cfg["artifact_root"]) / "manifest_stats.json").read_text(
                    encoding="utf-8"
                )
            )["samples"]
        )
        if len(frame) != expected:
            raise RuntimeError(f"OpenPI prediction coverage mismatch: {len(frame)} != {expected}")
        frame = frame.sort_values("sample_key").reset_index(drop=True)
        _atomic_parquet(pa.Table.from_pandas(frame, preserve_index=False), evaluation_dir / "predictions.parquet")
        episode_frame = _episode_frame(frame)
        _atomic_parquet(
            pa.Table.from_pandas(episode_frame, preserve_index=False),
            evaluation_dir / "episode_predictions.parquet",
        )
        bootstrap_samples = int(external_cfg["bootstrap_samples"])
        rollout = frame.loc[frame.collection == "rollout"].copy()
        demonstration = frame.loc[frame.collection == "demonstration"].copy()
        per_rollout = {
            str(name): _subset_report(
                group,
                bootstrap_samples=min(bootstrap_samples, 200),
                seed=int(config["seed"]),
            )
            for name, group in rollout.groupby("dataset_name", sort=True)
        }
        report = {
            "run_name": run_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_step": int(checkpoint.get("global_step", -1)),
            "primary_rollout": _subset_report(
                rollout, bootstrap_samples=bootstrap_samples, seed=int(config["seed"])
            ),
            "demonstration_diagnostic": _subset_report(
                demonstration, bootstrap_samples=bootstrap_samples, seed=int(config["seed"])
            ),
            "all_data_diagnostic": _subset_report(
                frame, bootstrap_samples=bootstrap_samples, seed=int(config["seed"])
            ),
            "per_rollout": per_rollout,
            "world_size": world_size,
            "per_rank_batch_size": batch_size,
            "peak_memory_gib_rank0": torch.cuda.max_memory_allocated(device) / (1024**3),
            "selection_use": "external_holdout_only",
        }
        atomic_json(evaluation_dir / "report.json", report)
        print(json.dumps(json_ready(report), ensure_ascii=False, indent=2), flush=True)
    payload = [report]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
        dist.destroy_process_group()
    return payload[0]


def _paired_bootstrap_delta(
    merged: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    success = merged.loc[merged.outcome == "success"].reset_index(drop=True)
    failure = merged.loc[merged.outcome == "failure"].reset_index(drop=True)
    fields = ("mean_prediction", "last_prediction")
    results = {}
    for field in fields:
        left_name = f"{field}_baseline"
        right_name = f"{field}_proposed"
        estimate = _auc(merged.outcome, merged[right_name]) - _auc(merged.outcome, merged[left_name])
        draws = []
        for _ in range(samples):
            draw = pd.concat(
                (
                    success.iloc[rng.integers(0, len(success), size=len(success))],
                    failure.iloc[rng.integers(0, len(failure), size=len(failure))],
                ),
                ignore_index=True,
            )
            draws.append(_auc(draw.outcome, draw[right_name]) - _auc(draw.outcome, draw[left_name]))
        results[field.replace("prediction", "auc_delta")] = {
            "estimate": float(estimate),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }
    return results


def compare_openpi(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    frames = {}
    episodes = {}
    for run_name in ("baseline", "proposed"):
        root = Path(config["output_root"]) / "student" / run_name / "external_evaluation/openpi_rollout"
        frames[run_name] = pq.read_table(root / "predictions.parquet").to_pandas()
        episodes[run_name] = pq.read_table(root / "episode_predictions.parquet").to_pandas()
    left = frames["baseline"].set_index("sample_key").sort_index()
    right = frames["proposed"].set_index("sample_key").sort_index()
    if not left.index.equals(right.index):
        raise RuntimeError("OpenPI baseline/proposed sample keys differ")
    rollout_left = left.loc[left.collection == "rollout"].reset_index()
    rollout_right = right.loc[right.collection == "rollout"].reset_index()

    ep_left = episodes["baseline"].loc[lambda frame: frame.collection == "rollout"]
    ep_right = episodes["proposed"].loc[lambda frame: frame.collection == "rollout"]
    merged = ep_left.merge(
        ep_right,
        on=["uuid", "dataset_name", "collection", "outcome"],
        suffixes=("_baseline", "_proposed"),
        validate="one_to_one",
    )
    report = {
        "baseline": _subset_report(
            rollout_left,
            bootstrap_samples=int(_cfg(config)["bootstrap_samples"]),
            seed=int(config["seed"]),
        ),
        "proposed": _subset_report(
            rollout_right,
            bootstrap_samples=int(_cfg(config)["bootstrap_samples"]),
            seed=int(config["seed"]),
        ),
        "paired_episode_auc_delta_proposed_minus_baseline": _paired_bootstrap_delta(
            merged,
            samples=int(_cfg(config)["bootstrap_samples"]),
            seed=int(config["seed"]),
        ),
        "mean_anchor_prediction_delta": float(
            (right.loc[left.index, "prediction"] - left["prediction"]).mean()
        ),
        "selection_use": "comparison_only_not_model_selection",
    }
    output_path = Path(config["output_root"]) / "openpi_comparison.json"
    atomic_json(output_path, report)
    print(json.dumps(json_ready(report), ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--config", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--workers", type=int, default=4)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--checkpoint", default="best.pt")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        build_openpi_manifest(args.config)
    elif args.command == "prepare":
        prepare_openpi(args.config, workers=args.workers)
    elif args.command == "evaluate":
        evaluate_openpi(args.config, run_name=args.run_name, checkpoint_name=args.checkpoint)
    elif args.command == "compare":
        compare_openpi(args.config)


if __name__ == "__main__":
    main()
