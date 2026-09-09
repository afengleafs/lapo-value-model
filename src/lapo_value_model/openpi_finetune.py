from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import random
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Sequence

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF

from .common import (
    atomic_json,
    distributed_context,
    set_seed,
    stable_fraction,
    validate_resume_plan,
)
from .config import load_config
from .extract import _add_bytes, _letterbox_jpeg
from .openpi_eval import _atomic_parquet, _cfg, build_openpi_manifest, external_success_value
from .student import (
    QwenValueModel,
    architecture_metadata,
    checkpoint_value_path,
    load_processor,
    normalize_value_path,
    process_student_batch,
    student_loss,
    trainable_state,
)
from .teacher import LAPOTeacher
from .train_student import _cosine_lambda, _unwrap


DENSE_FRAME_NAMES = ("t0.jpg", "t1.jpg", "t2.jpg", "future.jpg")


def _ft_cfg(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _cfg(config)["finetune"]
    except KeyError as exc:
        raise KeyError("config.external_eval.openpi.finetune is required") from exc


def finetune_artifact_root(config: dict[str, Any]) -> Path:
    name = str(_ft_cfg(config).get("artifact_name", "finetune"))
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"artifact_name must be one directory name, got {name!r}")
    return Path(_cfg(config)["artifact_root"]) / name


def finetune_output_root(
    config: dict[str, Any], output_name: str = "openpi_finetune_oof"
) -> Path:
    """Return an isolated OOF output directory below the configured output root."""
    name = str(output_name)
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"output_name must be one directory name, got {output_name!r}")
    return Path(config["output_root"]) / name


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_stratified_folds(
    episodes: pd.DataFrame, *, folds: int, seed: int
) -> pd.DataFrame:
    """Assign whole episodes by dataset_name x outcome with deterministic round robin."""
    required = {"uuid", "dataset_name", "outcome"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"Episode frame missing fold fields: {sorted(missing)}")
    if folds < 2:
        raise ValueError("At least two folds are required")
    if episodes.uuid.duplicated().any():
        raise ValueError("Episode UUIDs must be unique")
    assigned = []
    total_fold_counts = np.zeros(folds, dtype=np.int64)
    outcome_order = sorted(
        episodes.outcome.unique().tolist(),
        key=lambda outcome: (-int((episodes.outcome == outcome).sum()), str(outcome)),
    )
    # A continuous cursor per outcome avoids the large fold-0 surplus that would
    # result from restarting every small dataset x outcome stratum at fold 0.
    # The initial rotation of each outcome is chosen to keep total folds balanced.
    for outcome in outcome_order:
        outcome_rows = episodes.loc[episodes.outcome == outcome]
        outcome_size = len(outcome_rows)
        candidate_scores = []
        for start in range(folds):
            additions = np.zeros(folds, dtype=np.int64)
            for position in range(outcome_size):
                additions[(start + position) % folds] += 1
            projected = total_fold_counts + additions
            candidate_scores.append(
                (
                    int(projected.max() - projected.min()),
                    float(projected.std()),
                    start,
                )
            )
        cursor = min(candidate_scores)[2]
        for dataset_name in sorted(outcome_rows.dataset_name.unique().tolist()):
            group = outcome_rows.loc[outcome_rows.dataset_name == dataset_name]
            order = sorted(
                group.index.tolist(),
                key=lambda index: (
                    stable_fraction(str(group.loc[index, "uuid"]), seed),
                    str(group.loc[index, "uuid"]),
                ),
            )
            for position, index in enumerate(order):
                fold = (cursor + position) % folds
                assigned.append((index, fold))
                total_fold_counts[fold] += 1
            cursor = (cursor + len(order)) % folds
    result = episodes.copy()
    result["heldout_fold"] = -1
    for index, fold in assigned:
        result.loc[index, "heldout_fold"] = fold
    result["heldout_fold"] = result.heldout_fold.astype(np.int64)
    if (result.heldout_fold < 0).any():
        raise RuntimeError("Not every episode received a fold")
    return result


def dense_anchor_indices(
    *, length: int, history_offsets: Sequence[int], future_offset: int, stride: int
) -> list[int]:
    if stride < 1:
        raise ValueError("Anchor stride must be positive")
    low = abs(min(int(value) for value in history_offsets))
    high = int(length) - int(future_offset) - 1
    return list(range(low, high + 1, stride)) if high >= low else []


def _count_exact_advantage_windows(
    samples: pd.DataFrame, *, horizon_source_frames: int
) -> int:
    count = 0
    for _, group in samples.groupby("uuid", sort=False):
        anchors = set(group.anchor.astype(int).tolist())
        count += sum(int(anchor) + horizon_source_frames in anchors for anchor in anchors)
    return count


def build_dense_manifest(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    openpi_cfg = _cfg(config)
    ft_cfg = _ft_cfg(config)
    sparse_root = Path(openpi_cfg["artifact_root"])
    if not (sparse_root / "episodes.parquet").is_file():
        build_openpi_manifest(config_path)
    source = pq.read_table(sparse_root / "episodes.parquet").to_pandas()
    collections = tuple(str(value) for value in ft_cfg.get("training_collections", ["rollout"]))
    known_collections = set(source.collection.astype(str))
    if not collections or not set(collections).issubset(known_collections):
        raise RuntimeError(
            f"Invalid OpenPI training_collections={collections}; available={sorted(known_collections)}"
        )
    episodes = source.loc[source.collection.isin(collections)].copy().reset_index(drop=True)
    expected_inventory = {
        frozenset({"rollout"}): (302, Counter({"failure": 214, "success": 88})),
        frozenset({"rollout", "demonstration"}): (
            400,
            Counter({"failure": 214, "success": 186}),
        ),
    }
    expected = expected_inventory.get(frozenset(collections))
    if expected is not None and (
        len(episodes) != expected[0] or Counter(episodes.outcome) != expected[1]
    ):
        raise RuntimeError(
            f"Unexpected OpenPI inventory for {collections}: "
            f"episodes={len(episodes)}, outcomes={dict(Counter(episodes.outcome))}"
        )
    if episodes.uuid.duplicated().any():
        raise RuntimeError("Duplicate episode UUIDs in OpenPI training inventory")

    folds = int(ft_cfg["folds"])
    reused_folds = ft_cfg.get("reuse_rollout_folds_from")
    if reused_folds:
        reference_path = Path(str(reused_folds)).expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        reference = pq.read_table(reference_path, columns=["uuid", "heldout_fold"]).to_pandas()
        if reference.uuid.duplicated().any():
            raise RuntimeError(f"Duplicate UUIDs in fold reference: {reference_path}")
        reference_map = reference.set_index("uuid").heldout_fold
        rollout_mask = episodes.collection == "rollout"
        missing_rollout = sorted(set(episodes.loc[rollout_mask, "uuid"]) - set(reference_map.index))
        if missing_rollout:
            raise RuntimeError(
                f"Fold reference misses {len(missing_rollout)} rollout UUIDs: {missing_rollout[:8]}"
            )
        assigned_parts = []
        if rollout_mask.any():
            rollout = episodes.loc[rollout_mask].copy()
            rollout["heldout_fold"] = rollout.uuid.map(reference_map).astype(np.int64)
            if not set(rollout.heldout_fold).issubset(set(range(folds))):
                raise RuntimeError(f"Invalid rollout fold in {reference_path}")
            assigned_parts.append(rollout)
        remaining = episodes.loc[~rollout_mask].copy()
        if not remaining.empty:
            assigned_parts.append(
                assign_stratified_folds(remaining, folds=folds, seed=int(config["seed"]))
            )
        episodes = (
            pd.concat(assigned_parts, ignore_index=True)
            .sort_values(["dataset_name", "episode_index"])
            .reset_index(drop=True)
        )
    else:
        episodes = assign_stratified_folds(episodes, folds=folds, seed=int(config["seed"]))
    stride = int(ft_cfg["anchor_stride_source_frames"])
    model_fps = int(config["data"]["fps"])
    value_bins = int(config["student"]["value_bins"])
    tmax = json.loads(
        (Path(config["artifact_root"]) / "manifest_stats.json").read_text(encoding="utf-8")
    )["tmax"]
    samples: list[dict[str, Any]] = []
    for episode in episodes.to_dict("records"):
        offsets = [int(value) for value in episode["history_offsets"]]
        future_offset = int(episode["future_margin"])
        anchors = dense_anchor_indices(
            length=int(episode["length"]),
            history_offsets=offsets,
            future_offset=future_offset,
            stride=stride,
        )
        for anchor in anchors:
            remaining = int(episode["length"]) - 1 - anchor
            value = (
                external_success_value(
                    remaining_source_frames=remaining,
                    source_fps=int(episode["source_fps"]),
                    model_fps=model_fps,
                    tmax=float(tmax[episode["task_family"]]),
                )
                if episode["outcome"] == "success"
                else -1.0
            )
            value_bin = int(
                np.clip(round((value + 1.0) * (value_bins - 1)), 0, value_bins - 1)
            )
            sample_key = (
                f"openpi-ft-{episode['dataset_name']}-{int(episode['episode_index']):06d}-"
                f"{anchor:05d}"
            )
            samples.append(
                {
                    "sample_key": sample_key,
                    "dataset_name": episode["dataset_name"],
                    "collection": str(episode["collection"]),
                    "episode_index": int(episode["episode_index"]),
                    "uuid": episode["uuid"],
                    "length": int(episode["length"]),
                    "source_fps": int(episode["source_fps"]),
                    "outcome": episode["outcome"],
                    "anchor": anchor,
                    "progress": float(anchor / max(int(episode["length"]) - 1, 1)),
                    "history_indices": [anchor + offset for offset in offsets],
                    "future_index": anchor + future_offset,
                    "value": value,
                    "value_bin": value_bin,
                    "task_family": episode["task_family"],
                    "task_prompt": episode["task_prompt"],
                    "video_path": episode["video_path"],
                    "video_start_sec": float(episode["video_start_sec"]),
                    "heldout_fold": int(episode["heldout_fold"]),
                }
            )

    sample_frame = pd.DataFrame(samples)
    expected_anchor_counts = {
        frozenset({"rollout"}): 14_855,
        frozenset({"rollout", "demonstration"}): 18_071,
    }
    expected_anchors = expected_anchor_counts.get(frozenset(collections))
    if (
        sample_frame.sample_key.nunique() != len(sample_frame)
        or (expected_anchors is not None and len(sample_frame) != expected_anchors)
    ):
        raise RuntimeError(
            f"Dense OpenPI anchor gate failed: {len(sample_frame)} rows / "
            f"{sample_frame.sample_key.nunique()} unique"
        )
    horizon_model = int(ft_cfg["advantage_horizon_model_frames"])
    source_fps_values = set(episodes.source_fps.astype(int).tolist())
    if len(source_fps_values) != 1:
        raise RuntimeError(f"Mixed OpenPI FPS is unsupported: {source_fps_values}")
    source_fps = next(iter(source_fps_values))
    horizon_source = round(horizon_model * source_fps / model_fps)
    advantage_windows = _count_exact_advantage_windows(
        sample_frame, horizon_source_frames=horizon_source
    )
    mathematically_expected = len(sample_frame) - len(episodes) * math.ceil(horizon_source / stride)
    if advantage_windows != mathematically_expected:
        raise RuntimeError(
            f"Exact advantage window count mismatch: {advantage_windows} != {mathematically_expected}"
        )

    root = finetune_artifact_root(config)
    episode_path = root / "episodes.parquet"
    sample_path = root / "samples.parquet"
    _atomic_parquet(pa.Table.from_pandas(episodes, preserve_index=False), episode_path)
    _atomic_parquet(pa.Table.from_pandas(sample_frame, preserve_index=False), sample_path)
    manifest_hash = sha256_file(sample_path)
    fold_counts = {
        str(fold): {
            "episodes": int((episodes.heldout_fold == fold).sum()),
            "anchors": int((sample_frame.heldout_fold == fold).sum()),
            "outcomes": dict(Counter(episodes.loc[episodes.heldout_fold == fold, "outcome"])),
        }
        for fold in range(folds)
    }
    stats = {
        "episodes": len(episodes),
        "episode_outcomes": dict(Counter(episodes.outcome)),
        "episode_collections": dict(Counter(episodes.collection)),
        "anchors": len(sample_frame),
        "anchor_outcomes": dict(Counter(sample_frame.outcome)),
        "anchor_collections": dict(Counter(sample_frame.collection)),
        "training_collections": list(collections),
        "folds": folds,
        "fold_counts": fold_counts,
        "anchor_stride_source_frames": stride,
        "history_offsets_source_frames": episodes.history_offsets.iloc[0].tolist(),
        "future_offset_source_frames": int(episodes.future_margin.iloc[0]),
        "advantage_horizon_model_frames": horizon_model,
        "advantage_horizon_source_frames": horizon_source,
        "advantage_windows_exact": advantage_windows,
        "manifest_hash": manifest_hash,
        "demonstration_episodes_included": int(
            (episodes.collection == "demonstration").sum()
        ),
    }
    atomic_json(root / "manifest_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def _dense_output_base(shard_root: Path, sample: dict[str, Any]) -> Path:
    video = Path(str(sample["video_path"]))
    return shard_root / str(sample["dataset_name"]) / video.parent.name / video.stem


def _extract_dense_video_task(
    task: tuple[str, list[dict[str, Any]], str, int, int]
) -> dict[str, Any]:
    video_raw, samples, shard_root_raw, size, quality = task
    video_path = Path(video_raw)
    base = _dense_output_base(Path(shard_root_raw), samples[0])
    final_path = base.with_suffix(".tar")
    marker = base.with_suffix(".done.json")
    key_hash = hashlib.sha256(
        "\n".join(sorted(str(row["sample_key"]) for row in samples)).encode()
    ).hexdigest()
    if marker.is_file() and final_path.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("sample_key_hash") == key_hash and int(payload.get("samples", -1)) == len(
            samples
        ):
            return {"video": str(video_path), "status": "skipped", "samples": len(samples)}

    base.parent.mkdir(parents=True, exist_ok=True)
    requests: dict[int, list[tuple[str, int]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = str(sample["sample_key"])
        metadata[key] = sample
        indices = [int(value) for value in sample["history_indices"]] + [
            int(sample["future_index"])
        ]
        for slot, frame_index in enumerate(indices):
            target_us = round(
                (float(sample["video_start_sec"]) + frame_index / float(sample["source_fps"]))
                * 1_000_000
            )
            requests[target_us].append((key, slot))
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
                    f"Missed dense OpenPI frame in {video_path}: "
                    f"target_us={wanted[pointer]}, current_us={timestamp_us}"
                )
            while pointer < len(wanted) and abs(wanted[pointer] - timestamp_us) <= tolerance_us:
                encoded = _letterbox_jpeg(frame, size, quality)
                for sample_key, slot in requests[wanted[pointer]]:
                    found[sample_key][slot] = (encoded, timestamp_us / 1_000_000.0)
                pointer += 1
    if pointer != len(wanted):
        raise RuntimeError(f"Decoded {pointer}/{len(wanted)} timestamps from {video_path}")

    temporary = final_path.with_suffix(f".tar.tmp.{os.getpid()}")
    with tarfile.open(temporary, mode="w") as archive:
        for key in sorted(metadata):
            if len(found[key]) != len(DENSE_FRAME_NAMES):
                raise RuntimeError(f"Incomplete dense OpenPI sample {key}: {sorted(found[key])}")
            actual_pts = []
            for slot, frame_name in enumerate(DENSE_FRAME_NAMES):
                encoded, pts = found[key][slot]
                actual_pts.append(pts)
                _add_bytes(archive, f"{key}.{frame_name}", encoded)
            record = {name: value for name, value in metadata[key].items() if name != "video_path"}
            record["actual_pts_sec"] = actual_pts
            _add_bytes(
                archive,
                f"{key}.json",
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode(),
            )
    temporary.replace(final_path)
    atomic_json(
        marker,
        {
            "video": str(video_path),
            "samples": len(samples),
            "sample_key_hash": key_hash,
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


def _build_shard_index(root: Path, expected_keys: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "shards").glob("**/*.tar")):
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".json"):
                    rows.append(
                        {
                            "sample_key": member.name[: -len(".json")],
                            "tar_path": str(path),
                        }
                    )
    frame = pd.DataFrame(rows)
    keys = set(frame.sample_key.tolist()) if not frame.empty else set()
    if frame.sample_key.duplicated().any() or keys != expected_keys:
        raise RuntimeError(
            f"Dense shard coverage mismatch: rows={len(frame)}, unique={len(keys)}, "
            f"missing={len(expected_keys - keys)}, extra={len(keys - expected_keys)}"
        )
    _atomic_parquet(
        pa.Table.from_pandas(frame.sort_values("sample_key"), preserve_index=False),
        root / "shard_index.parquet",
    )
    return frame


def extract_dense_shards(config_path: str | Path, *, workers: int = 4) -> dict[str, Any]:
    config = load_config(config_path)
    root = finetune_artifact_root(config)
    if not (root / "samples.parquet").is_file():
        build_dense_manifest(config_path)
    samples = pq.read_table(root / "samples.parquet").to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["video_path"])].append(sample)
    tasks = [
        (
            video,
            rows,
            str(root / "shards"),
            int(config["data"]["image_size"]),
            int(config["data"]["jpeg_quality"]),
        )
        for video, rows in sorted(grouped.items())
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    print(f"Extracting {len(samples)} dense anchors from {len(tasks)} videos", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_dense_video_task, task): task[0] for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            video = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(tasks)}] {result['status']} {video}", flush=True)
            except Exception as exc:
                failures.append({"video": video, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {video}: {exc}", flush=True)
    if not failures:
        _build_shard_index(root, {str(row["sample_key"]) for row in samples})
    summary = {
        "videos": len(tasks),
        "samples": len(samples),
        "written": sum(row["status"] == "written" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failures": failures,
    }
    atomic_json(root / "extraction_summary.json", summary)
    if failures:
        raise RuntimeError(f"Dense OpenPI extraction failed for {len(failures)} videos")
    return summary


def prepare_openpi_finetune(config_path: str | Path, *, workers: int = 4) -> dict[str, Any]:
    manifest = build_dense_manifest(config_path)
    extraction = extract_dense_shards(config_path, workers=workers)
    return {"manifest": manifest, "extraction": extraction}


class DenseOpenPIDataset(Dataset[dict[str, Any]]):
    """Random-access dense dataset backed by per-worker cached tar handles."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.root = Path(artifact_root)
        manifest = pq.read_table(self.root / "samples.parquet").to_pandas()
        index = pq.read_table(self.root / "shard_index.parquet").to_pandas()
        merged = manifest.merge(index, on="sample_key", validate="one_to_one")
        if len(merged) != len(manifest):
            raise RuntimeError("Dense manifest and shard index do not align")
        self.rows = merged.to_dict("records")
        self._handles: dict[str, tarfile.TarFile] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.rows[index]
        path = str(metadata["tar_path"])
        archive = self._handles.get(path)
        if archive is None:
            archive = tarfile.open(path, mode="r:")
            self._handles[path] = archive
        key = str(metadata["sample_key"])
        images = []
        for name in DENSE_FRAME_NAMES:
            handle = archive.extractfile(f"{key}.{name}")
            if handle is None:
                raise RuntimeError(f"Missing {key}.{name} in {path}")
            images.append(Image.open(io.BytesIO(handle.read())).convert("RGB"))
        clean_metadata = {name: value for name, value in metadata.items() if name != "tar_path"}
        return {"key": key, "images": images, "metadata": clean_metadata}


class IndexBatchSampler(Sampler[list[int]]):
    def __init__(self, indices: Sequence[int], *, batch_size: int) -> None:
        self.indices = list(indices)
        self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.indices), self.batch_size):
            yield self.indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)


class BalancedEpisodeBatchSampler(Sampler[list[int]]):
    """Stateless step-addressable sampler: outcome -> episode -> anchor."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        eligible_indices: Sequence[int],
        global_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        start_step: int,
        end_step: int,
    ) -> None:
        if global_batch_size % (2 * world_size) != 0:
            raise ValueError("Global batch must split evenly across outcomes and ranks")
        self.rows = rows
        self.global_batch_size = int(global_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        by_outcome: dict[str, dict[str, list[int]]] = {
            "success": defaultdict(list),
            "failure": defaultdict(list),
        }
        for index in eligible_indices:
            row = rows[index]
            by_outcome[str(row["outcome"])][str(row["uuid"])].append(int(index))
        if not by_outcome["success"] or not by_outcome["failure"]:
            raise RuntimeError("Balanced sampler needs both success and failure episodes")
        self.by_outcome = {
            outcome: {uuid: sorted(indices) for uuid, indices in episodes.items()}
            for outcome, episodes in by_outcome.items()
        }

    def global_batch(self, step: int) -> list[int]:
        rng = np.random.default_rng(self.seed + int(step) * 1_000_003)
        half = self.global_batch_size // 2
        result: list[int] = []
        for outcome in ("success", "failure"):
            episodes = self.by_outcome[outcome]
            uuids = sorted(episodes)
            choices = rng.integers(0, len(uuids), size=half)
            for choice in choices:
                anchors = episodes[uuids[int(choice)]]
                result.append(anchors[int(rng.integers(0, len(anchors)))])
        rng.shuffle(result)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.end_step):
            global_indices = self.global_batch(step)
            yield global_indices[self.rank :: self.world_size]

    def __len__(self) -> int:
        return max(0, self.end_step - self.start_step)


def _collate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def _image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    value = TF.pil_to_tensor(image).float().div_(255.0)
    if tuple(value.shape[-2:]) != (size, size):
        value = TF.resize(value, [size, size], antialias=True)
    return value


def generate_openpi_latents(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    rank, _, world_size, device = distributed_context()
    if device.type != "cuda":
        raise RuntimeError("OpenPI latent generation requires GPU")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    set_seed(int(config["seed"]), rank)
    root = finetune_artifact_root(config)
    dataset = DenseOpenPIDataset(root)
    indices = list(range(rank, len(dataset), world_size))
    batch_size = int(config["teacher"]["micro_batch_size"])
    loader = DataLoader(
        dataset,
        batch_sampler=IndexBatchSampler(indices, batch_size=batch_size),
        num_workers=max(0, int(config["teacher"]["num_workers"]) // 2),
        pin_memory=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(int(config["seed"]) + rank),
    )
    checkpoint_path = Path(config["output_root"]) / "teacher/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = LAPOTeacher(int(config["teacher"]["latent_dim"])).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    output_dir = root / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    with torch.inference_mode():
        for rows in loader:
            current = torch.stack(
                [_image_tensor(row["images"][2], int(config["teacher"]["image_size"])) for row in rows]
            ).to(device, non_blocking=True)
            future = torch.stack(
                [_image_tensor(row["images"][3], int(config["teacher"]["image_size"])) for row in rows]
            ).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, _ = teacher.encode(current, future)
            rows_out.extend(
                {"sample_key": row["key"], "latent": latent}
                for row, latent in zip(rows, mu.float().cpu().tolist(), strict=True)
            )
    rank_path = output_dir / f"rank-{rank:03d}.parquet"
    _atomic_parquet(pa.Table.from_pylist(rows_out), rank_path)
    if world_size > 1:
        dist.barrier()
    summary: dict[str, Any] = {}
    if rank == 0:
        frames = [
            pq.read_table(output_dir / f"rank-{index:03d}.parquet").to_pandas()
            for index in range(world_size)
        ]
        frame = pd.concat(frames, ignore_index=True)
        expected = pq.read_table(root / "samples.parquet", columns=["sample_key"]).num_rows
        latent_values = np.stack(frame.latent.to_numpy())
        if (
            len(frame) != expected
            or frame.sample_key.nunique() != expected
            or latent_values.shape != (expected, int(config["teacher"]["latent_dim"]))
            or not np.isfinite(latent_values).all()
        ):
            raise RuntimeError("Dense OpenPI latent coverage/dimension/finiteness gate failed")
        summary = {
            "rows": len(frame),
            "unique_keys": int(frame.sample_key.nunique()),
            "latent_dim": latent_values.shape[1],
            "finite": True,
            "teacher_checkpoint": str(checkpoint_path),
            "teacher_checkpoint_sha256": sha256_file(checkpoint_path),
            "world_size": world_size,
        }
        atomic_json(output_dir / "summary.json", summary)
    payload = [summary]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
        dist.destroy_process_group()
    return payload[0]


def load_openpi_latents(root: str | Path) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    latent_root = Path(root) / "latents"
    summary_path = latent_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    world_size = int(json.loads(summary_path.read_text(encoding="utf-8"))["world_size"])
    paths = [latent_root / f"rank-{rank:03d}.parquet" for rank in range(world_size)]
    for path in paths:
        table = pq.read_table(path, columns=["sample_key", "latent"])
        for key, latent in zip(
            table["sample_key"].to_pylist(), table["latent"].to_pylist(), strict=True
        ):
            if str(key) in lookup:
                raise RuntimeError(f"Duplicate OpenPI latent: {key}")
            lookup[str(key)] = np.asarray(latent, dtype=np.float32)
    return lookup


def resolve_starting_checkpoint(
    config: dict[str, Any], override: str | Path | None = None
) -> Path:
    if override is not None:
        path = Path(override).expanduser().resolve()
    else:
        configured = _ft_cfg(config).get("starting_checkpoint")
        if configured:
            path = Path(configured).expanduser().resolve()
        else:
            selection_path = Path(config["output_root"]) / "final_selection.json"
            if not selection_path.is_file():
                raise FileNotFoundError(
                    f"No starting checkpoint configured and selection is absent: {selection_path}"
                )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            path = (
                Path(config["output_root"])
                / "student"
                / str(selection["selected_run"])
                / "best.pt"
            )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_steps(total_steps: int, save_every_steps: int) -> list[int]:
    if total_steps < 1 or save_every_steps < 1 or total_steps % save_every_steps:
        raise ValueError("Total steps must be a positive multiple of save_every_steps")
    return list(range(save_every_steps, total_steps + 1, save_every_steps))


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_last_link(checkpoint_path: Path, last_path: Path) -> None:
    temporary = last_path.with_suffix(last_path.suffix + f".tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    os.link(checkpoint_path, temporary)
    temporary.replace(last_path)


def _prediction_columns(row: dict[str, Any], prediction: float, entropy: float) -> dict[str, Any]:
    metadata = row["metadata"]
    return {
        "sample_key": row["key"],
        "dataset_name": metadata["dataset_name"],
        "collection": str(metadata["collection"]),
        "episode_index": int(metadata["episode_index"]),
        "uuid": metadata["uuid"],
        "anchor": int(metadata["anchor"]),
        "length": int(metadata["length"]),
        "source_fps": int(metadata["source_fps"]),
        "progress": float(metadata["progress"]),
        "outcome": metadata["outcome"],
        "task_family": metadata["task_family"],
        "target": float(metadata["value"]),
        "target_bin": int(metadata["value_bin"]),
        "prediction": float(prediction),
        "entropy": float(entropy),
        "heldout_fold": int(metadata["heldout_fold"]),
    }


@torch.inference_mode()
def predict_heldout(
    model: torch.nn.Module,
    processor: Any,
    dataset: DenseOpenPIDataset,
    *,
    fold: int,
    step: int,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    output_root: Path,
) -> Path:
    prediction_dir = output_root / f"fold-{fold}" / "predictions"
    final_path = prediction_dir / f"step-{step:05d}.parquet"
    skip = final_path.is_file() if rank == 0 else False
    if world_size > 1:
        payload = [skip]
        dist.broadcast_object_list(payload, src=0)
        skip = bool(payload[0])
    if skip:
        return final_path
    indices = [
        index
        for index, row in enumerate(dataset.rows)
        if int(row["heldout_fold"]) == int(fold)
    ]
    local_indices = indices[rank::world_size]
    loader = DataLoader(
        dataset,
        batch_sampler=IndexBatchSampler(local_indices, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        # Worker base seeds must not advance the model RNG.  Otherwise a resume
        # that skips an already-written prediction file would change dropout.
        generator=torch.Generator().manual_seed(91_000_000 + fold * 10_000 + step * 10 + rank),
    )
    model.eval()
    rows_out: list[dict[str, Any]] = []
    for rows in loader:
        # Future images exist only to build offline Teacher targets.  This call
        # deliberately consumes images[:3] and receives no latent lookup.
        inputs, _ = process_student_batch(processor, rows, None, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        probabilities = output["value_logits"].float().softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        rows_out.extend(
            _prediction_columns(row, prediction, entropy_value)
            for row, prediction, entropy_value in zip(
                rows,
                output["value"].float().cpu().tolist(),
                entropy.cpu().tolist(),
                strict=True,
            )
        )
    rank_path = prediction_dir / f"step-{step:05d}.rank-{rank:03d}.parquet"
    _atomic_parquet(pa.Table.from_pylist(rows_out), rank_path)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        frames = [
            pq.read_table(prediction_dir / f"step-{step:05d}.rank-{index:03d}.parquet").to_pandas()
            for index in range(world_size)
        ]
        frame = pd.concat(frames, ignore_index=True).sort_values("sample_key")
        expected_keys = {str(dataset.rows[index]["sample_key"]) for index in indices}
        actual_keys = set(frame.sample_key.tolist())
        if len(frame) != len(expected_keys) or actual_keys != expected_keys:
            raise RuntimeError(
                f"Fold {fold} step {step} held-out coverage mismatch: "
                f"rows={len(frame)}, expected={len(expected_keys)}"
            )
        if set(frame.heldout_fold.astype(int)) != {fold}:
            raise RuntimeError("Held-out prediction contains a training-fold episode")
        _atomic_parquet(pa.Table.from_pandas(frame, preserve_index=False), final_path)
    if world_size > 1:
        dist.barrier()
    return final_path


def _parameter_groups(
    model: torch.nn.Module, *, adapter_lr: float, head_lr: float
) -> list[dict[str, Any]]:
    adapters = []
    heads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("value_head") or name.startswith("latent_head"):
            heads.append(parameter)
        else:
            adapters.append(parameter)
    return [
        {"params": adapters, "lr": adapter_lr},
        {"params": heads, "lr": head_lr},
    ]


def _load_trainable_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor], *, source: str | Path
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {source}: {unexpected}")
    expected_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(expected_trainable.intersection(missing))
    if missing_trainable:
        raise RuntimeError(
            f"Checkpoint {source} is missing trainable keys: {missing_trainable[:16]}"
        )


def train_openpi_fold(
    config_path: str | Path,
    *,
    fold: int,
    starting_checkpoint: str | Path | None = None,
    output_name: str = "openpi_finetune_oof",
    lambda_latent_override: float | None = None,
    value_path: str = "direct",
) -> None:
    config = load_config(config_path)
    ft_cfg = _ft_cfg(config)
    folds = int(ft_cfg["folds"])
    if not 0 <= fold < folds:
        raise ValueError(f"Fold must be in [0, {folds}): {fold}")
    rank, local_rank, world_size, device = distributed_context()
    if device.type != "cuda":
        raise RuntimeError("OpenPI fine-tuning requires GPU")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    expected_world = int(ft_cfg.get("world_size_per_fold", 2))
    if world_size != expected_world:
        raise RuntimeError(f"OpenPI fold training requires world_size={expected_world}, got {world_size}")
    set_seed(int(config["seed"]) + fold * 10_000, rank)

    artifact_root = finetune_artifact_root(config)
    output_root = finetune_output_root(config, output_name)
    fold_root = output_root / f"fold-{fold}"
    if rank == 0:
        fold_root.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    manifest_stats = json.loads(
        (artifact_root / "manifest_stats.json").read_text(encoding="utf-8")
    )
    start_path = resolve_starting_checkpoint(config, starting_checkpoint)
    start_hash = sha256_file(start_path)
    start_state = torch.load(start_path, map_location="cpu", weights_only=False)
    if "trainable_model" not in start_state:
        raise RuntimeError(f"Not a Student checkpoint: {start_path}")
    value_path = normalize_value_path(value_path)
    starting_value_path = checkpoint_value_path(start_state)
    if starting_value_path != value_path:
        raise RuntimeError(
            f"Starting checkpoint value_path={starting_value_path!r} does not match "
            f"requested value_path={value_path!r}: {start_path}"
        )

    processor = load_processor(config)
    dataset = DenseOpenPIDataset(artifact_root)
    heldout_uuids = {
        str(row["uuid"]) for row in dataset.rows if int(row["heldout_fold"]) == fold
    }
    train_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if int(row["heldout_fold"]) != fold
    ]
    train_uuids = {str(dataset.rows[index]["uuid"]) for index in train_indices}
    if heldout_uuids & train_uuids:
        raise RuntimeError("Episode leakage between fold train and held-out sets")

    lambda_latent = (
        float(ft_cfg["lambda_latent"])
        if lambda_latent_override is None
        else float(lambda_latent_override)
    )
    if lambda_latent < 0:
        raise ValueError(f"lambda_latent must be non-negative, got {lambda_latent}")
    latent_lookup: dict[str, np.ndarray] | None = None
    if lambda_latent > 0:
        latent_lookup = load_openpi_latents(artifact_root)
        all_keys = {str(row["sample_key"]) for row in dataset.rows}
        if set(latent_lookup) != all_keys:
            raise RuntimeError(
                f"OpenPI latent coverage mismatch: {len(latent_lookup)} != {len(all_keys)}"
            )

    model: torch.nn.Module = QwenValueModel(
        config, trainable=True, include_latent_head=True, value_path=value_path
    ).to(device)
    _load_trainable_state(model, start_state["trainable_model"], source=start_path)
    architecture = architecture_metadata(model)
    starting_initialization = start_state.get("initialization")
    if isinstance(starting_initialization, dict):
        initialization_policy = str(starting_initialization.get("policy", "unknown"))
        reset_modules = sorted(
            {
                str(name).split(".", 1)[0]
                for name in starting_initialization.get("reset_parameter_names", [])
            }
        )
    else:
        initialization_policy = "full_checkpoint"
        reset_modules = []
    adapter_lr = float(ft_cfg["adapter_learning_rate"])
    head_lr = float(ft_cfg["head_learning_rate"])
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, adapter_lr=adapter_lr, head_lr=head_lr),
        weight_decay=float(config["student"]["weight_decay"]),
    )
    total_steps = int(ft_cfg["steps"])
    save_every = int(ft_cfg["save_every_steps"])
    save_steps = checkpoint_steps(total_steps, save_every)
    warmup = int(ft_cfg["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_lambda(step, total_steps, warmup)
    )
    micro_batch = int(ft_cfg.get("micro_batch_size", config["student"]["micro_batch_size"]))
    accumulation = int(ft_cfg.get("grad_accum_steps", 1))
    if accumulation != 1:
        raise RuntimeError("The exact OpenPI sampler currently requires grad_accum_steps=1")
    global_batch = micro_batch * world_size * accumulation
    training_plan = {
        "protocol": "openpi_episode_oof_v3",
        "fold": fold,
        "folds": folds,
        "steps": total_steps,
        "save_every_steps": save_every,
        "checkpoint_steps": save_steps,
        "world_size": world_size,
        "micro_batch_size": micro_batch,
        "global_batch_size": global_batch,
        "grad_accum_steps": accumulation,
        "sampler": "outcome_1to1_episode_uniform_anchor_uniform",
        "train_episodes": len(train_uuids),
        "heldout_episodes": len(heldout_uuids),
        "train_anchors": len(train_indices),
        "heldout_anchors": len(dataset) - len(train_indices),
        "manifest_hash": manifest_stats["manifest_hash"],
        "starting_checkpoint": str(start_path),
        "starting_checkpoint_sha256": start_hash,
        "adapter_learning_rate": adapter_lr,
        "head_learning_rate": head_lr,
        "warmup_steps": warmup,
        "scheduler": "cosine",
        "lambda_latent": lambda_latent,
        "architecture_version": architecture["version"],
        "value_path": architecture["value_path"],
        "value_feature_dim": architecture["value_feature_dim"],
        "initialization_policy": initialization_policy,
        "reset_modules": reset_modules,
        "selection_step": total_steps,
        "early_stopping": False,
    }

    global_step = 0
    last_path = fold_root / "last.pt"
    checkpoint = None
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        validate_resume_plan(checkpoint, training_plan, last_path)
        _load_trainable_state(model, checkpoint["trainable_model"], source=last_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    if checkpoint is not None:
        rng_states = checkpoint.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != world_size:
            raise RuntimeError("Resume checkpoint lacks per-rank RNG state")
        _restore_rng_state(rng_states[rank])

    evaluation_batch = int(
        ft_cfg.get(
            "evaluation_micro_batch_size",
            config["student"].get("validation_micro_batch_size", micro_batch),
        )
    )
    num_workers = int(ft_cfg.get("num_workers", config["student"]["num_workers"]))
    writer = SummaryWriter(fold_root / "tensorboard") if rank == 0 else None
    metrics_path = fold_root / "train_metrics.jsonl"
    predict_heldout(
        model,
        processor,
        dataset,
        fold=fold,
        step=global_step,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=evaluation_batch,
        num_workers=max(0, num_workers // 2),
        output_root=output_root,
    )
    if global_step >= total_steps:
        if writer:
            writer.close()
        if world_size > 1:
            dist.destroy_process_group()
        return

    sampler = BalancedEpisodeBatchSampler(
        dataset.rows,
        eligible_indices=train_indices,
        global_batch_size=global_batch,
        rank=rank,
        world_size=world_size,
        seed=int(config["seed"]) + fold * 10_000,
        start_step=global_step,
        end_step=total_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(
            int(config["seed"]) + fold * 10_000 + rank + global_step * 101
        ),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    for rows in loader:
        inputs, targets = process_student_batch(processor, rows, latent_lookup, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
            loss, loss_metrics = student_loss(output, targets, lambda_latent)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite OpenPI fold loss: "
                f"{ {name: float(value) for name, value in loss_metrics.items()} }"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if rank == 0 and (global_step <= 5 or global_step % 20 == 0):
            metric_values = {name: float(value) for name, value in loss_metrics.items()}
            payload = {
                "fold": fold,
                "step": global_step,
                "grad_norm": float(grad_norm),
                "learning_rates": scheduler.get_last_lr(),
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                **metric_values,
            }
            print(json.dumps(payload), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            if writer:
                for name, value in metric_values.items():
                    writer.add_scalar(f"train/{name}", value, global_step)

        if global_step in save_steps:
            local_rng = _capture_rng_state()
            if world_size > 1:
                gathered: list[Any] | None = [None] * world_size if rank == 0 else None
                dist.gather_object(local_rng, gathered, dst=0)
            else:
                gathered = [local_rng]
            checkpoint_path = fold_root / "checkpoints" / f"step-{global_step:05d}.pt"
            if rank == 0:
                state = {
                    "trainable_model": trainable_state(_unwrap(model)),
                    "architecture": architecture,
                    "initialization": starting_initialization,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "sampler_position": global_step,
                    "rng_states": gathered,
                    "fold": fold,
                    "fold_manifest_hash": manifest_stats["manifest_hash"],
                    "starting_checkpoint": str(start_path),
                    "starting_checkpoint_sha256": start_hash,
                    "training_plan": training_plan,
                    "config": config,
                }
                _atomic_torch_save(state, checkpoint_path)
                _atomic_last_link(checkpoint_path, last_path)
            if world_size > 1:
                dist.barrier()
            predict_heldout(
                model,
                processor,
                dataset,
                fold=fold,
                step=global_step,
                device=device,
                rank=rank,
                world_size=world_size,
                batch_size=evaluation_batch,
                num_workers=max(0, num_workers // 2),
                output_root=output_root,
            )
            model.train()

    if global_step != total_steps:
        raise RuntimeError(f"Fold {fold} ended at step {global_step}, expected {total_steps}")
    if writer:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()
