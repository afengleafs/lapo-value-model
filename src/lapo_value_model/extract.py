from __future__ import annotations

import io
import json
import os
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .common import atomic_json
from .config import load_config


FRAME_NAMES = ("t0.jpg", "t1.jpg", "t2.jpg", "future.jpg")


def _letterbox_jpeg(frame: av.VideoFrame, size: int, quality: int) -> bytes:
    image = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
    image.thumbnail((size, size), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def _output_base(shard_root: Path, video_path: Path) -> Path:
    source = "droid_success" if "droid_success" in video_path.parts else "droid_failure"
    chunk = video_path.parent.name
    return shard_root / source / chunk / video_path.stem


def _extract_video_task(task: tuple[str, list[dict[str, Any]], str, int, int, bool]) -> dict[str, Any]:
    video_path_raw, samples, shard_root_raw, size, quality, overwrite = task
    video_path = Path(video_path_raw)
    shard_root = Path(shard_root_raw)
    base = _output_base(shard_root, video_path)
    marker = base.with_suffix(".done.json")
    expected_outputs = [base.with_name(f"{base.name}-{split}.tar") for split in sorted({s["split"] for s in samples})]
    if marker.exists() and all(path.exists() for path in expected_outputs) and not overwrite:
        return {"video": str(video_path), "status": "skipped", "samples": len(samples)}

    base.parent.mkdir(parents=True, exist_ok=True)
    requests: dict[int, list[tuple[str, int]]] = defaultdict(list)
    metadata_by_key: dict[str, dict[str, Any]] = {}
    for sample in samples:
        indices = list(sample["history_indices"]) + [int(sample["future_index"])]
        metadata_by_key[sample["sample_key"]] = sample
        for slot, frame_index in enumerate(indices):
            # Integer microseconds make timestamp grouping stable across Python processes.
            target_us = round((float(sample["video_start_sec"]) + frame_index / float(sample["fps"])) * 1_000_000)
            requests[target_us].append((sample["sample_key"], slot))

    wanted = sorted(requests)
    found: dict[str, dict[int, tuple[bytes, float]]] = defaultdict(dict)
    pointer = 0
    tolerance_us = round(0.51 / float(samples[0]["fps"]) * 1_000_000)
    started = time.time()
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    for frame in container.decode(stream):
        if pointer >= len(wanted):
            break
        timestamp_us = round(float(frame.pts * stream.time_base) * 1_000_000)
        while pointer < len(wanted) and wanted[pointer] < timestamp_us - tolerance_us:
            raise RuntimeError(
                f"Missed requested frame in {video_path}: target_us={wanted[pointer]}, current_us={timestamp_us}"
            )
        if pointer < len(wanted) and abs(wanted[pointer] - timestamp_us) <= tolerance_us:
            encoded = _letterbox_jpeg(frame, size, quality)
            while pointer < len(wanted) and abs(wanted[pointer] - timestamp_us) <= tolerance_us:
                for sample_key, slot in requests[wanted[pointer]]:
                    found[sample_key][slot] = (encoded, timestamp_us / 1_000_000.0)
                pointer += 1
    container.close()
    if pointer != len(wanted):
        raise RuntimeError(f"Decoded {pointer}/{len(wanted)} requested timestamps from {video_path}")

    temporary_paths: dict[str, Path] = {}
    archives: dict[str, tarfile.TarFile] = {}
    try:
        for split in sorted({sample["split"] for sample in samples}):
            final_path = base.with_name(f"{base.name}-{split}.tar")
            temp_path = final_path.with_suffix(f".tar.tmp.{os.getpid()}")
            temporary_paths[split] = temp_path
            archives[split] = tarfile.open(temp_path, mode="w")
        for key in sorted(metadata_by_key):
            sample = metadata_by_key[key]
            if len(found[key]) != 4:
                raise RuntimeError(f"Incomplete sample {key}: {sorted(found[key])}")
            archive = archives[sample["split"]]
            actual_pts = []
            for slot, frame_name in enumerate(FRAME_NAMES):
                jpeg, pts = found[key][slot]
                actual_pts.append(pts)
                _add_bytes(archive, f"{key}.{frame_name}", jpeg)
            record = {k: v for k, v in sample.items() if k != "video_path"}
            record["actual_pts_sec"] = actual_pts
            _add_bytes(
                archive,
                f"{key}.json",
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )
    finally:
        for archive in archives.values():
            archive.close()
    outputs = []
    for split, temporary in temporary_paths.items():
        final_path = base.with_name(f"{base.name}-{split}.tar")
        temporary.replace(final_path)
        outputs.append(str(final_path))
    atomic_json(
        marker,
        {
            "video": str(video_path),
            "samples": len(samples),
            "timestamps": len(wanted),
            "outputs": outputs,
            "elapsed_seconds": time.time() - started,
        },
    )
    return {
        "video": str(video_path),
        "status": "written",
        "samples": len(samples),
        "elapsed_seconds": time.time() - started,
    }


def extract_shards(config_path: str | Path, workers: int = 8, overwrite: bool = False) -> None:
    config = load_config(config_path)
    artifact_root = Path(config["artifact_root"])
    sample_path = artifact_root / "samples.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(f"Run manifest first: {sample_path}")
    samples = pq.read_table(sample_path).to_pylist()
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
            overwrite,
        )
        for video_path, rows in sorted(grouped.items())
    ]
    print(f"Extracting {len(samples)} samples from {len(tasks)} packed videos with {workers} workers")
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_extract_video_task, task): task[0] for task in tasks}
        for index, future in enumerate(as_completed(future_map), 1):
            video = future_map[future]
            try:
                result = future.result()
                results.append(result)
                if index % 20 == 0 or index == len(tasks):
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
        raise RuntimeError(f"Extraction failed for {len(failures)} videos; see extraction_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_shards(args.config, workers=args.workers, overwrite=args.overwrite)

