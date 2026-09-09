from __future__ import annotations

import hashlib
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from .common import atomic_json
from .config import load_config
from .openpi_finetune import (
    DenseOpenPIDataset,
    _load_trainable_state,
    finetune_artifact_root,
    finetune_output_root,
    sha256_file,
)
from .procvlm_metrics import generate_procvlm_report
from .student import (
    QwenValueModel,
    checkpoint_value_path,
    load_processor,
    process_student_batch,
)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Latency samples must be non-empty and finite")
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _keys_sha256(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> QwenValueModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("global_step", -1)) < 0 or "trainable_model" not in checkpoint:
        raise RuntimeError(f"Invalid fine-tuning checkpoint: {checkpoint_path}")
    value_path = checkpoint_value_path(checkpoint)
    model = QwenValueModel(
        config,
        trainable=True,
        include_latent_head=True,
        value_path=value_path,
    )
    _load_trainable_state(model, checkpoint["trainable_model"], source=checkpoint_path)
    model.requires_grad_(False)
    model.eval().to(device)
    return model


def _forward(model: QwenValueModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**inputs)["value"]


def _end_to_end_pass(
    model: QwenValueModel,
    processor: Any,
    dataset: DenseOpenPIDataset,
    episode_indices: list[tuple[str, list[int]]],
    device: torch.device,
    *,
    round_index: int,
    position: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    pass_started = time.perf_counter()
    for uuid, indices in episode_indices:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        keys: list[str] = []
        for index in indices:
            row = dataset[index]
            keys.append(str(row["key"]))
            inputs, _ = process_student_batch(processor, [row], None, device)
            value = _forward(model, inputs)
            value.float().cpu()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        records.append(
            {
                "uuid": uuid,
                "anchors": len(indices),
                "seconds": float(elapsed),
                "milliseconds_per_anchor": float(1000.0 * elapsed / len(indices)),
                "sample_keys_sha256": _keys_sha256(keys),
            }
        )
    torch.cuda.synchronize(device)
    total = time.perf_counter() - pass_started
    anchors = sum(len(indices) for _, indices in episode_indices)
    return {
        "round": int(round_index),
        "position": int(position),
        "seconds": float(total),
        "anchors": int(anchors),
        "anchors_per_second": float(anchors / total),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "episodes": records,
    }


def _aggregate_end_to_end(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for run in rounds for record in run["episodes"]]
    total_seconds = sum(float(run["seconds"]) for run in rounds)
    total_anchors = sum(int(run["anchors"]) for run in rounds)
    return {
        "rounds": rounds,
        "seconds_per_episode": _summary([float(row["seconds"]) for row in records]),
        "milliseconds_per_anchor_episode_normalized": _summary(
            [float(row["milliseconds_per_anchor"]) for row in records]
        ),
        "anchors_per_second": float(total_anchors / total_seconds),
        "total_timed_seconds": float(total_seconds),
        "peak_allocated_bytes": max(int(run["peak_allocated_bytes"]) for run in rounds),
    }


def _kernel_benchmark(
    models: dict[str, QwenValueModel],
    cached_inputs: list[dict[str, torch.Tensor]],
    sample_keys: list[str],
    device: torch.device,
) -> dict[str, Any]:
    for model in models.values():
        for inputs_cpu in cached_inputs[:32]:
            inputs = {key: value.to(device) for key, value in inputs_cpu.items()}
            _forward(model, inputs)
        torch.cuda.synchronize(device)

    order_by_round = [
        ["baseline", "proposed"],
        ["proposed", "baseline"],
        ["baseline", "proposed"],
        ["proposed", "baseline"],
    ]
    measurements: dict[str, list[list[float]]] = {
        name: [[] for _ in cached_inputs] for name in models
    }
    peak: dict[str, int] = {name: 0 for name in models}
    for round_index, order in enumerate(order_by_round, start=1):
        for name in order:
            model = models[name]
            torch.cuda.reset_peak_memory_stats(device)
            for index, inputs_cpu in enumerate(cached_inputs):
                inputs = {key: value.to(device, non_blocking=False) for key, value in inputs_cpu.items()}
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _forward(model, inputs)
                end.record()
                end.synchronize()
                measurements[name][index].append(float(start.elapsed_time(end)))
            peak[name] = max(peak[name], int(torch.cuda.max_memory_allocated(device)))
            print(f"Kernel latency: round {round_index}/4 {name}", flush=True)

    result: dict[str, Any] = {
        "protocol": "preprocessed input, H2D excluded, batch=1, per-anchor median over four AB/BA rounds",
        "order_by_round": order_by_round,
        "warmup_forwards_per_model": min(32, len(cached_inputs)),
        "sample_keys": sample_keys,
        "sample_keys_sha256": _keys_sha256(sample_keys),
    }
    for name, rows in measurements.items():
        medians = [float(np.median(row)) for row in rows]
        result[name] = {
            "milliseconds_per_anchor": _summary(medians),
            "raw_repeats_ms": rows,
            "peak_allocated_bytes": peak[name],
        }
    return result


def benchmark_openpi_latency(
    config_path: str | Path,
    *,
    baseline_output_name: str = "openpi_finetune_oof",
    proposed_output_name: str = "openpi_finetune_oof_proposed",
    output_name: str = "openpi_procvlm_metrics",
    fold: int = 0,
    step: int = 2000,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Latency benchmark requires one visible GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU, for example HIP_VISIBLE_DEVICES=7")
    device = torch.device("cuda", 0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes / total_bytes < 0.8:
        raise RuntimeError(f"Visible GPU is not sufficiently idle: {free_bytes}/{total_bytes} bytes free")

    config = load_config(config_path)
    output_root = finetune_output_root(config, output_name)
    if not (output_root / "metrics_report.json").is_file():
        raise FileNotFoundError("Run evaluate-openpi-procvlm-metrics before latency")
    output_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "baseline": finetune_output_root(config, baseline_output_name),
        "proposed": finetune_output_root(config, proposed_output_name),
    }
    checkpoints = {
        name: root / f"fold-{fold}" / "checkpoints" / f"step-{step:05d}.pt"
        for name, root in sources.items()
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = DenseOpenPIDataset(finetune_artifact_root(config))
    heldout = [
        index for index, row in enumerate(dataset.rows) if int(row["heldout_fold"]) == int(fold)
    ]
    heldout.sort(key=lambda index: (str(dataset.rows[index]["uuid"]), int(dataset.rows[index]["anchor"])))
    by_episode: dict[str, list[int]] = {}
    for index in heldout:
        by_episode.setdefault(str(dataset.rows[index]["uuid"]), []).append(index)
    episode_indices = list(by_episode.items())
    keys = [str(dataset.rows[index]["sample_key"]) for index in heldout]
    if fold != 0 or step != 2000 or len(episode_indices) != 60 or len(heldout) != 2916:
        raise RuntimeError(
            f"Preregistered latency gate failed: fold={fold}, step={step}, "
            f"episodes={len(episode_indices)}, anchors={len(heldout)}"
        )

    processor = load_processor(config)
    models = {
        name: _load_model(config, path, device) for name, path in checkpoints.items()
    }
    print("Latency models loaded; checkpoint loading is excluded from timing", flush=True)

    for index in heldout:
        row = dataset[index]
        for image in row["images"]:
            image.load()
    print("Shared fold-0 image cache warmup complete", flush=True)

    for name, model in models.items():
        for index in heldout[:32]:
            row = dataset[index]
            inputs, _ = process_student_batch(processor, [row], None, device)
            _forward(model, inputs)
        torch.cuda.synchronize(device)
        print(f"Untimed model warmup complete: {name} (32 forwards)", flush=True)

    end_to_end_rounds: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    orders = [["baseline", "proposed"], ["proposed", "baseline"]]
    for round_index, order in enumerate(orders, start=1):
        for position, name in enumerate(order, start=1):
            print(f"End-to-end latency: round {round_index}/2 position {position}/2 {name}", flush=True)
            result = _end_to_end_pass(
                models[name],
                processor,
                dataset,
                episode_indices,
                device,
                round_index=round_index,
                position=position,
            )
            end_to_end_rounds[name].append(result)

    sample_positions = np.linspace(0, len(heldout) - 1, 256, dtype=np.int64)
    kernel_indices = [heldout[int(position)] for position in sample_positions]
    kernel_keys: list[str] = []
    cached_inputs: list[dict[str, torch.Tensor]] = []
    for index in kernel_indices:
        row = dataset[index]
        kernel_keys.append(str(row["key"]))
        inputs, _ = process_student_batch(processor, [row], None, torch.device("cpu"))
        cached_inputs.append({key: value.contiguous() for key, value in inputs.items()})
    kernel = _kernel_benchmark(models, cached_inputs, kernel_keys, device)

    report: dict[str, Any] = {
        "protocol": "fold-0 step-2000 online batch=1 latency; model loading excluded",
        "fold": int(fold),
        "step": int(step),
        "heldout_episodes": len(episode_indices),
        "heldout_anchors": len(heldout),
        "heldout_sample_keys_sha256": _keys_sha256(keys),
        "end_to_end": {
            "protocol": "warm filesystem cache, tar/JPEG + processor + H2D + forward + result D2H",
            "batch_size": 1,
            "order_by_round": orders,
            **{
                name: _aggregate_end_to_end(rounds)
                for name, rounds in end_to_end_rounds.items()
            },
        },
        "kernel": kernel,
        "environment": {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": int(total_bytes),
            "gpu_free_bytes_before_load": int(free_bytes),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")
            or os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dtype": "bfloat16",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "checkpoints": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in checkpoints.items()
            },
        },
    }
    atomic_json(output_root / "latency_report.json", report)
    generate_procvlm_report(output_root)
    return report
