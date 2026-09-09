from __future__ import annotations

import io
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.transforms import functional as TF


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _iter_tar(path: Path) -> Iterator[dict[str, Any]]:
    current_key: str | None = None
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, suffix = member.name.split(".", 1)
            if current_key is not None and key != current_key:
                raise RuntimeError(f"Non-contiguous sample {current_key} in {path}")
            current_key = key
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot extract {member.name} from {path}")
            payloads[suffix] = handle.read()
            if suffix == "json":
                record = json.loads(payloads.pop("json"))
                images = [
                    Image.open(io.BytesIO(payloads[name])).convert("RGB")
                    for name in ("t0.jpg", "t1.jpg", "t2.jpg", "future.jpg")
                ]
                yield {"key": key, "images": images, "metadata": record}
                current_key = None
                payloads = {}
    if current_key is not None or payloads:
        raise RuntimeError(f"Incomplete final sample in {path}")


class TarIterableDataset(IterableDataset):
    def __init__(
        self,
        shard_root: str | Path,
        split: str,
        *,
        repeat: bool,
        shuffle: bool,
        seed: int,
        balance_outcomes: bool = False,
    ) -> None:
        super().__init__()
        self.shard_root = Path(shard_root)
        self.split = split
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.balance_outcomes = balance_outcomes

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assigned(self, source: str | None = None) -> list[Path]:
        pattern = f"{source}/**/*-{self.split}.tar" if source else f"**/*-{self.split}.tar"
        shards = sorted(self.shard_root.glob(pattern))
        rank, world = _rank_world()
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        global_worker = rank * workers + worker_id
        global_workers = world * workers
        if global_workers > len(shards):
            # Tiny smoke sets may contain fewer shards than DDP workers. Replication is
            # preferable to leaving a rank empty and deadlocking collective training.
            assigned = [shards[global_worker % len(shards)]] if shards else []
        else:
            assigned = shards[global_worker::global_workers]
        if not assigned:
            raise RuntimeError(
                f"No {self.split} shards assigned to global worker {global_worker}/{global_workers}"
            )
        return assigned

    def _stream(self, shards: list[Path], salt: int) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed + self.epoch * 1009 + salt)
        while True:
            order = list(shards)
            if self.shuffle:
                rng.shuffle(order)
            for path in order:
                rows = list(_iter_tar(path))
                if self.shuffle:
                    rng.shuffle(rows)
                yield from rows
            if not self.repeat:
                break

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.balance_outcomes:
            yield from self._stream(self._assigned(), 0)
            return
        success = self._stream(self._assigned("droid_success"), 17)
        failure = self._stream(self._assigned("droid_failure"), 29)
        while True:
            yield next(success)
            yield next(failure)


def _image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    tensor = TF.pil_to_tensor(image).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (size, size):
        tensor = TF.resize(tensor, [size, size], antialias=True)
    return tensor


def teacher_collate(rows: list[dict[str, Any]], size: int) -> dict[str, Any]:
    current = torch.stack([_image_tensor(row["images"][2], size) for row in rows])
    future = torch.stack([_image_tensor(row["images"][3], size) for row in rows])
    return {
        "current": current,
        "future": future,
        "keys": [row["key"] for row in rows],
        "metadata": [row["metadata"] for row in rows],
    }


def make_teacher_loader(
    shard_root: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
    train: bool,
) -> tuple[TarIterableDataset, DataLoader]:
    dataset = TarIterableDataset(
        shard_root,
        split,
        repeat=train,
        shuffle=train,
        seed=seed,
        balance_outcomes=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=lambda rows: teacher_collate(rows, image_size),
        persistent_workers=False,
    )
    return dataset, loader


def student_raw_collate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def make_student_loader(
    shard_root: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    train: bool,
) -> tuple[TarIterableDataset, DataLoader]:
    dataset = TarIterableDataset(
        shard_root,
        split,
        repeat=train,
        shuffle=train,
        seed=seed,
        balance_outcomes=train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=student_raw_collate,
        persistent_workers=False,
    )
    return dataset, loader
