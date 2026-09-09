# LAPO + Qwen3-VL RECAP-style Value Model

This repository trains a strictly causal, language-conditioned 201-bin distributional value model on local DROID success/failure data. During training, a LAPO teacher extracts a 32D transition latent from the current frame and a future frame. The deployed model reads only three history frames, a coarse task-language prompt, and `observation.images.left_external`; it never sees future frames.

Method notes: [docs/recap_value_model_research.md](docs/recap_value_model_research.md). Architecture I/O: [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

## Hardware

All reported training and evaluation runs used **two Hygon BW1000_H DCUs** (a ROCm/DTK accelerator, not NVIDIA).

| Item | Spec |
|---|---|
| Accelerator | 2× Hygon BW1000_H DCU |
| Physical devices | GPU 6 and 7 (`HIP_VISIBLE_DEVICES=6,7` maps to in-process `cuda:0,cuda:1`) |
| Per-card memory | 64 GiB |
| Stack | DTK / ROCm PyTorch 2.7.1, Python 3.11.9, transformers 4.57.1, bf16 |
| Student train memory | micro-batch 160/card → measured `max_memory_allocated` ≈ **40.53 GiB/card** |
| Student val memory | micro-batch 80/card, so variable-length sequences do not push the cache toward the 64 GiB limit |
| OpenPI folds | 8-card parallel for folds 0–3, then GPUs 6,7 for fold 4 |
| Inference (batch 1) | ≈ 108 ms/anchor (≈ 9.2 anchors/s), peak ≈ 8.26 GiB/card |

On this ROCm stack, vision fused-attention backward produces non-finite LoRA gradients. Vision attention is therefore fixed to eager; language attention still uses the efficient kernel.

## Datasets

### DROID (pretraining)

LeRobot v3.0 ports of [DROID-COMMUNITY](https://droid-dataset.github.io/): [jnogga/droid_success](https://huggingface.co/datasets/jnogga/droid_success) and [jnogga/droid_failure](https://huggingface.co/datasets/jnogga/droid_failure). Robot: Franka Emika Panda + Robotiq 2F-85.

| Spec | Value |
|---|---|
| Format | LeRobot v3.0 |
| FPS | 15 |
| Source video | H.264 RGB, 720×1280 |
| Source cameras | `left_external`, `right_external`, `wrist` |
| Camera used here | **`observation.images.left_external` only** |
| Student history | frames `[-10, -5, 0]` ≈ 0.67 s / 0.33 s / now |
| Teacher future | `+5` frames ≈ 0.33 s |
| Student image | letterbox to 336×336, JPEG quality 90 |
| Teacher image | 224×224 |
| Anchors / episode | 8 |
| Split | 80 / 10 / 10 train / val / test |
| Raw episodes | 53,282 success + 13,747 failure |
| Eligible episodes | 52,475 success + 13,422 failure (19 shared stations) |
| Filters dropped | too-short, no-action, unshared-station, duplicate UUID, inconsistent failure labels |
| Anchors | **526,718** total; train / val / test = **419,344 / 53,772 / 53,602** |
| Anchors by outcome | 419,800 success / 106,918 failure |
| Language | coarse task-family prompt, shared by success and failure; the prompt does not contain the outcome label |

Task families (from the DROID manifest): `container_transfer`, `reposition`, `lid`, `fold_spread`, `slidable_open_close`, `clean`, `hang`, `button`, `pour`, `tool_use`, `twist`, `hinged_open_close`, `bagging`, `stir`, `curtain`, `multi_step` (failure only), and `other` (capped at 15% of the split).

### OpenPI FR3 plug (finetune / eval)

Private LeRobot collection `openpi_rollout`: Franka FR3 plug insertion at 30 Hz, cameras `camera_0` / `camera_1` at 480×640 RGB (AV1). This repo uses **`observation.images.camera_1`**.

| Protocol | Episodes | Anchors | Notes |
|---|---:|---:|---|
| 302-rollout OOF | 302 (88 success / 214 failure) | 14,855 | 98 demonstration episodes excluded; 5 episode-level folds stratified by `dataset_name × outcome` |
| all400 | 400 (98 demos + 302 rollouts) | 18,071 | seed `20260901`; 5-fold, 2,000 steps/fold, batch 16 |

OpenPI history is `[-20, -10, 0]` at 30 Hz (same wall-clock spacing as DROID). Teacher future is `+10` source frames. Anchors are taken every 10 source frames. Strict A50 uses 50 model frames = 100 source frames, which yields **11,835** A50 windows under that stride.

Report OpenPI numbers as **OOF finetune evaluation**, not as an untouched external holdout.

## Frozen training config

- Backbone: local `Qwen3-VL-2B-Instruct`, bf16.
- LoRA: rank 16; language q/k/v/o and vision qkv/proj.
- Value Head: `2048 → 512 → 256 → 201`, expectation over bins in `[-1, 0]`.
- Latent Head: `2048 → 512 → 128 → 32`, training-only.
- Student loss: categorical CE + `0.1 × Huber(latent)`.
- Two cards: physical GPUs 6 and 7; train micro-batch 160/card, accum 1, global effective batch 320; val micro-batch 80/card.

## Latent-hidden cascade (OpenPI all400)

The model supports `value_path=direct|latent_hidden`. The old baseline keeps a direct value head `2048→512→256→201`. The proposed model uses a `2048→512→128` latent trunk, feeds that 128D hidden state into a `128→512→256→201` value head, and aligns a separate `128→32` projection to the teacher latent. Value CE updates the latent trunk but does not flow through the 32D projection. Deployed proposed models keep the latent trunk and still read only three history frames and task language — no future frames and no teacher outputs.

Five-fold retraining entry point:

```bash
scripts/run_openpi_all400_latent128_seed20260901.sh
```

That protocol uses 400 episodes, batch size 16, 2,000 steps per fold, checkpoints at steps 1000/2000, and held-out OOF inference at steps 0/1000/2000. Baseline fully loads the old DROID value head. Proposed migrates the old LoRA / latent head and reseeds a new value head. The comparison is therefore the full architecture-plus-initialization package, not a single causal effect of the architecture alone.

## Environment

The project venv uses the system DTK/ROCm PyTorch:

```bash
.venv/bin/python -m compileall -q src
.venv/bin/python -m pytest -q
```

All GPU commands set `HIP_VISIBLE_DEVICES=6,7` explicitly.

## Official training pipeline

Each stage writes to its own directory. Frame extraction and checkpoints are resumable.

Fixed budgets below. “Sample presentations” include DDP padding / resampling and are not unique-anchor counts.

| Stage | micro-batch/card | global batch | optimizer steps | sample presentations |
|---|---:|---:|---:|---:|
| LAPO teacher, 20 epochs | 112 | 224 | 37,460 | 8,391,040 |
| screen-baseline | 160 | 320 | 400 | 128,000 |
| screen-proposed | 160 | 320 | 400 | 128,000 |
| screen-random | 160 | 320 | 400 | 128,000 |
| screen-shuffled | 160 | 320 | 400 | 128,000 |
| full baseline, 3 epochs | 160 | 320 | 3,933 | 1,258,560 |
| full proposed, 3 epochs | 160 | 320 | 3,933 | 1,258,560 |

Teacher uses AdamW at `5.25e-4`. Student LoRA adapter LR is `5e-5`; value/latent head LR is `5e-4`. The four screen runs share the same 400-step budget and only change the latent supervision mode. They are diagnostic ablations, not a substitute for the full baseline/proposed trains.

```bash
.venv/bin/lapo-value preflight --config configs/full.yaml
.venv/bin/lapo-value manifest --config configs/full.yaml
.venv/bin/lapo-value extract --config configs/full.yaml --workers 12

HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.train_teacher --config configs/full.yaml

HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.generate_latents --config configs/full.yaml

HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.train_student --config configs/full.yaml \
  --run-name proposed --latent-mode true
```

Official teacher latent generation has a quality gate: validation reconstruction gain must be > 0. Each student epoch writes rank-level validation Parquet; the best checkpoint is selected by macro Spearman.

## Required ablations

Four experiments must use the same optimizer-step budget:

```bash
# A. 400-step no-LAPO screen
... -m lapo_value_model.train_student --config configs/full.yaml \
  --run-name screen-baseline --latent-mode none --max-optimizer-steps 400

# B. 400-step proposed screen
... -m lapo_value_model.train_student --config configs/full.yaml \
  --run-name screen-proposed --latent-mode true --max-optimizer-steps 400

# C/D. 400-step controls
... -m lapo_value_model.train_student --config configs/full.yaml \
  --run-name screen-random --latent-mode random --max-optimizer-steps 400
... -m lapo_value_model.train_student --config configs/full.yaml \
  --run-name screen-shuffled --latent-mode shuffled --max-optimizer-steps 400
```

After screening, train full `baseline --latent-mode none` and `proposed --latent-mode true` in new directories: 3 epochs / 3,933 optimizer steps each. The final DROID-initialized model is chosen only by DROID validation macro-Spearman between those two full runs; ties take baseline. Later OpenPI 2k runs use strict episode-level OOF, so they must be reported as OOF finetune evaluation.

## OpenPI 302-rollout five-fold OOF finetune

Demonstration episodes (98) are fully excluded. Only the 302 rollout episodes (88 success / 214 failure) are used.

- Stratify by `dataset_name × outcome` and assign whole episodes to 5 folds; each episode is held out exactly once.
- Each fold initializes from the same final DROID checkpoint and trains 2,000 optimizer steps.
- Global batch 320; sample success/failure 1:1, then uniformly over episodes and uniformly over anchors inside an episode.
- LoRA LR `1e-5`, value/latent head LR `1e-4`, warmup 100, cosine decay.
- Atomically save a full resumable checkpoint every 100 steps (20 per fold, 100 total).
- Steps 0, 100, …, 2000 infer only on that fold’s held-out episodes. Step 2000 is the pre-registered primary result; do not pick a checkpoint from the OpenPI curve.

```bash
# 1. CPU: dense manifest, fold split, four-frame tars
.venv/bin/lapo-value prepare-openpi-finetune --config configs/full.yaml --workers 4

# 2. GPU: official LAPO teacher → 14,855 32D latents
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.cli generate-openpi-latents --config configs/full.yaml

# 3. 8-card parallel folds 0–3, then GPUs 6,7 for fold 4, then aggregate/report
scripts/run_openpi_finetune_oof.sh
```

Official outputs live under `outputs/full/openpi_finetune_oof/`: per-fold checkpoints/predictions, 21 OOF prediction/advantage/metrics files, `curve_data.parquet`, `oof_report.json`, and `report.html`. The report includes first/mean/last episode AUC, three value-gap metrics, MAE/RMSE, success-episode macro-Spearman, temporal monotonicity, 4-stage matched AUC, A50 effect size and q70 positive-rate, plus 2,000 outcome-stratified episode bootstrap 95% CIs.

## Test evaluation and deployment export

```bash
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.evaluate --config configs/full.yaml \
  --run-name proposed --checkpoint best.pt

.venv/bin/lapo-value export --config configs/full.yaml \
  --run-name proposed --checkpoint best.pt
```

The eval report includes MAE/RMSE, episode macro Spearman, success-trajectory temporal monotonicity, success/failure AUC, per-task metrics, stage-matched AUC, and advantage sign quality, and checks 100% test-anchor coverage.

Export directory: `outputs/full/student/proposed/export`

- standard PEFT adapter
- `deployment_state.safetensors` without the latent head
- 201-bin value head
- Qwen3-VL processor
- manifest and SHA-256 checksums

Reload:

```python
from lapo_value_model.deployment import load_exported_model

model, processor = load_exported_model("outputs/full/student/proposed/export", device="cuda")
assert model.latent_head is None
```

`configs/smoke.yaml` runs a small deterministic check of the full chain. Smoke metrics are not used to judge final model quality.
