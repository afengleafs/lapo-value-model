# LAPO + Qwen3‑VL RECAP-style Value Model

本项目在本地 DROID success/failure 数据上训练严格因果、语言条件的 201-bin 价值模型。训练期 LAPO teacher 从当前帧和未来帧提取 32D transition latent；部署模型只读取三帧历史、粗粒度任务语言以及 `observation.images.left_external`，不会读取未来帧。

详细方法调研见 [docs/recap_value_model_research.md](docs/recap_value_model_research.md)。原始实施方案位于相邻文件 `../lapo_qwenvl_value_model_plan_with_architecture.html`。

## 已固定的正式配置

- Backbone：本地 `Qwen3-VL-2B-Instruct`，bf16。
- LoRA：rank 16；语言 q/k/v/o 与视觉 qkv/proj。
- Value Head：`2048 → 512 → 256 → 201`，取 `[-1,0]` bin 概率期望。
- Latent Head：`2048 → 512 → 128 → 32`，仅训练期使用。
- Student loss：categorical CE + `0.1 × Huber(latent)`。
- 两卡：物理 GPU 6、7；训练 micro-batch 160/卡，累积 1，全局 effective batch 320；验证 micro-batch 80/卡。
- 全量 anchor：526,718；train/val/test = 419,344/53,772/53,602。

## Latent-hidden 串联重设计（OpenPI all400）

模型现支持 `value_path=direct|latent_hidden`。旧 Baseline 保持
`2048→512→256→201` 直接价值头；新 Proposed 使用
`2048→512→128` Latent Trunk，然后把 128D 隐层送入
`128→512→256→201` Value Head，同时由独立 `128→32` projection 对齐 Teacher latent。
价值 CE 会更新 Latent Trunk，但不会经过 32D projection。部署新 Proposed 时保留
Latent Trunk，仍只读取三帧历史和任务语言，不读取未来帧或 Teacher 输出。

对应五折重训入口为：

```bash
scripts/run_openpi_all400_latent128_seed20260901.sh
```

该协议使用 400 episodes、batch size 16、每折 2,000 steps，仅保存 step 1000/2000，
并保留 step 0/1000/2000 的 held-out OOF 推理。Baseline 完整加载旧 DROID Value Head，
Proposed 迁移旧 LoRA/Latent Head 并固定 seed 重置新 Value Head；因此比较的是完整
架构与初始化方案，而不是架构本身的单一因果效应。

两卡 Student 冒烟中，训练 micro-batch 160/卡的 `max_memory_allocated` 实测约 40.53 GiB/卡。验证 batch 单独设为 80/卡，以避免长时间验证中可变序列形状导致缓存逼近 64 GiB 上限。ROCm 上视觉 fused attention backward 会产生非有限 LoRA 梯度，本实现只把视觉 attention 固定为 eager；语言 attention 仍使用高效实现。

## 环境

项目 venv 使用系统 DTK/ROCm PyTorch：

```bash
cd /home/tione/notebook/users/fhh/lapo_value_model
.venv/bin/python -m compileall -q src
.venv/bin/python -m pytest -q
```

所有 GPU 命令都显式设置 `HIP_VISIBLE_DEVICES=6,7`，进程内对应 `cuda:0,cuda:1`。

## 正式训练流水线

各阶段均分离落盘；抽帧和 checkpoint 可恢复。

固定训练预算如下（“样本数”是包含 DDP 补齐/重采样的 sample presentations，不等同于去重 anchor 数）：

| 阶段 | micro-batch/卡 | 全局 batch | optimizer steps | sample presentations |
|---|---:|---:|---:|---:|
| LAPO Teacher，20 epochs | 112 | 224 | 37,460 | 8,391,040 |
| screen-baseline | 160 | 320 | 400 | 128,000 |
| screen-proposed | 160 | 320 | 400 | 128,000 |
| screen-random | 160 | 320 | 400 | 128,000 |
| screen-shuffled | 160 | 320 | 400 | 128,000 |
| full baseline，3 epochs | 160 | 320 | 3,933 | 1,258,560 |
| full proposed，3 epochs | 160 | 320 | 3,933 | 1,258,560 |

Teacher 使用 AdamW，学习率 `5.25e-4`。Student 的 LoRA adapter 学习率为 `5e-5`，Value/Latent Head 学习率为 `5e-4`。四组 screen 严格使用相同 400-step 预算，只改变 latent 监督模式；它们用于消融诊断，不替代后续完整 baseline/proposed 训练。

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

Teacher 的正式 latent 生成有质量门禁：validation reconstruction gain 必须大于 0。Student 每个 epoch 输出 rank-level validation Parquet，并以 macro Spearman 选择 best checkpoint。

## 必做 ablation

四个实验应使用相同 optimizer-step 预算：

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

screen 完成后另起新目录完整训练 `baseline --latent-mode none` 和 `proposed --latent-mode true`，每组 3 epochs / 3,933 optimizer steps。最终 DROID 初始化模型只按 DROID validation macro-Spearman 在这两个完整 run 之间选择；相同时选择 baseline。后续 OpenPI 2k 实验使用严格 episode-level OOF，因此须报告为“OOF 微调评估”，不能再称作 untouched external holdout。

## OpenPI 302-rollout 五折 OOF 微调

OpenPI 的 98 个 demonstration episodes 完全排除，只使用 302 个 rollout episodes（88 success / 214 failure）。30 Hz 原视频按 `[-20,-10,0]` 取三帧历史，Teacher future 为 `+10` 源帧；anchor 每 10 个源帧取一个，共 14,855 个。

- 按 `dataset_name × outcome` 分层、episode 整体分到 5 折，每个 episode 恰好 held-out 一次；
- 每 fold 从同一个 DROID 最终 checkpoint 初始化，训练 2,000 optimizer steps；
- 全局 batch 320；先按 success/failure 1:1，再均匀采 episode、均匀采 episode 内 anchor；
- LoRA LR `1e-5`，Value/Latent Head LR `1e-4`，warmup 100，cosine decay；
- 每 100 step 原子保存完整可恢复 checkpoint，共 20 个/fold、100 个；
- step 0、100…2000 都只在当折 held-out episodes 推理，step 2000 是预注册主结果，禁止按 OpenPI 曲线择优。

严格 A50 使用 `50 model frames = 100 source frames`。在 10-source-frame anchor stride 下，每个 episode 会少 10 个可配对起点，因此实际是 11,835 个 A50 窗口；12,137 对应的是 9 个 anchor 间隔，即 A45，不采用该错位口径。

```bash
# 1. CPU：生成密集 manifest、分折并抽取四帧 tar
.venv/bin/lapo-value prepare-openpi-finetune --config configs/full.yaml --workers 4

# 2. GPU：用正式 LAPO Teacher 生成 14,855 个 32D latent
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.cli generate-openpi-latents --config configs/full.yaml

# 3. 8 卡并行 fold 0–3，完成后用 6,7 跑 fold 4，并自动聚合/出报告
scripts/run_openpi_finetune_oof.sh
```

正式输出在 `outputs/full/openpi_finetune_oof/`：每折 checkpoint/prediction、21 个 OOF prediction/advantage/metrics 文件、`curve_data.parquet`、`oof_report.json` 和 `report.html`。报告包含 first/mean/last episode AUC、三种 value gap、MAE/RMSE、成功回合 macro-Spearman、时间单调性、4-stage matched AUC、A50 效应量与 q70 positive-rate，以及 2,000 次 outcome-stratified episode bootstrap 95% CI。

## Test 评估与部署导出

```bash
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m lapo_value_model.evaluate --config configs/full.yaml \
  --run-name proposed --checkpoint best.pt

.venv/bin/lapo-value export --config configs/full.yaml \
  --run-name proposed --checkpoint best.pt
```

评估报告包含 MAE/RMSE、episode macro Spearman、成功轨迹时间单调性、成功/失败 AUC、按任务分组指标、stage-matched AUC 与 advantage 符号质量，并核对 test anchor 100% 覆盖。

导出目录位于 `outputs/full/student/proposed/export`，包含：

- 标准 PEFT adapter；
- 不含 Latent Head 的 `deployment_state.safetensors`；
- 201-bin Value Head；
- Qwen3‑VL processor；
- manifest 和 SHA-256 校验和。

回载方式：

```python
from lapo_value_model.deployment import load_exported_model

model, processor = load_exported_model("outputs/full/student/proposed/export", device="cuda")
assert model.latent_head is None
```

`configs/smoke.yaml` 可对整条链路做小规模确定性验证，但 smoke 指标不用于判断最终模型质量。
