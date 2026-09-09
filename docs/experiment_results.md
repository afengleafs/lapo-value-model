# Experiment Results

Compiled from `outputs/full/` JSON reports (`metrics.jsonl`, `oof_report.json`, `openpi_oof_model_comparison.json`, `final_selection.json`, `metrics_report.json`, `latency_report.json`). Generated 2026-09-09. Hardware: 2× Hygon BW1000_H DCU (`HIP_VISIBLE_DEVICES=6,7`), DTK/ROCm PyTorch 2.7.1, bf16.

OpenPI numbers are **OOF finetune evaluation**, not an untouched external holdout. The DROID deployment model is chosen **only** by DROID validation macro-Spearman.

## 1. Headline results

| Setting | Finding |
|---|---|
| **DROID in-domain** | LAPO latent supervision **does not help**. Full baseline validation macro-Spearman **0.4129** vs proposed **0.2976**. Pre-registered rule selects **baseline** (`outputs/full/final`). |
| **OpenPI 302-rollout OOF** (step 2000) | Proposed wins **11 / 13** metrics. Mean-value AUC **0.8777** vs **0.8206** (bootstrap 95% CIs do not overlap). Latent supervision **helps transfer**. |
| **OpenPI all400, old architecture** | Proposed (latent32) wins **10 / 13** vs first direct baseline. Advantage shrinks vs the 302-only protocol. |
| **OpenPI all400, latent128 cascade** | Trade-off: best MAE (**0.2455**) and mean-value AUC (**0.9658**), large A50 success/failure separation, but worse in-trajectory ranking (macro-Spearman **0.695** vs **0.775**). Compares a full architecture+init package, not architecture alone. |
| **Latency** | ~108 ms/anchor at batch 1; baseline and proposed are essentially identical. |

## 2. Datasets and training budget

| | DROID pretraining | OpenPI 302-rollout OOF | OpenPI all400 |
|---|---|---|---|
| Episodes used | 52,475 success + 13,422 failure (eligible) | 302 (88 / 214); 98 demos excluded | 400 (98 demos + 302 rollouts) |
| Anchors | 526,718 (train/val/test 419,344 / 53,772 / 53,602) | 14,855 | 18,071 |
| A50 windows | — | 11,835 | 14,071 |
| Source size | 1.04 TiB + 258 GiB ≈ 1.29 TiB | included in 2.6 GiB collection | 2.6 GiB |
| Camera | `left_external`, 15 Hz, 720×1280 | `camera_1`, 30 Hz, 480×640 | same |
| History | [-10, -5, 0] | [-20, -10, 0] | [-20, -10, 0] |
| Student train | 2× GPU 6,7; micro-batch 160/card; global 320; 3 epochs / 3,933 steps | 5 folds × 2,000 steps; LoRA 1e-5, head 1e-4 | 5 folds × 2,000 steps; batch 16 |

## 3. LAPO teacher (DROID, 20 epochs / 37,460 steps)

ResNet-18 IDM + FiLM U-Net FDM, 32D latent, AdamW 5.25e-4, micro-batch 112/card × 2 cards. Quality gate: validation reconstruction gain > 0.

| Epoch | MSE ↓ | Copy-MSE | Reconstruction gain ↑ | SSIM ↑ |
|---:|---:|---:|---:|---:|
| 0 | 0.003029 | 0.002774 | -0.000255 | 0.6438 |
| 1 | 0.002519 | 0.002774 | +0.000255 | 0.8349 |
| 2 | 0.002983 | 0.002774 | -0.000209 | 0.8491 |
| 3 | 0.002214 | 0.002774 | +0.000560 | 0.8784 |
| 4 | 0.002119 | 0.002774 | +0.000655 | 0.9014 |
| 5 | 0.001950 | 0.002774 | +0.000824 | 0.9110 |
| 6 | 0.001983 | 0.002774 | +0.000792 | 0.8992 |
| 7 | 0.001816 | 0.002774 | +0.000958 | 0.9176 |
| 8 | 0.001715 | 0.002774 | +0.001059 | 0.9193 |
| 9 | 0.001677 | 0.002774 | +0.001097 | 0.9248 |
| 10 | 0.001584 | 0.002774 | +0.001191 | 0.9283 |
| 11 | 0.001549 | 0.002774 | +0.001225 | 0.9227 |
| 12 | 0.001397 | 0.002774 | +0.001378 | 0.9331 |
| 13 | 0.001362 | 0.002774 | +0.001412 | 0.9344 |
| 14 | 0.001297 | 0.002774 | +0.001478 | 0.9359 |
| 15 | 0.001280 | 0.002774 | +0.001494 | 0.9378 |
| 16 | 0.001256 | 0.002774 | +0.001518 | 0.9382 |
| 17 | 0.001243 | 0.002774 | +0.001531 | 0.9379 |
| 18 | 0.001239 | 0.002774 | +0.001535 | 0.9391 |
| 19 | 0.001248 | 0.002774 | +0.001527 | 0.9390 |

Best reconstruction gain is epoch 18: **+0.001535**. Final epoch 19: gain **+0.001527**, MSE 0.001248 vs copy-MSE 0.002774 (**55.0%** relative reduction), SSIM 0.9390 (epoch 0 was 0.6438).

## 4. DROID student, 400-step screens

Same optimizer-step budget (400). Only the latent supervision mode changes. Diagnostic only — not used for model selection.

| Run | Latent | MAE ↓ | RMSE ↓ | macro-Spearman ↑ | Success/failure AUC ↑ | Temporal monotonicity ↑ | Adv. sign acc. ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| screen-baseline | none | 0.3225 | 0.3931 | 0.1562 | 0.7468 | 0.5266 | 0.5265 |
| screen-proposed | true LAPO | 0.3388 | 0.4130 | 0.2196 | 0.7325 | 0.5518 | 0.5516 |
| screen-random | random | 0.3241 | 0.3944 | 0.1698 | 0.7200 | 0.5377 | 0.5377 |
| screen-shuffled | shuffled pairing | 0.3408 | 0.4164 | 0.1347 | 0.7201 | 0.5237 | 0.5237 |

At 400 steps, proposed Spearman (0.2196) beats baseline (0.1562), but random latent (0.1698) also beats baseline. The short-budget gain cannot be attributed cleanly to true transition supervision.

## 5. DROID student, full training (3 epochs / 3,933 steps)

Validation set: 53,772 anchors. Constant-predictor MAE = 0.2523.

| Epoch | Step | Run | MAE ↓ | RMSE ↓ | macro-Spearman ↑ | Success/failure AUC ↑ | Temporal monotonicity ↑ | Adv. sign acc. ↑ |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 1311 | baseline | 0.4049 | 0.5025 | 0.2056 | 0.7573 | 0.5462 | 0.5460 |
| 0 | 1311 | proposed | 0.3766 | 0.4536 | 0.0953 | 0.7589 | 0.5216 | 0.5217 |
| 1 | 2622 | baseline | 0.2736 | 0.3898 | 0.3496 | 0.8309 | 0.5695 | 0.5694 |
| 1 | 2622 | proposed | 0.2846 | 0.3928 | 0.2930 | 0.8165 | 0.5601 | 0.5600 |
| 2 | 3933 | baseline | 0.2393 | 0.3487 | 0.4129 | 0.8804 | 0.5887 | 0.5885 |
| 2 | 3933 | proposed | 0.2594 | 0.3643 | 0.2976 | 0.8578 | 0.5617 | 0.5616 |

### Selection

- Metric: `DROID validation macro_spearman`
- Rule: `baseline wins ties or has higher DROID validation macro_spearman`
- OpenPI used for selection: **False**
- **Selected run: `baseline`** (macro-Spearman 0.4129 vs 0.2976)
- Export: `outputs/full/final/` (PEFT adapter + 201-bin value head, no latent head)

Full baseline final MAE 0.2393 beats the constant baseline; proposed 0.2594 does not.

## 6. OpenPI 302-rollout five-fold OOF (pre-registered primary)

- Protocol: 5-fold episode-level OOF fine-tuning
- Status: OOF fine-tuning evaluation; no longer an untouched external holdout
- Selection policy: step 2000 is the preregistered primary result; intermediate checkpoints are learning-curve diagnostics only
- Episodes: 302 (success 88 / failure 214); demonstration episodes included: 0
- Anchors / A50 windows per step: 14,855 / 11,835
- Folds: 5; bootstrap: 2,000 outcome-stratified episode resamples
- Checkpoints evaluated: every 100 steps from 0 to 2000 (21 points)
- Paired comparison gates passed: **True**

### 6.1 Primary result (step 2000)

Proposed wins **11**, baseline wins **2** of 13 metrics.

| Metric | Direction | Baseline | Proposed | Δ (P−B) | Winner |
|---|---|---:|---:|---:|---|
| MAE | lower | 0.3404 | 0.3457 | +0.0053 | baseline |
| RMSE | lower | 0.3921 | 0.3853 | -0.0068 | proposed |
| Success macro-Spearman | higher | 0.5731 | 0.6030 | +0.0299 | proposed |
| Temporal monotonicity | higher | 0.5170 | 0.5275 | +0.0105 | proposed |
| First-anchor AUC | higher | 0.5105 | 0.5537 | +0.0432 | proposed |
| Mean-value AUC | higher | 0.8206 | 0.8777 | +0.0571 | proposed |
| Last-value AUC | higher | 0.9256 | 0.9513 | +0.0258 | proposed |
| 4-stage AUC | higher | 0.6487 | 0.6476 | -0.0010 | baseline |
| A50 sigma | lower | 0.2255 | 0.1967 | -0.0287 | proposed |
| A50 sigma/scale | lower | 4.0422 | 3.5276 | -0.5146 | proposed |
| A50 success positive rate | higher | 0.3435 | 0.3457 | +0.0021 | proposed |
| A50 failure positive rate | lower | 0.2710 | 0.2696 | -0.0014 | proposed |
| A50 failure tail20 positive rate | lower | 0.4605 | 0.4079 | -0.0526 | proposed |

### 6.2 Bootstrap 95% CI (step 2000)

| Metric | Baseline | 95% CI | Proposed | 95% CI |
|---|---:|---|---:|---|
| Mean-value AUC | 0.8206 | [0.7678, 0.8713] | 0.8777 | [0.8354, 0.9179] |
| Last-value AUC | 0.9256 | [0.8875, 0.9570] | 0.9513 | [0.9255, 0.9727] |
| First-anchor AUC | 0.5105 | [0.4372, 0.5820] | 0.5537 | [0.4840, 0.6248] |
| Success−failure gap (mean) | 0.1235 | [0.0996, 0.1484] | 0.1161 | [0.0972, 0.1368] |
| Success−failure gap (last) | 0.4333 | [0.3865, 0.4774] | 0.4472 | [0.4069, 0.4834] |
| A50 success pos. rate | 0.3435 | [0.3253, 0.3617] | 0.3457 | [0.3267, 0.3652] |
| A50 failure pos. rate | 0.2710 | [0.2602, 0.2820] | 0.2696 | [0.2587, 0.2805] |
| A50 failure tail20 pos. rate | 0.4605 | [0.3214, 0.5962] | 0.4079 | [0.2647, 0.5455] |

Mean-value AUC CIs do not overlap: baseline [0.7678, 0.8713] vs proposed [0.8354, 0.9179]. Proposed A50 σ/scale is 3.53 vs 4.04 (cleaner advantage).

### 6.3 Stage-matched AUC (container_transfer, step 2000)

| Stage group | Baseline | Proposed | Δ |
|---|---:|---:|---:|
| container_transfer:stage-0 | 0.5195 | 0.5305 | +0.0110 |
| container_transfer:stage-1 | 0.5978 | 0.5874 | -0.0104 |
| container_transfer:stage-2 | 0.6470 | 0.6413 | -0.0057 |
| container_transfer:stage-3 | 0.8303 | 0.8313 | +0.0010 |
| **macro (4-stage)** | 0.6487 | 0.6476 | -0.0010 |

### 6.4 Learning curve (OOF, every 500 steps plus endpoints)

| Step | MAE B / P | Mean-value AUC B / P | Last-value AUC B / P | Success Spearman B / P | A50 σ/scale B / P |
|---:|---|---|---|---|---|
| 0 | 0.5977 / 0.5058 | 0.4245 / 0.2340 | 0.4861 / 0.4561 | 0.1414 / -0.3648 | 0.656 / 2.178 |
| 500 | 0.3590 / 0.3607 | 0.8463 / 0.8978 | 0.9080 / 0.9221 | 0.6459 / 0.5807 | 2.732 / 2.593 |
| 1000 | 0.3486 / 0.3555 | 0.8079 / 0.8712 | 0.9088 / 0.9333 | 0.5960 / 0.6181 | 3.331 / 2.915 |
| 1500 | 0.3366 / 0.3471 | 0.8225 / 0.8805 | 0.9244 / 0.9441 | 0.5771 / 0.6328 | 3.939 / 3.393 |
| 2000 | 0.3404 / 0.3457 | 0.8206 / 0.8777 | 0.9256 / 0.9513 | 0.5731 / 0.6030 | 4.042 / 3.528 |

## 7. OpenPI all400 five-fold (400 episodes, 18,071 anchors, seed 20260901)

Demonstrations are included. Checkpoints at steps 0 / 1000 / 2000 only. Batch size 16 per fold.

Latent128 comparison caveat: Baseline uses its fully pretrained DROID value head while Proposed uses a newly initialized latent-hidden value head. The comparison therefore measures the complete architecture/initialization packages, not the isolated causal effect of architecture alone.

### 7.1 Four runs at step 2000

| Metric | direct baseline | direct baseline (rerun) | proposed (latent32) | proposed latent128 |
|---|---:|---:|---:|---:|
| MAE | 0.2560 | 0.2557 | 0.2613 | **0.2455** |
| RMSE | 0.3364 | **0.3338** | 0.3350 | 0.3420 |
| Success macro-Spearman | 0.7783 | 0.7746 | **0.7856** | 0.6954 |
| Temporal monotonicity | 0.5869 | 0.5906 | **0.5998** | 0.5802 |
| First-anchor AUC | 0.7834 | 0.7836 | **0.7971** | 0.7688 |
| Mean-value AUC | 0.9452 | 0.9519 | 0.9566 | **0.9658** |
| Last-value AUC | 0.9676 | 0.9695 | **0.9760** | 0.9718 |
| 4-stage AUC | 0.7805 | **0.7849** | 0.7636 | 0.7570 |
| A50 σ/scale | 3.1735 | 3.0901 | **2.9232** | 3.2499 |
| A50 success pos. rate | 0.2955 | 0.2979 | 0.3121 | **0.4032** |
| A50 failure pos. rate | 0.3044 | 0.3020 | 0.2881 | **0.1986** |
| A50 failure tail20 pos. rate | **0.4079** | 0.4474 | 0.4211 | 0.4342 |

Bold = best in row.

### 7.2 Paired: direct baseline (first) vs proposed latent32 — proposed 10 : 3

| Metric | Baseline | Proposed | Δ | Winner |
|---|---:|---:|---:|---|
| MAE | 0.2560 | 0.2613 | +0.0054 | baseline |
| RMSE | 0.3364 | 0.3350 | -0.0014 | proposed |
| Success macro-Spearman | 0.7783 | 0.7856 | +0.0073 | proposed |
| Temporal monotonicity | 0.5869 | 0.5998 | +0.0130 | proposed |
| First-anchor AUC | 0.7834 | 0.7971 | +0.0137 | proposed |
| Mean-value AUC | 0.9452 | 0.9566 | +0.0114 | proposed |
| Last-value AUC | 0.9676 | 0.9760 | +0.0084 | proposed |
| 4-stage AUC | 0.7805 | 0.7636 | -0.0169 | baseline |
| A50 sigma | 0.1770 | 0.1630 | -0.0140 | proposed |
| A50 sigma/scale | 3.1735 | 2.9232 | -0.2503 | proposed |
| A50 success positive rate | 0.2955 | 0.3121 | +0.0166 | proposed |
| A50 failure positive rate | 0.3044 | 0.2881 | -0.0163 | proposed |
| A50 failure tail20 positive rate | 0.4079 | 0.4211 | +0.0132 | baseline |

### 7.3 Paired: direct baseline (rerun) vs proposed latent128 — proposed 6 : 7

| Metric | Baseline (rerun) | latent128 | Δ | Winner |
|---|---:|---:|---:|---|
| MAE | 0.2557 | 0.2455 | -0.0102 | proposed |
| RMSE | 0.3338 | 0.3420 | +0.0081 | baseline |
| Success macro-Spearman | 0.7746 | 0.6954 | -0.0792 | baseline |
| Temporal monotonicity | 0.5906 | 0.5802 | -0.0104 | baseline |
| First-anchor AUC | 0.7836 | 0.7688 | -0.0148 | baseline |
| Mean-value AUC | 0.9519 | 0.9658 | +0.0138 | proposed |
| Last-value AUC | 0.9695 | 0.9718 | +0.0023 | proposed |
| 4-stage AUC | 0.7849 | 0.7570 | -0.0278 | baseline |
| A50 sigma | 0.1724 | 0.1813 | +0.0089 | baseline |
| A50 sigma/scale | 3.0901 | 3.2499 | +0.1598 | baseline |
| A50 success positive rate | 0.2979 | 0.4032 | +0.1053 | proposed |
| A50 failure positive rate | 0.3020 | 0.1986 | -0.1034 | proposed |
| A50 failure tail20 positive rate | 0.4474 | 0.4342 | -0.0132 | proposed |

latent128 characteristic trade-off: MAE 0.2455 and mean-value AUC 0.9658 are best of the four; A50 positive-rate gap is +0.2046 (success 0.4032 vs failure 0.1986) against −0.0041 for the rerun baseline (0.2979 vs 0.3020). Cost: success macro-Spearman 0.6954 vs 0.7746 (−0.079) and 4-stage AUC 0.7570 vs 0.7849.

### 7.4 all400 learning curve (steps 0 / 1000 / 2000)

| Step | Run | MAE | Mean-value AUC | Success Spearman |
|---:|---|---:|---:|---:|
| 0 | direct baseline | 0.4996 | 0.6082 | 0.2263 |
| 0 | direct baseline (rerun) | 0.4996 | 0.6080 | 0.2267 |
| 0 | proposed (latent32) | 0.4443 | 0.5771 | -0.4516 |
| 0 | proposed latent128 | 0.4214 | 0.3099 | -0.3740 |
| 1000 | direct baseline | 0.2643 | 0.9159 | 0.7758 |
| 1000 | direct baseline (rerun) | 0.2637 | 0.9245 | 0.7769 |
| 1000 | proposed (latent32) | 0.2669 | 0.9315 | 0.7488 |
| 1000 | proposed latent128 | 0.2466 | 0.9618 | 0.6801 |
| 2000 | direct baseline | 0.2560 | 0.9452 | 0.7783 |
| 2000 | direct baseline (rerun) | 0.2557 | 0.9519 | 0.7746 |
| 2000 | proposed (latent32) | 0.2613 | 0.9566 | 0.7856 |
| 2000 | proposed latent128 | 0.2455 | 0.9658 | 0.6954 |

## 8. ProcVLM metrics and latency (302-rollout, step 2000)

Progress transform: `clip(value + 1, 0, 1)`. MCC95 uses threshold 0.95.

| Model | EPR50 episode-macro | 95% CI | EPR50 pooled | MCC95 (last anchor) | MCC95 (frame) |
|---|---:|---|---:|---:|---:|
| baseline | 5.2257 | [5.1107, 5.3342] | 14.13 | 0.2023 | 0.1498 |
| proposed | 4.7889 | [4.6827, 4.8962] | 13.83 | 0.3230 | 0.2684 |

Proposed MCC95 is higher at both last-anchor (0.323 vs 0.202) and frame (0.268 vs 0.150): high-confidence success predictions match outcomes better.

End-to-end latency, batch=1, warmed cache, includes tar/JPEG + processor + H2D + forward + D2H:

| Model | anchors/s | ms/anchor (mean) | ms/anchor median [min, max] | Peak allocated |
|---|---:|---:|---|---:|
| baseline | 9.20 | 108.7 | 108.7 [107.1, 109.9] | 8.26 GiB |
| proposed | 9.32 | 107.3 | 107.2 [106.7, 108.9] | 8.26 GiB |

## 9. Run inventory

| Stage | Path under `outputs/full/` |
|---|---|
| Teacher | `teacher/` |
| DROID screens | `student/screen-{baseline,proposed,random,shuffled}/` |
| DROID full trains | `student/baseline/`, `student/proposed/` |
| Selected export | `final/`, `final_selection.json` |
| OpenPI 302 baseline OOF | `openpi_finetune_oof/` |
| OpenPI 302 proposed OOF | `openpi_finetune_oof_proposed/` |
| OpenPI 302 comparison | `openpi_finetune_oof_comparison/` |
| OpenPI all400 direct baseline | `openpi_all400_b16_2k_baseline_seed20260901/` |
| OpenPI all400 direct baseline rerun | `openpi_all400_b16_2k_direct_baseline_rerun_seed20260901/` |
| OpenPI all400 proposed latent32 | `openpi_all400_b16_2k_proposed_seed20260901/` |
| OpenPI all400 proposed latent128 | `openpi_all400_b16_2k_latent128_proposed_seed20260901/` |
| ProcVLM + latency | `openpi_procvlm_metrics/` |
| Interactive HTML summary | [`experiment_summary.html`](../experiment_summary.html) (repo root) |

Stopped / non-primary: `openpi_finetune_oof_b160_stopped_step1700_20260831T0932/` (batch-160 OOF, stopped at step 1700; superseded by the batch-320 2k protocol above).

