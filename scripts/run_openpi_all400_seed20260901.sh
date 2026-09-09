#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

config=configs/openpi_all400_seed20260901.yaml
baseline_output=openpi_all400_b16_2k_baseline_seed20260901
proposed_output=openpi_all400_b16_2k_proposed_seed20260901
comparison_output=openpi_all400_comparison_seed20260901
curve_output=openpi_all400_curves_seed20260901
baseline_checkpoint="$project_root/outputs/full/student/baseline/best.pt"
proposed_checkpoint="$project_root/outputs/full/student/proposed/best.pt"

check_sha256() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA256 mismatch for $path: $actual != $expected" >&2
        exit 1
    fi
}

check_sha256 "$baseline_checkpoint" 5c16f90bbb61c06893fe157a6af299491199aa599a2237285ea332d85d9b93f0
check_sha256 "$proposed_checkpoint" d8154e98111f91eacfd0860d180170b88e93b88daf5eeddd26053a109451b067

.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit(f"Need 8 visible GPUs, got {torch.cuda.device_count()}")
for index in range(8):
    free, total = torch.cuda.mem_get_info(index)
    required = 40 if index in (3, 4) else 20
    if free < required * 1024**3:
        raise SystemExit(
            f"GPU {index} has only {free / 1024**3:.1f} GiB free; need {required} GiB"
        )
    print(
        f"GPU {index}: {torch.cuda.get_device_name(index)}, "
        f"free={free / 1024**3:.1f}/{total / 1024**3:.1f} GiB",
        flush=True,
    )
PY

.venv/bin/lapo-value prepare-openpi-finetune --config "$config" --workers 8
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=2 -m lapo_value_model.cli \
    generate-openpi-latents --config "$config"

run_fold() {
    local model_name=$1
    local fold=$2
    local device=$3
    local output_name=$4
    local starting_checkpoint=$5
    local lambda_latent=$6
    local fold_dir="outputs/full/${output_name}/fold-${fold}"
    mkdir -p "$fold_dir"
    echo "$(date --iso-8601=seconds) starting ${model_name} fold ${fold} on GPU ${device}" \
        | tee -a "$fold_dir/console.log"
    HIP_VISIBLE_DEVICES="$device" .venv/bin/python -m lapo_value_model.cli \
        train-openpi-fold \
        --config "$config" \
        --fold "$fold" \
        --starting-checkpoint "$starting_checkpoint" \
        --output-name "$output_name" \
        --lambda-latent "$lambda_latent" \
        2>&1 | tee -a "$fold_dir/console.log"
}

pids=()
baseline_devices=(3 4 5 6 7)
proposed_devices=(0 1 2 3 4)
for fold in 0 1 2 3 4; do
    run_fold baseline "$fold" "${baseline_devices[$fold]}" \
        "$baseline_output" "$baseline_checkpoint" 0 &
    pids+=("$!")
    run_fold proposed "$fold" "${proposed_devices[$fold]}" \
        "$proposed_output" "$proposed_checkpoint" 0.1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one all400 fold failed; rerun this launcher to resume." >&2
    exit "$status"
fi

for output_name in "$baseline_output" "$proposed_output"; do
    .venv/bin/lapo-value evaluate-openpi-oof \
        --config "$config" --output-name "$output_name"
    .venv/bin/lapo-value report-openpi-oof \
        --config "$config" --output-name "$output_name"
done

.venv/bin/lapo-value compare-openpi-oof \
    --config "$config" \
    --baseline-output-name "$baseline_output" \
    --proposed-output-name "$proposed_output" \
    --comparison-output-name "$comparison_output"
.venv/bin/lapo-value plot-openpi-oof-curves \
    --config "$config" \
    --baseline-output-name "$baseline_output" \
    --proposed-output-name "$proposed_output" \
    --output-name "$curve_output"
