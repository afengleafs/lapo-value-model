#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"
export PYTHONUNBUFFERED=1
# Ten concurrent folds otherwise oversubscribe the 384 host cores during image
# preprocessing.  These limits apply only to proposed; the baseline is untouched.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

output_name=openpi_finetune_oof_proposed
starting_checkpoint="$project_root/outputs/full/student/proposed/best.pt"

run_fold() {
    local fold=$1
    local device=$2
    local fold_dir="outputs/full/${output_name}/fold-${fold}"
    mkdir -p "$fold_dir"
    echo "$(date --iso-8601=seconds) starting proposed fold ${fold} on physical GPU ${device}" \
        | tee -a "$fold_dir/console.log"
    HIP_VISIBLE_DEVICES="$device" .venv/bin/python -m lapo_value_model.cli train-openpi-fold \
        --config configs/full.yaml \
        --fold "$fold" \
        --starting-checkpoint "$starting_checkpoint" \
        --output-name "$output_name" \
        2>&1 | tee -a "$fold_dir/console.log"
}

pids=()
fold_devices=(0 1 2 3 4)
for fold in 0 1 2 3 4; do
    run_fold "$fold" "${fold_devices[$fold]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one proposed fold failed; baseline processes were not touched." >&2
    exit "$status"
fi

.venv/bin/lapo-value evaluate-openpi-oof \
    --config configs/full.yaml \
    --output-name "$output_name"
.venv/bin/lapo-value report-openpi-oof \
    --config configs/full.yaml \
    --output-name "$output_name"

# Baseline started earlier but may still be producing its aggregate report.
while [[ ! -f outputs/full/openpi_finetune_oof/oof_report.json ]]; do
    echo "$(date --iso-8601=seconds) waiting for baseline OOF report before comparison"
    sleep 30
done
.venv/bin/lapo-value compare-openpi-oof --config configs/full.yaml
