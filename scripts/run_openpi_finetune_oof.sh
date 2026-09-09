#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"
export PYTHONUNBUFFERED=1

starting_checkpoint=${1:-}
checkpoint_args=()
if [[ -n "$starting_checkpoint" ]]; then
    checkpoint_args=(--starting-checkpoint "$starting_checkpoint")
fi

run_fold() {
    local fold=$1
    local device=$2
    local fold_dir="outputs/full/openpi_finetune_oof/fold-${fold}"
    mkdir -p "$fold_dir"
    echo "$(date --iso-8601=seconds) starting fold ${fold} on physical GPU ${device}" \
        | tee -a "$fold_dir/console.log"
    HIP_VISIBLE_DEVICES="$device" .venv/bin/python -m lapo_value_model.cli train-openpi-fold \
        --config configs/full.yaml \
        --fold "$fold" \
        "${checkpoint_args[@]}" \
        2>&1 | tee -a "$fold_dir/console.log"
}

pids=()
fold_devices=(3 4 5 6 7)
for fold in 0 1 2 3 4; do
    run_fold "$fold" "${fold_devices[$fold]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

.venv/bin/lapo-value evaluate-openpi-oof --config configs/full.yaml
.venv/bin/lapo-value report-openpi-oof --config configs/full.yaml
