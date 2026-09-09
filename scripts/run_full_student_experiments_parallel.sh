#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"
export PYTHONUNBUFFERED=1

run_full() {
    local run_name=$1
    local latent_mode=$2
    local devices=$3
    local output_dir="outputs/full/student/${run_name}"
    mkdir -p "$output_dir"
    if [[ -f "$output_dir/best.pt" ]]; then
        echo "$(date --iso-8601=seconds) skip ${run_name}: best.pt already exists"
        return
    fi
    HIP_VISIBLE_DEVICES="$devices" .venv/bin/python -m torch.distributed.run \
        --standalone --nproc_per_node=2 \
        -m lapo_value_model.train_student \
        --config configs/full.yaml \
        --run-name "$run_name" \
        --latent-mode "$latent_mode" \
        2>&1 | tee -a "$output_dir/console.log"
}

run_full baseline none 0,1 &
baseline_pid=$!
run_full proposed true 2,3 &
proposed_pid=$!
status=0
wait "$baseline_pid" || status=1
wait "$proposed_pid" || status=1
exit "$status"
