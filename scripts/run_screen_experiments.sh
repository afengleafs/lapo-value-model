#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"

export HIP_VISIBLE_DEVICES=6,7
export PYTHONUNBUFFERED=1

run_screen() {
    local run_name=$1
    local latent_mode=$2
    local output_dir="outputs/full/student/${run_name}"

    mkdir -p "$output_dir"
    if [[ -f "$output_dir/best.pt" ]]; then
        echo "$(date --iso-8601=seconds) skip ${run_name}: best.pt already exists"
        return
    fi

    echo "$(date --iso-8601=seconds) start ${run_name} latent_mode=${latent_mode}"
    .venv/bin/python -m torch.distributed.run \
        --standalone --nproc_per_node=2 \
        -m lapo_value_model.train_student \
        --config configs/full.yaml \
        --run-name "$run_name" \
        --latent-mode "$latent_mode" \
        --max-optimizer-steps 400 \
        2>&1 | tee -a "$output_dir/console.log"
    echo "$(date --iso-8601=seconds) complete ${run_name}"
}

run_screen screen-baseline none
run_screen screen-proposed true
run_screen screen-random random
run_screen screen-shuffled shuffled

echo "$(date --iso-8601=seconds) all screen experiments complete"
