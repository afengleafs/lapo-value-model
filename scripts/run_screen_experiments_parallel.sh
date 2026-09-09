#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"

export PYTHONUNBUFFERED=1

run_screen() {
    local run_name=$1
    local latent_mode=$2
    local devices=$3
    local output_dir="outputs/full/student/${run_name}"

    mkdir -p "$output_dir"
    if [[ -f "$output_dir/best.pt" ]]; then
        echo "$(date --iso-8601=seconds) skip ${run_name}: best.pt already exists"
        return
    fi

    echo "$(date --iso-8601=seconds) start ${run_name} latent_mode=${latent_mode} devices=${devices}" \
        | tee -a "$output_dir/console.log"
    HIP_VISIBLE_DEVICES="$devices" .venv/bin/python -m torch.distributed.run \
        --standalone --nproc_per_node=2 \
        -m lapo_value_model.train_student \
        --config configs/full.yaml \
        --run-name "$run_name" \
        --latent-mode "$latent_mode" \
        --max-optimizer-steps 400 \
        2>&1 | tee -a "$output_dir/console.log"
    echo "$(date --iso-8601=seconds) complete ${run_name}" \
        | tee -a "$output_dir/console.log"
}

run_screen screen-baseline none 0,1 &
baseline_pid=$!
run_screen screen-proposed true 2,3 &
proposed_pid=$!
run_screen screen-random random 4,5 &
random_pid=$!
run_screen screen-shuffled shuffled 6,7 &
shuffled_pid=$!

status=0
for pid in "$baseline_pid" "$proposed_pid" "$random_pid" "$shuffled_pid"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [[ "$status" -eq 0 ]]; then
    echo "$(date --iso-8601=seconds) all parallel screen experiments complete"
else
    echo "$(date --iso-8601=seconds) one or more parallel screen experiments failed" >&2
fi
exit "$status"
