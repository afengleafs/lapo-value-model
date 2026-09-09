#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"
export PYTHONUNBUFFERED=1

log() {
    echo "$(date --iso-8601=seconds) $*"
}

log "waiting for four screen validations"
while pgrep -f 'lapo_value_model.train_student.*screen-.*max-optimizer-steps 400' >/dev/null; do
    sleep 30
done
for run in screen-baseline screen-proposed screen-random screen-shuffled; do
    if [[ ! -f "outputs/full/student/${run}/best.pt" ]]; then
        log "missing screen checkpoint: ${run}/best.pt"
        exit 1
    fi
done
log "all screen runs complete"

log "waiting for dense OpenPI extraction"
while pgrep -f 'lapo-value prepare-openpi-finetune' >/dev/null; do
    sleep 30
done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/openpi_rollout/finetune/extraction_summary.json")
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["samples"] == 14855 and not payload["failures"], payload
assert Path("artifacts/openpi_rollout/finetune/shard_index.parquet").is_file()
PY
log "dense OpenPI extraction complete"

log "starting full DROID baseline/proposed and OpenPI Teacher latent generation"
scripts/run_full_student_experiments_parallel.sh \
    > outputs/full/full_student_parallel.log 2>&1 &
full_pid=$!
mkdir -p artifacts/openpi_rollout/finetune/latents
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    -m lapo_value_model.cli generate-openpi-latents \
    --config configs/full.yaml \
    > artifacts/openpi_rollout/finetune/latents/console.log 2>&1 &
latent_pid=$!

status=0
wait "$full_pid" || status=1
wait "$latent_pid" || status=1
if [[ "$status" -ne 0 ]]; then
    log "full DROID training or OpenPI latent generation failed"
    exit "$status"
fi
log "full DROID runs and OpenPI latents complete"

.venv/bin/lapo-value select-final --config configs/full.yaml \
    > outputs/full/final_selection.log 2>&1
log "DROID initialization checkpoint selected"

scripts/run_openpi_finetune_oof.sh
log "OpenPI OOF fine-tuning, aggregation, and report complete"
