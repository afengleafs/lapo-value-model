#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/tione/notebook/users/fhh/lapo_value_model
cd "$project_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

config=configs/openpi_all400_seed20260901.yaml
baseline_output=openpi_all400_b16_2k_direct_baseline_rerun_seed20260901
proposed_output=openpi_all400_b16_2k_latent128_proposed_seed20260901
comparison_output=openpi_all400_latent128_comparison_seed20260901
curve_output=openpi_all400_latent128_curves_seed20260901
three_group_output=openpi_all400_latent128_three_group_value_curves_seed20260901
baseline_checkpoint="$project_root/outputs/full/student/baseline/best.pt"
proposed_source="$project_root/outputs/full/student/proposed/best.pt"
proposed_init_dir="$project_root/outputs/full/student/proposed_latent128_openpi_init_seed20260901"
proposed_checkpoint="$proposed_init_dir/initial.pt"

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
check_sha256 "$proposed_source" d8154e98111f91eacfd0860d180170b88e93b88daf5eeddd26053a109451b067

.venv/bin/python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq
import torch

root = Path("artifacts/openpi_rollout/finetune_all400_seed20260901")
stats = json.loads((root / "manifest_stats.json").read_text())
assert stats["manifest_hash"] == "7d77d7c4d4c99c530c3845b7c9afb5ba9316327ecf08bf7f349e9ffc2bafa447"
assert stats["episodes"] == 400 and stats["anchors"] == 18071
assert stats["advantage_windows_exact"] == 14071
latents = json.loads((root / "latents/summary.json").read_text())
assert latents["finite"] and latents["rows"] == 18071 and latents["unique_keys"] == 18071
assert latents["latent_dim"] == 32
assert pq.read_metadata(root / "samples.parquet").num_rows == 18071

if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit(f"Need 8 visible GPUs, got {torch.cuda.device_count()}")
shared = {3, 4}
for index in range(8):
    free, total = torch.cuda.mem_get_info(index)
    required = 38 if index in shared else 20
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

mkdir -p "$proposed_init_dir"
if [[ ! -f "$proposed_checkpoint" ]]; then
    .venv/bin/lapo-value create-cascaded-init \
        --config "$config" \
        --source-checkpoint "$proposed_source" \
        --output-checkpoint "$proposed_checkpoint" \
        --seed 20260901 \
        2>&1 | tee "$proposed_init_dir/console.log"
fi

.venv/bin/python - "$proposed_checkpoint" <<'PY'
import json
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert checkpoint["architecture"] == {
    "version": 2,
    "value_path": "latent_hidden",
    "value_feature_dim": 128,
    "latent_dim": 32,
}
initialization = checkpoint["initialization"]
assert initialization["policy"] == "copy_lora_and_latent_reset_value_head"
assert initialization["seed"] == 20260901
assert initialization["reset_parameter_names"]
assert all(name.startswith("value_head.") for name in initialization["reset_parameter_names"])
print(json.dumps({
    "proposed_checkpoint": str(path),
    "architecture": checkpoint["architecture"],
    "copied_parameters": len(initialization["copied_parameter_names"]),
    "reset_parameters": len(initialization["reset_parameter_names"]),
}, indent=2), flush=True)
PY

run_fold() {
    local model_name=$1
    local fold=$2
    local device=$3
    local output_name=$4
    local starting_checkpoint=$5
    local lambda_latent=$6
    local value_path=$7
    local fold_dir="outputs/full/${output_name}/fold-${fold}"
    mkdir -p "$fold_dir"
    echo "$(date --iso-8601=seconds) starting ${model_name} fold ${fold} on GPU ${device}" \
        | tee -a "$fold_dir/console.log"
    HIP_VISIBLE_DEVICES="$device" .venv/bin/lapo-value train-openpi-fold \
        --config "$config" \
        --fold "$fold" \
        --starting-checkpoint "$starting_checkpoint" \
        --output-name "$output_name" \
        --lambda-latent "$lambda_latent" \
        --value-path "$value_path" \
        2>&1 | tee -a "$fold_dir/console.log"
}

pids=()
baseline_devices=(0 1 2 3 4)
proposed_devices=(5 6 7 3 4)
for fold in 0 1 2 3 4; do
    run_fold baseline "$fold" "${baseline_devices[$fold]}" \
        "$baseline_output" "$baseline_checkpoint" 0 direct &
    pids+=("$!")
    run_fold proposed "$fold" "${proposed_devices[$fold]}" \
        "$proposed_output" "$proposed_checkpoint" 0.1 latent_hidden &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one latent128 all400 fold failed; rerun this launcher to resume." >&2
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
.venv/bin/lapo-value plot-openpi-oof-three-group-values \
    --config "$config" \
    --baseline-output-name "$baseline_output" \
    --proposed-output-name "$proposed_output" \
    --output-name "$three_group_output" \
    --step 2000
