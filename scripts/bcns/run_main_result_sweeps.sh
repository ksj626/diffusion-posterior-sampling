#!/usr/bin/env bash
set -euo pipefail

SECTION="main"
GPUS="0"
SAVE_ROOT="results/bcns_main"
MODEL_CONFIG="configs/model_config.yaml"
DIFFUSION_CONFIG="configs/diffusion_config.yaml"
TASK_CONFIG="configs/inpainting_bcns_flow_be_struct_projected_ffhq.yaml"
NUM_IMAGES="64"
NUM_VISUALIZE="10"
SEED="6000"
RECORD_EVERY="50"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bcns/run_main_result_sweeps.sh [options]

Options:
  --section main|solver|frequency|all
  --gpus 0,1,2,3
  --save_root DIR
  --model_config PATH
  --diffusion_config PATH
  --task_config PATH
  --num_images N
  --num_visualize N
  --seed N
  --record_every N
  --dry_run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --section) SECTION="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --save_root) SAVE_ROOT="$2"; shift 2 ;;
    --model_config) MODEL_CONFIG="$2"; shift 2 ;;
    --diffusion_config) DIFFUSION_CONFIG="$2"; shift 2 ;;
    --task_config) TASK_CONFIG="$2"; shift 2 ;;
    --num_images) NUM_IMAGES="$2"; shift 2 ;;
    --num_visualize) NUM_VISUALIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --record_every) RECORD_EVERY="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SECTION" in
  main|solver|frequency|all) ;;
  *) echo "--section must be one of main, solver, frequency, all." >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ ${#GPU_ARRAY[@]} -eq 0 || -z "${GPU_ARRAY[0]}" ]]; then
  echo "At least one GPU id is required." >&2
  exit 2
fi

RUNNER="scripts/bcns/run_bcns_step4_ablation.py"
PIDS=()
JOB_INDEX=0

wait_batch() {
  local status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  PIDS=()
  if [[ "$status" -ne 0 ]]; then
    echo "One or more sweep jobs failed." >&2
    exit "$status"
  fi
}

launch_job() {
  local gpu="${GPU_ARRAY[$((JOB_INDEX % ${#GPU_ARRAY[@]}))]}"
  JOB_INDEX=$((JOB_INDEX + 1))
  local cmd=(
    python "$RUNNER"
    --model_config "$MODEL_CONFIG"
    --diffusion_config "$DIFFUSION_CONFIG"
    --task_config "$TASK_CONFIG"
    --gpu 0
    --num_images "$NUM_IMAGES"
    --num_visualize "$NUM_VISUALIZE"
    --seed "$SEED"
    --record_every "$RECORD_EVERY"
    "$@"
  )
  echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
    PIDS+=("$!")
    if [[ ${#PIDS[@]} -ge ${#GPU_ARRAY[@]} ]]; then
      wait_batch
    fi
  fi
}

run_main() {
  local masks=(
    thick_scratch_24
    text_mask
    freeform_medium
    center_box_96
    center_box_128
    center_keep_96
    center_keep_128
    global_random_50_60
  )
  local steps=(20 50 100)
  local mask step
  for mask in "${masks[@]}"; do
    for step in "${steps[@]}"; do
      launch_job \
        --save_dir "$SAVE_ROOT/main/$mask/steps_${step}" \
        --mask_mode "$mask" \
        --sampling_steps "$step" \
        --ablation_set bcns_main_methods
    done
  done
}

run_solver() {
  local masks=(freeform_medium center_box_128 center_keep_128 global_random_50_60)
  local ablations=(bcns_main_solver_ablation_harmonic bcns_main_solver_ablation_poisson)
  local mask ablation
  for ablation in "${ablations[@]}"; do
    for mask in "${masks[@]}"; do
      launch_job \
        --save_dir "$SAVE_ROOT/solver/$ablation/$mask" \
        --mask_mode "$mask" \
        --sampling_steps 100 \
        --ablation_set "$ablation"
    done
  done
}

run_frequency() {
  local masks=(freeform_medium text_mask center_box_128)
  local mask
  for mask in "${masks[@]}"; do
    launch_job \
      --save_dir "$SAVE_ROOT/frequency/$mask" \
      --mask_mode "$mask" \
      --sampling_steps 100 \
      --ablation_set bcns_main_frequency_ablation
  done
}

case "$SECTION" in
  main) run_main ;;
  solver) run_solver ;;
  frequency) run_frequency ;;
  all) run_main; run_solver; run_frequency ;;
esac

wait_batch
