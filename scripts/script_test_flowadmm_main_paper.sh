#!/usr/bin/env bash

set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"

MODEL_ROOT_DIR="${MODEL_ROOT_DIR:-model}"
RESULTS_ROOT_DIR="${RESULTS_ROOT_DIR:-results_submission}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_BATCH="${MAX_BATCH:-25}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"

run_case() {
  local dataset="$1"
  local problem="$2"
  shift 2

  echo "run: dataset=${dataset} problem=${problem}"
  uv run python main.py --opts \
    dataset "${dataset}" \
    eval_split "${EVAL_SPLIT}" \
    model ot \
    model_root_dir "${MODEL_ROOT_DIR}" \
    results_root_dir "${RESULTS_ROOT_DIR}" \
    problem "${problem}" \
    method flowadmm \
    max_batch "${MAX_BATCH}" \
    batch_size_ip "${BATCH_SIZE_IP}" \
    "$@"
}

# CelebA
run_case celeba denoising \
  tau 5.0 t_min 0.5 t_max 0.95 gamma 1.0 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 1 K_avg_late 41 \
  K_avg_switch_frac 0.5 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 100

run_case celeba gaussian_deblurring_FFT \
  tau 0.5 t_min 0.5 t_max 0.95 gamma 0.5 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 1 K_avg_late 41 \
  K_avg_switch_frac 0.5 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 100

run_case celeba superresolution \
  tau 0.5 t_min 0.3 t_max 0.95 gamma 1.0 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 3 K_avg_late 35 \
  K_avg_switch_frac 0.6 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 100

run_case celeba random_inpainting \
  tau 0.25 t_min 0.3 t_max 0.95 gamma 0.5 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 4 K_avg_late 29 \
  K_avg_switch_frac 0.5 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 100

run_case celeba inpainting \
  tau 1.0 t_min 0.1 t_max 0.95 gamma 2.0 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 4 K_avg_late 35 \
  K_avg_switch_frac 0.7 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 100

# AFHQ-Cat
run_case afhq_cat denoising \
  tau 5.0 t_min 0.5 t_max 0.95 gamma 1.0 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 1 K_avg_late 41 \
  K_avg_switch_frac 0.5 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 100

run_case afhq_cat gaussian_deblurring_FFT \
  tau 0.25 t_min 0.5 t_max 0.95 gamma 0.5 \
  K_avg 5 K_avg_schedule constant denoiser_chunk_size 32 \
  steps_admm 100

run_case afhq_cat superresolution \
  tau 0.25 t_min 0.3 t_max 0.95 gamma 1.0 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 4 K_avg_late 29 \
  K_avg_switch_frac 0.5 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 500

run_case afhq_cat random_inpainting \
  tau 0.125 t_min 0.3 t_max 0.95 gamma 0.5 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 3 K_avg_late 33 \
  K_avg_switch_frac 0.5 K_avg_switch_frac_2 0.9 denoiser_chunk_size 32 \
  steps_admm 200

run_case afhq_cat inpainting \
  tau 0.5 t_min 0.1 t_max 0.9 gamma 2.0 \
  K_avg 5 K_avg_schedule three_phase K_avg_early 1 K_avg_mid 3 K_avg_late 19 \
  K_avg_switch_frac 0.6 K_avg_switch_frac_2 0.8 denoiser_chunk_size 32 \
  steps_admm 100
