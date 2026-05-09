#!/usr/bin/env bash

set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"

RESULTS_ROOT_DIR="${RESULTS_ROOT_DIR:-results_paper_baselines}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_BATCH="${MAX_BATCH:-25}"
BATCH_SIZE_IP="${BATCH_SIZE_IP:-4}"
DATASET=afhq_cat

run_case() {
  local model="$1"
  local method="$2"
  local problem="$3"
  shift 3

  echo "run: dataset=${DATASET} model=${model} method=${method} problem=${problem}"
  uv run python main.py --opts \
    dataset "${DATASET}" \
    eval_split "${EVAL_SPLIT}" \
    model "${model}" \
    results_root_dir "${RESULTS_ROOT_DIR}" \
    problem "${problem}" \
    method "${method}" \
    max_batch "${MAX_BATCH}" \
    batch_size_ip "${BATCH_SIZE_IP}" \
    "$@"
}

# OT-ODE
run_case ot ot_ode denoising steps_ode 100 start_time 0.3 gamma gamma_t
run_case ot ot_ode gaussian_deblurring_FFT steps_ode 100 start_time 0.3 gamma gamma_t
run_case ot ot_ode superresolution steps_ode 100 start_time 0.1 gamma constant
run_case ot ot_ode random_inpainting steps_ode 100 start_time 0.1 gamma constant
run_case ot ot_ode inpainting steps_ode 100 start_time 0.1 gamma gamma_t

# D-Flow
run_case ot d_flow denoising steps_euler 6 start_time 0.0 lmbda 0.001 alpha 0.1 max_iter 3 LBFGS_iter 20
run_case ot d_flow gaussian_deblurring_FFT steps_euler 6 start_time 0.0 lmbda 0.001 alpha 0.1 max_iter 7 LBFGS_iter 20
run_case ot d_flow superresolution steps_euler 6 start_time 0.0 lmbda 0.001 alpha 0.1 max_iter 10 LBFGS_iter 20
run_case ot d_flow inpainting steps_euler 6 start_time 0.0 lmbda 0.001 alpha 0.1 max_iter 9 LBFGS_iter 20
run_case ot d_flow random_inpainting steps_euler 6 start_time 0.0 lmbda 0.001 alpha 0.1 max_iter 20 LBFGS_iter 20

# Flow-Priors
run_case ot flow_priors denoising start_time 0.0 K 1 N 100 eta 0.01 lmbda 100
run_case ot flow_priors gaussian_deblurring_FFT start_time 0.0 K 1 N 100 eta 0.01 lmbda 1000
run_case ot flow_priors superresolution start_time 0.0 K 1 N 100 eta 0.1 lmbda 10000
run_case ot flow_priors random_inpainting start_time 0.0 K 1 N 100 eta 0.01 lmbda 10000
run_case ot flow_priors inpainting start_time 0.0 K 1 N 100 eta 0.01 lmbda 10000

# PnP-Flow1
run_case ot pnp_flow denoising steps_pnp 100 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 1 alpha 0.8
run_case ot pnp_flow gaussian_deblurring_FFT steps_pnp 500 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 1 alpha 0.01
run_case ot pnp_flow superresolution steps_pnp 500 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 1 alpha 0.01
run_case ot pnp_flow random_inpainting steps_pnp 200 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 1 alpha 0.01
run_case ot pnp_flow inpainting steps_pnp 100 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 1 alpha 0.5

# PnP-Flow5
run_case ot pnp_flow denoising steps_pnp 100 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 5 alpha 0.8
run_case ot pnp_flow gaussian_deblurring_FFT steps_pnp 500 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 5 alpha 0.01
run_case ot pnp_flow superresolution steps_pnp 500 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 5 alpha 0.01
run_case ot pnp_flow random_inpainting steps_pnp 200 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 5 alpha 0.01
run_case ot pnp_flow inpainting steps_pnp 100 lr_pnp 1.0 gamma_style alpha_1_minus_t num_samples 5 alpha 0.5

# Flower1-OT
run_case ot flower denoising steps 100 num_samples 1
run_case ot flower gaussian_deblurring_FFT steps 100 num_samples 1
run_case ot flower superresolution steps 500 num_samples 1
run_case ot flower random_inpainting steps 200 num_samples 1
run_case ot flower inpainting steps 100 num_samples 1

# Flower5-OT
run_case ot flower denoising steps 100 num_samples 5
run_case ot flower gaussian_deblurring_FFT steps 100 num_samples 5
run_case ot flower superresolution steps 500 num_samples 5
run_case ot flower random_inpainting steps 200 num_samples 5
run_case ot flower inpainting steps 100 num_samples 5

# PnP-GS
run_case gradient_step pnp_gs denoising lr_pnp 1.0 alpha 1.0 algo pgd max_iter 1 sigma_factor 1.0
run_case gradient_step pnp_gs gaussian_deblurring_FFT lr_pnp 2.0 alpha 0.5 algo pgd max_iter 35 sigma_factor 1.8
run_case gradient_step pnp_gs superresolution lr_pnp 2.0 alpha 1.0 algo pgd max_iter 20 sigma_factor 1.8
run_case gradient_step pnp_gs random_inpainting algo hqs max_iter 20

# DiffPIR
run_case diffusion pnp_diff denoising lmbda 1.0 zeta 1.0 max_iter 100
run_case diffusion pnp_diff gaussian_deblurring_FFT lmbda 1000.0 zeta 1.0 max_iter 100
run_case diffusion pnp_diff superresolution lmbda 100.0 zeta 1.0 max_iter 100
run_case diffusion pnp_diff random_inpainting lmbda 1.0 zeta 1.0 max_iter 100
