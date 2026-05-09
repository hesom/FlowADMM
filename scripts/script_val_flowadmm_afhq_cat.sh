#!/usr/bin/env bash

set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"

DATASET=afhq_cat
MODEL=ot
METHOD=flowadmm
EVAL_SPLIT=val
MAX_BATCH=1
BATCH_SIZE_IP=32

run_case() {
  local problem="$1"
  local steps="$2"
  shift 2

  uv run python main.py --opts \
    dataset "${DATASET}" \
    eval_split "${EVAL_SPLIT}" \
    model "${MODEL}" \
    problem "${problem}" \
    method "${METHOD}" \
    max_batch "${MAX_BATCH}" \
    batch_size_ip "${BATCH_SIZE_IP}" \
    steps_admm "${steps}" \
    "$@"
}

run_grid() {
  local problem="$1"
  local steps="$2"

  for tau in 0.125 0.25 0.5 1.0 2.0 5.0 10.0; do
    for t_min in 0.1 0.2 0.3 0.5; do
      for t_max in 0.9 0.95; do
        for gamma in 0.5 1.0 2.0; do
          run_case "${problem}" "${steps}" \
            tau "${tau}" t_min "${t_min}" t_max "${t_max}" gamma "${gamma}" \
            K_avg 5 K_avg_schedule constant

          for switch_1 in 50 60 70; do
            for switch_2 in 80 90; do
              for K_avg_mid in 1 3 4; do
                early_steps=$((steps * switch_1 / 100))
                mid_steps=$((steps * (switch_2 - switch_1) / 100))
                late_steps=$((steps - early_steps - mid_steps))
                if [ "${late_steps}" -le 0 ]; then
                  continue
                fi

                late_num=$((steps * 5 - early_steps - mid_steps * K_avg_mid))
                if [ "${late_num}" -le 0 ]; then
                  continue
                fi
                if [ $((late_num % late_steps)) -ne 0 ]; then
                  continue
                fi

                K_avg_late=$((late_num / late_steps))
                run_case "${problem}" "${steps}" \
                  tau "${tau}" t_min "${t_min}" t_max "${t_max}" gamma "${gamma}" \
                  K_avg 5 K_avg_schedule three_phase \
                  K_avg_early 1 K_avg_mid "${K_avg_mid}" K_avg_late "${K_avg_late}" \
                  K_avg_switch_frac "$(printf "0.%02d" "${switch_1}")" \
                  K_avg_switch_frac_2 "$(printf "0.%02d" "${switch_2}")"
              done
            done
          done
        done
      done
    done
  done
}

run_grid denoising 100
run_grid gaussian_deblurring_FFT 100
run_grid superresolution 500
run_grid random_inpainting 200
run_grid inpainting 100
