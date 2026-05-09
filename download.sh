#!/usr/bin/env bash

set -euo pipefail

FILE="${1:-}"

prepare_afhq_cat_layout() {
  local root=./data/afhq_cat

  if [ ! -d "${root}" ]; then
    echo "Error: expected ${root} to exist."
    exit 1
  fi

  # The public AFHQ archive ships official train/val splits, while this repo
  # expects train/test plus a small validation subset carved out of test/cat.
  if [ -d "${root}/val/cat" ] && [ ! -d "${root}/test/cat" ]; then
    mv "${root}/val" "${root}/test"
    mkdir -p "${root}/val/cat" "${root}/val/dog" "${root}/val/wild"
  fi
}

case "${FILE}" in
  pretrained-network-celeba)
    OUTPUT_DIR=./model/celeba/ot
    mkdir -p "${OUTPUT_DIR}"
    gdown --id 1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6 -O "${OUTPUT_DIR}/model_final.pt"
    ;;
  pretrained-network-afhq-cat)
    OUTPUT_DIR=./model/afhq_cat/ot
    mkdir -p "${OUTPUT_DIR}"
    gdown --id 1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF -O "${OUTPUT_DIR}/model_final.pt"
    ;;
  pretrained-denoiser-celeba)
    OUTPUT_DIR=./model/celeba/gradient_step
    mkdir -p "${OUTPUT_DIR}"
    gdown --id 1ZqBeafErEogaXFupW0ZSLL7P9QoRA-lN -O "${OUTPUT_DIR}/model_final.pt"
    ;;
  pretrained-denoiser-afhq-cat)
    OUTPUT_DIR=./model/afhq_cat/gradient_step
    mkdir -p "${OUTPUT_DIR}"
    gdown --id 17AXI9p17c7h_xaI19qDcTT2u9_wu0DQY -O "${OUTPUT_DIR}/model_final.pt"
    ;;
  afhq-cat-dataset)
    DEST_DIR=./data
    mkdir -p "${DEST_DIR}"
    ZIP_FILE=./data/afhq.zip
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    wget -N "${URL}" -O "${ZIP_FILE}"
    unzip "${ZIP_FILE}" -d "${DEST_DIR}"
    rm "${ZIP_FILE}"
    if [ -d ./data/afhq ] && [ ! -d ./data/afhq_cat ]; then
      mv ./data/afhq ./data/afhq_cat
    fi
    prepare_afhq_cat_layout
    bash scripts/afhq_validation_images.sh
    ;;
  celeba-dataset)
    DEST_DIR=./data/celeba
    ZIP_FILE="${DEST_DIR}/celeba-dataset.zip"
    IMG_DIR="${DEST_DIR}/img_align_celeba"
    NESTED_IMG_DIR="${IMG_DIR}/img_align_celeba"
    mkdir -p "${DEST_DIR}"
    kaggle datasets download jessicali9530/celeba-dataset -p "${DEST_DIR}"
    if [ -f "${ZIP_FILE}" ]; then
      unzip -q "${ZIP_FILE}" -d "${DEST_DIR}"
      rm "${ZIP_FILE}"
      if [ -d "${NESTED_IMG_DIR}" ]; then
        mv "${IMG_DIR}" "${IMG_DIR}_old"
        mv "${IMG_DIR}_old/img_align_celeba" "${IMG_DIR}"
        rmdir "${IMG_DIR}_old"
      fi
    else
      echo "Error: expected ${ZIP_FILE} after Kaggle download."
      exit 1
    fi
    ;;
  *)
    cat <<'EOF'
Usage: bash download.sh <target>

Targets:
  celeba-dataset
  afhq-cat-dataset
  pretrained-network-celeba
  pretrained-network-afhq-cat
  pretrained-denoiser-celeba
  pretrained-denoiser-afhq-cat
EOF
    exit 1
    ;;
esac
