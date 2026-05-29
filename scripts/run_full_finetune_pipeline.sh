#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_DIR="${ROOT_DIR}/pi05"

SRC_DIR=""
RAW_NAME=""
REPO_ID=""
EXP_NAME=""
PROMPT="fold the cloth"
GPUS="0,1,2,3"
LOG_PATH=""
OPENPI_DATA_HOME_VALUE="${OPENPI_DATA_HOME:-${ROOT_DIR}/.cache/openpi}"
HF_HOME_VALUE="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
HF_LEROBOT_HOME_VALUE="${HF_LEROBOT_HOME:-${ROOT_DIR}/.cache/lerobot}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_full_finetune_pipeline.sh \
    --src-dir /path/to/raw_robotwin_hdf5 \
    --raw-name fold \
    --repo-id fold_repo \
    --exp-name fold_full_bs32_fsdp4 \
    [--prompt "fold the cloth"] \
    [--gpus 0,1,2,3] \
    [--log-path ./logs/train.log]

This script performs:
  1. RoboTwin/ARX qpos HDF5 -> ALOHA raw layout.
  2. ALOHA raw layout -> LeRobot repo with 224x224 image features.
  3. Compute normalization stats.
  4. Start pi05 full fine-tuning.

Before running, make sure pi05/src/openpi/training/config.py has:
  name="pi05_aloha_full_base"
  repo_id="<your --repo-id>"
  num_train_steps / batch_size / fsdp_devices / save_interval set as desired.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-dir) SRC_DIR="$2"; shift 2 ;;
    --raw-name) RAW_NAME="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${SRC_DIR}" || -z "${RAW_NAME}" || -z "${REPO_ID}" || -z "${EXP_NAME}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 2
fi

RAW_DIR="${PI05_DIR}/training_data/${RAW_NAME}"

echo "[1/4] Convert source HDF5 to ALOHA raw layout"
"${PI05_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/convert_robotwin_qpos_to_aloha_raw.py" \
  --src-dir "${SRC_DIR}" \
  --dst-dir "${RAW_DIR}" \
  --instruction "${PROMPT}" \
  --overwrite

echo "[2/4] Convert ALOHA raw layout to LeRobot repo: ${REPO_ID}"
cd "${PI05_DIR}"
HF_HOME="${HF_HOME_VALUE}" \
HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
"${PI05_DIR}/.venv/bin/python" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_resume_224.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id "${REPO_ID}" \
  --task "${PROMPT}" \
  --order-mode numeric \
  --dataset-config.image-writer-processes 32 \
  --dataset-config.image-writer-threads 2

echo "[3/4] Compute normalization stats for pi05_aloha_full_base"
HF_HOME="${HF_HOME_VALUE}" \
HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
"${PI05_DIR}/.venv/bin/python" scripts/compute_norm_stats.py \
  --config-name pi05_aloha_full_base

echo "[4/4] Start full fine-tuning"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_VALUE}"
export HF_HOME="${HF_HOME_VALUE}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENPI_PLOT_LOSS="${OPENPI_PLOT_LOSS:-1}"

mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}" "${HF_LEROBOT_HOME}"
if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
fi

TRAIN_CMD=(
  bash finetune.sh
  pi05_aloha_full_base
  "${EXP_NAME}"
  "${GPUS}"
)

if [[ -n "${LOG_PATH}" ]]; then
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
  "${TRAIN_CMD[@]}"
fi
