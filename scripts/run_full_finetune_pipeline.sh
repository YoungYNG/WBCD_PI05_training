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
NUM_STEPS="50000"
SAVE_INTERVAL="5000"
KEEP_PERIOD="5000"
BATCH_SIZE="32"
FSDP_DEVICES="4"
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
    [--num-steps 50000] \
    [--save-interval 5000] \
    [--keep-period 5000] \
    [--batch-size 32] \
    [--fsdp-devices 4] \
    [--log-path ./logs/train.log]

This script performs:
  1. RoboTwin/ARX qpos HDF5 -> ALOHA raw layout.
  2. ALOHA raw layout -> LeRobot repo.
  3. Compute normalization stats.
  4. Start pi05 full fine-tuning.

Before running, make sure pi05/src/openpi/training/config.py has:
  name="pi05_aloha_full_base"
  repo_id="<your --repo-id>"
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
    --num-steps) NUM_STEPS="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --keep-period) KEEP_PERIOD="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --fsdp-devices) FSDP_DEVICES="$2"; shift 2 ;;
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
"${PI05_DIR}/.venv/bin/python" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw_dir "${RAW_DIR}" \
  --repo_id "${REPO_ID}"

echo "[3/4] Compute normalization stats for pi05_aloha_full_base"
"${PI05_DIR}/.venv/bin/python" scripts/compute_norm_stats.py \
  --config-name pi05_aloha_full_base

echo "[4/4] Start full fine-tuning"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_VALUE}"
export HF_HOME="${HF_HOME_VALUE}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}" "${HF_LEROBOT_HOME}"
if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
fi

TRAIN_CMD=(
  "${PI05_DIR}/.venv/bin/python" scripts/train.py
  pi05_aloha_full_base
  --exp-name="${EXP_NAME}"
  --overwrite
  --num-train-steps="${NUM_STEPS}"
  --save-interval="${SAVE_INTERVAL}"
  --keep-period="${KEEP_PERIOD}"
  --batch-size="${BATCH_SIZE}"
  --fsdp-devices="${FSDP_DEVICES}"
)

if [[ -n "${LOG_PATH}" ]]; then
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
  "${TRAIN_CMD[@]}"
fi
