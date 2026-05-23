#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-robotwin}"

echo "[1/3] Create or update conda env: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ROOT_DIR}/env/robotwin-conda-full.yml" --prune
else
  conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/env/robotwin-conda-full.yml"
fi

echo "[2/3] Create relative cache directories"
mkdir -p "${ROOT_DIR}/.cache/openpi" "${ROOT_DIR}/.cache/huggingface" "${ROOT_DIR}/.cache/lerobot" "${ROOT_DIR}/logs"

echo "[3/3] Sync pi05 uv environment"
echo "Run the following commands in your shell:"
echo
echo "  conda activate ${ENV_NAME}"
echo "  cd ${ROOT_DIR}/pi05"
echo "  GIT_LFS_SKIP_SMUDGE=1 uv sync"
echo
echo "After that, use ${ROOT_DIR}/pi05/.venv/bin/python for pi05 commands."
echo "For the uv environment, also refer to the RoboTwin Pi0.5 docs:"
echo "  https://robotwin-platform.github.io/doc/usage/Pi05.html"
