train_config_name=$1
model_name=$2
gpu_use=$3
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

export CUDA_VISIBLE_DEVICES=$gpu_use
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$repo_root/.cache/openpi}"
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$repo_root/.cache/lerobot}"
echo $CUDA_VISIBLE_DEVICES
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME"
cd "$script_dir"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" "$script_dir/.venv/bin/python" scripts/train.py "$train_config_name" --exp-name="$model_name" --overwrite
