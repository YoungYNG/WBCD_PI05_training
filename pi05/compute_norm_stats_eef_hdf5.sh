train_config_name=${1:-pi05_base_eef_hdf5_lora}
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$repo_root/.cache/openpi}"
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$repo_root/.cache/lerobot}"
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME"
cd "$script_dir"
"$script_dir/.venv/bin/python" scripts/compute_norm_stats.py --config-name "$train_config_name"
