#!/bin/bash
set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}

train_config_name=${1:-pi05_base_aloha_lora}
model_name=${2:-demo_clean}
checkpoint_id=${3:-30000}
gpu_id=${4:-0}
max_episodes=${5:-2}
stride=${6:-200}
max_frames_per_episode=${7:-10}

export CUDA_VISIBLE_DEVICES=${gpu_id}

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/../.venv/bin/activate"
cd "${script_dir}/../../.."

PYTHONWARNINGS=ignore::UserWarning \
python policy/pi05/pi05_rtc_eval_qpos/eval_lerobot_qpos_actions.py \
    --train-config-name "${train_config_name}" \
    --model-name "${model_name}" \
    --checkpoint-id "${checkpoint_id}" \
    --max-episodes "${max_episodes}" \
    --stride "${stride}" \
    --max-frames-per-episode "${max_frames_per_episode}" \
    --pi0-step 50 \
    --warmup-runs 1

