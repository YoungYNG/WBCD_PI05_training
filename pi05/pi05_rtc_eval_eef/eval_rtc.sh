#!/bin/bash
set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}

if [ -z "$DISPLAY" ]; then
    if [ ! -f /tmp/.X99-lock ]; then
        Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 &
        sleep 1
    fi
    export DISPLAY=:99
fi
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}

task_name=${1}
task_config=${2}
train_config_name=${3:-pi05_base_eef_hdf5_lora}
model_name=${4:-deformable_eef_hdf5_run}
checkpoint_id=${5:-040000}
seed=${6:-0}
gpu_id=${7:-0}
test_num=${8:-100}
rtc_inference_delay=${9:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/../.venv/bin/activate"
cd "${script_dir}/../../.."

PYTHONWARNINGS=ignore::UserWarning \
python policy/pi05/pi05_rtc_eval/eval_policy_rtc.py \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --checkpoint_id "${checkpoint_id}" \
    --seed "${seed}" \
    --test_num "${test_num}" \
    --pi0_step 50 \
    --rtc_execution_horizon 10 \
    --rtc_max_guidance_weight 10.0 \
    --rtc_prefix_attention_schedule exp \
    --rtc_inference_delay "${rtc_inference_delay}" \
    --rtc_refill_threshold 10 \
    --eef_prompt "deformable manipulation <control_mode> end effector <control_mode>" \
    --head_camera_mask_value 0
