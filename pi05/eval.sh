#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

# Start virtual display if none is available (needed for NVIDIA Vulkan/SAPIEN)
if [ -z "$DISPLAY" ]; then
    if [ ! -f /tmp/.X99-lock ]; then
        Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 &
        sleep 1
    fi
    export DISPLAY=:99
fi
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source "$(dirname "$0")/.venv/bin/activate"
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
