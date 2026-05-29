# WBCD PI05 训练流程

这个仓库整理了一条最小可复现流程，用于把 RoboTwin 风格的双臂 qpos 数据转成 LeRobot 数据集，并对 PI05 做全量微调。

整体流程：

```text
RoboTwin HDF5
  -> 可选：resize_with_pad 到 224x224
  -> ALOHA raw HDF5
  -> LeRobot repo
  -> 计算归一化统计
  -> PI05 训练
```

仓库不包含数据、checkpoint、日志、缓存和虚拟环境。

## 1. 配环境

先按 RoboTwin / PI05 官方方式准备环境。这里默认已经有 `robotwin` 环境：

```bash
conda activate robotwin
pip install uv
```

然后在本仓库内创建 PI05 的 `.venv`：

```bash
cd WBCD_PI05_training/pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ..
```

后续所有 PI05 脚本都用：

```bash
pi05/.venv/bin/python
```

建议先设置缓存路径：

```bash
cd WBCD_PI05_training

export ROOT_DIR="$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot"
export OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" logs
```

## 2. 准备数据

原始数据目录格式：

```text
/path/to/source_data/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

每个 HDF5 至少包含：

```text
/data/demo_0/observations/qpos
/data/demo_0/observations/images/head
/data/demo_0/observations/images/left_wrist
/data/demo_0/observations/images/right_wrist
```

训练监督方式是 next-qpos：

```text
observation.state = qpos[:-1]
action            = qpos[1:]
```

相机映射：

```text
head        -> cam_high
left_wrist  -> cam_left_wrist
right_wrist -> cam_right_wrist
```

## 3. 可选：确认原始 HDF5 图像 resize 到 224x224

如果数据已经按同样方式 resize 好，就直接把 `SRC_DIR` 指向已有数据目录。

## 4. HDF5 转 ALOHA raw

设置任务变量：

```bash
cd "$ROOT_DIR"

export SRC_DIR="/path/to/source_data_resize224"
export TASK_NAME="your_task_name"
export RAW_DIR="$ROOT_DIR/pi05/training_data/$TASK_NAME"
export REPO_ID="${TASK_NAME}_repo"
export PROMPT="Pick up the two corners of the white garment on the table, place it over the clothing board on the blue rack, and smooth it flat."
```

转换：

```bash
pi05/.venv/bin/python scripts/convert_robotwin_qpos_to_aloha_raw.py \
  --src-dir "$SRC_DIR" \
  --dst-dir "$RAW_DIR" \
  --instruction "$PROMPT" \
  --overwrite \
  2>&1 | tee "logs/convert_${TASK_NAME}_to_aloha_raw.log"
```

## 5. ALOHA raw 转 LeRobot

使用 224 版本转换脚本。这个脚本会把图像 feature shape 写成 `[3, 224, 224]`，并支持断点续转。

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
./.venv/bin/python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_resume_224.py \
  --raw-dir "$RAW_DIR" \
  --repo-id "$REPO_ID" \
  --task "$PROMPT" \
  --order-mode numeric \
  --dataset-config.image-writer-processes 32 \
  --dataset-config.image-writer-threads 2 \
  2>&1 | tee "$ROOT_DIR/logs/convert_${TASK_NAME}_aloha_to_lerobot.log"
```

并发建议先用 `32 processes x 2 threads`。如果机器和磁盘都比较强，可以再试 `48 x 2`。

## 6. 改训练配置

打开：

```text
pi05/src/openpi/training/config.py
```

修改 `pi05_aloha_full_base`：

```python
data=LeRobotAlohaDataConfig(
    repo_id="your_task_name_repo",
    adapt_to_pi=False,
    ...
)

weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params"
)

num_train_steps=20_000
batch_size=40
num_workers=16
save_interval=5_000
keep_period=5_000
fsdp_devices=4
wandb_enabled=False
```

如果要从本地 checkpoint 继续做参数初始化，把 `weight_loader` 改成：

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "/path/to/checkpoints/pi05_aloha_full_base/<exp>/<step>/params"
)
```

注意：这种方式只加载模型参数，不加载 optimizer 状态，也不会继承 step；新训练会从 step 0 开始。

## 7. 计算归一化

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_aloha_full_base \
  2>&1 | tee "$ROOT_DIR/logs/compute_norm_stats_${REPO_ID}.log"
```

成功后会生成：

```text
pi05/assets/pi05_aloha_full_base/<repo_id>/norm_stats.json
```

## 8. 开始训练

示例：4 卡、全局 batch size 40、训练 20k steps、每 5k 保存一次。

```bash
cd "$ROOT_DIR/pi05"

export EXP_NAME="${TASK_NAME}_pi05_full_bs40_fsdp4_20k_save5k"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MPLCONFIGDIR=/tmp/matplotlib-cache \
OPENPI_PLOT_LOSS=1 \
bash finetune.sh \
  pi05_aloha_full_base \
  "$EXP_NAME" \
  0,1,2,3 \
  2>&1 | tee "$ROOT_DIR/logs/train_${EXP_NAME}.log"
```

checkpoint 路径：

```text
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/
```

## 9. Loss 图

训练时默认每 5 step 更新一次 loss 图：

```text
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/plots/loss_curve.png
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/plots/loss_points.csv
```

如果不想边训练边画图：

```bash
export OPENPI_PLOT_LOSS=0
```

也可以训练后从 log 补画：

```bash
cd "$ROOT_DIR/pi05"

MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python scripts/plot_train_loss.py \
  "$ROOT_DIR/logs/train_${EXP_NAME}.log"
```

输出会放在 log 同目录下：

```text
logs/train_<exp_name>_loss_curve.png
logs/train_<exp_name>_loss_points.csv
```

## 备注

- `batch_size` 必须能被可见 GPU 数量整除。
- 我们实测 4 卡时 `fsdp_devices=4`，`batch_size=32~40` 比较稳。
- PI05 默认输入是当前三路图像、当前 state 和 prompt，输出 action chunk；默认没有历史帧窗口或循环记忆。
