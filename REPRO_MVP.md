# PI05 全量微调最小复现流程

本文档给出一条最短可复现链路，用于在 RoboTwin 风格 qpos 数据上全量微调 PI05：

```text
RoboTwin HDF5 -> ALOHA raw -> LeRobot repo -> norm stats -> full fine-tuning
```

下面命令默认从仓库根目录执行：

```bash
cd WBCD_PI05_training
export ROOT_DIR="$(pwd)"
```

## 0. 数据格式

源数据放在一个目录下：

```text
/path/to/source_hdf5/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

每个 HDF5 至少需要包含：

```text
/data/demo_0/observations/qpos
/data/demo_0/actions
/data/demo_0/observations/images/head
/data/demo_0/observations/images/head_lengths
/data/demo_0/observations/images/left_wrist
/data/demo_0/observations/images/left_wrist_lengths
/data/demo_0/observations/images/right_wrist
/data/demo_0/observations/images/right_wrist_lengths
```

转换脚本使用 next-qpos 监督：

```text
observation.state = qpos[:-1]
action            = qpos[1:]
```

相机映射：

```text
head        -> observation.images.cam_high
left_wrist  -> observation.images.cam_left_wrist
right_wrist -> observation.images.cam_right_wrist
```

## 1. 环境

创建轻量 conda 环境：

```bash
conda env create -n robotwin_pi05_mvp -f env/robotwin-pi05-mvp.yml
conda activate robotwin_pi05_mvp
```

创建 PI05 的 uv 环境：

```bash
export UV_CACHE_DIR="$ROOT_DIR/.cache/uv"
export TMPDIR="$ROOT_DIR/.cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

cd "$ROOT_DIR/pi05"
GIT_LFS_SKIP_SMUDGE=1 \
UV_CACHE_DIR="$UV_CACHE_DIR" \
TMPDIR="$TMPDIR" \
uv sync --default-index https://pypi.org/simple
```

快速检查：

```bash
cd "$ROOT_DIR"
pi05/.venv/bin/python -c "import openpi; print('openpi ok')"
pi05/.venv/bin/python -c "import jax; print('jax', jax.__version__)"
```

## 2. 设置路径变量

根据自己的数据和任务修改这些变量：

```bash
cd "$ROOT_DIR"

export SRC_DIR="/path/to/source_hdf5"
export TASK_NAME="fold_mvp"
export REPO_ID="${TASK_NAME}_repo"
export EXP_NAME="${TASK_NAME}_full_bs96_fsdp4_50k"
export PROMPT="Pick up the two corners of the white garment on the table, place it over the clothing board on the blue rack, and smooth it flat."

export OUT_DIR="$ROOT_DIR/outputs/$TASK_NAME"
export RAW_DIR="$ROOT_DIR/pi05/training_data/$TASK_NAME"

export OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR/logs" "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME"
```

## 3. HDF5 转 ALOHA Raw

```bash
cd "$ROOT_DIR"

pi05/.venv/bin/python scripts/convert_robotwin_qpos_to_aloha_raw.py \
  --src-dir "$SRC_DIR" \
  --dst-dir "$RAW_DIR" \
  --instruction "$PROMPT" \
  --overwrite \
  2>&1 | tee "$OUT_DIR/logs/convert_to_aloha_raw.log"
```

检查输出：

```bash
find "$RAW_DIR" -maxdepth 2 -type f | sort | head
```

预期输出结构：

```text
$RAW_DIR/episode_0/episode_0.hdf5
$RAW_DIR/episode_0/instructions.json
```

## 4. ALOHA Raw 转 LeRobot

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
./.venv/bin/python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir "$RAW_DIR" \
  --repo-id "$REPO_ID" \
  2>&1 | tee "$OUT_DIR/logs/convert_to_lerobot.log"
```

检查 LeRobot 数据：

```bash
cd "$ROOT_DIR"

pi05/.venv/bin/python - <<'PY'
from pathlib import Path
import json, os

repo = Path(os.environ["HF_LEROBOT_HOME"]) / os.environ["REPO_ID"]
info = json.load(open(repo / "meta/info.json"))

print("repo:", repo)
print("total_episodes:", info["total_episodes"])
print("total_frames:", info["total_frames"])
print("fps:", info["fps"])
for k, v in info["features"].items():
    print(k, v["dtype"], v["shape"])
PY
```

重要特征应包含：

```text
observation.state float32 [14]
action float32 [14]
observation.images.cam_high image [3, 480, 640]
observation.images.cam_left_wrist image [3, 480, 640]
observation.images.cam_right_wrist image [3, 480, 640]
```

## 5. 修改训练配置

打开：

```text
pi05/src/openpi/training/config.py
```

在 `pi05_aloha_full_base` 中设置：

```python
data=LeRobotAlohaDataConfig(
    repo_id="fold_mvp_repo",
    adapt_to_pi=False,
    ...
)
```

如果使用 8 张可见 GPU，并且每 4 张组成一个 FSDP group，使用：

```python
num_train_steps=50_000
batch_size=96
save_interval=5_000
keep_period=5_000
fsdp_devices=4
wandb_enabled=False
```

注意：`repo_id` 必须和 LeRobot 转换时的 `--repo-id` 一致。

## 6. 计算归一化统计

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
PYTHONUNBUFFERED=1 \
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_aloha_full_base \
  2>&1 | tee "$OUT_DIR/logs/compute_norm_stats.log"
```

成功后应生成：

```text
pi05/assets/pi05_aloha_full_base/<repo_id>/norm_stats.json
```

## 7. 全量微调

8 张可见 GPU，全局 batch size 96，FSDP group size 4，训练 50k step，每 5k step 保留一次 checkpoint：

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
PYTHONUNBUFFERED=1 \
bash finetune.sh \
  pi05_aloha_full_base \
  "$EXP_NAME" \
  0,1,2,3,4,5,6,7 \
  2>&1 | tee "$OUT_DIR/logs/train_${EXP_NAME}.log"
```

checkpoint 路径：

```text
pi05/checkpoints/pi05_aloha_full_base/<exp_name>
```

确认训练是否正确启动：

```bash
grep -E "Loaded norm stats|data_config|local_batch_size|Step 0|Progress on" \
  "$OUT_DIR/logs/train_${EXP_NAME}.log" | head -100
```

预期看到：

```text
Loaded norm stats from .../assets/pi05_aloha_full_base/<repo_id>
repo_id='<repo_id>'
local_batch_size: 96
Step 0: ...
```

## 8. 断点续训

如果是断点续训，不要使用会覆盖实验目录的 `finetune.sh`。

直接使用 `scripts/train.py`：

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
PYTHONUNBUFFERED=1 \
./.venv/bin/python scripts/train.py \
  pi05_aloha_full_base \
  --exp-name="$EXP_NAME" \
  --resume \
  --no-overwrite \
  --num-train-steps=50000 \
  --save-interval=5000 \
  --keep-period=5000 \
  --batch-size=96 \
  --fsdp-devices=4 \
  2>&1 | tee "$OUT_DIR/logs/train_${EXP_NAME}_resume.log"
```

## 我们复现时遇到的问题

### 1. LeRobot 转换中途卡住

我们遇到过原始转换脚本在某个 episode 附近卡住。常见原因是异步 image writer 压力过大，或者 repo 中已经有半写入的数据。

处理方式：

```text
1. 先检查已经生成了多少个 parquet。
2. 只删除未完成的 episode parquet/images。
3. 从下一个干净的 episode 继续转换。
```

如果仓库里有 resume-safe 转换脚本，可以使用：

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
./.venv/bin/python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_resume.py \
  --raw-dir "$RAW_DIR" \
  --repo-id "$REPO_ID" \
  --task "$PROMPT" \
  --order-mode numeric \
  --end-output-index <num_episodes> \
  --dataset-config.image-writer-processes 10 \
  --dataset-config.image-writer-threads 5
```

### 2. 磁盘空间不足

我们遇到的磁盘占用主要来自 HuggingFace datasets cache 和全量微调 checkpoint。

检查：

```bash
df -h "$ROOT_DIR"
du -sh "$HF_HOME/datasets" 2>/dev/null || true
du -sh "$HF_LEROBOT_HOME"/* 2>/dev/null | sort -h | tail
du -sh "$ROOT_DIR/pi05/checkpoints/pi05_aloha_full_base"/* 2>/dev/null | sort -h | tail
```

可以安全清理的 cache：

```bash
rm -rf "$HF_HOME/datasets/parquet"
```

不要删除当前还要继续训练或评估的 checkpoints，也不要误删正在使用的 LeRobot repo。

### 3. 磁盘满后留下半写入 episode

典型现象：

```text
meta/info.json 里 total_episodes=N
但 data/chunk-000/episode_00000N.parquet 已经存在
```

只删除这个多出来的未完成 episode：

```bash
rm -f "$HF_LEROBOT_HOME/$REPO_ID/data/chunk-000/episode_00000N.parquet"
rm -rf "$HF_LEROBOT_HOME/$REPO_ID/images"
```

然后从 `N` 继续转换。

### 4. repo_id 不一致

如果 `repo_id` 过期或写错，norm stats 和训练可能会读到旧数据。

检查三处：

```bash
echo "$REPO_ID"
grep -n 'repo_id=' "$ROOT_DIR/pi05/src/openpi/training/config.py"
find "$HF_LEROBOT_HOME/$REPO_ID" -maxdepth 2 -type f | head
find "$ROOT_DIR/pi05/assets/pi05_aloha_full_base/$REPO_ID" -name norm_stats.json
```

### 5. `fsdp_devices` 不是 GPU 数量

当可见 GPU 数量为 8，且 `fsdp_devices=4` 时，JAX mesh 是：

```text
(data_axis=2, fsdp_axis=4)
```

也就是说 8 张 GPU 都会被使用，只是被分成两个 data-parallel group，每个 group 内做 4 卡 FSDP。

### 6. `compute_norm_stats` 显示的 batch 数小于帧数

这是正常现象。例如：

```text
Generating train split: 234859 examples
Computing stats: 7339 batches
```

`7339` 是 batch 数，不是帧数。

### 7. PyAV / uv / wheel 问题

我们遇到过：

```text
PyAV source build fails
open3d wheel unavailable
uv resolves wrong platform wheels
```

常用处理方式：

```bash
export UV_CACHE_DIR="$ROOT_DIR/.cache/uv"
export TMPDIR="$ROOT_DIR/.cache/tmp"
GIT_LFS_SKIP_SMUDGE=1 uv sync --default-index https://pypi.org/simple
```

仓库中的 `pi05/pyproject.toml` 也对部分依赖做了 pin/override。
