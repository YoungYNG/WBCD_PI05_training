# WBCD pi05 Full Fine-Tuning Pipeline

这个仓库整理了我们已经跑通的 pi05 qpos 版本全量微调流程。流程从原始 RoboTwin/ARX HDF5 数据开始，先转换成 ALOHA raw layout，再转换成 LeRobot repo，随后计算归一化统计，最后启动 `pi05_aloha_full_base` 全量微调。

所有命令都按“clone 后在仓库根目录执行”来写，默认使用相对路径。仓库不会依赖 `/root/gpufree-data/ljw/...` 这类本机路径。

## 数据假设

当前默认假设数据格式和之前 30 条 fold 数据一致：

```text
episode_0.hdf5
episode_1.hdf5
...
```

每个 HDF5 内部包含：

```text
/data/demo_0/observations/qpos
/data/demo_0/observations/images/head
/data/demo_0/observations/images/left_wrist
/data/demo_0/observations/images/right_wrist
```

输出动作使用之前跑通的 next-qpos 监督方式：

```text
observation.qpos = qpos_all[:-1]
action           = qpos_all[1:]
```

相机映射为：

```text
head        -> cam_high
left_wrist  -> cam_left_wrist
right_wrist -> cam_right_wrist
```

## 目录结构

```text
WBCD_pi05_training/
  README.md
  .gitignore
  env/
    robotwin-conda-full.yml
    robotwin-conda-history.yml
    pi05-runtime-notes.md
  scripts/
    setup_env.sh
    convert_robotwin_qpos_to_aloha_raw.py
    run_full_finetune_pipeline.sh
  pi05/
    examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
    scripts/compute_norm_stats.py
    scripts/train.py
    src/openpi/training/config.py
    ...
```

仓库没有包含数据、checkpoint、`.venv`、缓存和训练日志。这些内容体积很大，已经通过 `.gitignore` 排除。

## 环境

为了对齐当前已经跑通的服务器环境，仓库里保存了完整 conda 导出：

```text
env/robotwin-conda-full.yml
```

这个文件来自当前可运行的 `robotwin` 环境，并且已经去掉了本机 `prefix`，别人 clone 后可以直接创建：

```bash
conda env create -n robotwin -f env/robotwin-conda-full.yml
conda activate robotwin
```

如果本机已经有 `robotwin` 环境，可以更新：

```bash
conda env update -n robotwin -f env/robotwin-conda-full.yml --prune
conda activate robotwin
```

仓库也保留了一个最小历史记录：

```text
env/robotwin-conda-history.yml
```

但复现时优先使用 `robotwin-conda-full.yml`。

注意：之前训练虽然是在 `robotwin` shell 里启动，但真正运行 pi05 训练、归一化、LeRobot 转换的解释器是：

```bash
pi05/.venv/bin/python
```

也就是说，环境分两层：

```text
robotwin conda env      提供系统级 Python 包、CUDA 相关包、工具命令、uv 等
pi05/.venv via uv       真正执行 pi05/openpi 训练代码
```

新机器 clone 后，需要在 `pi05` 目录下用 uv 重建 `.venv`。uv 环境建议优先参考 RoboTwin 官方 Pi0.5 文档：

```text
https://robotwin-platform.github.io/doc/usage/Pi05.html
```

官方文档的核心步骤是：

```bash
conda activate robotwin
pip install uv
cd pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ..
```

本仓库也保留了 `pi05/pyproject.toml` 和 `pi05/uv.lock`，用于让 `uv sync` 对齐 pi05/openpi 运行依赖：

```bash
cd pi05
GIT_LFS_SKIP_SMUDGE=1 \
uv sync
cd ..
```

也可以使用仓库里的辅助脚本：

```bash
bash scripts/setup_env.sh robotwin
conda activate robotwin
cd pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ..
```

如果目标机器不能联网，需要提前准备好 Python 包、LeRobot 依赖、JAX CUDA wheel、OpenPI 预训练权重缓存和 tokenizer 缓存。

## 相对缓存路径

为了避免硬编码本机路径，建议把缓存都放在仓库根目录的 `.cache/` 下：

```bash
export ROOT_DIR="$(pwd)"
export OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME" logs
```

说明：

```text
OPENPI_DATA_HOME   OpenPI 预训练权重/tokenizer 缓存
HF_HOME            HuggingFace datasets/cache
HF_LEROBOT_HOME    LeRobot repo 输出目录
```

如果你已经有共享缓存，也可以把这些变量指向共享缓存路径。

## Step 1: 原始 HDF5 转 ALOHA Raw

假设你的新数据放在仓库外部：

```text
/path/to/source_hdf5
```

先进入仓库根目录：

```bash
cd WBCD_pi05_training
export ROOT_DIR="$(pwd)"
```

转换到 pi05 的 raw 目录：

```bash
pi05/.venv/bin/python scripts/convert_robotwin_qpos_to_aloha_raw.py \
  --src-dir /path/to/source_hdf5 \
  --dst-dir "$ROOT_DIR/pi05/training_data/fold" \
  --instruction "fold the cloth" \
  --overwrite
```

这里的 `--instruction` 就是训练 prompt。脚本会把它写进每个 episode 目录下的 `instructions.json`：

```json
{
  "instructions": [
    "fold the cloth"
  ]
}
```

如果要换任务 prompt，只需要改这个参数，例如：

```bash
pi05/.venv/bin/python scripts/convert_robotwin_qpos_to_aloha_raw.py \
  --src-dir /path/to/source_hdf5 \
  --dst-dir "$ROOT_DIR/pi05/training_data/pick_object" \
  --instruction "pick up the object" \
  --overwrite
```

后续 LeRobot 转换脚本会读取这个 `instructions.json`，并把 prompt 写入每一帧的 `task` 字段。训练配置里已经设置了：

```python
base_config=DataConfig(
    prompt_from_task=True,
)
```

所以训练时模型实际使用的 prompt 来源是：

```text
--instruction / --prompt
-> instructions.json
-> LeRobot frame["task"]
-> prompt_from_task=True
-> model input prompt
```

转换后的结构应该类似：

```text
pi05/training_data/fold/
  episode_0/
    episode_0.hdf5
    instructions.json
  episode_1/
    episode_1.hdf5
    instructions.json
```

每个输出 HDF5 内部应该有：

```text
/action
/observations/qpos
/observations/images/cam_high
/observations/images/cam_left_wrist
/observations/images/cam_right_wrist
```

## Step 2: ALOHA Raw 转 LeRobot

用本仓库里的转换脚本：

```bash
cd "$ROOT_DIR/pi05"

HF_HOME="$ROOT_DIR/.cache/huggingface" \
HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot" \
./.venv/bin/python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw_dir ./training_data/fold \
  --repo_id fold_repo
```

LeRobot 数据会写到：

```text
$HF_LEROBOT_HOME/fold_repo
```

也就是默认：

```text
WBCD_pi05_training/.cache/lerobot/fold_repo
```

转换完成后可以检查：

```bash
cd "$ROOT_DIR"

pi05/.venv/bin/python - <<'PY'
from pathlib import Path
import json
import os

repo = Path(os.environ.get("HF_LEROBOT_HOME", ".cache/lerobot")) / "fold_repo"
info = json.load(open(repo / "meta/info.json"))

print("repo exists:", repo.exists())
print("total_episodes:", info["total_episodes"])
print("total_frames:", info["total_frames"])
print("fps:", info["fps"])
print("features:")
for k, v in info["features"].items():
    print(" ", k, v["dtype"], v["shape"])
PY
```

之前 30 条 fold 数据的检查结果是：

```text
total_episodes: 30
total_frames: 54829
fps: 50
observation.state float32 [14]
action float32 [14]
observation.images.cam_high image [3, 480, 640]
observation.images.cam_left_wrist image [3, 480, 640]
observation.images.cam_right_wrist image [3, 480, 640]
```

## Step 3: 检查训练配置

全量微调用的配置文件是：

```text
pi05/src/openpi/training/config.py
```

配置名：

```text
pi05_aloha_full_base
```

当前关键配置应为：

```python
TrainConfig(
    name="pi05_aloha_full_base",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotAlohaDataConfig(
        repo_id="fold_repo",
        adapt_to_pi=False,
        ...
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=50_000,
    batch_size=32,
    log_interval=100,
    save_interval=5_000,
    keep_period=5_000,
    fsdp_devices=4,
    wandb_enabled=False,
)
```

判断它是全量微调的关键是：

```python
model=pi0_config.Pi0Config(pi05=True)
```

并且没有这些 LoRA 字段：

```python
paligemma_variant="gemma_2b_lora"
action_expert_variant="gemma_300m_lora"
freeze_filter=...
```

如果换新数据集，只需要把：

```python
repo_id="fold_repo"
```

改成新 LeRobot repo 名字，例如：

```python
repo_id="new_task_repo"
```

注意 `repo_id` 必须和 Step 2 的 `--repo_id` 一致。

## Step 4: 计算归一化统计

训练前必须计算 norm stats：

```bash
cd "$ROOT_DIR/pi05"

OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi" \
HF_HOME="$ROOT_DIR/.cache/huggingface" \
HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot" \
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_aloha_full_base
```

成功后会写到：

```text
pi05/assets/pi05_aloha_full_base/<repo_id>
```

例如：

```text
pi05/assets/pi05_aloha_full_base/fold_repo
```

如果遇到：

```text
OSError: [Errno 28] No space left on device
```

说明 HuggingFace datasets/LeRobot cache 或 checkpoint 占满了磁盘，需要先清理无用缓存或旧 checkpoint。不要删除当前要续训或评估的 checkpoint。

## Step 5: 开始全量微调

从头训练，4 卡，batch size 32，训练 5w step，每 5k step 保留一次 checkpoint：

```bash
cd "$ROOT_DIR/pi05"
mkdir -p "$ROOT_DIR/logs"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi" \
HF_HOME="$ROOT_DIR/.cache/huggingface" \
HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONUNBUFFERED=1 \
./.venv/bin/python scripts/train.py \
  pi05_aloha_full_base \
  --exp-name=fold_full_bs32_fsdp4_50k \
  --overwrite \
  --num-train-steps=50000 \
  --save-interval=5000 \
  --keep-period=5000 \
  --batch-size=32 \
  --fsdp-devices=4 \
  > "$ROOT_DIR/logs/fold_full_bs32_fsdp4_50k.log" 2>&1
```

checkpoint 会写到：

```text
pi05/checkpoints/pi05_aloha_full_base/fold_full_bs32_fsdp4_50k/
```

全量微调 checkpoint 的体积通常很大。之前看到单个 full checkpoint 大约：

```text
params       12G
train_state  31G
total        42G
```

如果只看到十几 G，通常要检查是不是跑成了 LoRA。

## 断点续训

断点续训不要用 `finetune.sh`，因为它带 `--overwrite`，会覆盖实验目录。

续训必须直接调用 `scripts/train.py`，并使用：

```text
--resume --no-overwrite
```

例如从已有实验目录继续训到 35000：

```bash
cd "$ROOT_DIR/pi05"
mkdir -p "$ROOT_DIR/logs"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi" \
HF_HOME="$ROOT_DIR/.cache/huggingface" \
HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONUNBUFFERED=1 \
./.venv/bin/python scripts/train.py \
  pi05_aloha_full_base \
  --exp-name=fold_full_bs32_fsdp4_50k \
  --resume \
  --no-overwrite \
  --num-train-steps=35000 \
  --save-interval=5000 \
  --keep-period=5000 \
  --batch-size=32 \
  --fsdp-devices=4 \
  > "$ROOT_DIR/logs/fold_full_bs32_fsdp4_50k_resume.log" 2>&1
```

## 一键流程脚本

仓库提供了封装脚本：

```bash
cd "$ROOT_DIR"

bash scripts/run_full_finetune_pipeline.sh \
  --src-dir /path/to/source_hdf5 \
  --raw-name fold \
  --repo-id fold_repo \
  --exp-name fold_full_bs32_fsdp4_50k \
  --prompt "fold the cloth" \
  --gpus 0,1,2,3 \
  --num-steps 50000 \
  --save-interval 5000 \
  --keep-period 5000 \
  --batch-size 32 \
  --fsdp-devices 4 \
  --log-path "$ROOT_DIR/logs/fold_full_bs32_fsdp4_50k.log"
```

一键脚本里的 `--prompt` 会传给第一步数据转换脚本的 `--instruction`。也就是说，如果使用一键脚本，修改训练 prompt 的位置就是这里：

```bash
--prompt "fold the cloth"
```

例如换成抓取任务：

```bash
bash scripts/run_full_finetune_pipeline.sh \
  --src-dir /path/to/source_hdf5 \
  --raw-name pick_object \
  --repo-id pick_object_repo \
  --exp-name pick_object_full_bs32_fsdp4_50k \
  --prompt "pick up the object" \
  --gpus 0,1,2,3 \
  --num-steps 50000 \
  --save-interval 5000 \
  --keep-period 5000 \
  --batch-size 32 \
  --fsdp-devices 4 \
  --log-path "$ROOT_DIR/logs/pick_object_full_bs32_fsdp4_50k.log"
```

这个脚本会自动设置默认相对缓存路径：

```text
$ROOT_DIR/.cache/openpi
$ROOT_DIR/.cache/huggingface
$ROOT_DIR/.cache/lerobot
```

并依次执行：

1. `scripts/convert_robotwin_qpos_to_aloha_raw.py`
2. `pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`
3. `pi05/scripts/compute_norm_stats.py`
4. `pi05/scripts/train.py`

运行前仍然要确认 `pi05/src/openpi/training/config.py` 里的 `repo_id` 和封装脚本的 `--repo-id` 一致。

## 常见检查命令

看当前训练进程：

```bash
ps -eo pid,etime,cmd | grep 'scripts/train.py'
```

看 checkpoint 大小：

```bash
du -h --max-depth=2 pi05/checkpoints/pi05_aloha_full_base/<exp-name> | sort -h
```

看 GPU：

```bash
nvidia-smi
```

或：

```bash
nvitop
```

检查 LeRobot repo：

```bash
find "$ROOT_DIR/.cache/lerobot/fold_repo" -maxdepth 2 -type f | head
```

## 注意事项

1. 新数据格式必须和之前 30 条 fold 数据一致，否则 `convert_robotwin_qpos_to_aloha_raw.py` 需要改 HDF5 key。
2. `repo_id` 必须在 LeRobot 转换、`config.py`、归一化和训练四处保持一致。
3. 全量微调非常占磁盘，每 5k 一个 checkpoint 时，单个 checkpoint 约 42G。
4. 不要在还有用的 checkpoint 目录上运行带 `--overwrite` 的训练命令。
5. 如果只是续训，必须使用 `--resume --no-overwrite`。
