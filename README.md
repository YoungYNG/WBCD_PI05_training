# WBCD PI05 Training

This repo is a compact reproduction package for fine-tuning PI05 on RoboTwin-style bimanual qpos data.

The expected workflow is:

```text
RoboTwin HDF5
  -> optional 224x224 resize_with_pad preprocessing
  -> ALOHA raw HDF5
  -> LeRobot repo
  -> norm stats
  -> PI05 full fine-tuning
```

The scripts here intentionally do not include datasets, checkpoints, logs, HuggingFace caches, or virtual environments.

## 1. Environment

Use the official RoboTwin PI05 environment as the base environment:

```bash
conda activate robotwin
pip install uv
```

Then create the PI05 Python environment inside this repo:

```bash
cd WBCD_PI05_training/pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ..
```

For our current server setup, training and data conversion are run with:

```bash
pi05/.venv/bin/python
```

Recommended cache variables:

```bash
cd WBCD_PI05_training

export ROOT_DIR="$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_LEROBOT_HOME="$ROOT_DIR/.cache/lerobot"
export OPENPI_DATA_HOME="$ROOT_DIR/.cache/openpi"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" logs
```

If your machine already has shared HuggingFace/OpenPI caches, point these variables to the shared cache paths instead.

## 2. Data Format

The raw RoboTwin data directory should contain one HDF5 file per episode:

```text
/path/to/source_data/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

Each HDF5 should contain the RoboTwin layout:

```text
/data/demo_0/observations/qpos
/data/demo_0/observations/images/head
/data/demo_0/observations/images/left_wrist
/data/demo_0/observations/images/right_wrist
```

The conversion uses next-qpos supervision:

```text
observation.state = qpos[:-1]
action            = qpos[1:]
```

Camera mapping:

```text
head        -> cam_high
left_wrist  -> cam_left_wrist
right_wrist -> cam_right_wrist
```

## 3. Optional: Resize Raw HDF5 Images To 224x224

If the raw HDF5 images are not already 224x224, resize them first. The resize implementation is an OpenCV equivalent of OpenPI `resize_with_pad`: aspect-ratio preserving resize with centered black padding.

```bash
cd "$ROOT_DIR"

export SRC_DIR="/path/to/source_data"
export RESIZED_DIR="/path/to/source_data_resize224"

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
pi05/.venv/bin/python scripts/resize_robotwin_raw_hdf5_images_224.py \
  --src-dir "$SRC_DIR" \
  --dst-dir "$RESIZED_DIR" \
  --workers 32 \
  --overwrite \
  2>&1 | tee logs/resize_to_224.log
```

If the data is already resized with the same `resize_with_pad` behavior, skip this step and set `SRC_DIR` to the existing resized directory.

## 4. Convert RoboTwin HDF5 To ALOHA Raw

Set task-specific variables:

```bash
cd "$ROOT_DIR"

export SRC_DIR="/path/to/source_data_resize224"
export TASK_NAME="fold_5_28_120_resize224_45fps"
export RAW_DIR="$ROOT_DIR/pi05/training_data/$TASK_NAME"
export REPO_ID="${TASK_NAME}_repo"
export PROMPT="Pick up the two corners of the white garment on the table, place it over the clothing board on the blue rack, and smooth it flat."
```

Run conversion:

```bash
pi05/.venv/bin/python scripts/convert_robotwin_qpos_to_aloha_raw.py \
  --src-dir "$SRC_DIR" \
  --dst-dir "$RAW_DIR" \
  --instruction "$PROMPT" \
  --overwrite \
  2>&1 | tee "logs/convert_${TASK_NAME}_to_aloha_raw.log"
```

Expected output:

```text
$RAW_DIR/episode_0/episode_0.hdf5
$RAW_DIR/episode_0/instructions.json
```

## 5. Convert ALOHA Raw To LeRobot

Use the 224-specific LeRobot converter. It writes image feature shape as `[3, 224, 224]` and supports resume-safe conversion.

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

For faster machines, `32 x 2` is a safe starting point. We usually test `24 x 2`, `32 x 2`, and `48 x 2`, then keep the best stable setting.

Check metadata:

```bash
cd "$ROOT_DIR"

pi05/.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["HF_LEROBOT_HOME"]) / os.environ["REPO_ID"]
info = json.loads((repo / "meta/info.json").read_text())

print("repo:", repo)
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("fps:", info.get("fps"))
for k, v in info.get("features", {}).items():
    if k in ("observation.state", "action") or "observation.images" in k:
        print(k, v)
PY
```

Expected image features:

```text
observation.images.cam_high       shape [3, 224, 224]
observation.images.cam_left_wrist shape [3, 224, 224]
observation.images.cam_right_wrist shape [3, 224, 224]
```

## 6. Configure Training

Edit:

```text
pi05/src/openpi/training/config.py
```

Update the `pi05_aloha_full_base` config:

```python
TrainConfig(
    name="pi05_aloha_full_base",
    ...
    data=LeRobotAlohaDataConfig(
        repo_id="your_lerobot_repo",
        adapt_to_pi=False,
        ...
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    batch_size=40,
    num_workers=16,
    save_interval=5_000,
    keep_period=5_000,
    fsdp_devices=4,
    wandb_enabled=False,
)
```

Set `repo_id` to the same value used in LeRobot conversion:

```python
repo_id="<REPO_ID>"
```

For continued fine-tuning from a local checkpoint, set `weight_loader` to the checkpoint `params` directory:

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "/path/to/checkpoints/pi05_aloha_full_base/<exp>/<step>/params"
)
```

This loads model parameters only. It does not resume optimizer state or training step. Use `--resume` only when you intentionally want to resume the same training run state.

## 7. Compute Norm Stats

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

Expected output:

```text
pi05/assets/pi05_aloha_full_base/<repo_id>/norm_stats.json
```

## 8. Train

Example: 4 GPUs, global batch size 40, 20k steps, checkpoint every 5k steps:

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

Checkpoints:

```text
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/5000
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/10000
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/15000
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/20000
```

Automatic loss plot:

```text
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/plots/loss_curve.png
pi05/checkpoints/pi05_aloha_full_base/<exp_name>/plots/loss_points.csv
```

Set `OPENPI_PLOT_LOSS=0` to disable live plotting.

## 9. Plot Loss From An Existing Log

If the run was started without live plotting, create the curve from the log:

```bash
cd "$ROOT_DIR/pi05"

MPLCONFIGDIR=/tmp/matplotlib-cache \
./.venv/bin/python scripts/plot_train_loss.py \
  "$ROOT_DIR/logs/train_${EXP_NAME}.log"
```

This writes:

```text
logs/train_<exp_name>_loss_curve.png
logs/train_<exp_name>_loss_points.csv
```

## Notes

- `batch_size` must be divisible by the number of visible JAX devices.
- In our tests, `fsdp_devices=4, batch_size=32-40` was stable on 4 GPUs.
- The model consumes the current three camera images, current state, and prompt, then predicts an action chunk. It does not use recurrent memory or a history window by default.
- Keep data, checkpoints, logs, and caches outside git. The `.gitignore` already excludes the common output paths.
