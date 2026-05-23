# PI05 RTC Evaluation

This directory contains an isolated RoboTwin evaluation entrypoint for the
`pi05_base_eef_hdf5_lora` checkpoint with RTC-style action chunking.

The checkpoint trained with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/python scripts/train.py pi05_base_eef_hdf5_lora \
--exp-name=deformable_eef_hdf5_run \
--resume
```

is loaded from:

```text
policy/pi05/checkpoints/pi05_base_eef_hdf5_lora/deformable_eef_hdf5_run/<checkpoint_id>
```

## Run

From the RoboTwin root:

```bash
bash policy/pi05/pi05_rtc_eval/eval_rtc.sh \
  <task_name> \
  <task_config> \
  pi05_base_eef_hdf5_lora \
  deformable_eef_hdf5_run \
  10000 \
  0 \
  0 \
  100 \
  0
```

Arguments:

```text
1 task_name
2 task_config
3 train_config_name, default pi05_base_eef_hdf5_lora
4 model_name, default deformable_eef_hdf5_run
5 checkpoint_id, default 040000
6 seed, default 0
7 gpu_id, default 0
8 test_num, default 100
9 rtc_inference_delay, default 0
```

For synchronous RoboTwin simulation, `rtc_inference_delay=0` is the least
surprising default because no robot actions are executed while the model is
blocking on inference. If you want to mimic a real controller where several
actions are consumed during policy inference, pass a positive delay, for example
`4`.

The default `rtc_refill_threshold` is `10`, matching the default
`rtc_execution_horizon`. This means the evaluator requests the next chunk while
10 actions from the previous chunk are still unexecuted, so the RTC overlap is
actually used.

## EEF Training Alignment

This evaluator is aligned with `HDF5EEFDataConfig` and
`process_deformable_eef_data.py`:

- `cam_high` is a fixed black mask image, matching the training HDF5 data.
- `cam_left_wrist` uses RoboTwin's left wrist RGB image.
- `cam_right_wrist` uses RoboTwin's right wrist RGB image.
- `state` is `left_xyzquat,left_gripper,right_xyzquat,right_gripper`.
- `prompt` is fixed to
  `deformable manipulation <control_mode> end effector <control_mode>`.
- output actions are executed with `TASK_ENV.take_action(action, action_type="ee")`.

The state/action layout is 16-D:

```text
left_xyzquat(7), left_gripper(1), right_xyzquat(7), right_gripper(1)
```

## What RTC Means Here

LeRobot's PI0 RTC implementation injects prefix consistency inside the diffusion
denoising loop through `predict_action_chunk(..., prev_chunk_left_over=...)`.
The OpenPI JAX PI05 policy used by this RoboTwin checkout exposes only
`policy.infer(obs) -> action_chunk`, and the JAX `sample_actions` method does
not accept RTC denoising kwargs.

Because of that API boundary, this implementation applies RTC-style behavior at
the rollout layer:

- keeps an action queue,
- tracks unexecuted actions from the previous chunk,
- blends the overlapping prefix of the next chunk with the previous leftover
  using the configured RTC prefix attention schedule,
- replaces the queue after accounting for `rtc_inference_delay`.

This gives RTC-style real-time chunk continuity without modifying OpenPI core
model code.
