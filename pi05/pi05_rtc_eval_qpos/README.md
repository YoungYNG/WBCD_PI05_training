# PI05 Qpos RTC Evaluation

This directory contains an isolated RoboTwin evaluation entrypoint for the
`pi05_base_aloha_lora` / `demo_clean` checkpoint with RTC-style action chunking.

It is aligned with the training command:

```bash
cd /root/gpufree-data/ljw/RoboTwin/policy/pi05
PYTHONUNBUFFERED=1 bash finetune.sh pi05_base_aloha_lora demo_clean 0,1,2,3
```

The checkpoint is loaded from:

```text
policy/pi05/checkpoints/pi05_base_aloha_lora/demo_clean/<checkpoint_id>
```

## Run

From the RoboTwin root:

```bash
bash policy/pi05/pi05_rtc_eval_qpos/eval_rtc.sh \
  <task_name> \
  <task_config> \
  pi05_base_aloha_lora \
  demo_clean \
  30000 \
  0 \
  0 \
  100 \
  0
```

Arguments:

```text
1 task_name
2 task_config
3 train_config_name, default pi05_base_aloha_lora
4 model_name, default demo_clean
5 checkpoint_id, default 040000
6 seed, default 0
7 gpu_id, default 0
8 test_num, default 100
9 rtc_inference_delay, default 0
```

## Training Alignment

The training config is `LeRobotAlohaDataConfig` with:

```text
repo_id = demo_clean_repo
adapt_to_pi = False
prompt_from_task = True
use_delta_joint_actions = True
```

The local LeRobot dataset metadata shows:

```text
state/action shape: 14
state/action layout:
  left_waist,left_shoulder,left_elbow,left_forearm_roll,left_wrist_angle,left_wrist_rotate,left_gripper,
  right_waist,right_shoulder,right_elbow,right_forearm_roll,right_wrist_angle,right_wrist_rotate,right_gripper

cameras:
  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist

prompt:
  Pick up the two corners of the white garment on the table, place it over the clothing board on the blue rack, and smooth it flat.
```

This evaluator therefore:

- uses real RGB from all three RoboTwin cameras, no mask image,
- uses `observation["joint_action"]["vector"]` as the 14-D qpos state,
- uses the fixed prompt above from `demo_clean_repo/meta/tasks.jsonl`,
- executes actions with `TASK_ENV.take_action(action, action_type="qpos")`.

OpenPI's training transform converts absolute qpos chunks to delta joint actions
for learning and converts model outputs back to absolute qpos at inference via
the configured output transforms, so the rollout receives executable absolute
qpos actions.

## RTC

The evaluator keeps an action queue and refills it when 10 actions remain. The
new chunk is blended with the unexecuted prefix from the previous chunk using an
RTC-style prefix schedule, then the queue is replaced after accounting for
`rtc_inference_delay`.

For synchronous RoboTwin simulation, `rtc_inference_delay=0` is the default
because no actions are consumed while policy inference blocks. Use a positive
value, for example `4`, only if you want to simulate real-controller latency.
