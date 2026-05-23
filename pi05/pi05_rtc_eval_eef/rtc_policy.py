import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PI05_DIR = os.path.dirname(CURRENT_DIR)
if PI05_DIR not in sys.path:
    sys.path.append(PI05_DIR)

from pi_model import PI0  # noqa: E402


class RTCAttentionSchedule(str, Enum):
    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclass
class RTCConfig:
    enabled: bool = True
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    inference_delay: int = 0
    refill_threshold: int = 0


@dataclass
class EEFInferenceConfig:
    prompt: str = "deformable manipulation <control_mode> end effector <control_mode>"
    head_camera_mask_value: int = 0


class RTCActionQueue:
    """Numpy action queue compatible with LeRobot's RTC queue semantics."""

    def __init__(self, cfg: RTCConfig):
        self.cfg = cfg
        self.queue: np.ndarray | None = None
        self.original_queue: np.ndarray | None = None
        self.last_index = 0
        self.lock = Lock()

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        with self.lock:
            if self.queue is None:
                return 0
            return max(0, len(self.queue) - self.last_index)

    def get(self) -> np.ndarray | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index].copy()
            self.last_index += 1
            return action

    def get_left_over(self) -> np.ndarray | None:
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].copy()

    def merge(self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int) -> None:
        original_actions = np.asarray(original_actions)
        processed_actions = np.asarray(processed_actions)
        if original_actions.ndim != 2 or processed_actions.ndim != 2:
            raise ValueError("RTC action chunks must be 2D arrays shaped (time, action_dim)")

        with self.lock:
            if self.cfg.enabled:
                delay = int(np.clip(real_delay, 0, min(len(original_actions), len(processed_actions))))
                self.original_queue = original_actions[delay:].copy()
                self.queue = processed_actions[delay:].copy()
                self.last_index = 0
                return

            if self.queue is None:
                self.original_queue = original_actions.copy()
                self.queue = processed_actions.copy()
                self.last_index = 0
                return

            self.original_queue = np.concatenate([self.original_queue, original_actions], axis=0)[self.last_index :]
            self.queue = np.concatenate([self.queue, processed_actions], axis=0)[self.last_index :]
            self.last_index = 0


def _linweights(start: int, end: int, total: int) -> np.ndarray:
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linspace(1.0, 0.0, linspace_steps + 2, dtype=np.float32)[1:-1]


def _prefix_weights(cfg: RTCConfig, start: int, end: int, total: int) -> np.ndarray:
    start = min(start, end, total)
    end = min(end, total)

    if cfg.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
        weights = np.zeros(total, dtype=np.float32)
        weights[:start] = 1.0
        return weights

    if cfg.prefix_attention_schedule == RTCAttentionSchedule.ONES:
        weights = np.ones(total, dtype=np.float32)
        weights[end:] = 0.0
        return weights

    weights = _linweights(start, end, total)
    if cfg.prefix_attention_schedule == RTCAttentionSchedule.EXP and len(weights) > 0:
        weights = weights * np.expm1(weights) / (np.e - 1.0)

    leading = np.ones(min(start, total), dtype=np.float32)
    trailing = np.zeros(max(total - end, 0), dtype=np.float32)
    return np.concatenate([leading, weights, trailing], axis=0)[:total]


def rtc_blend_actions(
    new_actions: np.ndarray,
    prev_chunk_left_over: np.ndarray | None,
    cfg: RTCConfig,
) -> np.ndarray:
    """Blend a new action chunk with unexecuted actions from the previous chunk.

    LeRobot's full RTC implementation applies this consistency objective inside
    the diffusion denoising loop. OpenPI's JAX PI0/PI05 policy in this repository
    does not expose that hook, so this adapter applies RTC-style prefix
    consistency at the chunk level before replacing the rollout queue.
    """

    new_actions = np.asarray(new_actions)
    if not cfg.enabled or prev_chunk_left_over is None or len(prev_chunk_left_over) == 0:
        return new_actions.copy()

    overlap = min(len(new_actions), len(prev_chunk_left_over), cfg.execution_horizon)
    if overlap <= 0:
        return new_actions.copy()

    blended = new_actions.copy()
    weights = _prefix_weights(
        cfg,
        start=cfg.inference_delay,
        end=overlap,
        total=len(new_actions),
    )[:overlap]

    # A mild gain keeps max_guidance_weight meaningful in this chunk-level
    # adapter without letting it extrapolate beyond the previous chunk.
    gain = cfg.max_guidance_weight / (cfg.max_guidance_weight + 1.0)
    weights = np.clip(weights * gain, 0.0, 1.0).astype(new_actions.dtype, copy=False)
    blended[:overlap] = weights[:, None] * prev_chunk_left_over[:overlap] + (1.0 - weights[:, None]) * new_actions[:overlap]
    return blended


def _black_like(image: np.ndarray, value: int = 0) -> np.ndarray:
    return np.full_like(image, fill_value=value)


def encode_eef_obs(task_env, observation, cfg: EEFInferenceConfig):
    """Encode observations exactly like the HDF5 EEF training dataset.

    Training data layout:
      images/cam_high: fixed black mask image
      images/cam_left_wrist: left wrist RGB
      images/cam_right_wrist: right wrist RGB
      state: left_xyzquat,left_gripper,right_xyzquat,right_gripper
    """

    left_wrist = observation["observation"]["left_camera"]["rgb"]
    right_wrist = observation["observation"]["right_camera"]["rgb"]
    head_mask = _black_like(left_wrist, cfg.head_camera_mask_value)

    input_rgb_arr = [
        head_mask,
        right_wrist,
        left_wrist,
    ]

    input_state = np.asarray(
        task_env.robot.get_left_ee_pose()
        + [task_env.robot.get_left_gripper_val()]
        + task_env.robot.get_right_ee_pose()
        + [task_env.robot.get_right_gripper_val()],
        dtype=np.float32,
    )
    return input_rgb_arr, input_state


class RTCPI0(PI0):
    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        rtc_config: RTCConfig,
        eef_config: EEFInferenceConfig | None = None,
    ):
        super().__init__(train_config_name, model_name, checkpoint_id, pi0_step)
        self.rtc_config = rtc_config
        self.eef_config = eef_config or EEFInferenceConfig()
        self.action_queue = RTCActionQueue(rtc_config)
        self.last_infer_ms = 0.0

    def reset_obsrvationwindows(self):
        super().reset_obsrvationwindows()
        self.action_queue.clear()
        self.last_infer_ms = 0.0

    def maybe_refill_action_queue(self, task_env, observation) -> None:
        if self.action_queue.qsize() > self.rtc_config.refill_threshold:
            return

        if self.observation_window is None:
            self.set_language(self.eef_config.prompt)

        input_rgb_arr, input_state = encode_eef_obs(task_env, observation, self.eef_config)
        self.update_observation_window(input_rgb_arr, input_state)

        prev_actions = self.action_queue.get_left_over()
        start_time = time.monotonic()
        new_actions = self.get_action()[: self.pi0_step]
        self.last_infer_ms = (time.monotonic() - start_time) * 1000.0

        processed_actions = rtc_blend_actions(new_actions, prev_actions, self.rtc_config)
        self.action_queue.merge(
            original_actions=new_actions,
            processed_actions=processed_actions,
            real_delay=self.rtc_config.inference_delay,
        )

    def next_action(self, task_env, observation) -> np.ndarray | None:
        self.maybe_refill_action_queue(task_env, observation)
        return self.action_queue.get()
