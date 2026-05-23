"""
Convert EEF hdf5 episodes produced by scripts/process_deformable_eef_data.py to LeRobot v2 format.
"""

import dataclasses
import fnmatch
import json
import os
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


CONTROL_MODE_TAG = "<control_mode> end effector <control_mode>"


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _eef_names() -> list[str]:
    return [
        "left_x",
        "left_y",
        "left_z",
        "left_qx",
        "left_qy",
        "left_qz",
        "left_qw",
        "left_gripper",
        "right_x",
        "right_y",
        "right_z",
        "right_qx",
        "right_qy",
        "right_qz",
        "right_qw",
        "right_gripper",
    ]


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    eef_dims = _eef_names()
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(eef_dims),),
            "names": [eef_dims],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(eef_dims),),
            "names": [eef_dims],
        },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _load_images(ep: h5py.File, camera: str) -> np.ndarray:
    ds = ep[f"/observations/images/{camera}"]
    if ds.ndim == 4:
        return ds[:]
    frames = []
    for payload in ds:
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode {camera} frame")
        frames.append(image)
    return np.asarray(frames)


def load_raw_episode_data(ep_path: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:]).float()
        action = torch.from_numpy(ep["/action"][:]).float()
        imgs_per_cam = {
            camera: _load_images(ep, camera)
            for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        }
    return imgs_per_cam, state, action


def _episode_instruction(ep_path: Path, fallback: str) -> str:
    instruction_path = ep_path.parent / "instructions.json"
    if not instruction_path.exists():
        return fallback
    with instruction_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    instructions = payload.get("instructions") or [fallback]
    return str(instructions[0])


def _with_control_mode(instruction: str) -> str:
    if CONTROL_MODE_TAG in instruction:
        return instruction
    return f"{instruction} {CONTROL_MODE_TAG}"


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        instruction = _with_control_mode(_episode_instruction(ep_path, task))

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset


def port_eef(
    raw_dir: Path,
    repo_id: str,
    task: str = "deformable manipulation",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> None:
    if not raw_dir.exists():
        raise ValueError(f"raw_dir does not exist: {raw_dir}")

    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(Path(root) / filename)
    hdf5_files = sorted(hdf5_files)
    if not hdf5_files:
        raise ValueError(f"No hdf5 episodes found under {raw_dir}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="dual_arm_eef",
        mode=mode,
        dataset_config=dataset_config,
    )
    populate_dataset(dataset, hdf5_files, task=task, episodes=episodes)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_eef)
