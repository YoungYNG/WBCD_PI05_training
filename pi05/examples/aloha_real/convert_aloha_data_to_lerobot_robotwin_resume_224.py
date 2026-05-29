"""
Resume-safe converter from RoboTwin/Aloha-style HDF5 episodes to LeRobot v2.

This script is intentionally separate from convert_aloha_data_to_lerobot_robotwin.py
because the original converter removes the target repo on startup. This version can
append new episodes to an existing local LeRobot repo and is useful for large
conversions that may stall midway.
"""

import dataclasses
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import torch
import tqdm
import tyro


MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

CAMERAS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class ConvertConfig:
    raw_dir: Path
    repo_id: str
    raw_repo_id: str | None = None
    task: str = "DEBUG"
    push_to_hub: bool = False
    is_mobile: bool = False
    mode: Literal["video", "image"] = "image"
    dataset_config: DatasetConfig = DatasetConfig()
    order_mode: Literal["oswalk", "numeric"] = "oswalk"
    start_output_index: int | None = None
    end_output_index: int | None = None
    overwrite: bool = False
    validate_existing: bool = True


def _episode_number(path: Path) -> int:
    matches = re.findall(r"episode[_-]?(\d+)", str(path))
    return int(matches[-1]) if matches else 10**12


def collect_hdf5_files(raw_dir: Path, order_mode: str) -> list[Path]:
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(Path(root) / filename)

    if order_mode == "numeric":
        hdf5_files.sort(key=lambda p: (_episode_number(p), str(p)))

    return hdf5_files


def dataset_features(mode: Literal["video", "image"]) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        }

    return features


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"],
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=dataset_features(mode),
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def open_dataset_for_append(
    repo_id: str,
    repo_root: Path,
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    """Create a minimal LeRobotDataset object that appends without loading all parquet files."""
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=repo_root)
    obj.repo_id = obj.meta.repo_id
    obj.root = obj.meta.root
    obj.revision = None
    obj.tolerance_s = dataset_config.tolerance_s
    obj.image_writer = None
    if dataset_config.image_writer_processes or dataset_config.image_writer_threads:
        obj.start_image_writer(dataset_config.image_writer_processes, dataset_config.image_writer_threads)
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.episode_data_index = None
    obj.video_backend = dataset_config.video_backend
    return obj


def load_raw_images_per_camera(ep: h5py.File) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in CAMERAS:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image for camera {camera}")
                imgs_array.append(img)
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(ep_path: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        imgs_per_cam = load_raw_images_per_camera(ep)

    if state.shape != action.shape:
        raise ValueError(f"state/action shape mismatch in {ep_path}: {state.shape} vs {action.shape}")
    if state.shape[1] != len(MOTORS):
        raise ValueError(f"Expected {len(MOTORS)} state dims in {ep_path}, got {state.shape}")

    for camera, imgs in imgs_per_cam.items():
        if len(imgs) != state.shape[0]:
            raise ValueError(f"{camera} frame count mismatch in {ep_path}: {len(imgs)} vs {state.shape[0]}")

    return imgs_per_cam, state, action


def load_instruction(ep_path: Path, fallback_task: str) -> str:
    instructions_path = ep_path.parent / "instructions.json"
    if not instructions_path.exists():
        return fallback_task

    with open(instructions_path) as f_instr:
        instruction_dict = json.load(f_instr)

    instructions = instruction_dict.get("instructions") or []
    if not instructions:
        return fallback_task
    return str(np.random.choice(instructions))


def save_order_manifest(repo_root: Path, raw_dir: Path, order_mode: str, hdf5_files: list[Path]) -> None:
    manifest_path = repo_root / "meta" / "conversion_order_manifest.json"
    manifest = {
        "raw_dir": str(raw_dir.resolve()),
        "order_mode": order_mode,
        "num_files": len(hdf5_files),
        "files": [str(path) for path in hdf5_files],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def validate_existing_dataset(repo_root: Path) -> int:
    info_path = repo_root / "meta" / "info.json"
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata under {repo_root}")

    info = json.load(open(info_path))
    episodes = [json.loads(line) for line in open(episodes_path)]
    parquets = sorted(repo_root.glob("data/chunk-*/episode_*.parquet"))

    if info["total_episodes"] != len(episodes) or len(episodes) != len(parquets):
        raise ValueError(
            "Existing repo is inconsistent: "
            f"info total_episodes={info['total_episodes']}, "
            f"episodes.jsonl={len(episodes)}, parquet={len(parquets)}"
        )

    return int(info["total_episodes"])


def append_episode(dataset: LeRobotDataset, ep_path: Path, task: str) -> None:
    imgs_per_cam, state, action = load_raw_episode_data(ep_path)
    instruction = load_instruction(ep_path, fallback_task=task)
    num_frames = state.shape[0]

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


def port_aloha_resume(config: ConvertConfig) -> None:
    raw_dir = config.raw_dir
    repo_id = config.repo_id
    repo_root = HF_LEROBOT_HOME / repo_id

    if not raw_dir.exists():
        raise ValueError(f"raw_dir does not exist: {raw_dir}")

    hdf5_files = collect_hdf5_files(raw_dir, config.order_mode)
    if not hdf5_files:
        raise ValueError(f"No hdf5 files found under {raw_dir}")

    if repo_root.exists() and config.overwrite:
        shutil.rmtree(repo_root)

    if repo_root.exists():
        completed = validate_existing_dataset(repo_root) if config.validate_existing else int(
            json.load(open(repo_root / "meta" / "info.json"))["total_episodes"]
        )
        dataset = open_dataset_for_append(repo_id, repo_root, config.dataset_config)
        print(f"Resuming existing repo: {repo_root}")
        print(f"Completed output episodes: {completed}")
    else:
        dataset = create_empty_dataset(
            repo_id=repo_id,
            robot_type="mobile_aloha" if config.is_mobile else "aloha",
            mode=config.mode,
            dataset_config=config.dataset_config,
        )
        completed = 0
        print(f"Created new repo: {repo_root}")

    save_order_manifest(repo_root, raw_dir, config.order_mode, hdf5_files)

    start = completed if config.start_output_index is None else config.start_output_index
    end = len(hdf5_files) if config.end_output_index is None else min(config.end_output_index, len(hdf5_files))

    if start != completed:
        raise ValueError(
            "start_output_index must equal the number of completed episodes when appending. "
            f"Got start_output_index={start}, completed={completed}."
        )
    if end <= start:
        print(f"Nothing to do: start={start}, end={end}, total raw files={len(hdf5_files)}")
        return

    print(f"Raw hdf5 files: {len(hdf5_files)}")
    print(f"Converting output episode indices [{start}, {end})")
    print(f"First source file: {hdf5_files[start]}")
    print(f"Last source file: {hdf5_files[end - 1]}")

    try:
        for output_idx in tqdm.tqdm(range(start, end)):
            ep_path = hdf5_files[output_idx]
            print(f"\n[episode {output_idx}] source={ep_path}", flush=True)
            append_episode(dataset, ep_path, config.task)
    finally:
        dataset.stop_image_writer()

    if config.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    port_aloha_resume(tyro.cli(ConvertConfig))
