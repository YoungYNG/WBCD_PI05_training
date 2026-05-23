#!/usr/bin/env python3
"""Convert RoboTwin/ARX qpos HDF5 episodes to the ALOHA raw layout used by pi05.

Input format expected by this script:

    <src-dir>/episode_0.hdf5
    <src-dir>/episode_1.hdf5
    ...

Each source HDF5 file should contain one demo at `/data/demo_0` with:

    observations/qpos
    observations/images/head
    observations/images/left_wrist
    observations/images/right_wrist

The output format is the raw ALOHA-style directory consumed by:

    pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py

Output layout:

    <dst-dir>/episode_0/episode_0.hdf5
    <dst-dir>/episode_0/instructions.json
    <dst-dir>/episode_1/episode_1.hdf5
    <dst-dir>/episode_1/instructions.json

The training target follows the qpos next-state convention used in our previous
full-finetune runs:

    observations/qpos = qpos_all[:-1]
    action            = qpos_all[1:]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


CAMERA_MAP = {
    "head": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse episode index from file name: {path.name}") from exc


def read_padded_image_bytes(demo: h5py.Group, cam_name: str, count: int) -> list[bytes]:
    image_ds = demo[f"observations/images/{cam_name}"]
    length_name = f"observations/images/{cam_name}_lengths"

    frames: list[bytes] = []
    if length_name in demo:
        lengths = demo[length_name][()]
        for row, n in zip(image_ds[:count], lengths[:count], strict=False):
            frames.append(bytes(row[: int(n)]))
    else:
        for row in image_ds[:count]:
            if isinstance(row, bytes):
                frames.append(row.rstrip(b"\0"))
            else:
                frames.append(bytes(row).rstrip(b"\0"))
    return frames


def write_fixed_string_dataset(group: h5py.Group, name: str, frames: list[bytes]) -> None:
    if not frames:
        raise ValueError(f"No frames for {name}")
    max_len = max(len(x) for x in frames)
    padded = [x.ljust(max_len, b"\0") for x in frames]
    group.create_dataset(name, data=padded, dtype=f"S{max_len}")


def convert_episode(src_path: Path, dst_dir: Path, instruction_text: str, overwrite: bool) -> None:
    idx = episode_index(src_path)
    ep_dir = dst_dir / f"episode_{idx}"
    dst_path = ep_dir / f"episode_{idx}.hdf5"
    instr_path = ep_dir / "instructions.json"

    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"{dst_path} already exists. Use --overwrite to replace it.")

    ep_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        demo = src["/data/demo_0"]

        qpos_all = demo["observations/qpos"][()].astype(np.float32)
        qpos = qpos_all[:-1]
        action = qpos_all[1:]
        frame_count = qpos.shape[0]

        image_frames = {
            dst_name: read_padded_image_bytes(demo, src_name, frame_count)
            for src_name, dst_name in CAMERA_MAP.items()
        }

    with h5py.File(dst_path, "w") as dst:
        dst.create_dataset("action", data=action)

        obs = dst.create_group("observations")
        obs.create_dataset("qpos", data=qpos)

        images = obs.create_group("images")
        for dst_name, frames in image_frames.items():
            write_fixed_string_dataset(images, dst_name, frames)

    with instr_path.open("w", encoding="utf-8") as f:
        json.dump({"instructions": [instruction_text]}, f, indent=2)

    print(f"wrote {dst_path} frames={frame_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="Directory containing source episode_*.hdf5 files.")
    parser.add_argument("--dst-dir", type=Path, required=True, help="Output raw ALOHA-style directory.")
    parser.add_argument("--instruction", default="fold the cloth", help="Task prompt written to instructions.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted episode files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_files = sorted(args.src_dir.glob("episode_*.hdf5"), key=episode_index)
    if not src_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {args.src_dir}")

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"found {len(src_files)} source episodes")
    print(f"source: {args.src_dir}")
    print(f"target: {args.dst_dir}")

    for src_path in src_files:
        convert_episode(src_path, args.dst_dir, args.instruction, args.overwrite)

    print(f"done: {args.dst_dir}")


if __name__ == "__main__":
    main()
