#!/usr/bin/env python3
"""Create a 224x224-image copy of ALOHA raw HDF5 episodes.

State/action frame counts are kept unchanged. Images are decoded from the raw
HDF5 byte-string datasets and written back as fixed-length PNG byte strings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time

import cv2
import h5py
import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def read_image_bytes(ds: h5py.Dataset, idx: int) -> bytes:
    value = ds[idx]
    if isinstance(value, bytes):
        return value.rstrip(b"\0")
    return bytes(value).rstrip(b"\0")


def resize_with_pad_opencv(image_bgr: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    cur_height, cur_width = image_bgr.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    return cv2.copyMakeBorder(resized, pad_h0, pad_h1, pad_w0, pad_w1, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def decode_resize_encode_png(raw_bytes: bytes) -> bytes:
    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("cv2.imdecode failed")
    resized_bgr = resize_with_pad_opencv(image_bgr, 224, 224)
    ok, encoded_png = cv2.imencode(".png", resized_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("cv2.imencode PNG failed")
    return encoded_png.tobytes()


def write_fixed_string_dataset(group: h5py.Group, name: str, frames: list[bytes]) -> None:
    max_len = max(len(frame) for frame in frames)
    padded = [frame.ljust(max_len, b"\0") for frame in frames]
    group.create_dataset(name, data=padded, dtype=f"S{max_len}")


def convert_episode(src_path: Path, src_root: Path, dst_root: Path, overwrite: bool) -> tuple[int, int, float]:
    start_time = time.perf_counter()
    rel = src_path.relative_to(src_root)
    dst_path = dst_root / rel
    instr_src = src_path.parent / "instructions.json"
    instr_dst = dst_path.parent / "instructions.json"

    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"{dst_path} exists; pass --overwrite to replace it")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        action = src["action"][()]
        qpos = src["observations/qpos"][()]
        dst.create_dataset("action", data=action)
        obs = dst.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        images_out = obs.create_group("images")

        num_frames = int(qpos.shape[0])
        for camera in CAMERAS:
            src_ds = src[f"observations/images/{camera}"]
            frames = [decode_resize_encode_png(read_image_bytes(src_ds, i)) for i in range(num_frames)]
            write_fixed_string_dataset(images_out, camera, frames)

    if instr_src.exists():
        shutil.copy2(instr_src, instr_dst)

    return episode_index(src_path), int(qpos.shape[0]), time.perf_counter() - start_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_files = sorted(args.src_dir.glob("episode_*/episode_*.hdf5"), key=episode_index)
    if args.end_index is None:
        args.end_index = len(src_files)
    selected = src_files[args.start_index : args.end_index]
    if not selected:
        raise ValueError("No episodes selected")

    print(f"source: {args.src_dir}")
    print(f"target: {args.dst_dir}")
    print(f"episodes: [{args.start_index}, {args.end_index}) count={len(selected)}")
    print(f"workers: {args.workers}")

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(convert_episode, p, args.src_dir, args.dst_dir, args.overwrite) for p in selected]
        for future in futures:
            idx, frames, seconds = future.result()
            print(f"wrote episode_{idx} frames={frames} seconds={seconds:.2f}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"done: {args.dst_dir}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"episodes_per_second={len(selected) / elapsed:.4f}")


if __name__ == "__main__":
    main()
