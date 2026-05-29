#!/usr/bin/env python3
"""Resize images inside original RoboTwin HDF5 episodes to 224x224.

Input layout:
    <src-dir>/episode_0.hdf5
    <src-dir>/episode_1.hdf5

This keeps the original RoboTwin HDF5 structure intact and only replaces:
    /data/demo_0/observations/images/head
    /data/demo_0/observations/images/head_lengths
    /data/demo_0/observations/images/left_wrist
    /data/demo_0/observations/images/left_wrist_lengths
    /data/demo_0/observations/images/right_wrist
    /data/demo_0/observations/images/right_wrist_lengths

All other groups, datasets, and attrs are copied unchanged. Frames are not
subsampled. The resize is an OpenCV equivalent of resize_with_pad: aspect-ratio
preserving resize with centered black padding to 224x224.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import cv2
import h5py
import numpy as np


CAMERAS = ("head", "left_wrist", "right_wrist")
IMAGE_GROUP = "data/demo_0/observations/images"


def episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse episode index from file name: {path.name}") from exc


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def should_skip(name: str) -> bool:
    for camera in CAMERAS:
        if name == f"{IMAGE_GROUP}/{camera}" or name == f"{IMAGE_GROUP}/{camera}_lengths":
            return True
    return False


def copy_tree_except_images(src_obj: h5py.Group | h5py.File, dst_obj: h5py.Group | h5py.File, prefix: str = "") -> None:
    copy_attrs(src_obj.attrs, dst_obj.attrs)
    for key, value in src_obj.items():
        name = f"{prefix}/{key}" if prefix else key
        if should_skip(name):
            continue
        if isinstance(value, h5py.Group):
            group = dst_obj.create_group(key)
            copy_tree_except_images(value, group, name)
        elif isinstance(value, h5py.Dataset):
            dataset = dst_obj.create_dataset(key, data=value[()], dtype=value.dtype)
            copy_attrs(value.attrs, dataset.attrs)
        else:
            raise TypeError(f"Unsupported HDF5 object at {name}: {type(value)}")


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


def read_encoded_frame(image_ds: h5py.Dataset, lengths: np.ndarray | None, idx: int) -> bytes:
    if lengths is not None:
        return bytes(image_ds[idx][: int(lengths[idx])])
    value = image_ds[idx]
    if isinstance(value, bytes):
        return value.rstrip(b"\0")
    return bytes(value).rstrip(b"\0")


def resize_encoded_frame(raw_bytes: bytes) -> bytes:
    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("cv2.imdecode failed")
    resized_bgr = resize_with_pad_opencv(image_bgr, 224, 224)
    ok, encoded_png = cv2.imencode(".png", resized_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("cv2.imencode PNG failed")
    return encoded_png.tobytes()


def resize_image_dataset(src_images: h5py.Group, dst_images: h5py.Group, camera: str) -> None:
    image_ds = src_images[camera]
    length_name = f"{camera}_lengths"
    lengths = src_images[length_name][()] if length_name in src_images else None

    resized_frames: list[bytes] = []
    if image_ds.ndim == 4:
        for frame in image_ds:
            resized = resize_with_pad_opencv(frame, 224, 224)
            ok, encoded_png = cv2.imencode(".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise ValueError("cv2.imencode PNG failed")
            resized_frames.append(encoded_png.tobytes())
    else:
        for idx in range(image_ds.shape[0]):
            resized_frames.append(resize_encoded_frame(read_encoded_frame(image_ds, lengths, idx)))

    max_len = max(len(frame) for frame in resized_frames)
    padded = np.zeros((len(resized_frames), max_len), dtype=np.uint8)
    out_lengths = np.empty((len(resized_frames),), dtype=np.int32)
    for idx, frame in enumerate(resized_frames):
        raw = np.frombuffer(frame, dtype=np.uint8)
        padded[idx, : raw.shape[0]] = raw
        out_lengths[idx] = raw.shape[0]

    dst = dst_images.create_dataset(camera, data=padded, dtype=np.uint8)
    copy_attrs(image_ds.attrs, dst.attrs)
    dst_len = dst_images.create_dataset(length_name, data=out_lengths, dtype=np.int32)
    if length_name in src_images:
        copy_attrs(src_images[length_name].attrs, dst_len.attrs)


def convert_episode(src_path: Path, src_root: Path, dst_root: Path, overwrite: bool) -> tuple[int, int, float]:
    started = time.perf_counter()
    dst_path = dst_root / src_path.relative_to(src_root)
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"{dst_path} exists; pass --overwrite to replace it")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_tree_except_images(src, dst)
        src_images = src[IMAGE_GROUP]
        dst_images = dst[IMAGE_GROUP]
        for camera in CAMERAS:
            resize_image_dataset(src_images, dst_images, camera)
        frame_count = int(src_images[CAMERAS[0]].shape[0])

    return episode_index(src_path), frame_count, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_files = sorted(args.src_dir.glob("episode_*.hdf5"), key=episode_index)
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
        futures = [executor.submit(convert_episode, src_path, args.src_dir, args.dst_dir, args.overwrite) for src_path in selected]
        for future in futures:
            idx, frames, seconds = future.result()
            print(f"wrote episode_{idx} frames={frames} seconds={seconds:.2f}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"done: {args.dst_dir}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"episodes_per_second={len(selected) / elapsed:.4f}")


if __name__ == "__main__":
    main()
