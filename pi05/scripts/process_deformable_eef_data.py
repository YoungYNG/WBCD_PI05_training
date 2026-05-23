import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


LEFT_HAND_PREFIX = "left_hand_"
RIGHT_HAND_PREFIX = "right_hand_"
DEFAULT_INSTRUCTION = "deformable manipulation"


def _read_numeric_table(path: Path, expected_cols: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns in {path}, got {data.shape[1]}")
    return data


def _read_timestamps(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "header_stamp" in reader.fieldnames:
            for row in reader:
                try:
                    values.append(float(row["header_stamp"]))
                except (TypeError, ValueError):
                    continue
        else:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                numeric = []
                for part in parts:
                    try:
                        numeric.append(float(part))
                    except ValueError:
                        pass
                if not numeric:
                    continue
                values.append(numeric[-1])
    if not values:
        raise ValueError(f"No numeric timestamps found in {path}")
    return np.asarray(values, dtype=np.float64)


def _interp_columns(table: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    src_ts = table[:, 0]
    values = table[:, 1:]
    out = np.empty((query_ts.shape[0], values.shape[1]), dtype=np.float32)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(query_ts, src_ts, values[:, i]).astype(np.float32)
    return out


def _find_child_dir(session_dir: Path, prefix: str) -> Path:
    matches = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {prefix} directory in {session_dir}, got {len(matches)}")
    return matches[0]


def _iter_session_dirs(raw_dir: Path) -> list[Path]:
    sessions = []
    for path in raw_dir.rglob("session_*"):
        if not path.is_dir():
            continue
        try:
            _find_child_dir(path, LEFT_HAND_PREFIX)
            _find_child_dir(path, RIGHT_HAND_PREFIX)
        except ValueError:
            continue
        sessions.append(path)
    return sorted(sessions)


def _load_hand_stream(hand_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    trajectory = _read_numeric_table(hand_dir / "Merged_Trajectory" / "merged_trajectory.txt", 8)
    gripper = _read_numeric_table(hand_dir / "Clamp_Data" / "clamp_data_tum.txt", 2)
    timestamps = _read_timestamps(hand_dir / "RGB_Images" / "timestamps.csv")
    video_path = hand_dir / "RGB_Images" / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    return trajectory, gripper, timestamps, video_path


def _encode_video_frames(video_path: Path, frame_count: int, resize: tuple[int, int]) -> list[bytes]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    try:
        for _ in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize)
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError(f"Could not JPEG-encode frame from {video_path}")
            frames.append(encoded.tobytes())
    finally:
        cap.release()
    return frames


def _fixed_width_bytes(encoded_frames: list[bytes]) -> tuple[list[bytes], int]:
    if not encoded_frames:
        raise ValueError("Cannot store an empty image stream")
    max_len = max(len(frame) for frame in encoded_frames)
    return [frame.ljust(max_len, b"\0") for frame in encoded_frames], max_len


def _make_mask_stream(frame_count: int, resize: tuple[int, int], value: int) -> list[bytes]:
    width, height = resize
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not JPEG-encode mask frame")
    payload = encoded.tobytes()
    return [payload] * frame_count


def _write_image_dataset(group: h5py.Group, name: str, encoded_frames: list[bytes]) -> None:
    padded, max_len = _fixed_width_bytes(encoded_frames)
    group.create_dataset(name, data=padded, dtype=f"S{max_len}")


def _build_episode(session_dir: Path, output_path: Path, instruction: str, image_size: tuple[int, int], mask_value: int) -> int:
    left_dir = _find_child_dir(session_dir, LEFT_HAND_PREFIX)
    right_dir = _find_child_dir(session_dir, RIGHT_HAND_PREFIX)

    left_traj, left_gripper, left_img_ts, left_video = _load_hand_stream(left_dir)
    right_traj, right_gripper, right_img_ts, right_video = _load_hand_stream(right_dir)

    frame_count = min(len(left_img_ts), len(right_img_ts))
    if frame_count < 2:
        raise ValueError(f"Session {session_dir} has fewer than two aligned image timestamps")

    timeline = left_img_ts[:frame_count]
    min_ts = max(left_traj[0, 0], right_traj[0, 0], left_gripper[0, 0], right_gripper[0, 0])
    max_ts = min(left_traj[-1, 0], right_traj[-1, 0], left_gripper[-1, 0], right_gripper[-1, 0])
    valid = (timeline >= min_ts) & (timeline <= max_ts)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.shape[0] < 2:
        raise ValueError(f"Session {session_dir} has fewer than two timestamps inside trajectory/gripper overlap")

    # Keep a contiguous prefix/slice so video frame indices remain simple and deterministic.
    start = int(valid_indices[0])
    stop = int(valid_indices[-1]) + 1
    timeline = timeline[start:stop]
    raw_frame_count = stop

    left_images_all = _encode_video_frames(left_video, raw_frame_count, image_size)
    right_images_all = _encode_video_frames(right_video, raw_frame_count, image_size)
    usable_count = min(len(left_images_all) - start, len(right_images_all) - start, len(timeline))
    if usable_count < 2:
        raise ValueError(f"Session {session_dir} has fewer than two decoded frames")

    timeline = timeline[:usable_count]
    left_images = left_images_all[start : start + usable_count]
    right_images = right_images_all[start : start + usable_count]

    left_pose = _interp_columns(left_traj, timeline)
    right_pose = _interp_columns(right_traj, timeline)
    left_grip = _interp_columns(left_gripper, timeline)[:, 0:1]
    right_grip = _interp_columns(right_gripper, timeline)[:, 0:1]

    full_state = np.concatenate([left_pose, left_grip, right_pose, right_grip], axis=1).astype(np.float32)
    qpos = full_state[:-1]
    action = full_state[1:]
    image_count = qpos.shape[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["control_mode"] = "end effector"
        f.attrs["source_session"] = str(session_dir)
        f.attrs["state_layout"] = "left_xyzquat,left_gripper,right_xyzquat,right_gripper"
        f.create_dataset("action", data=action)
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        images = obs.create_group("images")
        _write_image_dataset(images, "cam_high", _make_mask_stream(image_count, image_size, mask_value))
        _write_image_dataset(images, "cam_left_wrist", left_images[:image_count])
        _write_image_dataset(images, "cam_right_wrist", right_images[:image_count])

    instruction_path = output_path.parent / "instructions.json"
    with instruction_path.open("w", encoding="utf-8") as f:
        json.dump({"instructions": [instruction]}, f, indent=2)

    return image_count


def _episode_index(path: Path) -> int:
    try:
        return int(path.stem.replace("episode_", "").replace("episode", ""))
    except ValueError:
        return -1


def _is_complete_episode(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return (
                "/observations/qpos" in f
                and "/action" in f
                and "/observations/images/cam_high" in f
                and "/observations/images/cam_left_wrist" in f
                and "/observations/images/cam_right_wrist" in f
                and int(f["/observations/qpos"].shape[0]) > 0
                and int(f["/action"].shape[0]) > 0
            )
    except Exception:
        return False


def _resume_count(output_dir: Path) -> int:
    paths = sorted(output_dir.glob("episode_*/episode_*.hdf5"), key=_episode_index)
    count = 0
    for path in paths:
        expected = output_dir / f"episode_{count}" / f"episode_{count}.hdf5"
        if path != expected or not _is_complete_episode(path):
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Deformable Manipulation raw data to EEF hdf5 episodes.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/Deformable Manipulation"),
        help="Root directory of the Deformable Manipulation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data_eef/deformable_manipulation_eef"),
        help="Output directory containing episode_*/episode_*.hdf5.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap on processed sessions.")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mask-value", type=int, default=0, choices=range(0, 256))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip an existing contiguous prefix of complete output episodes and continue from the next session.",
    )
    args = parser.parse_args()

    sessions = _iter_session_dirs(args.raw_dir)
    if not sessions:
        raise ValueError(f"No valid sessions found under {args.raw_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = _resume_count(args.output_dir) if args.resume else 0
    if args.resume and written:
        print(f"resume enabled: found {written} complete existing episodes, continuing from session index {written}")

    for session in sessions[written:]:
        ep_dir = args.output_dir / f"episode_{written}"
        ep_path = ep_dir / f"episode_{written}.hdf5"
        try:
            frame_count = _build_episode(
                session,
                ep_path,
                args.instruction,
                image_size=(args.width, args.height),
                mask_value=args.mask_value,
            )
        except Exception as exc:
            print(f"skip {session}: {exc}")
            continue
        print(f"wrote {ep_path} with {frame_count} frames from {session}")
        written += 1
        if args.max_episodes is not None and written >= args.max_episodes:
            break

    print(f"processed {written} / {len(sessions)} sessions")


if __name__ == "__main__":
    main()
