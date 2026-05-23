#!/usr/bin/env python3
"""Offline qpos action evaluation on the LeRobot demo_clean_repo dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtc_policy import QposInferenceConfig, RTCConfig, RTCPI0  # noqa: E402


DEFAULT_DATASET_DIR = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", ROOT.parent / ".cache" / "lerobot")) / "demo_clean_repo"
DEFAULT_PROMPT = (
    "Pick up the two corners of the white garment on the table, "
    "place it over the clothing board on the blue rack, and smooth it flat."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PI05 qpos checkpoint on local LeRobot parquet data.")
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train-config-name", default="pi05_base_aloha_lora")
    parser.add_argument("--model-name", default="demo_clean")
    parser.add_argument("--checkpoint-id", default="30000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--pi0-step", type=int, default=50, help="Number of predicted actions to compare from each chunk.")
    parser.add_argument("--stride", type=int, default=100, help="Frame stride within each episode.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=20)
    parser.add_argument("--max-total-frames", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "pi05_rtc_eval_qpos" / "offline_eval_outputs")
    return parser.parse_args()


def _decode_lerobot_image(value) -> np.ndarray:
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is None:
            raise ValueError(f"LeRobot image dict has no bytes field: {value.keys()}")
        value = payload
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Expected image bytes or dict, got {type(value)!r}")
    with Image.open(io.BytesIO(value)) as img:
        return np.asarray(img.convert("RGB"))


def _episode_paths(dataset_dir: pathlib.Path, max_episodes: int | None) -> list[pathlib.Path]:
    pattern = dataset_dir / "data" / "chunk-*" / "episode_*.parquet"
    paths = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet episodes found with pattern {pattern}")
    return paths[:max_episodes] if max_episodes is not None else paths


def _target_actions(df: pd.DataFrame, frame_index: int, horizon: int) -> np.ndarray:
    end = min(frame_index + horizon, len(df))
    actions = np.stack(df.iloc[frame_index:end]["action"].to_numpy()).astype(np.float32)
    if len(actions) < horizon:
        actions = np.concatenate([actions, np.repeat(actions[-1:], horizon - len(actions), axis=0)], axis=0)
    return actions


def _make_obs(row: pd.Series, prompt: str) -> dict:
    return {
        "images": {
            "cam_high": _decode_lerobot_image(row["observation.images.cam_high"]),
            "cam_left_wrist": _decode_lerobot_image(row["observation.images.cam_left_wrist"]),
            "cam_right_wrist": _decode_lerobot_image(row["observation.images.cam_right_wrist"]),
        },
        "state": np.asarray(row["observation.state"], dtype=np.float32),
        "prompt": prompt,
    }


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum(np.square(target - np.mean(target))))
    r2 = float("nan") if denom == 0.0 else float(1.0 - np.sum(np.square(err)) / denom)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def _per_dim_metrics(pred: np.ndarray, target: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for dim in range(pred.shape[-1]):
        row = {"dim": dim}
        row.update(_metrics(pred[..., dim], target[..., dim]))
        rows.append(row)
    return rows


def _latency_stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "latency_ms_mean": float(np.mean(arr)),
        "latency_ms_std": float(np.std(arr)),
        "latency_ms_min": float(np.min(arr)),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p90": float(np.percentile(arr, 90)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "latency_ms_max": float(np.max(arr)),
    }


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    model = RTCPI0(
        args.train_config_name,
        args.model_name,
        args.checkpoint_id,
        args.pi0_step,
        RTCConfig(enabled=False, refill_threshold=0),
        QposInferenceConfig(prompt=args.prompt),
    )
    model.set_language(args.prompt)

    all_pred = []
    all_target = []
    per_frame_rows = []
    latency_ms = []
    total_frames = 0

    for episode_num, path in enumerate(_episode_paths(args.dataset_dir, args.max_episodes)):
        df = pd.read_parquet(path)
        frame_indices = list(range(0, len(df), args.stride))
        if args.max_frames_per_episode is not None:
            frame_indices = frame_indices[: args.max_frames_per_episode]

        for frame_index in frame_indices:
            row = df.iloc[frame_index]
            obs = _make_obs(row, args.prompt)
            target = _target_actions(df, frame_index, args.pi0_step)

            input_rgb_arr = [obs["images"]["cam_high"], obs["images"]["cam_right_wrist"], obs["images"]["cam_left_wrist"]]
            model.update_observation_window(input_rgb_arr, obs["state"])

            # First call includes JIT/compilation overhead; record it separately by skipping warmups.
            for _ in range(args.warmup_runs if total_frames == 0 else 0):
                _ = model.get_action()[: args.pi0_step]

            start = time.perf_counter()
            pred = np.asarray(model.get_action()[: args.pi0_step], dtype=np.float32)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            pred = pred[:, : target.shape[-1]]
            all_pred.append(pred)
            all_target.append(target)
            latency_ms.append(elapsed_ms)

            row_metrics = {
                "episode_path": str(path),
                "episode_num": episode_num,
                "frame_index": frame_index,
                "latency_ms": elapsed_ms,
                "compare_horizon": args.pi0_step,
            }
            row_metrics.update(_metrics(pred.reshape(-1), target.reshape(-1)))
            per_frame_rows.append(row_metrics)

            total_frames += 1
            if args.max_total_frames is not None and total_frames >= args.max_total_frames:
                break

        if args.max_total_frames is not None and total_frames >= args.max_total_frames:
            break

    if not all_pred:
        raise ValueError("No frames evaluated")

    pred_all = np.concatenate(all_pred, axis=0)
    target_all = np.concatenate(all_target, axis=0)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "train_config_name": args.train_config_name,
        "model_name": args.model_name,
        "checkpoint_id": args.checkpoint_id,
        "prompt": args.prompt,
        "num_eval_frames": total_frames,
        "num_compared_action_vectors": int(pred_all.shape[0]),
        "action_dim": int(pred_all.shape[-1]),
        "predicted_actions_per_inference": args.pi0_step,
        "stride": args.stride,
        "warmup_runs": args.warmup_runs,
    }
    summary.update(_metrics(pred_all.reshape(-1), target_all.reshape(-1)))
    summary.update(_latency_stats(latency_ms))
    summary["per_dim"] = _per_dim_metrics(pred_all, target_all)

    out_dir = args.output_dir / f"{args.train_config_name}_{args.model_name}_{args.checkpoint_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "per_frame_metrics.csv", per_frame_rows)
    _write_csv(out_dir / "per_dim_metrics.csv", summary["per_dim"])

    print(json.dumps({k: v for k, v in summary.items() if k != "per_dim"}, indent=2))
    print(f"Saved offline eval outputs to {out_dir}")
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
