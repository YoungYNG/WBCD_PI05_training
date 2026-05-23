#!/usr/bin/env python3
"""Evaluate a pi05 EEF checkpoint on converted HDF5 episodes.

The script reads episodes produced by process_deformable_eef.sh, runs the
trained policy on each selected frame, and compares predicted EEF actions
against the HDF5 action targets.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import pathlib
import sys
from typing import Iterable

import cv2
import h5py
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi import transforms  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.shared import download  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "processed_data_eef" / "deformable_manipulation_eef"
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "pi05_base_eef_hdf5_lora" / "deformable_eef_hdf5_run" / "10000"
DEFAULT_OUT_DIR = ROOT / "eval_outputs" / "deformable_eef_hdf5_run_10000"


@dataclasses.dataclass(frozen=True)
class RenameActionToActions(transforms.DataTransformFn):
    """Policy.infer emits 'action'; normalization stats are keyed by 'actions'."""

    def __call__(self, data: dict) -> dict:
        if "action" in data and "actions" not in data:
            data = dict(data)
            data["actions"] = data.pop("action")
        return data


def _decode_image(payload) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG image from HDF5 payload")
    return image


def _iter_episode_paths(data_dir: pathlib.Path, max_episodes: int | None) -> list[pathlib.Path]:
    paths = sorted(data_dir.glob("episode_*/episode_*.hdf5"))
    if not paths:
        paths = sorted(data_dir.glob("episode*.hdf5"))
    if not paths:
        raise ValueError(f"No HDF5 episodes found under {data_dir}")
    if max_episodes is not None:
        paths = paths[:max_episodes]
    return paths


def _sample_episode_paths(
    data_dir: pathlib.Path,
    *,
    max_episodes: int | None,
    episode_sample_count: int | None,
    seed: int,
) -> list[pathlib.Path]:
    paths = _iter_episode_paths(data_dir, max_episodes=None)
    if episode_sample_count is not None and episode_sample_count < len(paths):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(paths), size=episode_sample_count, replace=False)
        paths = [paths[i] for i in sorted(selected.tolist())]
    if max_episodes is not None:
        paths = paths[:max_episodes]
    return paths


def _episode_indices(length: int, stride: int, max_frames_per_episode: int | None) -> list[int]:
    indices = list(range(0, length, stride))
    if max_frames_per_episode is not None:
        indices = indices[:max_frames_per_episode]
    return indices


def _uniform_episode_indices(length: int, count: int, compare_horizon: int) -> list[int]:
    if count <= 0:
        return []
    max_start = max(0, length - max(1, compare_horizon))
    if count == 1:
        return [0]
    return np.linspace(0, max_start, num=count, dtype=np.int64).tolist()


def _target_actions(action_ds, frame_index: int, length: int, horizon: int) -> np.ndarray:
    action_end = min(frame_index + horizon, length)
    actions = action_ds[frame_index:action_end].astype(np.float32)
    if actions.shape[0] < horizon:
        pad = np.repeat(actions[-1:], horizon - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return actions


def _make_obs(ep: h5py.File, frame_index: int, target_seq: np.ndarray, prompt: str) -> dict:
    return {
        "images": {
            "cam_high": _decode_image(ep["/observations/images/cam_high"][frame_index]),
            "cam_left_wrist": _decode_image(ep["/observations/images/cam_left_wrist"][frame_index]),
            "cam_right_wrist": _decode_image(ep["/observations/images/cam_right_wrist"][frame_index]),
        },
        "state": ep["/observations/qpos"][frame_index].astype(np.float32),
        "actions": target_seq,
        "prompt": prompt,
    }


def _load_policy(
    train_config_name: str,
    checkpoint_dir: pathlib.Path,
    asset_id: str | None,
    num_steps: int,
    pytorch_device: str | None,
) -> tuple[_policy.Policy, _config.TrainConfig]:
    train_config = _config.get_config(train_config_name)
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    asset_id = asset_id or data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is required to load normalization stats")

    weight_path = checkpoint_dir / "model.safetensors"
    is_pytorch = weight_path.exists()
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", asset_id)
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    policy = _policy.Policy(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            RenameActionToActions(),
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
    return policy, train_config


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum(np.square(target - np.mean(target))))
    r2 = float("nan") if denom == 0.0 else float(1.0 - np.sum(np.square(err)) / denom)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def _dimension_metrics(pred: np.ndarray, target: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for dim in range(pred.shape[-1]):
        row = {"dim": dim}
        row.update(_metrics(pred[..., dim], target[..., dim]))
        rows.append(row)
    return rows


def _write_csv(path: pathlib.Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    policy, train_config = _load_policy(
        args.train_config,
        args.checkpoint_dir,
        args.asset_id,
        args.num_steps,
        args.pytorch_device,
    )
    horizon = int(train_config.model.action_horizon)
    action_dim = int(args.action_dim)
    compare_horizon = int(args.compare_horizon or horizon)
    if compare_horizon <= 0:
        raise ValueError("--compare-horizon must be positive")
    if compare_horizon > horizon:
        raise ValueError(f"--compare-horizon ({compare_horizon}) cannot exceed model action_horizon ({horizon})")

    data_dirs = args.data_dirs or [args.data_dir]
    sampled_paths: list[tuple[str, pathlib.Path]] = []
    for data_dir_index, data_dir in enumerate(data_dirs):
        paths = _sample_episode_paths(
            data_dir,
            max_episodes=args.max_episodes,
            episode_sample_count=args.episode_sample_count,
            seed=args.episode_sample_seed + data_dir_index,
        )
        sampled_paths.extend((str(data_dir), path) for path in paths)

    all_pred = []
    all_target = []
    per_episode_rows = []
    per_dataset_values: dict[str, dict[str, list[np.ndarray] | int]] = {}
    prediction_rows = []
    frame_count = 0

    for episode_num, (dataset_id, path) in enumerate(sampled_paths):
        ep_pred = []
        ep_target = []
        with h5py.File(path, "r") as ep:
            length = int(ep["/observations/qpos"].shape[0])
            if args.uniform_frames_per_episode is not None:
                indices = _uniform_episode_indices(length, args.uniform_frames_per_episode, compare_horizon)
            else:
                indices = _episode_indices(length, args.stride, args.max_frames_per_episode)
            for frame_index in indices:
                target_seq = _target_actions(ep["/action"], frame_index, length, horizon)
                obs = _make_obs(ep, frame_index, target_seq, args.prompt)
                out = policy.infer(obs)
                pred_seq = np.asarray(out["actions"], dtype=np.float32)[..., :action_dim]
                target_cmp = target_seq[..., :action_dim]
                if args.first_action_only:
                    pred_cmp = pred_seq[0]
                    target_cmp = target_cmp[0]
                else:
                    pred_cmp = pred_seq[:compare_horizon]
                    target_cmp = target_cmp[:compare_horizon]

                ep_pred.append(pred_cmp)
                ep_target.append(target_cmp)
                prediction_rows.append(
                    {
                        "dataset": dataset_id,
                        "episode_path": str(path),
                        "episode_num": episode_num,
                        "frame_index": frame_index,
                        "compare_horizon": 1 if args.first_action_only else compare_horizon,
                        "pred_first": json.dumps(np.asarray(pred_seq[0]).tolist()),
                        "target_first": json.dumps(np.asarray(target_seq[0, :action_dim]).tolist()),
                    }
                )
                frame_count += 1
                if args.max_total_frames is not None and frame_count >= args.max_total_frames:
                    break

        if ep_pred:
            ep_pred_arr = np.asarray(ep_pred, dtype=np.float32)
            ep_target_arr = np.asarray(ep_target, dtype=np.float32)
            row = {
                "dataset": dataset_id,
                "episode_path": str(path),
                "episode_num": episode_num,
                "num_eval_frames": len(ep_pred),
            }
            row.update(_metrics(ep_pred_arr.reshape(-1), ep_target_arr.reshape(-1)))
            per_episode_rows.append(row)
            all_pred.append(ep_pred_arr)
            all_target.append(ep_target_arr)
            if dataset_id not in per_dataset_values:
                per_dataset_values[dataset_id] = {"pred": [], "target": [], "num_eval_frames": 0, "num_episodes": 0}
            per_dataset_values[dataset_id]["pred"].append(ep_pred_arr)
            per_dataset_values[dataset_id]["target"].append(ep_target_arr)
            per_dataset_values[dataset_id]["num_eval_frames"] += len(ep_pred)
            per_dataset_values[dataset_id]["num_episodes"] += 1

        logging.info("evaluated %s frames from %s", len(ep_pred), path)
        if args.max_total_frames is not None and frame_count >= args.max_total_frames:
            break

    if not all_pred:
        raise ValueError("No frames were evaluated")

    pred = np.concatenate([x.reshape(-1, action_dim) for x in all_pred], axis=0)
    target = np.concatenate([x.reshape(-1, action_dim) for x in all_target], axis=0)

    summary = {
        "data_dir": str(args.data_dir),
        "data_dirs": [str(p) for p in data_dirs],
        "checkpoint_dir": str(args.checkpoint_dir),
        "train_config": args.train_config,
        "asset_id": args.asset_id,
        "num_episodes_seen": len(per_episode_rows),
        "num_eval_frames": frame_count,
        "num_compared_action_vectors": int(pred.shape[0]),
        "action_dim": action_dim,
        "first_action_only": bool(args.first_action_only),
        "compare_horizon": 1 if args.first_action_only else compare_horizon,
        "stride": args.stride,
        "uniform_frames_per_episode": args.uniform_frames_per_episode,
        "episode_sample_count": args.episode_sample_count,
        "episode_sample_seed": args.episode_sample_seed,
        "num_steps": args.num_steps,
    }
    summary.update(_metrics(pred.reshape(-1), target.reshape(-1)))
    summary["per_dim"] = _dimension_metrics(pred, target)
    summary["per_dataset"] = {}
    for dataset_id, values in per_dataset_values.items():
        dataset_pred = np.concatenate([x.reshape(-1, action_dim) for x in values["pred"]], axis=0)
        dataset_target = np.concatenate([x.reshape(-1, action_dim) for x in values["target"]], axis=0)
        dataset_summary = {
            "num_episodes": values["num_episodes"],
            "num_eval_frames": values["num_eval_frames"],
            "num_compared_action_vectors": int(dataset_pred.shape[0]),
        }
        dataset_summary.update(_metrics(dataset_pred.reshape(-1), dataset_target.reshape(-1)))
        dataset_summary["per_dim"] = _dimension_metrics(dataset_pred, dataset_target)
        summary["per_dataset"][dataset_id] = dataset_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "per_episode_metrics.csv", per_episode_rows)
    if args.save_predictions:
        _write_csv(args.output_dir / "predictions.csv", prediction_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--data-dirs",
        type=pathlib.Path,
        nargs="+",
        default=None,
        help="Optional list of HDF5 episode directories. If set, evaluates all of them in one run.",
    )
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--train-config", default="pi05_base_eef_hdf5_lora")
    parser.add_argument("--asset-id", default="deformable_eef_hdf5")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt", default="deformable manipulation <control_mode> end effector <control_mode>")
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion sampling steps.")
    parser.add_argument("--stride", type=int, default=10, help="Evaluate every Nth frame.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=20)
    parser.add_argument("--max-total-frames", type=int, default=None)
    parser.add_argument(
        "--episode-sample-count",
        type=int,
        default=None,
        help="Randomly sample this many episodes from each data directory before evaluation.",
    )
    parser.add_argument("--episode-sample-seed", type=int, default=0)
    parser.add_argument(
        "--uniform-frames-per-episode",
        type=int,
        default=None,
        help="Uniformly sample this many start frames per episode, e.g. 10 gives linspace over the episode.",
    )
    parser.add_argument(
        "--compare-horizon",
        type=int,
        default=None,
        help="When --no-first-action-only is used, compare only the first N actions of the predicted chunk.",
    )
    parser.add_argument("--first-action-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--pytorch-device", default=None)
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps({k: v for k, v in summary.items() if k != "per_dim"}, indent=2))


if __name__ == "__main__":
    main()
