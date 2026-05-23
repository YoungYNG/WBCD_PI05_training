import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
PI05_DIR = ROOT_DIR / "policy" / "pi05"
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "policy"))
sys.path.append(str(ROOT_DIR / "description" / "utils"))
sys.path.append(str(PI05_DIR))
sys.path.append(str(Path(__file__).resolve().parent))

from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from script.eval_policy import class_decorator, get_camera_config, get_embodiment_config  # noqa: E402
from rtc_policy import QposInferenceConfig, RTCConfig, RTCAttentionSchedule, RTCPI0  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenPI PI05 on RoboTwin with RTC-style action chunking.")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--train_config_name", default="pi05_base_aloha_lora")
    parser.add_argument("--model_name", default="demo_clean")
    parser.add_argument("--checkpoint_id", default="040000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--instruction_type", default="unseen")
    parser.add_argument("--pi0_step", type=int, default=50)
    parser.add_argument("--rtc_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc_execution_horizon", type=int, default=10)
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--rtc_prefix_attention_schedule", choices=[x.value for x in RTCAttentionSchedule], default="exp")
    parser.add_argument("--rtc_inference_delay", type=int, default=0)
    parser.add_argument("--rtc_refill_threshold", type=int, default=10)
    parser.add_argument(
        "--prompt",
        default=(
            "Pick up the two corners of the white garment on the table, "
            "place it over the clothing board on the blue rack, and smooth it flat."
        ),
        help="Prompt from demo_clean_repo/meta/tasks.jsonl used during training.",
    )
    return parser.parse_args()


def build_task_args(cli_args):
    with open(ROOT_DIR / "task_config" / f"{cli_args.task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = cli_args.task_name
    args["task_config"] = cli_args.task_config
    args["ckpt_setting"] = f"{cli_args.model_name}_{cli_args.checkpoint_id}_rtc"

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment):
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["policy_name"] = "pi05_rtc"
    args["eval_mode"] = True
    return args


def prepare_video(args, save_dir):
    if not args["eval_video_log"]:
        return None, None

    camera_config = get_camera_config(args["camera"]["head_camera_type"])
    video_size = f"{camera_config['w']}x{camera_config['h']}"
    args["eval_video_save_dir"] = save_dir
    return save_dir, video_size


def create_ffmpeg(task_env, video_size):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
        ],
        stdin=subprocess.PIPE,
    )


def evaluate(cli_args):
    os.chdir(ROOT_DIR)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_args = build_task_args(cli_args)

    save_dir = ROOT_DIR / "eval_result" / cli_args.task_name / "pi05_rtc" / cli_args.task_config / task_args["ckpt_setting"] / current_time
    save_dir.mkdir(parents=True, exist_ok=True)
    _, video_size = prepare_video(task_args, save_dir)

    rtc_config = RTCConfig(
        enabled=cli_args.rtc_enabled,
        execution_horizon=cli_args.rtc_execution_horizon,
        max_guidance_weight=cli_args.rtc_max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule(cli_args.rtc_prefix_attention_schedule),
        inference_delay=cli_args.rtc_inference_delay,
        refill_threshold=cli_args.rtc_refill_threshold,
    )

    model = RTCPI0(
        train_config_name=cli_args.train_config_name,
        model_name=cli_args.model_name,
        checkpoint_id=cli_args.checkpoint_id,
        pi0_step=cli_args.pi0_step,
        rtc_config=rtc_config,
        qpos_config=QposInferenceConfig(prompt=cli_args.prompt),
    )

    task_env = class_decorator(cli_args.task_name)
    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = 100000 * (1 + cli_args.seed)
    clear_cache_freq = task_args["clear_cache_freq"]

    print(f"\033[34mTask Name: {cli_args.task_name}\033[0m")
    print("\033[34mPolicy Name: pi05_rtc\033[0m")
    print(f"\033[34mCheckpoint: {cli_args.train_config_name}/{cli_args.model_name}/{cli_args.checkpoint_id}\033[0m")
    print(f"\033[34mRTC Config: {rtc_config}\033[0m")

    while succ_seed < cli_args.test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue
        except Exception as exc:
            print(" -------------")
            print("Error: ", exc)
            print(traceback.format_exc())
            print(" -------------")
            task_env.close_env()
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        if task_env.plan_success and task_env.check_success():
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq
        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
        # demo_clean_repo has one task prompt in meta/tasks.jsonl. Keep eval
        # prompt fixed to the same text instead of sampling task descriptions.
        task_env.set_instruction(instruction=cli_args.prompt)

        if task_env.eval_video_path is not None:
            task_env._set_eval_video_ffmpeg(create_ffmpeg(task_env, video_size))

        succ = False
        model.reset_obsrvationwindows()
        while task_env.take_action_cnt < task_env.step_lim:
            observation = task_env.get_obs()
            action = model.next_action(task_env, observation)
            if action is None:
                break

            task_env.take_action(action, action_type="qpos")
            if task_env.eval_success:
                succ = True
                break

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()

        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()
        task_env.test_num += 1

        print(
            f"\033[93m{cli_args.task_name}\033[0m | \033[94mpi05_rtc\033[0m | "
            f"\033[92m{cli_args.task_config}\033[0m | \033[91m{task_args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
            f"\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m, last infer: {model.last_infer_ms:.1f} ms\n"
        )
        now_seed += 1

    result_path = save_dir / "_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_time}\n\n")
        f.write(f"Prompt: {cli_args.prompt}\n\n")
        f.write("State Layout: left_6_joints,left_gripper,right_6_joints,right_gripper\n\n")
        f.write("Camera Layout: cam_high,cam_left_wrist,cam_right_wrist all use real RGB\n\n")
        f.write(f"Checkpoint: {cli_args.train_config_name}/{cli_args.model_name}/{cli_args.checkpoint_id}\n\n")
        f.write(f"RTC Config: {rtc_config}\n\n")
        f.write(str(task_env.suc / cli_args.test_num))
    print(f"Data has been saved to {result_path}")


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    evaluate(parse_args())
