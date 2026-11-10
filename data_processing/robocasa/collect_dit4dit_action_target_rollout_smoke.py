#!/usr/bin/env python3
"""Collect one DiT4DiT RoboCasa rollout with state-vs-action-target mesh previews.

This is a diagnostic/smoke script for the "intention mesh" question.  It runs a
DiT4DiT policy against a live RoboCasa GR1 environment and renders, at every
environment step:

    RGB rollout frame | current-state robot mesh | policy action-target mesh

The action-target mesh is rendered by direct qpos assignment for the policy's
absolute arm/waist joint targets.  It is not produced by ``env.step`` and does
not include scene contact resolution.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import h5py
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
DATA_PROCESSING_ROOT = CURRENT_DIR.parent
LOCAL_ROBOSUITE_ROOT = REPO_ROOT / "robosuite"
LOCAL_ROBOCASA_ROOT = REPO_ROOT / "robocasa-gr1-tabletop-tasks"
LOCAL_DIT4DIT_ROOT = REPO_ROOT / "third_party" / "DiT4DiT"

for package_root in (LOCAL_DIT4DIT_ROOT, LOCAL_ROBOSUITE_ROOT, LOCAL_ROBOCASA_ROOT, DATA_PROCESSING_ROOT):
    package_root_str = str(package_root)
    if package_root.exists() and package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

from video_io import VideoWriter  # noqa: E402
try:
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402
except ImportError:
    WebsocketClientPolicy = None
from dit4dit_rollout_rendering import (  # noqa: E402
    DROID_STYLE_DEPTH_FAR_M,
    DROID_STYLE_DEPTH_NEAR_M,
    collect_eef_geom_ids,
    collect_robot_geom_ids,
    disable_geom_collisions,
    get_camera_id,
    hide_ik_indicator_sites,
    import_runtime_modules,
    override_camera_pose,
    render_depth_colormap_frame,
    render_robot_only_outputs,
    reset_to,
    _camera_xmat_to_quat,
)
from dit4dit_action_helpers import qpos_limits_for_indices, update_env_after_qpos  # noqa: E402


DEFAULT_ENV_NAME = "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
DEFAULT_CKPT = (
    REPO_ROOT
    / "data"
    / "checkpoints"
    / "dit4dit-model"
    / "dit4dit_robocasa_gr1"
    / "final_model"
    / "pytorch_model.pt"
)
DEFAULT_OUT_DIR = CURRENT_DIR / "outputs" / "dit4dit_robocasa_gr1_door_close_rollout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--policy-ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6398)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--num-ddim-steps", type=int, default=10)
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera", default="egoview")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--clip-fps", type=float, default=16.0)
    parser.add_argument("--clip-frames", type=int, default=81)
    parser.add_argument("--source-span", type=int, default=101)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--background-color", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--robot-only-rgb-style", choices=("flat", "shaded"), default="shaded")
    parser.add_argument("--robot-only-flat-color", type=int, nargs=3, default=[255, 255, 255])
    parser.add_argument("--depth-near-m", type=float, default=DROID_STYLE_DEPTH_NEAR_M)
    parser.add_argument("--depth-far-m", type=float, default=DROID_STYLE_DEPTH_FAR_M)
    parser.add_argument("--robot-depth-near-m", type=float, default=DROID_STYLE_DEPTH_NEAR_M)
    parser.add_argument("--robot-depth-far-m", type=float, default=DROID_STYLE_DEPTH_FAR_M)
    parser.add_argument(
        "--static-mode",
        choices=("none", "first_frame", "camera_follow"),
        default="none",
        help=(
            "Static RGB/depth rendering mode. camera_follow freezes the scene at rollout start "
            "but follows the live camera trajectory."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("debug", "production"),
        default="debug",
        help=(
            "debug writes all diagnostic full-length videos. production keeps policy settings unchanged "
            "but writes only the clip81 modalities needed for training."
        ),
    )
    parser.add_argument(
        "--write-shadow-step-nominal",
        action="store_true",
        help=(
            "Also run a second contact-free shadow environment with the same actions and render "
            "the post-env.step robot-only nominal mesh. This keeps controller dynamics but removes "
            "robot-scene contact."
        ),
    )
    parser.add_argument("--mujoco-gl", default="egl", choices=("egl", "osmesa", "glfw", "unset"))
    parser.add_argument("--warmstart-hdf5", type=Path, default=None)
    parser.add_argument("--warmstart-demo", default="demo_906")
    parser.add_argument("--warmstart-state-index", type=int, default=236)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_mujoco_gl(mujoco_gl: str) -> None:
    if mujoco_gl == "unset":
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        return
    os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    if mujoco_gl == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def make_env(env_name: str, *, seed: int):
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401

    return gym.make(env_name, enable_render=True, seed=seed)


def low_env(env):
    return env.unwrapped.env


def observation_from_current_low_env(env) -> dict[str, Any]:
    sim_env = low_env(env)
    if sim_env.viewer_get_obs:
        raw_obs = sim_env.viewer._get_observations(force_update=True)
    else:
        raw_obs = sim_env._get_observations(force_update=True)
    basic_obs = env.unwrapped.get_basic_observation(raw_obs)
    return env.unwrapped.get_groot_observation(basic_obs)


def reset_rollout_start(env, args: argparse.Namespace, robosuite_module) -> dict[str, Any]:
    if args.warmstart_hdf5 is None:
        obs, _ = env.reset(seed=args.seed)
        return obs
    dataset_path = args.warmstart_hdf5.expanduser().resolve()
    # Gymnasium's OrderEnforcing wrapper requires reset() before any step().
    env.reset(seed=args.seed)
    sim_env = low_env(env)
    with h5py.File(dataset_path, "r") as handle:
        demo_group = handle["data"][args.warmstart_demo]
        state_index = int(args.warmstart_state_index)
        model_xml = hide_ik_indicator_sites(demo_group.attrs["model_file"])
        ep_meta = demo_group.attrs.get("ep_meta", None)
        reset_to(
            sim_env,
            {"states": demo_group["states"][state_index], "model": model_xml, "ep_meta": ep_meta},
            robosuite_module,
        )
    return observation_from_current_low_env(env)


def policy_image_from_obs(obs: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    video = obs.get("video.ego_view_bg_crop_pad_res256_freq20")
    if video is None:
        video = obs.get("video.ego_view_pad_res256_freq20")
    if video is None:
        raise KeyError("Missing GR1 ego video observation")
    return cv2.resize(np.asarray(video, dtype=np.uint8), tuple(image_size), interpolation=cv2.INTER_AREA)


def policy_state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    chunks = []
    for key in ("state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"):
        value = np.asarray(obs[key], dtype=np.float32)[None, :]
        chunks.append(np.concatenate([np.sin(value), np.cos(value)], axis=-1))
    state = np.concatenate(chunks, axis=-1)
    return np.pad(state, ((0, 0), (0, 64 - state.shape[1])), mode="constant").astype(np.float32)


def load_action_stats(policy_ckpt_path: Path) -> dict[str, Any]:
    stats_path = policy_ckpt_path.expanduser().resolve().parents[1] / "dataset_statistics.json"
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if list(stats.keys()) != ["gr1"]:
        raise ValueError(f"Expected single gr1 stats entry in {stats_path}, got {list(stats.keys())}")
    return stats["gr1"]["action"]


def unnormalize_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    mask = np.asarray(action_stats.get("mask", np.ones_like(action_stats["min"], dtype=bool)), dtype=bool)
    action_high = np.asarray(action_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_stats["min"], dtype=np.float32)
    normalized_actions = np.clip(normalized_actions[..., :29], -1.0, 1.0)
    return np.where(
        mask,
        (normalized_actions + 1.0) / 2.0 * (action_high - action_low) + action_low,
        normalized_actions,
    ).astype(np.float32, copy=False)


class Dit4DitPolicyClient:
    def __init__(self, *, host: str, port: int, policy_ckpt_path: Path, image_size: tuple[int, int], n_action_steps: int, num_ddim_steps: int):
        if WebsocketClientPolicy is None:
            raise RuntimeError("DiT4DiT deployment client is unavailable; install the vendored DiT4DiT package.")
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.action_stats = load_action_stats(policy_ckpt_path)
        self.image_size = image_size
        self.n_action_steps = int(n_action_steps)
        self.num_ddim_steps = int(num_ddim_steps)

    def close(self) -> None:
        self.client.close()

    def step(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        response = self.client.predict_action(
            {
                "examples": [
                    {
                        "image": [policy_image_from_obs(obs, self.image_size)],
                        "lang": str(obs["annotation.human.coarse_action"]),
                        "state": policy_state_from_obs(obs),
                    }
                ],
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": self.num_ddim_steps,
            }
        )
        normalized = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
        raw = unnormalize_actions(normalized, self.action_stats)[0, : self.n_action_steps]
        return {
            "action.left_arm": raw[:, :7],
            "action.right_arm": raw[:, 7:14],
            "action.left_hand": raw[:, 14:20],
            "action.right_hand": raw[:, 20:26],
            "action.waist": raw[:, 26:29],
        }


def unbatch_action_chunk(action: dict[str, Any]) -> dict[str, np.ndarray]:
    chunk: dict[str, np.ndarray] = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for {key}, got {arr.shape}")
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected [T,D] action chunk for {key}, got {arr.shape}")
        chunk[key] = arr
    return chunk


def action_at(chunk: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[index], dtype=np.float32).copy() for key, value in chunk.items()}


def gripper_limits(env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    controller = env.robots[0].composite_controller
    limits = {}
    for part_name in ("left_gripper", "right_gripper"):
        qpos_indices = np.asarray(controller.part_controllers[part_name].qpos_index, dtype=np.int64)
        limits[part_name] = qpos_limits_for_indices(env, qpos_indices)
    return limits


def apply_action_target_qpos(
    env,
    action: dict[str, np.ndarray],
    *,
    dt: float,
    limits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    controller = env.robots[0].composite_controller
    qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()

    qpos[np.asarray(controller.part_controllers["left"].qpos_index, dtype=np.int64)] = np.asarray(
        action["action.left_arm"], dtype=np.float64
    )
    qpos[np.asarray(controller.part_controllers["right"].qpos_index, dtype=np.int64)] = np.asarray(
        action["action.right_arm"], dtype=np.float64
    )
    qpos[np.asarray(controller.part_controllers["torso"].qpos_index, dtype=np.int64)] = np.asarray(
        action["action.waist"], dtype=np.float64
    )

    for part_name, action_key in (("left_gripper", "action.left_hand"), ("right_gripper", "action.right_hand")):
        gripper = controller.grippers[part_name]
        part_controller = controller.part_controllers[part_name]
        qpos_indices = np.asarray(part_controller.qpos_index, dtype=np.int64)
        formatted_qvel = np.asarray(gripper.format_action(action[action_key]), dtype=np.float64)
        low, high = limits[part_name]
        qpos[qpos_indices] = np.clip(qpos[qpos_indices] + formatted_qvel * dt, low, high)

    env.sim.data.qpos[:] = qpos
    env.sim.data.qvel[:] = 0
    update_env_after_qpos(env)


def render_action_target_frame(
    env,
    action: dict[str, np.ndarray],
    *,
    camera_name: str,
    width: int,
    height: int,
    robot_geom_ids: np.ndarray,
    background_rgb: np.ndarray,
    flat_robot_rgb: np.ndarray,
    rgb_style: str,
    dt: float,
    limits: dict[str, tuple[np.ndarray, np.ndarray]],
    robosuite_camera_utils_module,
) -> np.ndarray:
    qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
    try:
        apply_action_target_qpos(env, action, dt=dt, limits=limits)
        mesh_frame, _ = render_robot_only_outputs(
            env,
            camera_name=camera_name,
            width=width,
            height=height,
            robot_geom_ids=robot_geom_ids,
            background_rgb=background_rgb,
            rgb_style=rgb_style,
            flat_robot_rgb=flat_robot_rgb,
            near_m=None,
            far_m=None,
            robosuite_camera_utils_module=robosuite_camera_utils_module,
        )
        return mesh_frame
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        update_env_after_qpos(env)


def render_action_target_eef_depth(
    env,
    action: dict[str, np.ndarray],
    *,
    camera_name: str,
    width: int,
    height: int,
    eef_geom_ids: np.ndarray,
    background_rgb: np.ndarray,
    flat_robot_rgb: np.ndarray,
    rgb_style: str,
    dt: float,
    limits: dict[str, tuple[np.ndarray, np.ndarray]],
    near_m: float,
    far_m: float,
    robosuite_camera_utils_module,
) -> np.ndarray:
    qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
    try:
        apply_action_target_qpos(env, action, dt=dt, limits=limits)
        _, depth_frame = render_robot_only_outputs(
            env,
            camera_name=camera_name,
            width=width,
            height=height,
            robot_geom_ids=eef_geom_ids,
            background_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            rgb_style=rgb_style,
            flat_robot_rgb=flat_robot_rgb,
            near_m=near_m,
            far_m=far_m,
            robosuite_camera_utils_module=robosuite_camera_utils_module,
        )
        if depth_frame is None:
            raise RuntimeError("Expected EEF depth frame")
        return depth_frame
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        update_env_after_qpos(env)


def safe_success(info: dict[str, Any]) -> bool:
    value = info.get("success", False)
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, (list, tuple)):
        return bool(any(value))
    return bool(value)


def maybe_unlink(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}")
        path.unlink()


def source_indices_for_clip(clip_frames: int, source_fps: float, clip_fps: float) -> np.ndarray:
    return np.rint(np.arange(clip_frames) * source_fps / clip_fps).astype(np.int64)


def write_clip81(path: Path, frames: list[np.ndarray], *, args: argparse.Namespace) -> None:
    if len(frames) < args.source_span:
        return
    maybe_unlink(path, args.overwrite)
    source_indices = source_indices_for_clip(args.clip_frames, args.source_fps, args.clip_fps)
    writer = VideoWriter(path, fps=args.clip_fps, frame_size=(frames[0].shape[1], frames[0].shape[0]), input_colorspace="rgb")
    try:
        for idx in source_indices:
            writer.write(frames[int(idx)])
    finally:
        writer.close()


def current_camera_pose(env, *, camera_name: str, mujoco_module) -> tuple[np.ndarray, np.ndarray]:
    camera_id = get_camera_id(env.sim.model, camera_name, mujoco_module)
    env.sim.forward()
    position = np.array(env.sim.data.cam_xpos[camera_id], copy=True)
    quaternion = _camera_xmat_to_quat(env.sim.data.cam_xmat[camera_id], mujoco_module)
    return position, quaternion


def main() -> None:
    args = parse_args()
    configure_mujoco_gl(args.mujoco_gl)
    if args.source_span <= int(source_indices_for_clip(args.clip_frames, args.source_fps, args.clip_fps).max()):
        raise ValueError("--source-span is too short for clip fps/frame settings")
    if args.output_mode == "production" and not args.write_shadow_step_nominal:
        raise ValueError("--output-mode production requires --write-shadow-step-nominal")
    if args.depth_far_m <= args.depth_near_m:
        raise ValueError("--depth-far-m must be greater than --depth-near-m")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "actions": out_dir / "rollout_actions_and_success.npz",
        "summary": out_dir / "summary.json",
    }
    if args.output_mode == "debug":
        paths.update(
            {
                "rgb": out_dir / "rollout_rgb_20fps.mp4",
                "state_mesh": out_dir / "state_mesh_20fps.mp4",
                "action_mesh": out_dir / "action_target_mesh_20fps.mp4",
                "action_eef_depth": out_dir / "action_target_eef_depth_20fps.mp4",
                "compare": out_dir / "compare_rgb_state_action_20fps.mp4",
                "compare_clip81": out_dir / "compare_rgb_state_action_last101_clips81_16fps.mp4",
            }
        )
        if args.write_shadow_step_nominal:
            paths.update(
                {
                    "shadow_nominal_mesh": out_dir / "shadow_step_nominal_mesh_20fps.mp4",
                    "shadow_nominal_eef_depth": out_dir / "shadow_step_nominal_eef_depth_20fps.mp4",
                    "compare_shadow": out_dir / "compare_rgb_state_target_shadow_20fps.mp4",
                    "compare_shadow_clip81": out_dir / "compare_rgb_state_target_shadow_last101_clips81_16fps.mp4",
                }
            )
    else:
        paths.update(
            {
                "rgb_clip81": out_dir / "rollout_rgb_last101_clips81_16fps.mp4",
                "static_rgb_clip81": out_dir / "static_rgb_last101_clips81_16fps.mp4",
                "static_depth_clip81": out_dir / "static_depth_last101_clips81_16fps.mp4",
                "shadow_nominal_mesh_clip81": out_dir / "shadow_step_nominal_mesh_last101_clips81_16fps.mp4",
                "shadow_nominal_eef_depth_clip81": out_dir / "shadow_step_nominal_eef_depth_last101_clips81_16fps.mp4",
            }
        )
    for key, path in paths.items():
        if key not in {
            "compare_clip81",
            "compare_shadow_clip81",
            "rgb_clip81",
            "static_rgb_clip81",
            "static_depth_clip81",
            "shadow_nominal_mesh_clip81",
            "shadow_nominal_eef_depth_clip81",
        }:
            maybe_unlink(path, args.overwrite)

    runtime_args = argparse.Namespace(mujoco_gl=args.mujoco_gl)
    mujoco_module, robosuite_module, robosuite_camera_utils = import_runtime_modules(runtime_args)
    env = make_env(args.env_name, seed=args.seed)
    policy = Dit4DitPolicyClient(
        policy_ckpt_path=args.policy_ckpt_path.expanduser(),
        host=args.host,
        port=args.port,
        image_size=tuple(args.image_size),
        n_action_steps=args.n_action_steps,
        num_ddim_steps=args.num_ddim_steps,
    )

    rgb_writer = None
    state_writer = None
    action_writer = None
    depth_writer = None
    compare_writer = None
    shadow_env = None
    static_env = None
    shadow_writer = None
    shadow_depth_writer = None
    shadow_compare_writer = None
    if args.output_mode == "debug":
        rgb_writer = VideoWriter(paths["rgb"], fps=args.source_fps, frame_size=(args.width, args.height), input_colorspace="rgb")
        state_writer = VideoWriter(paths["state_mesh"], fps=args.source_fps, frame_size=(args.width, args.height), input_colorspace="rgb")
        action_writer = VideoWriter(paths["action_mesh"], fps=args.source_fps, frame_size=(args.width, args.height), input_colorspace="rgb")
        depth_writer = VideoWriter(paths["action_eef_depth"], fps=args.source_fps, frame_size=(args.width, args.height), input_colorspace="rgb")
        compare_writer = VideoWriter(
            paths["compare"], fps=args.source_fps, frame_size=(args.width * 3, args.height), input_colorspace="rgb"
        )
    if args.output_mode == "debug" and args.write_shadow_step_nominal:
        shadow_writer = VideoWriter(
            paths["shadow_nominal_mesh"],
            fps=args.source_fps,
            frame_size=(args.width, args.height),
            input_colorspace="rgb",
        )
        shadow_depth_writer = VideoWriter(
            paths["shadow_nominal_eef_depth"],
            fps=args.source_fps,
            frame_size=(args.width, args.height),
            input_colorspace="rgb",
        )
        shadow_compare_writer = VideoWriter(
            paths["compare_shadow"],
            fps=args.source_fps,
            frame_size=(args.width * 4, args.height),
            input_colorspace="rgb",
        )

    background_rgb = np.asarray(args.background_color, dtype=np.uint8).reshape(1, 1, 3)
    flat_robot_rgb = np.asarray(args.robot_only_flat_color, dtype=np.uint8).reshape(1, 1, 3)
    last_compare_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    last_shadow_compare_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    last_rgb_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    last_static_rgb_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    last_static_depth_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    last_shadow_mesh_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    last_shadow_depth_frames: deque[np.ndarray] = deque(maxlen=args.source_span)
    action_records: dict[str, list[np.ndarray]] = {}
    success_steps: list[int] = []
    rewards: list[float] = []
    chunk_count = 0
    step_count = 0
    shadow_done = False

    try:
        obs = reset_rollout_start(env, args, robosuite_module)
        sim_env = low_env(env)
        robot_geom_ids = collect_robot_geom_ids(sim_env)
        eef_geom_ids = collect_eef_geom_ids(sim_env)
        limits = gripper_limits(sim_env)
        static_sim_env = None
        static_first_rgb = None
        static_first_depth = None
        if args.static_mode != "none":
            static_env = make_env(args.env_name, seed=args.seed)
            reset_rollout_start(static_env, args, robosuite_module)
            static_sim_env = low_env(static_env)
            if args.static_mode == "first_frame":
                static_first_rgb = static_sim_env.sim.render(
                    height=args.height,
                    width=args.width,
                    camera_name=args.camera,
                )[::-1]
                static_first_depth = render_depth_colormap_frame(
                    static_sim_env,
                    camera_name=args.camera,
                    width=args.width,
                    height=args.height,
                    near_m=args.depth_near_m,
                    far_m=args.depth_far_m,
                    invalid_color=None,
                    robosuite_camera_utils_module=robosuite_camera_utils,
                )
        shadow_sim_env = None
        shadow_robot_geom_ids = None
        shadow_eef_geom_ids = None
        if args.write_shadow_step_nominal:
            shadow_env = make_env(args.env_name, seed=args.seed)
            reset_rollout_start(shadow_env, args, robosuite_module)
            shadow_sim_env = low_env(shadow_env)
            shadow_robot_geom_ids = collect_robot_geom_ids(shadow_sim_env)
            shadow_eef_geom_ids = collect_eef_geom_ids(shadow_sim_env)
            disable_geom_collisions(shadow_sim_env, shadow_robot_geom_ids)
            print(
                f"Shadow-step nominal enabled: disabled robot collision geoms={len(shadow_robot_geom_ids)}",
                flush=True,
            )
        while step_count < args.max_steps:
            chunk = unbatch_action_chunk(policy.step(obs))
            horizon = min(args.n_action_steps, min(value.shape[0] for value in chunk.values()))
            chunk_count += 1
            for local_idx in range(horizon):
                if step_count >= args.max_steps:
                    break
                action = action_at(chunk, local_idx)
                for key, value in action.items():
                    action_records.setdefault(key, []).append(value.copy())

                rgb_frame = sim_env.sim.render(height=args.height, width=args.width, camera_name=args.camera)[::-1]
                last_rgb_frames.append(rgb_frame.copy())
                if args.static_mode == "first_frame":
                    assert static_first_rgb is not None
                    assert static_first_depth is not None
                    last_static_rgb_frames.append(static_first_rgb.copy())
                    last_static_depth_frames.append(static_first_depth.copy())
                elif args.static_mode == "camera_follow":
                    assert static_sim_env is not None
                    position, quaternion = current_camera_pose(
                        sim_env,
                        camera_name=args.camera,
                        mujoco_module=mujoco_module,
                    )
                    with override_camera_pose(
                        static_sim_env,
                        camera_name=args.camera,
                        position=position,
                        quaternion=quaternion,
                        mujoco_module=mujoco_module,
                    ):
                        static_depth = render_depth_colormap_frame(
                            static_sim_env,
                            camera_name=args.camera,
                            width=args.width,
                            height=args.height,
                            near_m=args.depth_near_m,
                            far_m=args.depth_far_m,
                            invalid_color=None,
                            robosuite_camera_utils_module=robosuite_camera_utils,
                        )
                        static_rgb = static_sim_env.sim.render(
                            height=args.height,
                            width=args.width,
                            camera_name=args.camera,
                        )[::-1]
                    last_static_rgb_frames.append(static_rgb.copy())
                    last_static_depth_frames.append(static_depth.copy())
                if args.output_mode == "debug":
                    assert rgb_writer is not None
                    assert state_writer is not None
                    assert action_writer is not None
                    assert depth_writer is not None
                    assert compare_writer is not None
                    state_mesh, _ = render_robot_only_outputs(
                        sim_env,
                        camera_name=args.camera,
                        width=args.width,
                        height=args.height,
                        robot_geom_ids=robot_geom_ids,
                        background_rgb=background_rgb,
                        rgb_style=args.robot_only_rgb_style,
                        flat_robot_rgb=flat_robot_rgb,
                        near_m=None,
                        far_m=None,
                        robosuite_camera_utils_module=robosuite_camera_utils,
                    )
                    action_mesh = render_action_target_frame(
                        sim_env,
                        action,
                        camera_name=args.camera,
                        width=args.width,
                        height=args.height,
                        robot_geom_ids=robot_geom_ids,
                        background_rgb=background_rgb,
                        flat_robot_rgb=flat_robot_rgb,
                        rgb_style=args.robot_only_rgb_style,
                        dt=1.0 / args.source_fps,
                        limits=limits,
                        robosuite_camera_utils_module=robosuite_camera_utils,
                    )
                    action_depth = render_action_target_eef_depth(
                        sim_env,
                        action,
                        camera_name=args.camera,
                        width=args.width,
                        height=args.height,
                        eef_geom_ids=eef_geom_ids,
                        background_rgb=background_rgb,
                        flat_robot_rgb=flat_robot_rgb,
                        rgb_style=args.robot_only_rgb_style,
                        dt=1.0 / args.source_fps,
                        limits=limits,
                        near_m=args.robot_depth_near_m,
                        far_m=args.robot_depth_far_m,
                        robosuite_camera_utils_module=robosuite_camera_utils,
                    )
                    compare = np.concatenate([rgb_frame, state_mesh, action_mesh], axis=1)
                    rgb_writer.write(rgb_frame)
                    state_writer.write(state_mesh)
                    action_writer.write(action_mesh)
                    depth_writer.write(action_depth)
                    compare_writer.write(compare)
                    last_compare_frames.append(compare.copy())

                if args.write_shadow_step_nominal:
                    assert shadow_env is not None
                    assert shadow_sim_env is not None
                    assert shadow_robot_geom_ids is not None
                    assert shadow_eef_geom_ids is not None
                    if not shadow_done:
                        _, _, shadow_terminated, shadow_truncated, _ = shadow_env.step(action)
                        shadow_done = bool(shadow_terminated or shadow_truncated)
                    shadow_mesh, _ = render_robot_only_outputs(
                        shadow_sim_env,
                        camera_name=args.camera,
                        width=args.width,
                        height=args.height,
                        robot_geom_ids=shadow_robot_geom_ids,
                        background_rgb=background_rgb,
                        rgb_style=args.robot_only_rgb_style,
                        flat_robot_rgb=flat_robot_rgb,
                        near_m=None,
                        far_m=None,
                        robosuite_camera_utils_module=robosuite_camera_utils,
                    )
                    _, shadow_depth = render_robot_only_outputs(
                        shadow_sim_env,
                        camera_name=args.camera,
                        width=args.width,
                        height=args.height,
                        robot_geom_ids=shadow_eef_geom_ids,
                        background_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                        rgb_style=args.robot_only_rgb_style,
                        flat_robot_rgb=flat_robot_rgb,
                        near_m=args.robot_depth_near_m,
                        far_m=args.robot_depth_far_m,
                        robosuite_camera_utils_module=robosuite_camera_utils,
                    )
                    if shadow_depth is None:
                        raise RuntimeError("Expected shadow-step EEF depth frame")
                    last_shadow_mesh_frames.append(shadow_mesh.copy())
                    last_shadow_depth_frames.append(shadow_depth.copy())
                    if args.output_mode == "debug":
                        assert shadow_writer is not None
                        assert shadow_depth_writer is not None
                        assert shadow_compare_writer is not None
                        shadow_compare = np.concatenate([rgb_frame, state_mesh, action_mesh, shadow_mesh], axis=1)
                        shadow_writer.write(shadow_mesh)
                        shadow_depth_writer.write(shadow_depth)
                        shadow_compare_writer.write(shadow_compare)
                        last_shadow_compare_frames.append(shadow_compare.copy())

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                if safe_success(info):
                    success_steps.append(step_count)
                if step_count % 20 == 0:
                    print(
                        f"[step {step_count:04d}] reward={float(reward):.3f} "
                        f"success={bool(success_steps)} chunk={chunk_count}",
                        flush=True,
                    )
                step_count += 1
                if (terminated or truncated) or (args.terminate_on_success and success_steps):
                    break
            if (terminated or truncated) or (args.terminate_on_success and success_steps):
                break
    finally:
        policy.close()
        env.close()
        if shadow_env is not None:
            shadow_env.close()
        if static_env is not None:
            static_env.close()
        for writer in (
            rgb_writer,
            state_writer,
            action_writer,
            depth_writer,
            compare_writer,
            shadow_writer,
            shadow_depth_writer,
            shadow_compare_writer,
        ):
            if writer is not None:
                writer.close()

    if args.output_mode == "debug":
        write_clip81(paths["compare_clip81"], list(last_compare_frames), args=args)
        if args.write_shadow_step_nominal:
            write_clip81(paths["compare_shadow_clip81"], list(last_shadow_compare_frames), args=args)
    else:
        write_clip81(paths["rgb_clip81"], list(last_rgb_frames), args=args)
        if args.static_mode != "none":
            write_clip81(paths["static_rgb_clip81"], list(last_static_rgb_frames), args=args)
            write_clip81(paths["static_depth_clip81"], list(last_static_depth_frames), args=args)
        write_clip81(paths["shadow_nominal_mesh_clip81"], list(last_shadow_mesh_frames), args=args)
        write_clip81(paths["shadow_nominal_eef_depth_clip81"], list(last_shadow_depth_frames), args=args)
    np.savez_compressed(
        paths["actions"],
        **{key.replace(".", "_"): np.stack(values, axis=0).astype(np.float32) for key, values in action_records.items()},
        rewards=np.asarray(rewards, dtype=np.float32),
        success_steps=np.asarray(success_steps, dtype=np.int32),
    )
    summary = {
        "env_name": args.env_name,
        "policy_ckpt_path": str(args.policy_ckpt_path),
        "steps": step_count,
        "chunks": chunk_count,
        "success_steps": success_steps,
        "warmstart_hdf5": str(args.warmstart_hdf5.expanduser().resolve()) if args.warmstart_hdf5 else None,
        "warmstart_demo": args.warmstart_demo if args.warmstart_hdf5 else None,
        "warmstart_state_index": int(args.warmstart_state_index) if args.warmstart_hdf5 else None,
        "outputs": {key: str(path) for key, path in paths.items() if path.exists()},
        "output_mode": args.output_mode,
        "static_mode": args.static_mode,
        "note": (
            "action_target_mesh is direct qpos target rendering from live policy actions; no env.step/contact in that render. "
            "shadow_step_nominal_mesh, when present, is rendered after env.step(action) in a second environment with robot "
            "collision geoms disabled."
        ),
        "shadow_step_nominal": bool(args.write_shadow_step_nominal),
        "shadow_done": bool(shadow_done) if args.write_shadow_step_nominal else None,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
