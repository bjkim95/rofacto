#!/usr/bin/env python3
"""Extract clip-aligned nominal GR1 state sidecars for DiT4DiT RoboCasa rollouts.

This replays cached policy actions from each clip start in a collision-free
shadow RoboCasa environment, matching the nominal robot mesh rendering path.
It does not run the policy or render videos.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
DATA_PROCESSING_ROOT = CURRENT_DIR.parent
LOCAL_ROBOSUITE_ROOT = REPO_ROOT / "robosuite"
LOCAL_ROBOCASA_ROOT = REPO_ROOT / "robocasa-gr1-tabletop-tasks"

for package_root in (LOCAL_ROBOSUITE_ROOT, LOCAL_ROBOCASA_ROOT, DATA_PROCESSING_ROOT, CURRENT_DIR):
    package_root_str = str(package_root)
    if package_root.exists() and package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

from collect_dit4dit_action_target_rollout_smoke import source_indices_for_clip  # noqa: E402
from collect_dit4dit_clipstart_rollout_dataset import (  # noqa: E402
    DEFAULT_GR1_ENV_SUFFIX,
    action_for_step,
    as_low_env,
    disable_geom_collisions,
    env_id_from_task_spec,
    import_runtime_modules,
    reset_env_to_state,
)
from dit4dit_rollout_rendering import collect_robot_geom_ids  # noqa: E402


DATA_ROOT = REPO_ROOT / "data"
DEFAULT_DATASET_FILE = (
    DATA_ROOT
    / "dataset_files"
    / "robocasa_dit4dit_clipstart_480x832_balanced24_task_success_balanced_split"
    / "all_videos_clips81_no_prompt.txt"
)
DEFAULT_OUT_ROOT = DATA_ROOT / "robocasa" / "dit4dit_clipstart_480x832_balanced24"
CLIP_RE = re.compile(r"__c(?P<clip>[0-9]+)_(?P<start>[0-9]+)-(?P<end>[0-9]+)$")
EPISODE_RE = re.compile(r"_dit4dit_w(?P<worker>[0-9]+)_e(?P<global>[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output-subdir", default="nominal_state_values_clips81")
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--clip-fps", type=float, default=16.0)
    parser.add_argument("--clip-frames", type=int, default=81)
    parser.add_argument("--source-span", type=int, default=101)
    parser.add_argument("--droid-dim", type=int, default=7)
    parser.add_argument(
        "--expected-robocasa-dim",
        type=int,
        default=39,
        help="GR1 realized qpos state dim: left/right arms 7+7, hands 11+11, waist 3.",
    )
    parser.add_argument("--mujoco-gl", default="unset", choices=("egl", "osmesa", "glfw", "unset"))
    parser.add_argument(
        "--replay-mode",
        choices=("episode_reset", "clip_start"),
        default="episode_reset",
        help=(
            "episode_reset replays cached actions once from the deterministic episode reset seed. "
            "clip_start matches existing nominal mesh rendering but requires rollout state caches."
        ),
    )
    parser.add_argument("--enable-render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--array-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-missing-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def configure_mujoco_gl(mujoco_gl: str) -> None:
    if mujoco_gl == "unset":
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        return
    os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    if mujoco_gl == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def import_runtime_modules_for_state(args: argparse.Namespace):
    if args.mujoco_gl != "unset":
        return import_runtime_modules(args)
    try:
        import mujoco  # type: ignore
        import robocasa  # noqa: F401  # type: ignore
        import robosuite  # type: ignore
        from robosuite.utils import camera_utils as robosuite_camera_utils  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mujoco / robocasa / robosuite. Activate the RoboCasa render environment first."
        ) from exc
    return mujoco, robosuite, robosuite_camera_utils


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def rel_to_abs(data_root: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    return data_root.expanduser() / path


def parse_clip(path: Path) -> tuple[str, str, str, int, int, int]:
    episode_dir = path.parent.parent
    task = episode_dir.parent.name
    episode = episode_dir.name
    match = CLIP_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Unexpected clip stem: {path.stem}")
    camera_source = path.stem[: match.start()]
    if "__" not in camera_source:
        raise ValueError(f"Expected '<episode>__<camera>' prefix in {path.stem}")
    _episode_from_stem, camera = camera_source.rsplit("__", 1)
    return (
        task,
        episode,
        camera,
        int(match.group("clip")),
        int(match.group("start")),
        int(match.group("end")),
    )


def split_for_shard(items: list[Any], shard_index: int, num_shards: int) -> list[Any]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--array-index must be in [0, {num_shards}), got {shard_index}")
    if num_shards == 1:
        return list(items)
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def action_dict_from_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "action.left_arm": np.asarray(payload["action_left_arm"], dtype=np.float32),
            "action.right_arm": np.asarray(payload["action_right_arm"], dtype=np.float32),
            "action.left_hand": np.asarray(payload["action_left_hand"], dtype=np.float32),
            "action.right_hand": np.asarray(payload["action_right_hand"], dtype=np.float32),
            "action.waist": np.asarray(payload["action_waist"], dtype=np.float32),
        }


def load_action_cache(episode_dir: Path) -> dict[str, np.ndarray]:
    action_path = episode_dir / "rollout_actions_and_success.npz"
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    return action_dict_from_npz(action_path)


def load_rollout_cache(episode_dir: Path) -> dict[str, Any]:
    state_path = episode_dir / "rollout_states_and_cameras.npz"
    action_path = episode_dir / "rollout_actions_and_success.npz"
    model_path = episode_dir / "rollout_model.xml.gz"
    meta_path = episode_dir / "rollout_ep_meta.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    with np.load(state_path, allow_pickle=False) as payload:
        states_pre = np.asarray(payload["states_pre"], dtype=np.float64)
    return {
        "states_pre": states_pre,
        "actions": action_dict_from_npz(action_path),
        "model_xml": gzip.decompress(model_path.read_bytes()).decode("utf-8"),
        "ep_meta": meta_path.read_text(encoding="utf-8"),
    }


def episode_global_index(episode: str) -> int:
    match = EPISODE_RE.search(episode)
    if match is None:
        raise ValueError(f"Cannot parse global episode index from {episode}")
    return int(match.group("global"))


def qpos_index(controller, part_name: str) -> np.ndarray:
    return np.asarray(controller.part_controllers[part_name].qpos_index, dtype=np.int64)


def gr1_nominal_state(env) -> tuple[np.ndarray, dict[str, list[int]]]:
    controller = env.robots[0].composite_controller
    sim_qpos = np.asarray(env.sim.data.qpos, dtype=np.float64)
    parts = {
        "left_arm": qpos_index(controller, "left"),
        "right_arm": qpos_index(controller, "right"),
        "left_hand": qpos_index(controller, "left_gripper"),
        "right_hand": qpos_index(controller, "right_gripper"),
        "waist": qpos_index(controller, "torso"),
    }
    values = [sim_qpos[indices] for indices in parts.values()]
    return np.concatenate(values, axis=0).astype(np.float32), {
        name: indices.astype(int).tolist() for name, indices in parts.items()
    }


def replay_nominal_source(
    *,
    env,
    rollout: dict[str, Any],
    start: int,
    source_span: int,
    seed: int,
    robosuite_module,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], bool]:
    low = as_low_env(env)
    if hasattr(env, "reset"):
        env.reset(seed=seed)
    reset_env_to_state(
        low,
        state=rollout["states_pre"][start],
        model_xml=rollout["model_xml"],
        ep_meta=rollout["ep_meta"],
        robosuite_module=robosuite_module,
    )
    disable_geom_collisions(low, collect_robot_geom_ids(low))

    pre_values: list[np.ndarray] = []
    post_values: list[np.ndarray] = []
    component_indices: dict[str, list[int]] | None = None
    shadow_done = False
    for offset in range(source_span):
        action_index = start + offset
        value, component_indices = gr1_nominal_state(low)
        pre_values.append(value)
        if not shadow_done:
            action = action_for_step(rollout["actions"], action_index)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            shadow_done = bool(terminated or truncated)
        value, component_indices = gr1_nominal_state(low)
        post_values.append(value)
    assert component_indices is not None
    return (
        np.asarray(pre_values, dtype=np.float32),
        np.asarray(post_values, dtype=np.float32),
        component_indices,
        shadow_done,
    )


def replay_episode_from_reset(
    *,
    env,
    actions: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], bool]:
    low = as_low_env(env)
    if hasattr(env, "reset"):
        env.reset(seed=seed)
    disable_geom_collisions(low, collect_robot_geom_ids(low))

    total_steps = min(int(value.shape[0]) for value in actions.values())
    pre_values: list[np.ndarray] = []
    post_values: list[np.ndarray] = []
    component_indices: dict[str, list[int]] | None = None
    shadow_done = False
    for action_index in range(total_steps):
        value, component_indices = gr1_nominal_state(low)
        pre_values.append(value)
        if not shadow_done:
            action = action_for_step(actions, action_index)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            shadow_done = bool(terminated or truncated)
        value, component_indices = gr1_nominal_state(low)
        post_values.append(value)
    assert component_indices is not None
    return (
        np.asarray(pre_values, dtype=np.float32),
        np.asarray(post_values, dtype=np.float32),
        component_indices,
        shadow_done,
    )


def concat_action_source(rollout: dict[str, Any], start: int, source_span: int) -> np.ndarray:
    actions = rollout["actions"]
    values = []
    for offset in range(source_span):
        idx = start + offset
        values.append(
            np.concatenate(
                [
                    actions["action.left_arm"][idx],
                    actions["action.right_arm"][idx],
                    actions["action.left_hand"][idx],
                    actions["action.right_hand"][idx],
                    actions["action.waist"][idx],
                ],
                axis=0,
            )
        )
    return np.asarray(values, dtype=np.float32)


def concat_action_source_from_actions(actions: dict[str, np.ndarray], start: int, source_span: int) -> np.ndarray:
    values = []
    for offset in range(source_span):
        idx = start + offset
        values.append(
            np.concatenate(
                [
                    actions["action.left_arm"][idx],
                    actions["action.right_arm"][idx],
                    actions["action.left_hand"][idx],
                    actions["action.right_hand"][idx],
                    actions["action.waist"][idx],
                ],
                axis=0,
            )
        )
    return np.asarray(values, dtype=np.float32)


def write_sidecar(
    *,
    path: Path,
    nominal_pre_source: np.ndarray,
    nominal_post_source: np.ndarray,
    raw_action_source: np.ndarray,
    source_indices: np.ndarray,
    droid_dim: int,
    component_indices: dict[str, list[int]],
    metadata: dict[str, Any],
    overwrite: bool,
    validate_only: bool,
) -> str:
    if path.exists() and not overwrite:
        return "exists"
    nominal_pre_81 = nominal_pre_source[source_indices]
    nominal_post_81 = nominal_post_source[source_indices]
    raw_action_81 = raw_action_source[source_indices]
    union = np.zeros((nominal_post_81.shape[0], droid_dim + nominal_post_81.shape[1]), dtype=np.float32)
    union[:, droid_dim:] = nominal_post_81
    union_mask = np.zeros((union.shape[1],), dtype=np.float32)
    union_mask[droid_dim:] = 1.0
    if validate_only:
        return "validated"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(
        tmp_path,
        nominal_state_81=nominal_post_81.astype(np.float32, copy=False),
        nominal_state_pre_81=nominal_pre_81.astype(np.float32, copy=False),
        raw_action_81=raw_action_81.astype(np.float32, copy=False),
        union_nominal_state_81=union.astype(np.float32, copy=False),
        union_mask=union_mask.astype(np.float32, copy=False),
        nominal_state_source=nominal_post_source.astype(np.float32, copy=False),
        nominal_state_pre_source=nominal_pre_source.astype(np.float32, copy=False),
        raw_action_source=raw_action_source.astype(np.float32, copy=False),
        source_indices=source_indices.astype(np.int64, copy=False),
        component_indices=json.dumps(component_indices, sort_keys=True),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    tmp_npz = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if tmp_npz != tmp_path:
        tmp_npz.replace(path)
    else:
        tmp_path.replace(path)
    return "written"


def make_env(env_name: str, seed: int, *, enable_render: bool):
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    import robosuite  # noqa: F401

    return gym.make(env_name, enable_render=enable_render, seed=seed)


def main() -> None:
    args = parse_args()
    configure_mujoco_gl(args.mujoco_gl)
    if args.source_span <= int(source_indices_for_clip(args.clip_frames, args.source_fps, args.clip_fps).max()):
        raise ValueError("--source-span is too short for clip fps/frame settings")
    data_root = args.data_root.expanduser().resolve()
    grouped: dict[tuple[str, str], list[tuple[str, Path, int, int, int, Path]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    clips = read_lines(args.dataset_file)
    if args.limit_clips > 0:
        clips = clips[: args.limit_clips]
    for rel in clips:
        video_path = rel_to_abs(data_root, rel)
        task, episode, camera, clip_index, start, end = parse_clip(video_path)
        out_path = video_path.parent.parent / args.output_subdir / f"{video_path.stem}.npz"
        grouped[(task, episode)].append((camera, video_path, clip_index, start, end, out_path))
    episode_keys = split_for_shard(sorted(grouped), args.array_index, args.num_shards)
    grouped = {key: grouped[key] for key in episode_keys}

    mujoco_module, robosuite_module, _camera_utils = import_runtime_modules_for_state(args)
    del mujoco_module
    source_indices = source_indices_for_clip(args.clip_frames, args.source_fps, args.clip_fps)
    env = None
    current_env_name = None
    try:
        for episode_counter, ((task, episode), records) in enumerate(sorted(grouped.items()), start=1):
            env_name = env_id_from_task_spec(task)
            if not env_name.endswith(DEFAULT_GR1_ENV_SUFFIX):
                raise ValueError(f"Expected GR1 env for task={task}: {env_name}")
            if current_env_name != env_name:
                if env is not None:
                    env.close()
                env = make_env(env_name, seed=args.seed, enable_render=args.enable_render)
                current_env_name = env_name
            assert env is not None
            episode_dir = args.out_root.expanduser().resolve() / task / episode
            episode_nominal_pre = None
            episode_nominal_post = None
            episode_actions = None
            episode_component_indices = None
            episode_shadow_done = None
            rollout = None
            if args.replay_mode == "episode_reset":
                try:
                    episode_actions = load_action_cache(episode_dir)
                    episode_nominal_pre, episode_nominal_post, episode_component_indices, episode_shadow_done = (
                        replay_episode_from_reset(
                            env=env,
                            actions=episode_actions,
                            seed=args.seed + episode_global_index(episode),
                        )
                    )
                except FileNotFoundError as exc:
                    stats["missing_action_episodes"] += 1
                    stats["missing_action_clips"] += len(records)
                    print(f"SKIP missing action cache {episode_dir}: {exc}", file=sys.stderr, flush=True)
                    if args.skip_missing_cache:
                        continue
                    raise
            else:
                try:
                    rollout = load_rollout_cache(episode_dir)
                except FileNotFoundError as exc:
                    stats["missing_cache_episodes"] += 1
                    stats["missing_cache_clips"] += len(records)
                    print(f"SKIP missing cache {episode_dir}: {exc}", file=sys.stderr, flush=True)
                    if args.skip_missing_cache:
                        continue
                    raise
            print(
                f"[{args.array_index}/{args.num_shards}] episode {episode_counter}/{len(grouped)} "
                f"{episode} clips={len(records)}",
                flush=True,
            )
            for camera, video_path, clip_index, start, end, out_path in sorted(records, key=lambda item: item[2]):
                stats["clips_total"] += 1
                if args.skip_existing and out_path.is_file() and not args.overwrite:
                    stats["exists"] += 1
                    continue
                try:
                    if args.replay_mode == "episode_reset":
                        assert episode_nominal_pre is not None
                        assert episode_nominal_post is not None
                        assert episode_actions is not None
                        assert episode_component_indices is not None
                        assert episode_shadow_done is not None
                        nominal_pre = episode_nominal_pre[start : start + args.source_span]
                        nominal_post = episode_nominal_post[start : start + args.source_span]
                        component_indices = episode_component_indices
                        shadow_done = bool(episode_shadow_done)
                        raw_action = concat_action_source_from_actions(episode_actions, start, args.source_span)
                    else:
                        assert rollout is not None
                        nominal_pre, nominal_post, component_indices, shadow_done = replay_nominal_source(
                            env=env,
                            rollout=rollout,
                            start=start,
                            source_span=args.source_span,
                            seed=args.seed + start,
                            robosuite_module=robosuite_module,
                        )
                        raw_action = concat_action_source(rollout, start, args.source_span)
                    if nominal_post.shape[1] != args.expected_robocasa_dim:
                        raise ValueError(
                            f"Expected RoboCasa nominal state dim {args.expected_robocasa_dim}, "
                            f"got {nominal_post.shape[1]} for {video_path}"
                        )
                    metadata = {
                        "source_video": str(video_path),
                        "task": task,
                        "episode": episode,
                        "camera": camera,
                        "clip_index": int(clip_index),
                        "orig_start": int(start),
                        "orig_end": int(end),
                        "source_span": int(args.source_span),
                        "clip_frames": int(args.clip_frames),
                        "source_fps": float(args.source_fps),
                        "clip_fps": float(args.clip_fps),
                        "state_dim": int(nominal_post.shape[1]),
                        "union_dim": int(args.droid_dim + nominal_post.shape[1]),
                        "shadow_done": bool(shadow_done),
                        "replay_mode": args.replay_mode,
                    }
                    result = write_sidecar(
                        path=out_path,
                        nominal_pre_source=nominal_pre,
                        nominal_post_source=nominal_post,
                        raw_action_source=raw_action,
                        source_indices=source_indices,
                        droid_dim=args.droid_dim,
                        component_indices=component_indices,
                        metadata=metadata,
                        overwrite=args.overwrite,
                        validate_only=args.validate_only,
                    )
                    stats[result] += 1
                except Exception as exc:
                    stats[f"error:{type(exc).__name__}"] += 1
                    print(f"ERROR {video_path}: {exc}", file=sys.stderr, flush=True)
                    if not args.keep_going:
                        raise
        print(json.dumps(dict(sorted(stats.items())), indent=2, sort_keys=True))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
