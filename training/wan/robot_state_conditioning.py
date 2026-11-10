import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


_CLIP_SPAN_RE = re.compile(r"_(?P<start>[0-9]+)-(?P<end>[0-9]+)$")
_DROID_ROBOCASA_UNION_SOURCES = {
    "droid_robocasa_action_union",
    "droid_robocasa_union_action",
    "union_droid_robocasa_action",
}


def _panda_fk(joint_angles: np.ndarray) -> np.ndarray:
    def tf_mat(i: int, dh: list[list[float]]) -> np.ndarray:
        a, d, alpha, theta = dh[i]
        q = theta
        return np.array(
            [
                [np.cos(q), -np.sin(q), 0.0, a],
                [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    dh_params = [
        [0.0, 0.333, 0.0, float(joint_angles[0])],
        [0.0, 0.0, -np.pi / 2, float(joint_angles[1])],
        [0.0, 0.316, np.pi / 2, float(joint_angles[2])],
        [0.0825, 0.0, np.pi / 2, float(joint_angles[3])],
        [-0.0825, 0.384, -np.pi / 2, float(joint_angles[4])],
        [0.0, 0.0, np.pi / 2, float(joint_angles[5])],
        [0.088, 0.0, np.pi / 2, float(joint_angles[6])],
        [0.0, 0.107, 0.0, 0.0],
        [0.0, 0.0, 0.0, -np.pi / 4],
        [0.0, 0.1034, 0.0, 0.0],
    ]

    transform = np.eye(4, dtype=np.float64)
    for i in range(8):
        transform = transform @ tf_mat(i, dh_params)
    return transform


def _matrix_to_xyz_euler(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([transform[0, 3], transform[1, 3], transform[2, 3], roll, pitch, yaw], dtype=np.float32)


def _clip_span_start(video_path: Path) -> int:
    last_part = video_path.stem.split("__")[-1]
    match = _CLIP_SPAN_RE.search(last_part)
    if match is None:
        return 0
    return int(match.group("start"))


def _episode_dir_from_clip(video_path: Path) -> Path:
    episode_id = video_path.stem.split("__")[0]
    return video_path.parent.parent / "episodes" / episode_id


def _frame_indices_for_clip(
    video_path: Path,
    clip_plan: Optional[Dict[str, Any]],
    num_frames: Optional[int],
) -> np.ndarray:
    clip_plan = clip_plan or {}
    target_num_frames = int(num_frames or clip_plan.get("target_num_frames", 0) or 0)
    if target_num_frames <= 0:
        raise ValueError("num_frames or clip_plan['target_num_frames'] is required for robot state conditioning")

    absolute_start = _clip_span_start(video_path) + int(clip_plan.get("start_idx", 0))
    frame_skip = max(1, int(clip_plan.get("frame_skip", 1)))
    return absolute_start + np.arange(target_num_frames, dtype=np.int64) * frame_skip


def _select_frames(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if array.shape[0] <= 0:
        raise ValueError("Robot state array has zero frames")
    clamped = np.clip(indices, 0, array.shape[0] - 1)
    return array[clamped]


def _resize_sequence_np(array: np.ndarray, num_frames: int) -> np.ndarray:
    if array.shape[0] == int(num_frames):
        return array
    if array.shape[0] <= 0:
        raise ValueError("Cannot resize an empty sequence")
    if int(num_frames) <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    tensor = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))
    tensor = tensor.unsqueeze(0).transpose(1, 2).float()
    resized = F.interpolate(tensor, size=int(num_frames), mode="linear", align_corners=True)
    return resized.transpose(1, 2).squeeze(0).numpy().astype(np.float32, copy=False)


def _is_robocasa_clip(video_path: Path) -> bool:
    return "robocasa" in set(video_path.parts)


def _load_robocasa_union_action_condition(
    video_path: Path,
    *,
    num_frames: Optional[int],
    action_sidecar_subdir: str,
    union_droid_dim: int,
    union_robocasa_dim: int,
) -> torch.Tensor:
    sidecar_path = video_path.parent.parent / action_sidecar_subdir / f"{video_path.stem}.npz"
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Missing RoboCasa action sidecar: {sidecar_path}")

    expected_union_dim = int(union_droid_dim) + int(union_robocasa_dim)
    with np.load(sidecar_path, allow_pickle=False) as payload:
        if "union_action_81" in payload:
            action = payload["union_action_81"].astype(np.float32, copy=False)
        elif "union_nominal_state_81" in payload:
            action = payload["union_nominal_state_81"].astype(np.float32, copy=False)
        elif "action_81" in payload:
            robocasa_action = payload["action_81"].astype(np.float32, copy=False)
            if robocasa_action.ndim != 2 or robocasa_action.shape[1] != int(union_robocasa_dim):
                raise ValueError(
                    f"Expected RoboCasa action sidecar [T,{union_robocasa_dim}], "
                    f"got {robocasa_action.shape} from {sidecar_path}"
                )
            action = np.zeros((robocasa_action.shape[0], expected_union_dim), dtype=np.float32)
            action[:, int(union_droid_dim):] = robocasa_action
        elif "nominal_state_81" in payload:
            robocasa_action = payload["nominal_state_81"].astype(np.float32, copy=False)
            if robocasa_action.ndim != 2 or robocasa_action.shape[1] != int(union_robocasa_dim):
                raise ValueError(
                    f"Expected RoboCasa action sidecar [T,{union_robocasa_dim}], "
                    f"got {robocasa_action.shape} from {sidecar_path}"
                )
            action = np.zeros((robocasa_action.shape[0], expected_union_dim), dtype=np.float32)
            action[:, int(union_droid_dim):] = robocasa_action
        else:
            raise KeyError(
                f"{sidecar_path} does not contain union_action_81, union_nominal_state_81, "
                "action_81, or nominal_state_81"
            )

    if action.ndim != 2 or action.shape[1] != expected_union_dim:
        raise ValueError(
            f"Expected union action [T,{expected_union_dim}], got {action.shape} from {sidecar_path}"
        )
    if num_frames is not None and int(num_frames) > 0:
        action = _resize_sequence_np(action, int(num_frames))
    return torch.from_numpy(np.ascontiguousarray(action.astype(np.float32)))


def load_robot_state_condition(
    video_path: str | Path,
    *,
    source: Optional[str] = "nominal_eef",
    clip_plan: Optional[Dict[str, Any]] = None,
    num_frames: Optional[int] = None,
    joint_filename: str = "joint_position_isaac_nominal.npy",
    gripper_filename: str = "gripper_position_isaac_nominal.npy",
    action_sidecar_subdir: str = "action_values_clips81",
    union_droid_dim: int = 7,
    union_robocasa_dim: int = 24,
) -> torch.Tensor:
    source = str(source or "").strip().lower()
    if source in {"", "none"}:
        raise ValueError("Robot state source is disabled")
    if source in _DROID_ROBOCASA_UNION_SOURCES:
        video_path = Path(video_path)
        if _is_robocasa_clip(video_path):
            return _load_robocasa_union_action_condition(
                video_path,
                num_frames=num_frames,
                action_sidecar_subdir=action_sidecar_subdir,
                union_droid_dim=int(union_droid_dim),
                union_robocasa_dim=int(union_robocasa_dim),
            )

        droid_state = load_robot_state_condition(
            video_path,
            source="nominal_eef",
            clip_plan=clip_plan,
            num_frames=num_frames,
            joint_filename=joint_filename,
            gripper_filename=gripper_filename,
            action_sidecar_subdir=action_sidecar_subdir,
            union_droid_dim=union_droid_dim,
            union_robocasa_dim=union_robocasa_dim,
        )
        if int(droid_state.shape[-1]) != int(union_droid_dim):
            raise ValueError(
                f"DROID nominal EEF dim mismatch: expected {union_droid_dim}, got {droid_state.shape[-1]}"
            )
        robocasa_zeros = torch.zeros(
            droid_state.shape[0],
            int(union_robocasa_dim),
            dtype=droid_state.dtype,
        )
        return torch.cat([droid_state, robocasa_zeros], dim=-1)
    if source not in {"nominal_eef", "nominal_joint", "nominal_joint_gripper"}:
        raise ValueError(
            f"Unsupported robot state source: {source}. "
            "Use one of: nominal_eef, nominal_joint, nominal_joint_gripper, "
            "droid_robocasa_action_union."
        )

    video_path = Path(video_path)
    episode_dir = _episode_dir_from_clip(video_path)
    joint_path = episode_dir / joint_filename
    gripper_path = episode_dir / gripper_filename
    if not joint_path.is_file():
        raise FileNotFoundError(f"Missing nominal joint state file: {joint_path}")
    if source in {"nominal_eef", "nominal_joint_gripper"} and not gripper_path.is_file():
        raise FileNotFoundError(f"Missing nominal gripper state file: {gripper_path}")

    indices = _frame_indices_for_clip(video_path, clip_plan, num_frames)
    joints = _select_frames(np.load(joint_path), indices).astype(np.float32)
    if joints.ndim != 2 or joints.shape[1] < 7:
        raise ValueError(f"Expected joint array [T, >=7], got {joints.shape} from {joint_path}")
    joints = joints[:, :7]

    if source == "nominal_joint":
        state = joints
    else:
        gripper = _select_frames(np.load(gripper_path), indices).astype(np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if gripper.ndim != 2 or gripper.shape[1] < 1:
            raise ValueError(f"Expected gripper array [T] or [T, >=1], got {gripper.shape} from {gripper_path}")
        gripper = gripper[:, :1]
        if source == "nominal_joint_gripper":
            state = np.concatenate([joints, gripper], axis=1)
        else:
            eef = np.stack([_matrix_to_xyz_euler(_panda_fk(q)) for q in joints], axis=0)
            state = np.concatenate([eef, gripper], axis=1)

    return torch.from_numpy(np.ascontiguousarray(state.astype(np.float32)))


def align_state_adaln_to_latent_frames(
    state_adaln: Optional[torch.Tensor],
    latent_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    temporal_mode: str = "sample4",
) -> Optional[torch.Tensor]:
    """Align raw robot states to the VAE latent temporal grid.

    `window4_stack` maps 81 raw states to 21 latent states as:
    l0=[s0,s0,s0,s0], l1=[s1,s2,s3,s4], ..., l20=[s77,s78,s79,s80].

    `delta4_stack` follows the DreamDojo action-chunk convention:
    l0=[0,0,0,0], l1=[s1-s0,s2-s0,s3-s0,s4-s0], ...,
    l20=[s77-s76,s78-s76,s79-s76,s80-s76].
    """
    if state_adaln is None:
        return None

    mode = str(temporal_mode or "sample4").strip().lower()
    state_adaln = state_adaln.to(device=device, dtype=dtype)
    if state_adaln.dim() == 2:
        state_adaln = state_adaln.unsqueeze(0)
    if state_adaln.dim() != 3:
        raise ValueError(f"state_adaln must be [B,T,D] or [T,D], got {tuple(state_adaln.shape)}")

    if mode in {"none", "identity"}:
        if state_adaln.shape[1] != latent_frames:
            raise ValueError(
                f"state_adaln temporal_mode={mode!r} requires {latent_frames} frames, "
                f"got {state_adaln.shape[1]}"
            )
        return state_adaln

    raw_frames_for_ratio4 = (int(latent_frames) - 1) * 4 + 1

    if mode in {"sample4", "stride4", "stride4_sample", "point"}:
        if state_adaln.shape[1] == latent_frames:
            return state_adaln
        if state_adaln.shape[1] <= 1:
            return state_adaln.repeat(1, latent_frames, 1)
        if state_adaln.shape[1] == raw_frames_for_ratio4:
            indices = torch.arange(latent_frames, device=device) * 4
            return state_adaln.index_select(1, indices)
        state_bdt = state_adaln.transpose(1, 2).float()
        resized = F.interpolate(state_bdt, size=latent_frames, mode="linear", align_corners=True)
        return resized.transpose(1, 2).to(dtype=dtype)

    if mode in {"window4_stack", "stack4", "ratio4_stack"}:
        current_dim = int(state_adaln.shape[-1])
        if state_adaln.shape[1] == latent_frames and current_dim % 4 == 0:
            return state_adaln
        if state_adaln.shape[1] != raw_frames_for_ratio4:
            state_bdt = state_adaln.transpose(1, 2).float()
            state_adaln = F.interpolate(
                state_bdt,
                size=raw_frames_for_ratio4,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2).to(dtype=dtype)

        windows = []
        first = state_adaln[:, :1, :].expand(-1, 4, -1)
        windows.append(first.reshape(state_adaln.shape[0], 1, current_dim * 4))
        for latent_idx in range(1, int(latent_frames)):
            start = 1 + (latent_idx - 1) * 4
            window = state_adaln[:, start:start + 4, :]
            if window.shape[1] < 4:
                pad = window[:, -1:, :].expand(-1, 4 - window.shape[1], -1)
                window = torch.cat([window, pad], dim=1)
            windows.append(window.reshape(state_adaln.shape[0], 1, current_dim * 4))
        return torch.cat(windows, dim=1)

    if mode in {"delta4_stack", "dreamdojo_delta4", "dreamdojo_action", "action_delta4_stack"}:
        current_dim = int(state_adaln.shape[-1])
        if state_adaln.shape[1] == latent_frames and current_dim % 4 == 0:
            return state_adaln
        if state_adaln.shape[1] != raw_frames_for_ratio4:
            state_bdt = state_adaln.transpose(1, 2).float()
            state_adaln = F.interpolate(
                state_bdt,
                size=raw_frames_for_ratio4,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2).to(dtype=dtype)

        windows = [
            state_adaln.new_zeros((state_adaln.shape[0], 1, current_dim * 4))
        ]
        for latent_idx in range(1, int(latent_frames)):
            start = 1 + (latent_idx - 1) * 4
            baseline = state_adaln[:, start - 1:start, :]
            window = state_adaln[:, start:start + 4, :]
            if window.shape[1] < 4:
                pad = window[:, -1:, :].expand(-1, 4 - window.shape[1], -1)
                window = torch.cat([window, pad], dim=1)
            delta_window = window - baseline
            windows.append(delta_window.reshape(state_adaln.shape[0], 1, current_dim * 4))
        return torch.cat(windows, dim=1)

    raise ValueError(
        f"Unsupported state_adaln_temporal_mode={temporal_mode!r}. "
        "Use one of: sample4, window4_stack, delta4_stack."
    )
