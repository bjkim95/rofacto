#!/usr/bin/env python3
from __future__ import annotations

import argparse
import h5py
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from yourdfpy import URDF

try:
    from .video_io import VideoWriter
except ImportError:
    from video_io import VideoWriter

try:
    import torch
    from pytorch3d.renderer import (
        BlendParams,
        DirectionalLights,
        HardPhongShader,
        MeshRasterizer,
        RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    HAS_PYTORCH3D = True
except Exception:
    HAS_PYTORCH3D = False


LOGICAL_CAMERA_TO_SERIAL_KEY = {
    "exterior_image_1_left": "ext1_cam_serial",
    "exterior_image_2_left": "ext2_cam_serial",
    "wrist_image_left": "wrist_cam_serial",
}

SERIAL_KEY_TO_LOGICAL_CAMERA = {
    "ext1_cam_serial": "exterior_image_1_left",
    "ext2_cam_serial": "exterior_image_2_left",
    "wrist_cam_serial": "wrist_image_left",
    "wrist1_cam_serial": "wrist_image_left",
}

# Validated DROID robot model: Panda + Robotiq FT300 force-torque sensor + Robotiq 2F-85.
# The FT300 is physically present on the DROID arm (verified against the RGB overlay), so
# this is the single source of truth for every DROID render and for DexYCB IK.
ORIGINAL_URDF_PATH = Path(__file__).resolve().parent / "assets" / "panda_arm_robotiq_2f85_fts300.urdf"
DEFAULT_URDF_PATH = ORIGINAL_URDF_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless URDF overlay renderer for DROID-style camera metadata."
    )
    parser.add_argument("--video-path", type=Path, help="Input RGB video path.")
    parser.add_argument("--joint-path", type=Path, help="Joint trajectory (.npy/.npz/.json).")
    parser.add_argument(
        "--gripper-path",
        type=Path,
        default=None,
        help="Optional scalar gripper trajectory (.npy/.npz/.json). Defaults to sibling gripper_position.npy.",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="Robot URDF path.",
    )
    parser.add_argument(
        "--episode-meta-json",
        type=Path,
        default=None,
        help="Optional JSON emitted by extract_droid_episode.py.",
    )
    parser.add_argument(
        "--cam2base-json",
        type=Path,
        help="JSON containing DROID-style camera-to-base extrinsics.",
    )
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        help="JSON containing DROID-style intrinsics.",
    )
    parser.add_argument(
        "--camera-serials-json",
        type=Path,
        default=None,
        help="Optional DROID camera_serials.json used to map logical camera names to serials.",
    )
    parser.add_argument(
        "--raw-metadata-json",
        type=Path,
        default=None,
        help="Optional raw DROID metadata.json. Required for wrist raw-extrinsics rendering.",
    )
    parser.add_argument(
        "--raw-trajectory-h5",
        type=Path,
        default=None,
        help="Optional raw DROID trajectory.h5 used for wrist raw extrinsics and direct joint/gripper loading.",
    )
    parser.add_argument(
        "--use-raw-wrist-extrinsics",
        action="store_true",
        help="Use raw DROID observation/camera_extrinsics for wrist_image_left instead of improved cam2base.",
    )
    parser.add_argument(
        "--wrist-pose-source",
        type=str,
        choices=["trajectory", "metadata"],
        default="trajectory",
        help="Use per-frame wrist pose from trajectory.h5 or the first-frame metadata wrist pose.",
    )
    parser.add_argument("--episode-id", type=str, help="Episode identifier.")
    parser.add_argument(
        "--camera-name",
        type=str,
        default="exterior_image_1_left",
        help="Logical camera name, serial alias, serial number, or 'auto' for the first calibrated camera.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("overlay.mp4"),
        help="Rendered output video path.",
    )
    parser.add_argument(
        "--comparison-output-path",
        type=Path,
        default=None,
        help="Optional side-by-side video: URDF on white | overlay | original.",
    )
    parser.add_argument(
        "--robot-only-output-path",
        type=Path,
        default=None,
        help="Optional video rendered on a white background with the robot only.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Output FPS. If lower than the source FPS, frames are temporally downsampled before rendering.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--invert-gripper",
        action="store_true",
        help="Invert the scalar gripper signal before mapping it onto the URDF gripper joint.",
    )
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        default=None,
        metavar=("B", "G", "R"),
        help="Optional BGR override. Defaults to the DAE material colors.",
    )
    parser.add_argument(
        "--renderer-backend",
        type=str,
        choices=["pytorch3d", "pyrender", "pyrender_persistent"],
        default="pytorch3d",
        help="Headless renderer backend.",
    )
    parser.add_argument(
        "--pyrender-style",
        type=str,
        choices=["default", "matte", "ambient", "flat", "flat_dim"],
        default="flat",
        help="Lighting/material style for the pyrender backend.",
    )
    parser.add_argument(
        "--robot-only-background",
        type=str,
        choices=["white", "gray"],
        default="white",
        help="Background for robot-only outputs. gray uses uint8 127.",
    )
    parser.add_argument(
        "--appearance",
        type=str,
        choices=["mask", "shaded", "unlit"],
        default="shaded" if HAS_PYTORCH3D else "mask",
        help="Mask overlay or shaded CAD-style rendering.",
    )
    parser.add_argument(
        "--base-adjust-json",
        type=Path,
        default=None,
        help="Optional 4x4 matrix applied after URDF base to absorb frame mismatch.",
    )
    parser.add_argument(
        "--tool-adjust-json",
        type=Path,
        default=None,
        help="Optional tool-frame 4x4 or xyz/rpy applied at droid_tool_adjust_joint.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=0,
        help="Maximum faces to render. Use 0 to keep the full mesh.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Process only the first N frames.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame and reuse original pixels for skipped frames.",
    )
    parser.add_argument(
        "--robot-only-render-batch-size",
        type=int,
        default=1,
        help="Batch size for PyTorch3D robot-only rendering. Use 1 for legacy per-frame rendering.",
    )
    parser.add_argument(
        "--perturb-mode",
        type=str,
        choices=["none", "pick_joint_offset"],
        default="none",
        help="Optional trajectory perturbation applied before rendering.",
    )
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=42,
        help="Random seed for deterministic trajectory perturbation.",
    )
    parser.add_argument(
        "--perturb-joint-indices",
        type=str,
        default="1,3,5",
        help="Comma-separated arm joint indices to perturb.",
    )
    parser.add_argument(
        "--perturb-max-deg",
        type=float,
        default=5.0,
        help="Maximum absolute perturbation per selected joint in degrees.",
    )
    parser.add_argument(
        "--perturb-window-start",
        type=int,
        default=None,
        help="Optional inclusive start frame for manual perturb window override.",
    )
    parser.add_argument(
        "--perturb-window-end",
        type=int,
        default=None,
        help="Optional inclusive end frame for manual perturb window override.",
    )
    parser.add_argument(
        "--perturb-pre-frames",
        type=int,
        default=8,
        help="Number of frames used for smooth ramp-up before the perturb window.",
    )
    parser.add_argument(
        "--perturb-post-frames",
        type=int,
        default=8,
        help="Number of frames used for smooth ramp-down after the perturb window.",
    )
    parser.add_argument(
        "--perturb-meta-output",
        type=Path,
        default=None,
        help="Optional JSON path for perturb metadata.",
    )
    parser.add_argument(
        "--calibration-image-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
        help="Image size that the calibration intrinsics were estimated for.",
    )
    parser.add_argument(
        "--target-short-side",
        type=int,
        default=480,
        help="Downsample output frames so the shorter side is at most this size. Use 0 to keep source resolution.",
    )
    parser.add_argument(
        "--rgb-output-path",
        type=Path,
        default=None,
        help="Optional resized RGB video written at the same resolution as the render outputs.",
    )
    parser.add_argument(
        "--demo-output-dir",
        type=Path,
        default=None,
        help="If set, generate a synthetic demo instead of reading dataset files.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Synthetic demo frame size.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def h5_dataset_or_none(handle: h5py.File, candidates: tuple[str, ...]) -> np.ndarray | None:
    for key in candidates:
        if key in handle:
            return np.asarray(handle[key], dtype=np.float64)
    return None


def load_joint_gripper_from_raw_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        joints = h5_dataset_or_none(
            f,
            (
                "observation/robot_state/joint_positions",
                "action/joint_position",
                "action/robot_state/joint_positions",
            ),
        )
        gripper = h5_dataset_or_none(
            f,
            (
                "observation/robot_state/gripper_position",
                "action/gripper_position",
                "action/robot_state/gripper_position",
            ),
        )
    if joints is None or gripper is None:
        raise KeyError(f"Could not read joint/gripper trajectories from {path}")
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 1:
        joints = joints.reshape(-1, 1)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    return joints, gripper


def load_raw_cartesian_trajectory(path: Path) -> np.ndarray | None:
    with h5py.File(path, "r") as f:
        return h5_dataset_or_none(
            f,
            (
                "observation/robot_state/cartesian_position",
                "action/cartesian_position",
                "action/robot_state/cartesian_position",
            ),
        )


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def report_render_progress(
    *,
    label: str,
    processed_frames: int,
    total_frames: int,
    start_time: float,
    rendered_frames: int,
    reused_frames: int,
    force: bool = False,
) -> None:
    elapsed = max(1e-6, time.time() - start_time)
    throughput = processed_frames / elapsed
    remaining_frames = max(0, total_frames - processed_frames)
    eta_seconds = remaining_frames / max(throughput, 1e-6)
    progress = processed_frames / max(total_frames, 1)
    message = (
        f"[render] {label} | {processed_frames}/{total_frames} "
        f"({progress * 100.0:5.1f}%) | elapsed {format_duration(elapsed)} "
        f"| eta {format_duration(eta_seconds)} | fps {throughput:4.2f} "
        f"| rendered {rendered_frames} | reused {reused_frames}"
    )
    is_tty = sys.stderr.isatty()
    end = "\n" if force or not is_tty else ""
    prefix = "\r" if is_tty and not force else ""
    print(prefix + message, file=sys.stderr, end=end, flush=True)


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def matrix_to_euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = float(rotation[0, 2])
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = float(np.arcsin(sy))
    cy = float(np.cos(ry))
    if abs(cy) > 1e-8:
        rx = float(np.arctan2(-rotation[1, 2], rotation[2, 2]))
        rz = float(np.arctan2(-rotation[0, 1], rotation[0, 0]))
    else:
        rx = float(np.arctan2(rotation[2, 1], rotation[1, 1]))
        rz = 0.0
    return rx, ry, rz


def normalize_path_like(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")


def read_matrix(candidate: Any) -> np.ndarray | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        for key in (
            "matrix",
            "transform",
            "extrinsic",
            "cam2base",
            "camera_to_base",
            "camera_extrinsic",
            "camera_extrinsics",
            "cameraExtrinsics",
        ):
            if key in candidate:
                matrix = read_matrix(candidate[key])
                if matrix is not None:
                    return matrix
        if "R" in candidate and "t" in candidate:
            rot = np.asarray(candidate["R"], dtype=np.float64).reshape(3, 3)
            trans = np.asarray(candidate["t"], dtype=np.float64).reshape(3)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rot
            matrix[:3, 3] = trans
            return matrix
    if isinstance(candidate, list):
        arr = np.asarray(candidate, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
        if arr.size == 6:
            tx, ty, tz, rx, ry, rz = arr.reshape(6)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = euler_xyz_to_matrix(rx, ry, rz)
            matrix[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
            return matrix
    return None


def read_intrinsics(candidate: Any) -> np.ndarray | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        for key in ("K", "camera_matrix", "cameraMatrix", "intrinsics", "matrix"):
            if key in candidate:
                matrix = read_intrinsics(candidate[key])
                if matrix is not None:
                    return matrix
        if all(k in candidate for k in ("fx", "fy", "cx", "cy")):
            return np.array(
                [
                    [float(candidate["fx"]), 0.0, float(candidate["cx"])],
                    [0.0, float(candidate["fy"]), float(candidate["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
    if isinstance(candidate, list):
        arr = np.asarray(candidate, dtype=np.float64)
        if arr.shape == (3, 3):
            return arr
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.size == 4:
            fx, cx, fy, cy = arr.reshape(4)
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return None


def read_xyz_rpy(candidate: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        if "xyz" in candidate and "rpy" in candidate:
            xyz = np.asarray(candidate["xyz"], dtype=np.float64).reshape(3)
            rpy = np.asarray(candidate["rpy"], dtype=np.float64).reshape(3)
            return xyz, rpy
        matrix = read_matrix(candidate)
        if matrix is not None:
            return matrix[:3, 3], np.asarray(matrix_to_euler_xyz(matrix[:3, :3]), dtype=np.float64)
    if isinstance(candidate, list):
        arr = np.asarray(candidate, dtype=np.float64)
        if arr.size == 6:
            return arr[:3].reshape(3), arr[3:].reshape(3)
        matrix = read_matrix(candidate)
        if matrix is not None:
            return matrix[:3, 3], np.asarray(matrix_to_euler_xyz(matrix[:3, :3]), dtype=np.float64)
    return None


def find_nested_value(node: Any, target_key: str) -> Any:
    if isinstance(node, dict):
        if target_key in node:
            return node[target_key]
        for value in node.values():
            found = find_nested_value(value, target_key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_nested_value(value, target_key)
            if found is not None:
                return found
    return None


def find_episode_entry(node: Any, episode_id: str) -> Any:
    if isinstance(node, dict):
        if episode_id in node:
            return node[episode_id]
        if str(episode_id) in node:
            return node[str(episode_id)]
        for value in node.values():
            found = find_episode_entry(value, episode_id)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_episode_entry(value, episode_id)
            if found is not None:
                return found
    return None


def resolve_camera_candidates(camera_name: str, episode_camera_serials: dict[str, Any] | None) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if camera_name == "auto":
        if isinstance(episode_camera_serials, dict):
            for serial_key in ("ext1_cam_serial", "ext2_cam_serial", "wrist_cam_serial", "wrist1_cam_serial"):
                serial = episode_camera_serials.get(serial_key)
                if serial:
                    logical_name = SERIAL_KEY_TO_LOGICAL_CAMERA.get(serial_key, serial_key)
                    candidates.append((logical_name, str(serial)))
        return candidates

    candidates.append((camera_name, camera_name))
    if isinstance(episode_camera_serials, dict):
        serial_key = LOGICAL_CAMERA_TO_SERIAL_KEY.get(camera_name)
        if serial_key and episode_camera_serials.get(serial_key):
            candidates.append((camera_name, str(episode_camera_serials[serial_key])))
        if camera_name in episode_camera_serials:
            logical_name = SERIAL_KEY_TO_LOGICAL_CAMERA.get(camera_name, camera_name)
            candidates.append((logical_name, str(episode_camera_serials[camera_name])))
        for key, value in episode_camera_serials.items():
            if str(value) == camera_name:
                logical_name = SERIAL_KEY_TO_LOGICAL_CAMERA.get(key, key)
                candidates.append((logical_name, str(value)))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def load_episode_meta_defaults(path: Path) -> dict[str, Any]:
    meta = load_json(path)
    if not isinstance(meta, dict):
        raise ValueError(f"Expected dict episode meta JSON at {path}")
    defaults: dict[str, Any] = {}
    for key in ("resolved_episode_id", "episode_id"):
        if meta.get(key):
            defaults["episode_id"] = str(meta[key])
            break
    if meta.get("camera_name"):
        defaults["camera_name"] = str(meta["camera_name"])
    if meta.get("camera_serial"):
        defaults["camera_serial"] = str(meta["camera_serial"])
    for key in ("gripper_path", "joint_path"):
        if meta.get(key):
            candidate = path.parent / str(meta[key])
            if candidate.exists():
                defaults[key] = candidate
    if meta.get("default_urdf_path"):
        defaults["urdf_path"] = Path(str(meta["default_urdf_path"]))
    return defaults


def materialize_tool_adjusted_urdf(urdf_path: Path, tool_adjust_json: Path | None) -> tuple[Path, str | None]:
    if tool_adjust_json is None:
        return urdf_path, None

    adjust_blob = load_json(tool_adjust_json)
    parsed = read_xyz_rpy(adjust_blob)
    if parsed is None:
        raise ValueError(f"Failed to parse tool adjust from {tool_adjust_json}")
    xyz, rpy = parsed

    urdf_text = urdf_path.read_text(encoding="utf-8")
    xyz_text = " ".join(f"{value:.9f}" for value in xyz.tolist())
    rpy_text = " ".join(f"{value:.9f}" for value in rpy.tolist())
    joint_pattern = re.compile(
        r'(<joint\s+name="droid_tool_adjust_joint"[^>]*>)(.*?)(</joint>)',
        flags=re.DOTALL,
    )
    joint_match = joint_pattern.search(urdf_text)
    if joint_match is None:
        raise ValueError(f"Could not find droid_tool_adjust_joint in {urdf_path}")
    joint_body = joint_match.group(2)
    origin_pattern = re.compile(r'(^[ \t]*)<origin\b[^>]*/>', flags=re.MULTILINE)
    patched_body, num_subs = origin_pattern.subn(
        rf'\1<origin xyz="{xyz_text}" rpy="{rpy_text}"/>',
        joint_body,
        count=1,
    )
    if num_subs != 1:
        raise ValueError(f"Could not find origin tag inside droid_tool_adjust_joint in {urdf_path}")
    patched_text = (
        urdf_text[: joint_match.start(2)]
        + patched_body
        + urdf_text[joint_match.end(2) :]
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="droid_tool_adjust_", dir=str(urdf_path.parent)))
    temp_urdf_path = temp_dir / urdf_path.name
    temp_urdf_path.write_text(patched_text, encoding="utf-8")
    return temp_urdf_path, str(tool_adjust_json)


def load_camera_metadata(
    episode_id: str,
    camera_name: str,
    cam2base_json: Path,
    intrinsics_json: Path,
    camera_serials_json: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    extrinsic_blob = load_json(cam2base_json)
    intrinsics_blob = load_json(intrinsics_json)
    camera_serials_blob = load_json(camera_serials_json) if camera_serials_json is not None else None

    episode_extrinsic = find_episode_entry(extrinsic_blob, episode_id)
    episode_intrinsics = find_episode_entry(intrinsics_blob, episode_id)
    if episode_extrinsic is None:
        raise KeyError(f"Could not find episode '{episode_id}' in {cam2base_json}")
    if episode_intrinsics is None:
        raise KeyError(f"Could not find episode '{episode_id}' in {intrinsics_json}")
    if not isinstance(episode_extrinsic, dict) or not isinstance(episode_intrinsics, dict):
        raise ValueError("Episode calibration entries must be JSON objects")

    episode_camera_serials = None
    if camera_serials_blob is not None:
        episode_camera_serials = find_episode_entry(camera_serials_blob, episode_id)
        if episode_camera_serials is not None and not isinstance(episode_camera_serials, dict):
            raise ValueError(f"camera_serials entry for episode '{episode_id}' must be a JSON object")

    candidates = resolve_camera_candidates(camera_name, episode_camera_serials)
    if camera_name == "auto" and not candidates:
        raise KeyError(
            "camera_name='auto' requires camera_serials_json with a known calibrated camera for this episode"
        )

    for logical_name, camera_key in candidates:
        camera_extrinsic_entry = episode_extrinsic.get(camera_key)
        camera_intrinsics_entry = episode_intrinsics.get(camera_key)
        if camera_extrinsic_entry is None or camera_intrinsics_entry is None:
            continue
        cam2base = read_matrix(camera_extrinsic_entry)
        intrinsics = read_intrinsics(camera_intrinsics_entry)
        if cam2base is None or intrinsics is None:
            continue
        return cam2base, intrinsics, logical_name, camera_key

    available = sorted(k for k in episode_extrinsic.keys() if isinstance(k, str))
    raise KeyError(
        f"Could not resolve camera '{camera_name}' for episode '{episode_id}'. "
        f"Available extrinsic keys include: {available[:8]}"
    )


def load_camera_intrinsics_metadata(
    episode_id: str,
    camera_name: str,
    intrinsics_json: Path,
    camera_serials_json: Path | None = None,
) -> tuple[np.ndarray, str, str]:
    intrinsics_blob = load_json(intrinsics_json)
    camera_serials_blob = load_json(camera_serials_json) if camera_serials_json is not None else None

    episode_intrinsics = find_episode_entry(intrinsics_blob, episode_id)
    if episode_intrinsics is None:
        raise KeyError(f"Could not find episode '{episode_id}' in {intrinsics_json}")
    if not isinstance(episode_intrinsics, dict):
        raise ValueError("Episode intrinsics entry must be a JSON object")

    episode_camera_serials = None
    if camera_serials_blob is not None:
        episode_camera_serials = find_episode_entry(camera_serials_blob, episode_id)
        if episode_camera_serials is not None and not isinstance(episode_camera_serials, dict):
            raise ValueError(f"camera_serials entry for episode '{episode_id}' must be a JSON object")

    candidates = resolve_camera_candidates(camera_name, episode_camera_serials)
    if camera_name == "auto" and not candidates:
        raise KeyError(
            "camera_name='auto' requires camera_serials_json with a known calibrated camera for this episode"
        )
    for logical_name, camera_key in candidates:
        camera_intrinsics_entry = episode_intrinsics.get(camera_key)
        if camera_intrinsics_entry is None:
            continue
        intrinsics = read_intrinsics(camera_intrinsics_entry)
        if intrinsics is not None:
            return intrinsics, logical_name, camera_key

    available = sorted(k for k in episode_intrinsics.keys() if isinstance(k, str))
    raise KeyError(
        f"Could not resolve camera '{camera_name}' intrinsics for episode '{episode_id}'. "
        f"Available intrinsics keys include: {available[:8]}"
    )


def resolve_raw_wrist_serial(
    *,
    raw_metadata: dict[str, Any] | None,
    episode_id: str | None,
    camera_serials_json: Path | None,
) -> str:
    if isinstance(raw_metadata, dict):
        for key in ("wrist_cam_serial", "wrist1_cam_serial"):
            value = raw_metadata.get(key)
            if value:
                return str(value)
    if episode_id is not None and camera_serials_json is not None:
        serials_blob = load_json(camera_serials_json)
        episode_camera_serials = find_episode_entry(serials_blob, episode_id)
        if isinstance(episode_camera_serials, dict):
            for key in ("wrist_cam_serial", "wrist1_cam_serial"):
                value = episode_camera_serials.get(key)
                if value:
                    return str(value)
    raise KeyError("Could not resolve wrist camera serial from raw metadata or camera_serials.json")


def load_raw_wrist_pose_sequence(
    *,
    raw_metadata_json: Path | None,
    raw_trajectory_h5: Path,
    wrist_serial: str,
    pose_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = load_json(raw_metadata_json) if raw_metadata_json is not None else None
    summary: dict[str, Any] = {
        "wrist_serial": wrist_serial,
        "pose_source": pose_source,
    }

    if pose_source == "metadata":
        if not isinstance(metadata, dict) or metadata.get("wrist_cam_extrinsics") is None:
            raise KeyError("wrist_cam_extrinsics is missing from raw metadata.json")
        pose_vector = np.asarray(metadata["wrist_cam_extrinsics"], dtype=np.float64).reshape(6)
        pose_matrix = read_matrix(pose_vector.tolist())
        if pose_matrix is None:
            raise ValueError("Failed to parse metadata wrist_cam_extrinsics")
        joints, _ = load_joint_gripper_from_raw_h5(raw_trajectory_h5)
        pose_sequence = np.repeat(pose_matrix[None, ...], joints.shape[0], axis=0)
        summary["trajectory_frames"] = int(joints.shape[0])
        summary["metadata_pose"] = pose_vector.tolist()
        return pose_sequence, summary

    left_key = f"{wrist_serial}_left"
    offset_key = f"{wrist_serial}_left_gripper_offset"
    with h5py.File(raw_trajectory_h5, "r") as f:
        if "observation" not in f or "camera_extrinsics" not in f["observation"]:
            raise KeyError(f"Could not find observation/camera_extrinsics in {raw_trajectory_h5}")
        camera_extrinsics = f["observation"]["camera_extrinsics"]
        if left_key not in camera_extrinsics:
            available = sorted(camera_extrinsics.keys())[:12]
            raise KeyError(
                f"Could not find wrist camera extrinsics key '{left_key}' in {raw_trajectory_h5}. "
                f"Available keys include: {available}"
            )
        pose_vectors = np.asarray(camera_extrinsics[left_key], dtype=np.float64)
        offset_vectors = None
        if offset_key in camera_extrinsics:
            offset_vectors = np.asarray(camera_extrinsics[offset_key], dtype=np.float64)

    if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6:
        raise ValueError(f"Expected raw wrist poses to have shape [T, 6], got {pose_vectors.shape}")

    pose_matrices = np.repeat(np.eye(4, dtype=np.float64)[None, ...], pose_vectors.shape[0], axis=0)
    for idx, pose_vector in enumerate(pose_vectors):
        matrix = read_matrix(pose_vector.tolist())
        if matrix is None:
            raise ValueError(f"Failed to parse wrist pose row {idx} from {raw_trajectory_h5}")
        pose_matrices[idx] = matrix

    summary["trajectory_frames"] = int(pose_vectors.shape[0])
    summary["first_pose"] = pose_vectors[0].tolist()
    summary["last_pose"] = pose_vectors[-1].tolist()
    if isinstance(metadata, dict) and metadata.get("wrist_cam_extrinsics") is not None:
        metadata_pose = np.asarray(metadata["wrist_cam_extrinsics"], dtype=np.float64).reshape(6)
        summary["metadata_pose"] = metadata_pose.tolist()
        summary["metadata_first_linf_error"] = float(np.max(np.abs(metadata_pose - pose_vectors[0])))
    if offset_vectors is not None:
        summary["gripper_offset_first"] = offset_vectors[0].tolist()
        summary["gripper_offset_mean_abs_step"] = float(np.mean(np.abs(np.diff(offset_vectors, axis=0)))) if offset_vectors.shape[0] > 1 else 0.0
    return pose_matrices, summary


def load_joint_trajectory(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        if "joint_position" in archive:
            data = archive["joint_position"]
        else:
            data = archive[archive.files[0]]
    elif path.suffix == ".json":
        blob = load_json(path)
        if isinstance(blob, dict):
            for key in ("joint_position", "joint_positions", "qpos", "trajectory"):
                if key in blob:
                    data = np.asarray(blob[key], dtype=np.float64)
                    break
            else:
                raise ValueError(f"No joint trajectory key found in {path}")
        else:
            data = np.asarray(blob, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported joint trajectory format: {path.suffix}")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D joint trajectory, got shape {data.shape}")
    return data


def load_gripper_trajectory(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        if "gripper_position" in archive:
            data = archive["gripper_position"]
        else:
            data = archive[archive.files[0]]
    elif path.suffix == ".json":
        blob = load_json(path)
        if isinstance(blob, dict):
            for key in ("gripper_position", "gripper_positions", "trajectory", "qpos"):
                if key in blob:
                    data = np.asarray(blob[key], dtype=np.float64)
                    break
            else:
                raise ValueError(f"No gripper trajectory key found in {path}")
        else:
            data = np.asarray(blob, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported gripper trajectory format: {path.suffix}")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    elif data.ndim > 1:
        data = data.reshape(data.shape[0], -1)[:, 0]
    return data.reshape(-1)


def maybe_resolve_gripper_path(joint_path: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    sibling = joint_path.with_name("gripper_position.npy")
    if sibling.exists():
        return sibling
    return None


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (matrix @ homog.T).T[:, :3]


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6
    pixels = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        pts = points_cam[valid]
        proj = (intrinsics @ pts.T).T
        pixels[valid, 0] = proj[:, 0] / proj[:, 2]
        pixels[valid, 1] = proj[:, 1] / proj[:, 2]
    return pixels, valid


def default_mesh_rgb() -> np.ndarray:
    return np.array([230, 235, 237], dtype=np.uint8)


def robotiq_pad_rgb() -> np.ndarray:
    return np.array([26, 26, 26], dtype=np.uint8)


def is_robotiq_finger_pad_mesh(mesh: Any) -> bool:
    """Detect Robotiq fingertip pad box primitives when URDF material is lost.

    yourdfpy/trimesh can drop the material on primitive box visuals. In the
    Robotiq 85 URDF used here, the two finger pads are box primitives with this
    exact size, while the visual meshes are loaded from DAE files.
    """
    try:
        vertices = np.asarray(getattr(mesh, "vertices", None))
        faces = np.asarray(getattr(mesh, "faces", None))
        extents = np.asarray(getattr(mesh, "extents", None), dtype=np.float64)
    except Exception:
        return False
    if vertices.shape[0] != 8 or faces.shape[0] != 12 or extents.size != 3:
        return False
    return bool(np.allclose(np.sort(extents), np.sort([0.022, 0.00635, 0.0375]), atol=1e-5))


def material_main_rgb(mesh: Any) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is not None:
        for attr in ("main_color", "baseColorFactor", "diffuse"):
            if hasattr(material, attr):
                color = np.asarray(getattr(material, attr))
                if color.size >= 3:
                    color = color.reshape(-1)[:3]
                    if np.issubdtype(color.dtype, np.floating) and np.max(color) <= 1.0:
                        color = color * 255.0
                    return np.clip(np.round(color), 0, 255).astype(np.uint8)
    try:
        color_visual = visual.to_color()
        vertex_colors = np.asarray(getattr(color_visual, "vertex_colors", None))
        if vertex_colors.size >= 3:
            color = vertex_colors.reshape(-1)[:3]
            return np.clip(np.round(color), 0, 255).astype(np.uint8)
    except Exception:
        pass
    return default_mesh_rgb()


def resolve_mesh_rgb(mesh: Any, mesh_index: int, visual_rgbs: list[np.ndarray | None]) -> np.ndarray:
    # DAE visuals can split into many scene geometries, so URDF visual order is
    # not a reliable color source here. Use the mesh material itself, with an
    # explicit fallback only for primitive Robotiq pads whose material is lost.
    if is_robotiq_finger_pad_mesh(mesh):
        return robotiq_pad_rgb()
    return material_main_rgb(mesh)


def color_to_rgb(color: Any) -> np.ndarray | None:
    if color is None:
        return None
    rgba = getattr(color, "rgba", color)
    try:
        values = np.asarray(rgba)
    except Exception:
        return None
    if values.size < 3:
        return None
    values = values.reshape(-1)[:3]
    if np.issubdtype(values.dtype, np.floating) and np.max(values) <= 1.0:
        values = values * 255.0
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def robot_visual_material_rgbs(robot: URDF) -> list[np.ndarray | None]:
    """Return URDF visual material colors in visual traversal order.

    Do not assume this list aligns one-to-one with scene geometries: DAE files
    often expand into multiple submeshes. Primitive Robotiq pads are handled by
    geometry detection in ``resolve_mesh_rgb`` instead.
    """
    rgbs: list[np.ndarray | None] = []
    robot_xml = getattr(robot, "robot", None)
    for link in getattr(robot_xml, "links", []) or []:
        for visual in getattr(link, "visuals", []) or []:
            material = getattr(visual, "material", None)
            rgbs.append(color_to_rgb(getattr(material, "color", None)))
    return rgbs


def resolve_uniform_bgr(color_override: tuple[int, int, int] | None, vertex_colors_rgb: np.ndarray | None) -> tuple[int, int, int]:
    if color_override is not None:
        return tuple(int(v) for v in color_override)
    if vertex_colors_rgb is not None and vertex_colors_rgb.size > 0:
        mean_rgb = np.round(vertex_colors_rgb.mean(axis=0)).astype(np.uint8)
        return tuple(int(v) for v in mean_rgb[::-1])
    return tuple(int(v) for v in default_mesh_rgb()[::-1])


def resolve_render_frame_size(
    frame_size: tuple[int, int],
    target_short_side: int,
) -> tuple[int, int]:
    width, height = frame_size
    if target_short_side <= 0:
        return width, height
    short_side = min(width, height)
    if short_side <= 0 or short_side <= target_short_side:
        return width, height
    scale = float(target_short_side) / float(short_side)
    resized_width = max(2, int(round(width * scale)))
    resized_height = max(2, int(round(height * scale)))
    resized_width -= resized_width % 2
    resized_height -= resized_height % 2
    return resized_width, resized_height


def build_output_frame_indices(
    total_frames: int,
    source_fps: float,
    output_fps: float,
) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((0,), dtype=np.int64)
    if source_fps <= 0.0 or output_fps <= 0.0 or output_fps >= source_fps:
        return np.arange(total_frames, dtype=np.int64)
    output_count = max(1, int(round(total_frames * output_fps / source_fps)))
    indices = np.round(np.linspace(0, total_frames - 1, output_count)).astype(np.int64)
    if indices.size <= 1:
        return indices
    keep = np.ones(indices.shape[0], dtype=bool)
    keep[1:] = indices[1:] != indices[:-1]
    return indices[keep]


def resize_frame_to_size(frame: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def build_pytorch3d_context(
    *,
    frame_size: tuple[int, int],
    base_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    appearance: str,
) -> dict[str, Any]:
    if not HAS_PYTORCH3D:
        raise RuntimeError("PyTorch3D backend requested but import failed")

    width, height = frame_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rotation = torch.as_tensor(base_to_camera[:3, :3], dtype=torch.float32, device=device)[None]
    translation = torch.as_tensor(base_to_camera[:3, 3], dtype=torch.float32, device=device)[None]
    camera_matrix = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)[None]
    image_size = torch.as_tensor([[height, width]], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(
        R=rotation,
        tvec=translation,
        camera_matrix=camera_matrix,
        image_size=image_size,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=0,
        ),
    )
    lights = None
    shader = None
    if appearance == "shaded":
        lights = DirectionalLights(
            device=device,
            direction=((0.2, 0.2, 1.0),),
            ambient_color=((0.6, 0.6, 0.6),),
            diffuse_color=((0.35, 0.35, 0.35),),
            specular_color=((0.05, 0.05, 0.05),),
        )
        shader = HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        )
    return {
        "device": device,
        "appearance": appearance,
        "cameras": cameras,
        "lights": lights,
        "rasterizer": rasterizer,
        "shader": shader,
        "height": height,
        "width": width,
    }


def build_pytorch3d_context_batch(
    *,
    frame_size: tuple[int, int],
    base_to_camera_batch: np.ndarray,
    intrinsics: np.ndarray,
    appearance: str,
) -> dict[str, Any]:
    if not HAS_PYTORCH3D:
        raise RuntimeError("PyTorch3D backend requested but import failed")

    width, height = frame_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_to_camera_batch = np.asarray(base_to_camera_batch, dtype=np.float32)
    if base_to_camera_batch.ndim != 3 or base_to_camera_batch.shape[1:] != (4, 4):
        raise ValueError(f"Expected base_to_camera_batch with shape [B,4,4], got {base_to_camera_batch.shape}")
    batch_size = int(base_to_camera_batch.shape[0])

    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.ndim == 2:
        intrinsics_batch = np.broadcast_to(intrinsics[None, ...], (batch_size, 3, 3)).copy()
    elif intrinsics.ndim == 3 and intrinsics.shape[0] == batch_size:
        intrinsics_batch = intrinsics
    else:
        raise ValueError(f"Expected intrinsics shape [3,3] or [B,3,3], got {intrinsics.shape}")

    rotation = torch.as_tensor(base_to_camera_batch[:, :3, :3], dtype=torch.float32, device=device)
    translation = torch.as_tensor(base_to_camera_batch[:, :3, 3], dtype=torch.float32, device=device)
    camera_matrix = torch.as_tensor(intrinsics_batch, dtype=torch.float32, device=device)
    image_size = torch.as_tensor([[height, width]], dtype=torch.float32, device=device).expand(batch_size, -1)
    cameras = cameras_from_opencv_projection(
        R=rotation,
        tvec=translation,
        camera_matrix=camera_matrix,
        image_size=image_size,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=0,
        ),
    )
    lights = None
    shader = None
    if appearance == "shaded":
        lights = DirectionalLights(
            device=device,
            direction=((0.2, 0.2, 1.0),),
            ambient_color=((0.6, 0.6, 0.6),),
            diffuse_color=((0.35, 0.35, 0.35),),
            specular_color=((0.05, 0.05, 0.05),),
        )
        shader = HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        )
    return {
        "device": device,
        "appearance": appearance,
        "cameras": cameras,
        "lights": lights,
        "rasterizer": rasterizer,
        "shader": shader,
        "height": height,
        "width": width,
    }


def resolve_vertex_rgb_array(
    *,
    vertices_count: int,
    faces: np.ndarray,
    color: tuple[int, int, int] | None,
    vertex_colors_rgb: np.ndarray | None,
) -> np.ndarray:
    if color is None and vertex_colors_rgb is not None:
        return np.asarray(vertex_colors_rgb, dtype=np.float32)
    rgb_color_array = np.asarray(
        resolve_uniform_bgr(color_override=color, vertex_colors_rgb=vertex_colors_rgb)[::-1],
        dtype=np.float32,
    )
    return np.tile(rgb_color_array[None, :], (vertices_count, 1))


def render_mesh_pytorch3d(
    *,
    context: dict[str, Any],
    vertices_base: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int] | None,
    vertex_colors_rgb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    device = context["device"]
    verts = torch.as_tensor(vertices_base, dtype=torch.float32, device=device)[None]
    faces_t = torch.as_tensor(faces, dtype=torch.int64, device=device)[None]
    rgb_color_array = resolve_vertex_rgb_array(
        vertices_count=int(verts.shape[1]),
        faces=faces,
        color=color,
        vertex_colors_rgb=vertex_colors_rgb,
    )
    if context["appearance"] == "mask":
        meshes = Meshes(verts=verts, faces=faces_t)
        fragments = context["rasterizer"](meshes)
        mask = fragments.pix_to_face[0, ..., 0] >= 0
        rendered_bgr = np.zeros((context["height"], context["width"], 3), dtype=np.uint8)
        rendered_bgr[mask.detach().cpu().numpy()] = np.asarray(
            resolve_uniform_bgr(color_override=color, vertex_colors_rgb=vertex_colors_rgb),
            dtype=np.uint8,
        )
        return rendered_bgr, mask.detach().cpu().numpy()

    if context["appearance"] == "unlit":
        meshes = Meshes(verts=verts, faces=faces_t)
        fragments = context["rasterizer"](meshes)
        pix_to_face = fragments.pix_to_face[0, ..., 0]
        mask = pix_to_face >= 0
        face_colors_rgb = torch.as_tensor(
            rgb_color_array[faces].mean(axis=1),
            dtype=torch.float32,
            device=device,
        )
        rendered_rgb = torch.zeros((context["height"], context["width"], 3), dtype=torch.float32, device=device)
        if bool(mask.any()):
            rendered_rgb[mask] = face_colors_rgb[pix_to_face[mask]]
        rendered_rgb_np = rendered_rgb.clamp(0.0, 255.0).to(torch.uint8).detach().cpu().numpy()
        rendered_bgr = cv2.cvtColor(rendered_rgb_np, cv2.COLOR_RGB2BGR)
        return rendered_bgr, mask.detach().cpu().numpy()

    vertex_colors = torch.as_tensor(rgb_color_array, dtype=torch.float32, device=device) / 255.0
    vertex_colors = vertex_colors[None]
    meshes = Meshes(
        verts=verts,
        faces=faces_t,
        textures=TexturesVertex(verts_features=vertex_colors),
    )
    fragments = context["rasterizer"](meshes)
    mask = fragments.pix_to_face[0, ..., 0] >= 0
    rendered = context["shader"](
        fragments,
        meshes,
        cameras=context["cameras"],
        lights=context["lights"],
    )[0, ..., :3].detach().cpu().numpy()
    rendered_bgr = cv2.cvtColor((rendered * 255.0).clip(0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return rendered_bgr, mask.detach().cpu().numpy()


def render_mesh_batch_pytorch3d(
    *,
    context: dict[str, Any],
    vertices_base_batch: np.ndarray,
    faces_batch: np.ndarray,
    color: tuple[int, int, int] | None,
    vertex_colors_rgb_batch: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    device = context["device"]
    vertices_base_batch = np.asarray(vertices_base_batch, dtype=np.float32)
    faces_batch = np.asarray(faces_batch, dtype=np.int64)
    if vertices_base_batch.ndim != 3:
        raise ValueError(f"Expected vertices_base_batch shape [B,V,3], got {vertices_base_batch.shape}")
    if faces_batch.ndim == 2:
        faces_batch = np.broadcast_to(
            faces_batch[None, ...],
            (vertices_base_batch.shape[0], faces_batch.shape[0], faces_batch.shape[1]),
        ).copy()
    if faces_batch.ndim != 3:
        raise ValueError(f"Expected faces_batch shape [B,F,3], got {faces_batch.shape}")

    verts = torch.as_tensor(vertices_base_batch, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces_batch, dtype=torch.int64, device=device)
    if color is None and vertex_colors_rgb_batch is not None:
        rgb_color_array = np.asarray(vertex_colors_rgb_batch, dtype=np.float32)
    else:
        rgb = np.asarray(
            resolve_uniform_bgr(
                color_override=color,
                vertex_colors_rgb=None if vertex_colors_rgb_batch is None else vertex_colors_rgb_batch.reshape(-1, 3),
            )[::-1],
            dtype=np.float32,
        )
        rgb_color_array = np.tile(rgb[None, None, :], (verts.shape[0], verts.shape[1], 1))

    if context["appearance"] == "mask":
        meshes = Meshes(verts=verts, faces=faces_t)
        fragments = context["rasterizer"](meshes)
        masks = fragments.pix_to_face[..., 0] >= 0
        rendered_bgr = np.zeros((verts.shape[0], context["height"], context["width"], 3), dtype=np.uint8)
        mask_np = masks.detach().cpu().numpy()
        mask_color = np.asarray(
            resolve_uniform_bgr(
                color_override=color,
                vertex_colors_rgb=None if vertex_colors_rgb_batch is None else vertex_colors_rgb_batch.reshape(-1, 3),
            ),
            dtype=np.uint8,
        )
        rendered_bgr[mask_np] = mask_color
        return rendered_bgr, mask_np

    if context["appearance"] == "unlit":
        meshes = Meshes(verts=verts, faces=faces_t)
        fragments = context["rasterizer"](meshes)
        pix_to_face = fragments.pix_to_face[..., 0]
        masks = pix_to_face >= 0
        rgb_tensor = torch.as_tensor(rgb_color_array, dtype=torch.float32, device=device)
        face_colors_rgb = torch.stack(
            [rgb_tensor[i][faces_t[i]].mean(dim=1) for i in range(int(verts.shape[0]))],
            dim=0,
        )
        rendered_rgb = torch.zeros(
            (verts.shape[0], context["height"], context["width"], 3),
            dtype=torch.float32,
            device=device,
        )
        for i in range(int(verts.shape[0])):
            mask_i = masks[i]
            if bool(mask_i.any()):
                rendered_rgb[i][mask_i] = face_colors_rgb[i][pix_to_face[i][mask_i]]
        rendered_rgb_np = rendered_rgb.clamp(0.0, 255.0).to(torch.uint8).detach().cpu().numpy()
        rendered_bgr = rendered_rgb_np[..., ::-1].copy()
        return rendered_bgr, masks.detach().cpu().numpy()

    vertex_colors = torch.as_tensor(rgb_color_array, dtype=torch.float32, device=device) / 255.0
    meshes = Meshes(
        verts=verts,
        faces=faces_t,
        textures=TexturesVertex(verts_features=vertex_colors),
    )
    fragments = context["rasterizer"](meshes)
    masks = fragments.pix_to_face[..., 0] >= 0
    rendered = context["shader"](
        fragments,
        meshes,
        cameras=context["cameras"],
        lights=context["lights"],
    )[..., :3].detach().cpu().numpy()
    rendered_rgb = (rendered * 255.0).clip(0.0, 255.0).astype(np.uint8)
    rendered_bgr = rendered_rgb[..., ::-1].copy()
    return rendered_bgr, masks.detach().cpu().numpy()


def camera_pose_opengl_from_base_to_camera(base_to_camera: np.ndarray) -> np.ndarray:
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(base_to_camera) @ cv_to_gl


def pyrender_style_config(style: str) -> dict[str, Any]:
    if style == "matte":
        return {
            "ambient": np.array([0.45, 0.45, 0.45, 1.0]),
            "directional_intensity": 0.65,
            "roughness": 1.0,
            "metallic": 0.0,
            "flags_extra": 0,
            "color_scale": 1.0,
        }
    if style == "ambient":
        return {
            "ambient": np.array([0.62, 0.62, 0.62, 1.0]),
            "directional_intensity": 0.0,
            "roughness": 1.0,
            "metallic": 0.0,
            "flags_extra": 0,
            "color_scale": 1.0,
        }
    if style == "flat":
        return {
            "ambient": np.array([1.0, 1.0, 1.0, 1.0]),
            "directional_intensity": 0.0,
            "roughness": 1.0,
            "metallic": 0.0,
            "flags_extra": "flat",
            "color_scale": 1.0,
        }
    if style == "flat_dim":
        return {
            "ambient": np.array([1.0, 1.0, 1.0, 1.0]),
            "directional_intensity": 0.0,
            "roughness": 1.0,
            "metallic": 0.0,
            "flags_extra": "flat",
            "color_scale": 0.68,
        }
    return {
        "ambient": np.array([0.58, 0.58, 0.58, 1.0]),
        "directional_intensity": 2.0,
        "roughness": 0.75,
        "metallic": 0.0,
        "flags_extra": 0,
        "color_scale": 1.0,
    }


def build_pyrender_context(
    *,
    robot: URDF,
    frame_size: tuple[int, int],
    base_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    style: str,
) -> dict[str, Any]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    width, height = frame_size
    style_config = pyrender_style_config(style)
    scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 1.0]), ambient_light=style_config["ambient"])
    visual_rgbs = robot_visual_material_rgbs(robot)
    parts: list[dict[str, Any]] = []
    for mesh_index, geom_node in enumerate(list(robot.scene.graph.nodes_geometry)):
        mesh = robot.scene.geometry[geom_node]
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            continue
        rgb = resolve_mesh_rgb(mesh, mesh_index, visual_rgbs)
        rgb = np.clip(
            np.round(np.asarray(rgb, dtype=np.float32) * float(style_config["color_scale"])),
            0,
            255,
        ).astype(np.uint8)
        tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0, 1.0),
            metallicFactor=float(style_config["metallic"]),
            roughnessFactor=float(style_config["roughness"]),
        )
        render_mesh = pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)
        transform, _ = robot.scene.graph.get(geom_node)
        node = scene.add(render_mesh, pose=np.asarray(transform, dtype=np.float64))
        parts.append({"geom_node": geom_node, "scene_node": node})

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=20.0)
    camera_pose = camera_pose_opengl_from_base_to_camera(base_to_camera)
    camera_node = scene.add(camera, pose=camera_pose)
    light_node = None
    directional_intensity = float(style_config["directional_intensity"])
    if directional_intensity > 0.0:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=directional_intensity)
        light_node = scene.add(light, pose=camera_pose)
    render_flags = pyrender.RenderFlags.RGBA
    if style_config["flags_extra"] == "flat":
        render_flags |= pyrender.RenderFlags.FLAT
    return {
        "pyrender": pyrender,
        "scene": scene,
        "parts": parts,
        "camera_node": camera_node,
        "light_node": light_node,
        "renderer": pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height),
        "render_flags": render_flags,
        "style": style,
    }


def update_pyrender_context(
    *,
    context: dict[str, Any],
    robot: URDF,
    joint_values: np.ndarray,
    base_to_camera: np.ndarray,
) -> None:
    robot.update_cfg(joint_values)
    scene = context["scene"]
    for part in context["parts"]:
        transform, _ = robot.scene.graph.get(part["geom_node"])
        scene.set_pose(part["scene_node"], pose=np.asarray(transform, dtype=np.float64))
    camera_pose = camera_pose_opengl_from_base_to_camera(base_to_camera)
    scene.set_pose(context["camera_node"], pose=camera_pose)
    if context["light_node"] is not None:
        scene.set_pose(context["light_node"], pose=camera_pose)


def render_pyrender_frame(context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    color_rgb, depth = context["renderer"].render(context["scene"], flags=context["render_flags"])
    color_rgb = color_rgb[..., :3]
    mask = depth > 0.0
    return cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), mask


def close_pyrender_context(context: dict[str, Any] | None) -> None:
    if context is None:
        return
    renderer = context.get("renderer")
    if renderer is not None:
        renderer.delete()


def blend_render(
    *,
    background_bgr: np.ndarray,
    rendered_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not np.any(mask):
        return background_bgr.copy()
    composite = background_bgr.copy()
    blended = (
        alpha * rendered_bgr[mask].astype(np.float32)
        + (1.0 - alpha) * background_bgr[mask].astype(np.float32)
    )
    composite[mask] = np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)
    return composite


def composite_render(
    *,
    background_bgr: np.ndarray,
    rendered_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if not np.any(mask):
        return background_bgr.copy()
    composite = background_bgr.copy()
    composite[mask] = rendered_bgr[mask]
    return composite


def subsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    keep = np.linspace(0, faces.shape[0] - 1, max_faces, dtype=np.int64)
    return faces[keep]


def get_robot_mesh(robot: URDF, joint_values: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot.update_cfg(joint_values)
    scene_meshes = robot.scene.dump(concatenate=False)
    visual_rgbs = robot_visual_material_rgbs(robot)
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_colors_parts: list[np.ndarray] = []
    vertex_offset = 0

    for mesh_index, mesh in enumerate(scene_meshes):
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            continue
        rgb = resolve_mesh_rgb(mesh, mesh_index, visual_rgbs)
        vertex_colors = np.tile(rgb[None, :], (vertices.shape[0], 1))

        vertices_parts.append(vertices)
        faces_parts.append(faces + vertex_offset)
        vertex_colors_parts.append(vertex_colors)
        vertex_offset += vertices.shape[0]

    if not vertices_parts:
        raise RuntimeError("URDF scene contains no renderable meshes")

    vertices = np.concatenate(vertices_parts, axis=0)
    faces = np.concatenate(faces_parts, axis=0)
    vertex_colors = np.concatenate(vertex_colors_parts, axis=0)
    faces = subsample_faces(faces, max_faces)
    return vertices, faces, vertex_colors


def build_robot_mesh_cache(robot: URDF, max_faces: int) -> dict[str, Any] | None:
    """Cache static URDF geometry so per-frame rendering only applies graph transforms."""
    try:
        nodes = list(robot.scene.graph.nodes_geometry)
    except Exception:
        return None
    if not nodes:
        return None

    visual_rgbs = robot_visual_material_rgbs(robot)
    parts: list[dict[str, Any]] = []
    faces_parts: list[np.ndarray] = []
    vertex_colors_parts: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_index, node in enumerate(nodes):
        try:
            mesh = robot.scene.geometry[node]
        except Exception:
            return None
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            continue
        rgb = resolve_mesh_rgb(mesh, mesh_index, visual_rgbs)
        parts.append({"node": node, "vertices_local": vertices})
        faces_parts.append(faces + vertex_offset)
        vertex_colors_parts.append(np.tile(rgb[None, :], (vertices.shape[0], 1)))
        vertex_offset += vertices.shape[0]

    if not parts:
        return None
    faces = np.concatenate(faces_parts, axis=0)
    return {
        "parts": parts,
        "faces": subsample_faces(faces, max_faces),
        "vertex_colors": np.concatenate(vertex_colors_parts, axis=0),
    }


def get_robot_mesh_cached(
    robot: URDF,
    cache: dict[str, Any] | None,
    joint_values: np.ndarray,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache is None:
        return get_robot_mesh(robot, joint_values, max_faces=max_faces)

    robot.update_cfg(joint_values)
    vertices_parts: list[np.ndarray] = []
    try:
        for part in cache["parts"]:
            transform, _ = robot.scene.graph.get(part["node"])
            vertices_local = np.asarray(part["vertices_local"], dtype=np.float64)
            vertices_parts.append(vertices_local @ transform[:3, :3].T + transform[:3, 3])
    except Exception:
        return get_robot_mesh(robot, joint_values, max_faces=max_faces)
    return (
        np.concatenate(vertices_parts, axis=0),
        np.asarray(cache["faces"], dtype=np.int32),
        np.asarray(cache["vertex_colors"], dtype=np.uint8),
    )


def get_gripper_joint_spec(robot: URDF) -> tuple[int, float, float] | None:
    for index, joint in enumerate(robot.actuated_joints):
        joint_name = joint.name.lower()
        if "finger_joint" not in joint_name and "gripper" not in joint_name:
            continue
        joint_limit = getattr(joint, "limit", None)
        lower = float(getattr(joint_limit, "lower", 0.0) or 0.0)
        upper = float(getattr(joint_limit, "upper", 1.0) or 1.0)
        return index, lower, upper
    return None


def normalize_gripper_signal(values: np.ndarray, invert: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values, dtype=np.float64)

    finite_values = values[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if min_value >= -1e-6 and max_value <= 0.043:
        normalized = values / 0.04
    elif min_value >= -1e-6 and max_value <= 0.09:
        normalized = values / 0.085
    elif min_value >= -1e-6 and max_value <= 1.05:
        normalized = values
    elif min_value >= -1e-6 and max_value <= 255.0:
        normalized = values / 255.0
    else:
        span = max_value - min_value
        normalized = (values - min_value) / span if span > 1e-8 else np.zeros_like(values)

    normalized = np.clip(normalized, 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized


def parse_index_csv(value: str) -> list[int]:
    if not value.strip():
        return []
    parsed: list[int] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        parsed.append(int(item))
    return parsed


def resolve_manual_window(
    total_frames: int,
    start: int | None,
    end: int | None,
) -> tuple[int, int] | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("--perturb-window-start and --perturb-window-end must be provided together")
    if total_frames <= 0:
        return 0, -1
    start_idx = max(0, min(int(start), total_frames - 1))
    end_idx = max(0, min(int(end), total_frames - 1))
    if end_idx < start_idx:
        raise ValueError(
            f"Invalid perturb window: start={start_idx} end={end_idx} for total_frames={total_frames}"
        )
    return start_idx, end_idx


def detect_pick_window(
    *,
    total_frames: int,
    gripper: np.ndarray | None,
    manual_window: tuple[int, int] | None,
) -> tuple[int, int, str, dict[str, Any]]:
    details: dict[str, Any] = {"total_frames": int(total_frames)}
    if total_frames <= 0:
        return 0, -1, "empty", details
    if manual_window is not None:
        start_idx, end_idx = manual_window
        details["manual_window"] = [int(start_idx), int(end_idx)]
        return start_idx, end_idx, "manual", details

    if gripper is not None:
        values = np.asarray(gripper, dtype=np.float64).reshape(-1)
        usable = min(total_frames, values.shape[0])
        if usable >= 3:
            deriv = np.diff(values[:usable])
            abs_deriv = np.abs(deriv)
            peak_idx = int(np.argmax(abs_deriv))
            peak_value = float(abs_deriv[peak_idx])
            threshold = max(0.05, 0.35 * float(np.max(abs_deriv)))
            details["gripper_peak_frame"] = int(peak_idx)
            details["gripper_peak_abs_delta"] = peak_value
            details["gripper_threshold"] = threshold
            if peak_value >= threshold:
                start_idx = max(0, peak_idx - 6)
                end_idx = min(total_frames - 1, peak_idx + 10)
                return start_idx, end_idx, "auto_gripper", details

    fallback_len = max(4, int(round(total_frames * 0.2)))
    center = total_frames // 2
    start_idx = max(0, center - fallback_len // 2)
    end_idx = min(total_frames - 1, start_idx + fallback_len - 1)
    details["fallback_window"] = [int(start_idx), int(end_idx)]
    return start_idx, end_idx, "fallback_center", details


def build_cosine_window_envelope(
    *,
    total_frames: int,
    start_idx: int,
    end_idx: int,
    pre_frames: int,
    post_frames: int,
) -> np.ndarray:
    envelope = np.zeros(total_frames, dtype=np.float64)
    if total_frames <= 0 or end_idx < start_idx:
        return envelope

    hold_start = max(0, start_idx)
    hold_end = min(total_frames - 1, end_idx)
    envelope[hold_start : hold_end + 1] = 1.0

    if pre_frames > 0:
        pre_start = max(0, hold_start - pre_frames)
        pre_len = hold_start - pre_start
        if pre_len > 0:
            phases = np.arange(1, pre_len + 1, dtype=np.float64) / float(pre_len + 1)
            envelope[pre_start:hold_start] = 0.5 - 0.5 * np.cos(np.pi * phases)

    if post_frames > 0:
        post_end = min(total_frames - 1, hold_end + post_frames)
        post_len = post_end - hold_end
        if post_len > 0:
            phases = np.arange(1, post_len + 1, dtype=np.float64) / float(post_len + 1)
            envelope[hold_end + 1 : post_end + 1] = 0.5 + 0.5 * np.cos(np.pi * phases)

    return envelope


def get_arm_joint_limits(
    robot: URDF,
    arm_dim: int,
    gripper_spec: tuple[int, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    lowers = np.full(arm_dim, -np.inf, dtype=np.float64)
    uppers = np.full(arm_dim, np.inf, dtype=np.float64)
    gripper_index = None if gripper_spec is None else int(gripper_spec[0])
    for joint_index, joint in enumerate(robot.actuated_joints[:arm_dim]):
        if gripper_index is not None and joint_index == gripper_index:
            continue
        joint_limit = getattr(joint, "limit", None)
        lower = getattr(joint_limit, "lower", None)
        upper = getattr(joint_limit, "upper", None)
        if lower is not None:
            lowers[joint_index] = float(lower)
        if upper is not None:
            uppers[joint_index] = float(upper)
    return lowers, uppers


def clip_joints_to_limits(
    joints: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> tuple[np.ndarray, int]:
    clipped = np.asarray(joints, dtype=np.float64).copy()
    clamp_count = 0
    for joint_index in range(clipped.shape[1]):
        lower = float(lower_limits[joint_index])
        upper = float(upper_limits[joint_index])
        if np.isfinite(lower):
            before = clipped[:, joint_index].copy()
            clipped[:, joint_index] = np.maximum(clipped[:, joint_index], lower)
            clamp_count += int(np.count_nonzero(before != clipped[:, joint_index]))
        if np.isfinite(upper):
            before = clipped[:, joint_index].copy()
            clipped[:, joint_index] = np.minimum(clipped[:, joint_index], upper)
            clamp_count += int(np.count_nonzero(before != clipped[:, joint_index]))
    return clipped, clamp_count


def apply_joint_trajectory_perturbation(
    *,
    robot: URDF,
    joints: np.ndarray,
    gripper: np.ndarray | None,
    gripper_spec: tuple[int, float, float] | None,
    mode: str,
    seed: int,
    joint_indices: list[int],
    max_deg: float,
    manual_window: tuple[int, int] | None,
    pre_frames: int,
    post_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "enabled": mode != "none",
        "mode": mode,
        "seed": int(seed),
    }
    original_joints = np.asarray(joints, dtype=np.float64)
    if mode == "none":
        metadata["reason"] = "disabled"
        return original_joints, metadata
    if mode != "pick_joint_offset":
        raise ValueError(f"Unsupported perturb mode: {mode}")
    if original_joints.ndim != 2:
        raise ValueError(f"Expected joints to be 2D, got shape {original_joints.shape}")
    if original_joints.shape[0] <= 0 or original_joints.shape[1] <= 0:
        metadata["enabled"] = False
        metadata["reason"] = "empty_joints"
        return original_joints, metadata

    arm_dim = int(original_joints.shape[1])
    unique_indices: list[int] = []
    for joint_index in joint_indices:
        if joint_index < 0 or joint_index >= arm_dim:
            raise ValueError(
                f"Perturb joint index {joint_index} is out of range for arm_dim={arm_dim}"
            )
        if gripper_spec is not None and int(gripper_spec[0]) == joint_index:
            continue
        if joint_index not in unique_indices:
            unique_indices.append(joint_index)
    if not unique_indices:
        metadata["enabled"] = False
        metadata["reason"] = "no_valid_joint_indices"
        metadata["joint_indices"] = []
        return original_joints, metadata

    total_frames = int(original_joints.shape[0])
    start_idx, end_idx, window_source, window_details = detect_pick_window(
        total_frames=total_frames,
        gripper=gripper,
        manual_window=manual_window,
    )
    envelope = build_cosine_window_envelope(
        total_frames=total_frames,
        start_idx=start_idx,
        end_idx=end_idx,
        pre_frames=max(0, int(pre_frames)),
        post_frames=max(0, int(post_frames)),
    )
    rng = np.random.default_rng(int(seed))
    max_rad = float(np.deg2rad(max(0.0, float(max_deg))))
    magnitudes = rng.uniform(0.35 * max_rad, max_rad, size=len(unique_indices)) if max_rad > 0.0 else np.zeros(
        len(unique_indices),
        dtype=np.float64,
    )
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=len(unique_indices))
    delta_selected = signs * magnitudes
    delta_vector = np.zeros(arm_dim, dtype=np.float64)
    for offset_index, joint_index in enumerate(unique_indices):
        delta_vector[joint_index] = delta_selected[offset_index]

    perturbed = original_joints + envelope[:, None] * delta_vector[None, :]
    lower_limits, upper_limits = get_arm_joint_limits(
        robot=robot,
        arm_dim=arm_dim,
        gripper_spec=gripper_spec,
    )
    perturbed, clamp_count = clip_joints_to_limits(
        joints=perturbed,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    metadata.update(
        {
            "joint_indices": [int(v) for v in unique_indices],
            "delta_q_rad": [float(delta_vector[idx]) for idx in unique_indices],
            "delta_q_deg": [float(np.rad2deg(delta_vector[idx])) for idx in unique_indices],
            "max_deg": float(max_deg),
            "window": {
                "start": int(start_idx),
                "end": int(end_idx),
                "source": window_source,
                "details": window_details,
            },
            "pre_frames": int(max(0, int(pre_frames))),
            "post_frames": int(max(0, int(post_frames))),
            "applied_frame_range": [
                int(np.argmax(envelope > 0.0)) if bool(np.any(envelope > 0.0)) else 0,
                int(total_frames - 1 - np.argmax((envelope > 0.0)[::-1])) if bool(np.any(envelope > 0.0)) else -1,
            ],
            "envelope_max": float(np.max(envelope)),
            "joint_limit_clamp_count": int(clamp_count),
        }
    )
    return perturbed, metadata


def compose_joint_values(
    robot: URDF,
    arm_joint_values: np.ndarray,
    gripper_signal: float | None,
    gripper_spec: tuple[int, float, float] | None,
) -> np.ndarray:
    arm_joint_values = np.asarray(arm_joint_values, dtype=np.float64).reshape(-1)
    target_dim = int(robot.num_actuated_joints)
    if arm_joint_values.shape[0] >= target_dim:
        return arm_joint_values[:target_dim]

    padded = np.zeros(target_dim, dtype=np.float64)
    padded[: arm_joint_values.shape[0]] = arm_joint_values
    if gripper_signal is None or gripper_spec is None:
        return padded

    gripper_index, lower, upper = gripper_spec
    if gripper_index >= arm_joint_values.shape[0]:
        padded[gripper_index] = lower + float(gripper_signal) * (upper - lower)
    return padded


def get_video_meta(video_path: Path) -> tuple[cv2.VideoCapture, int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, width, height, fps, frames


def scale_intrinsics_to_frame(
    intrinsics: np.ndarray,
    calibration_image_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> np.ndarray:
    src_width, src_height = calibration_image_size
    dst_width, dst_height = frame_size
    if src_width <= 0 or src_height <= 0:
        return intrinsics.copy()
    scaled = intrinsics.copy().astype(np.float64)
    scale_x = dst_width / float(src_width)
    scale_y = dst_height / float(src_height)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def render_video(
    video_path: Path,
    output_path: Path | None,
    comparison_output_path: Path | None,
    robot_only_output_path: Path | None,
    rgb_output_path: Path | None,
    robot: URDF,
    joints: np.ndarray,
    gripper: np.ndarray | None,
    base_to_camera: np.ndarray,
    base_to_camera_sequence: np.ndarray | None,
    intrinsics: np.ndarray,
    calibration_image_size: tuple[int, int],
    fps_override: float | None,
    color: tuple[int, int, int] | None,
    alpha: float,
    renderer_backend: str,
    appearance: str,
    max_faces: int,
    frame_limit: int | None,
    frame_step: int,
    target_short_side: int,
    robot_only_render_batch_size: int = 1,
    pyrender_style: str = "flat",
    robot_only_background: str = "white",
) -> dict[str, Any]:
    if renderer_backend not in {"pytorch3d", "pyrender", "pyrender_persistent"}:
        raise ValueError(f"Unsupported renderer backend: {renderer_backend}")
    use_pyrender = renderer_backend in {"pyrender", "pyrender_persistent"}
    if output_path is None and comparison_output_path is None and robot_only_output_path is None and rgb_output_path is None:
        raise ValueError("At least one output path must be provided.")
    cap, source_width, source_height, fps, frame_count = get_video_meta(video_path)
    width, height = resolve_render_frame_size(
        frame_size=(source_width, source_height),
        target_short_side=target_short_side,
    )
    render_intrinsics = scale_intrinsics_to_frame(
        intrinsics=intrinsics,
        calibration_image_size=calibration_image_size,
        frame_size=(width, height),
    )
    total_frames = min(frame_count if frame_count > 0 else joints.shape[0], joints.shape[0])
    if gripper is not None:
        total_frames = min(total_frames, gripper.shape[0])
    if base_to_camera_sequence is not None:
        total_frames = min(total_frames, base_to_camera_sequence.shape[0])
    if frame_limit is not None:
        total_frames = min(total_frames, frame_limit)
    output_fps = float(fps_override or fps or 30.0)
    output_frame_indices = build_output_frame_indices(
        total_frames=total_frames,
        source_fps=float(fps or 0.0),
        output_fps=output_fps,
    )
    output_frame_total = int(output_frame_indices.shape[0])
    gripper_spec = get_gripper_joint_spec(robot)
    robot_mesh_cache = None if use_pyrender else build_robot_mesh_cache(robot, max_faces=max_faces)

    overlay_writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = VideoWriter(
            path=output_path,
            fps=output_fps,
            frame_size=(width, height),
            input_colorspace="bgr",
        )
    rgb_writer = None
    if rgb_output_path is not None:
        rgb_output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_writer = VideoWriter(
            path=rgb_output_path,
            fps=output_fps,
            frame_size=(width, height),
            input_colorspace="bgr",
        )
    robot_only_writer = None
    if robot_only_output_path is not None:
        robot_only_output_path.parent.mkdir(parents=True, exist_ok=True)
        robot_only_writer = VideoWriter(
            path=robot_only_output_path,
            fps=output_fps,
            frame_size=(width, height),
            input_colorspace="bgr",
        )
    comparison_writer = None
    if comparison_output_path is not None:
        comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_writer = VideoWriter(
            path=comparison_output_path,
            fps=output_fps,
            frame_size=(width * 3, height),
            input_colorspace="bgr",
        )

    render_context = None
    pyrender_context = None
    if use_pyrender:
        if color is not None:
            raise ValueError("pyrender backend does not support --color override yet.")
        if max_faces > 0:
            raise ValueError("pyrender backend does not support --max-faces; use a low-poly URDF instead.")
        pyrender_context = build_pyrender_context(
            robot=robot,
            frame_size=(width, height),
            base_to_camera=base_to_camera,
            intrinsics=render_intrinsics,
            style=pyrender_style,
        )
    elif base_to_camera_sequence is None:
        render_context = build_pytorch3d_context(
            frame_size=(width, height),
            base_to_camera=base_to_camera,
            intrinsics=render_intrinsics,
            appearance=appearance,
        )
    background_fill = 127 if robot_only_background == "gray" else 255
    robot_only_background_bgr = np.full((height, width, 3), background_fill, dtype=np.uint8)
    render_outputs_requested = overlay_writer is not None or robot_only_writer is not None or comparison_writer is not None
    rendered_frames = 0
    reused_frames = 0
    selected_frames = 0
    progress_label = next(
        (
            path.name
            for path in (output_path, robot_only_output_path, comparison_output_path, rgb_output_path)
            if path is not None
        ),
        "render",
    )
    progress_start_time = time.time()
    progress_interval = 10
    render_batch_size = max(1, int(robot_only_render_batch_size))
    pending_entries: list[dict[str, Any]] = []
    pending_render_count = 0
    static_batch_context_cache: dict[int, dict[str, Any]] = {}

    def report_processed_progress(processed_frames: int) -> None:
        if (
            processed_frames == 1
            or processed_frames == output_frame_total
            or processed_frames % progress_interval == 0
        ):
            report_render_progress(
                label=progress_label,
                processed_frames=processed_frames,
                total_frames=max(1, output_frame_total),
                start_time=progress_start_time,
                rendered_frames=rendered_frames,
                reused_frames=reused_frames,
            )

    def render_entries(render_entries: list[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray]]:
        if not render_entries:
            return []
        if use_pyrender:
            if pyrender_context is None:
                raise RuntimeError("pyrender context was not initialized.")
            outputs: list[tuple[np.ndarray, np.ndarray]] = []
            for entry in render_entries:
                update_pyrender_context(
                    context=pyrender_context,
                    robot=robot,
                    joint_values=entry["joint_values"],
                    base_to_camera=entry["base_to_camera"],
                )
                outputs.append(render_pyrender_frame(pyrender_context))
            return outputs
        if len(render_entries) == 1:
            entry = render_entries[0]
            current_render_context = render_context
            if current_render_context is None:
                current_render_context = build_pytorch3d_context(
                    frame_size=(width, height),
                    base_to_camera=entry["base_to_camera"],
                    intrinsics=render_intrinsics,
                    appearance=appearance,
                )
            return [
                render_mesh_pytorch3d(
                    context=current_render_context,
                    vertices_base=entry["vertices"],
                    faces=entry["faces"],
                    color=color,
                    vertex_colors_rgb=entry["vertex_colors_rgb"],
                )
            ]

        try:
            vertices_batch = np.stack([entry["vertices"] for entry in render_entries], axis=0)
            faces_batch = np.stack([entry["faces"] for entry in render_entries], axis=0)
            vertex_colors_batch = np.stack([entry["vertex_colors_rgb"] for entry in render_entries], axis=0)
        except ValueError:
            outputs: list[tuple[np.ndarray, np.ndarray]] = []
            for entry in render_entries:
                current_render_context = render_context
                if current_render_context is None:
                    current_render_context = build_pytorch3d_context(
                        frame_size=(width, height),
                        base_to_camera=entry["base_to_camera"],
                        intrinsics=render_intrinsics,
                        appearance=appearance,
                    )
                outputs.append(
                    render_mesh_pytorch3d(
                        context=current_render_context,
                        vertices_base=entry["vertices"],
                        faces=entry["faces"],
                        color=color,
                        vertex_colors_rgb=entry["vertex_colors_rgb"],
                    )
                )
            return outputs

        batch_size = int(vertices_batch.shape[0])
        if base_to_camera_sequence is None:
            batch_context = static_batch_context_cache.get(batch_size)
            if batch_context is None:
                base_to_camera_batch = np.repeat(base_to_camera[None, ...], batch_size, axis=0)
                batch_context = build_pytorch3d_context_batch(
                    frame_size=(width, height),
                    base_to_camera_batch=base_to_camera_batch,
                    intrinsics=render_intrinsics,
                    appearance=appearance,
                )
                static_batch_context_cache[batch_size] = batch_context
        else:
            base_to_camera_batch = np.stack([entry["base_to_camera"] for entry in render_entries], axis=0)
            batch_context = build_pytorch3d_context_batch(
                frame_size=(width, height),
                base_to_camera_batch=base_to_camera_batch,
                intrinsics=render_intrinsics,
                appearance=appearance,
            )
        rendered_batch, mask_batch = render_mesh_batch_pytorch3d(
            context=batch_context,
            vertices_base_batch=vertices_batch,
            faces_batch=faces_batch,
            color=color,
            vertex_colors_rgb_batch=vertex_colors_batch,
        )
        return [
            (rendered_batch[index], mask_batch[index])
            for index in range(int(rendered_batch.shape[0]))
        ]

    def flush_pending_entries() -> None:
        nonlocal pending_entries, pending_render_count, rendered_frames, reused_frames
        if not pending_entries:
            return
        render_entries_only = [entry for entry in pending_entries if entry["kind"] == "render"]
        render_outputs = render_entries(render_entries_only)
        render_output_iter = iter(render_outputs)

        for entry in pending_entries:
            frame = entry["frame"]
            if entry["kind"] == "reuse":
                if overlay_writer is not None:
                    overlay_writer.write(frame)
                if robot_only_writer is not None:
                    robot_only_writer.write(robot_only_background_bgr)
                if comparison_writer is not None:
                    comparison_writer.write(np.concatenate([robot_only_background_bgr, frame, frame], axis=1))
                reused_frames += 1
            else:
                rendered_bgr, mask = next(render_output_iter)
                urdf_only = None
                if robot_only_writer is not None or comparison_writer is not None:
                    urdf_only = composite_render(
                        background_bgr=robot_only_background_bgr,
                        rendered_bgr=rendered_bgr,
                        mask=mask,
                    )
                overlay = None
                if overlay_writer is not None or comparison_writer is not None:
                    overlay = blend_render(
                        background_bgr=frame,
                        rendered_bgr=rendered_bgr,
                        mask=mask,
                        alpha=alpha,
                    )
                if overlay_writer is not None and overlay is not None:
                    overlay_writer.write(overlay)
                if robot_only_writer is not None:
                    if urdf_only is None:
                        raise RuntimeError("robot_only output requested but URDF-only frame was not generated.")
                    robot_only_writer.write(urdf_only)
                if comparison_writer is not None:
                    if urdf_only is None or overlay is None:
                        raise RuntimeError("comparison output requested but its source frames were not generated.")
                    comparison_writer.write(np.concatenate([urdf_only, overlay, frame], axis=1))
                rendered_frames += 1
            report_processed_progress(rendered_frames + reused_frames)

        pending_entries = []
        pending_render_count = 0

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if selected_frames >= output_frame_total:
            break
        if frame_idx != int(output_frame_indices[selected_frames]):
            continue
        frame = resize_frame_to_size(frame, (width, height))
        if rgb_writer is not None:
            rgb_writer.write(frame)
        if not render_outputs_requested:
            selected_frames += 1
            report_processed_progress(selected_frames)
            continue
        if selected_frames % frame_step != 0:
            pending_entries.append({"kind": "reuse", "frame": frame})
            selected_frames += 1
            continue
        joint_values = compose_joint_values(
            robot=robot,
            arm_joint_values=joints[frame_idx],
            gripper_signal=None if gripper is None else float(gripper[frame_idx]),
            gripper_spec=gripper_spec,
        )
        entry: dict[str, Any] = {
            "kind": "render",
            "frame": frame,
            "base_to_camera": base_to_camera if base_to_camera_sequence is None else base_to_camera_sequence[frame_idx],
        }
        if use_pyrender:
            entry["joint_values"] = joint_values
        else:
            vertices, faces, vertex_colors_rgb = get_robot_mesh_cached(
                robot,
                robot_mesh_cache,
                joint_values,
                max_faces=max_faces,
            )
            entry.update(
                {
                    "vertices": vertices,
                    "faces": faces,
                    "vertex_colors_rgb": vertex_colors_rgb,
                }
            )
        pending_entries.append(entry)
        pending_render_count += 1
        selected_frames += 1
        if pending_render_count >= render_batch_size:
            flush_pending_entries()

    cap.release()
    flush_pending_entries()
    if overlay_writer is not None:
        overlay_writer.close()
    if rgb_writer is not None:
        rgb_writer.close()
    if robot_only_writer is not None:
        robot_only_writer.close()
    if comparison_writer is not None:
        comparison_writer.close()
    close_pyrender_context(pyrender_context)
    report_render_progress(
        label=progress_label,
        processed_frames=rendered_frames + reused_frames,
        total_frames=max(1, output_frame_total),
        start_time=progress_start_time,
        rendered_frames=rendered_frames,
        reused_frames=reused_frames,
        force=True,
    )
    return {
        "output_path": str(output_path),
        "rgb_output_path": str(rgb_output_path) if rgb_output_path is not None else None,
        "comparison_output_path": str(comparison_output_path) if comparison_output_path is not None else None,
        "robot_only_output_path": str(robot_only_output_path) if robot_only_output_path is not None else None,
        "video_frames_seen": total_frames,
        "output_frames_written": rendered_frames + reused_frames,
        "source_fps": float(fps or 0.0),
        "output_fps": output_fps,
        "rendered_frames": rendered_frames,
        "reused_frames": reused_frames,
        "source_frame_size": [source_width, source_height],
        "output_frame_size": [width, height],
        "robot_dof_used": int(robot.num_actuated_joints),
        "gripper_joint_used": gripper_spec[0] if gripper_spec is not None else None,
        "renderer_backend": renderer_backend,
        "appearance": appearance,
        "pyrender_style": pyrender_style if use_pyrender else None,
        "robot_only_background": robot_only_background,
        "target_short_side": target_short_side,
        "robot_only_render_batch_size": render_batch_size,
        "uses_robot_mesh_cache": robot_mesh_cache is not None,
        "uses_dynamic_camera": base_to_camera_sequence is not None,
    }


def generate_demo_inputs(
    output_dir: Path,
    image_size: tuple[int, int],
    urdf_path: Path,
) -> dict[str, Path | str]:
    width, height = image_size
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "demo_input.mp4"
    joint_path = output_dir / "demo_joints.npy"
    cam2base_path = output_dir / "demo_cam2base.json"
    intrinsics_path = output_dir / "demo_intrinsics.json"

    fps = 15.0
    writer = VideoWriter(
        path=video_path,
        fps=fps,
        frame_size=(width, height),
        input_colorspace="bgr",
    )

    num_frames = 45
    for idx in range(num_frames):
        frame = np.full((height, width, 3), 240, dtype=np.uint8)
        cv2.putText(
            frame,
            f"Synthetic DROID Overlay Demo {idx:03d}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.close()

    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    joints = np.stack(
        [
            0.35 * np.sin(2.0 * np.pi * t),
            -0.45 + 0.20 * np.sin(2.0 * np.pi * t + 0.3),
            0.25 * np.sin(2.0 * np.pi * t + 0.7),
            -1.60 + 0.15 * np.sin(2.0 * np.pi * t + 1.4),
            0.20 * np.sin(2.0 * np.pi * t + 2.0),
            1.60 + 0.10 * np.sin(2.0 * np.pi * t + 2.8),
            0.35 * np.sin(2.0 * np.pi * t + 3.4),
            np.full_like(t, 0.02),
        ],
        axis=1,
    )
    np.save(joint_path, joints)

    episode_id = "demo_episode"
    camera_name = "exterior_image_1_left"
    base_to_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.00],
            [0.0, 1.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 1.80],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cam2base = np.linalg.inv(base_to_camera)
    intrinsics = np.array(
        [[620.0, 0.0, width / 2.0], [0.0, 620.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with cam2base_path.open("w", encoding="utf-8") as f:
        json.dump({episode_id: {camera_name: {"cam2base": cam2base.tolist()}}}, f, indent=2)
    with intrinsics_path.open("w", encoding="utf-8") as f:
        json.dump({episode_id: {camera_name: {"K": intrinsics.tolist()}}}, f, indent=2)

    return {
        "video_path": video_path,
        "joint_path": joint_path,
        "cam2base_json": cam2base_path,
        "intrinsics_json": intrinsics_path,
        "episode_id": episode_id,
        "camera_name": camera_name,
        "urdf_path": urdf_path,
    }


def main() -> None:
    args = parse_args()

    if args.demo_output_dir is not None:
        defaults = generate_demo_inputs(
            output_dir=args.demo_output_dir,
            image_size=(args.image_size[0], args.image_size[1]),
            urdf_path=args.urdf_path,
        )
        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
        if args.output_path == Path("overlay.mp4"):
            args.output_path = args.demo_output_dir / "demo_overlay.mp4"
    elif args.episode_meta_json is not None:
        defaults = load_episode_meta_defaults(args.episode_meta_json)
        if args.episode_id is None and defaults.get("episode_id"):
            args.episode_id = defaults["episode_id"]
        if args.camera_name == "exterior_image_1_left" and defaults.get("camera_name"):
            args.camera_name = defaults["camera_name"]
        if args.gripper_path is None and defaults.get("gripper_path"):
            args.gripper_path = defaults["gripper_path"]
        if args.urdf_path == DEFAULT_URDF_PATH and defaults.get("urdf_path"):
            args.urdf_path = defaults["urdf_path"]

    if args.raw_trajectory_h5 is not None and args.joint_path is None:
        pass
    else:
        args.gripper_path = maybe_resolve_gripper_path(args.joint_path, args.gripper_path)

    required = ["video_path", "urdf_path", "intrinsics_json", "episode-id", "camera_name"]
    if args.raw_trajectory_h5 is None and args.joint_path is None:
        required.append("joint_path")
    if not args.use_raw_wrist_extrinsics:
        required.append("cam2base_json")
    if args.use_raw_wrist_extrinsics:
        required.extend(["raw_metadata_json", "raw_trajectory_h5"])
    missing = []
    for name in required:
        attr_name = name.replace("-", "_")
        if getattr(args, attr_name, None) is None:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if args.use_raw_wrist_extrinsics and args.camera_name not in ("wrist_image_left", "auto"):
        raise ValueError("--use-raw-wrist-extrinsics currently only supports camera_name='wrist_image_left' or 'auto'")

    resolved_urdf_path, resolved_tool_adjust = materialize_tool_adjusted_urdf(
        urdf_path=args.urdf_path,
        tool_adjust_json=args.tool_adjust_json,
    )
    robot = URDF.load(str(resolved_urdf_path))
    if args.raw_trajectory_h5 is not None and args.joint_path is None:
        joints, raw_gripper = load_joint_gripper_from_raw_h5(args.raw_trajectory_h5)
        gripper = normalize_gripper_signal(
            raw_gripper,
            invert=bool(args.invert_gripper),
        )
    else:
        joints = load_joint_trajectory(args.joint_path)
        gripper = None
        if args.gripper_path is not None and args.gripper_path.exists():
            gripper = normalize_gripper_signal(
                load_gripper_trajectory(args.gripper_path),
                invert=bool(args.invert_gripper),
            )
    gripper_spec = get_gripper_joint_spec(robot)
    perturb_manual_window = resolve_manual_window(
        total_frames=int(joints.shape[0]),
        start=args.perturb_window_start,
        end=args.perturb_window_end,
    )
    perturbed_joints, perturb_summary = apply_joint_trajectory_perturbation(
        robot=robot,
        joints=joints,
        gripper=gripper,
        gripper_spec=gripper_spec,
        mode=args.perturb_mode,
        seed=int(args.perturb_seed),
        joint_indices=parse_index_csv(args.perturb_joint_indices),
        max_deg=float(args.perturb_max_deg),
        manual_window=perturb_manual_window,
        pre_frames=int(args.perturb_pre_frames),
        post_frames=int(args.perturb_post_frames),
    )
    joints = perturbed_joints

    base_to_camera = np.eye(4, dtype=np.float64)
    base_to_camera_sequence = None
    raw_wrist_summary = None

    if args.use_raw_wrist_extrinsics:
        raw_metadata = load_json(args.raw_metadata_json)
        wrist_serial = resolve_raw_wrist_serial(
            raw_metadata=raw_metadata if isinstance(raw_metadata, dict) else None,
            episode_id=args.episode_id,
            camera_serials_json=args.camera_serials_json,
        )
        pose_sequence, raw_wrist_summary = load_raw_wrist_pose_sequence(
            raw_metadata_json=args.raw_metadata_json,
            raw_trajectory_h5=args.raw_trajectory_h5,
            wrist_serial=wrist_serial,
            pose_source=args.wrist_pose_source,
        )
        intrinsics, resolved_camera_name, resolved_camera_key = load_camera_intrinsics_metadata(
            episode_id=args.episode_id,
            camera_name="wrist_image_left" if args.camera_name == "auto" else args.camera_name,
            intrinsics_json=args.intrinsics_json,
            camera_serials_json=args.camera_serials_json,
        )
        base_to_camera_sequence = np.repeat(np.eye(4, dtype=np.float64)[None, ...], pose_sequence.shape[0], axis=0)
        for idx, pose_matrix in enumerate(pose_sequence):
            base_to_camera_sequence[idx] = np.linalg.inv(pose_matrix)
        base_to_camera = base_to_camera_sequence[0]
    else:
        cam2base, intrinsics, resolved_camera_name, resolved_camera_key = load_camera_metadata(
            episode_id=args.episode_id,
            camera_name=args.camera_name,
            cam2base_json=args.cam2base_json,
            intrinsics_json=args.intrinsics_json,
            camera_serials_json=args.camera_serials_json,
        )
        base_to_camera = np.linalg.inv(cam2base)

    if args.base_adjust_json is not None:
        base_adjust = read_matrix(load_json(args.base_adjust_json))
        if base_adjust is None:
            raise ValueError(f"Failed to parse base-adjust matrix from {args.base_adjust_json}")
        if base_to_camera_sequence is not None:
            base_to_camera_sequence = np.einsum("nij,jk->nik", base_to_camera_sequence, base_adjust)
            base_to_camera = base_to_camera_sequence[0]
        else:
            base_to_camera = base_to_camera @ base_adjust

    summary = render_video(
        video_path=args.video_path,
        output_path=args.output_path,
        comparison_output_path=args.comparison_output_path,
        robot_only_output_path=args.robot_only_output_path,
        rgb_output_path=args.rgb_output_path,
        robot=robot,
        joints=joints,
        gripper=gripper,
        base_to_camera=base_to_camera,
        base_to_camera_sequence=base_to_camera_sequence,
        intrinsics=intrinsics,
        calibration_image_size=(args.calibration_image_size[0], args.calibration_image_size[1]),
        fps_override=args.fps,
        color=tuple(int(v) for v in args.color) if args.color is not None else None,
        alpha=float(args.alpha),
        renderer_backend=args.renderer_backend,
        appearance=args.appearance,
        max_faces=int(args.max_faces),
        frame_limit=args.frame_limit,
        frame_step=max(1, int(args.frame_step)),
        target_short_side=max(0, int(args.target_short_side)),
        robot_only_render_batch_size=max(1, int(args.robot_only_render_batch_size)),
        pyrender_style=str(args.pyrender_style),
        robot_only_background=str(args.robot_only_background),
    )
    summary["episode_id"] = args.episode_id
    summary["camera_name"] = resolved_camera_name
    summary["camera_key"] = resolved_camera_key
    summary["gripper_path"] = str(args.gripper_path) if args.gripper_path is not None else None
    summary["urdf_path"] = str(args.urdf_path)
    summary["tool_adjust_json"] = resolved_tool_adjust
    summary["raw_trajectory_h5"] = str(args.raw_trajectory_h5) if args.raw_trajectory_h5 is not None else None
    summary["raw_metadata_json"] = str(args.raw_metadata_json) if args.raw_metadata_json is not None else None
    summary["use_raw_wrist_extrinsics"] = bool(args.use_raw_wrist_extrinsics)
    summary["perturb"] = perturb_summary
    if raw_wrist_summary is not None:
        summary["raw_wrist_summary"] = raw_wrist_summary
    if args.perturb_meta_output is not None:
        args.perturb_meta_output.parent.mkdir(parents=True, exist_ok=True)
        with args.perturb_meta_output.open("w", encoding="utf-8") as f:
            json.dump(perturb_summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
