# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
"""DROID camera-pose helpers (intrinsics, extrinsics, wrist trajectory).

Copied from ``data_processing/droid_urdf_overlay/pi3_reconstruction.py`` so that
training code does NOT import the pi3 pipeline directly — pi3 is legacy and will
be retired in favour of FoundationStereo-based wrist reconstruction. The copies
here stand alone and can be kept/extended independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


__all__ = [
    "load_json",
    "euler_xyz_to_matrix",
    "read_matrix",
    "read_intrinsics",
    "read_intrinsics_size",
    "scale_intrinsics",
    "load_raw_wrist_pose_sequence",
]


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


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
    if isinstance(candidate, np.ndarray):
        arr = candidate.astype(np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
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
        if all(key in candidate for key in ("fx", "fy", "cx", "cy")):
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


def read_intrinsics_size(candidate: Any) -> tuple[int, int] | None:
    if isinstance(candidate, dict):
        width = candidate.get("width")
        height = candidate.get("height")
        if width is not None and height is not None:
            return int(width), int(height)
    return None


def scale_intrinsics(
    intrinsics: np.ndarray,
    src_size: tuple[int, int] | None,
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Rescale a 3x3 K to a new image size.

    ``src_size`` / ``dst_size`` are ``(width, height)``. When ``src_size`` is
    unknown the input is returned unchanged. Assumes no skew and that principal
    point + focal length scale linearly with image resolution.
    """
    width, height = dst_size
    if src_size is None:
        return intrinsics.copy().astype(np.float64)
    src_width, src_height = src_size
    if src_width <= 0 or src_height <= 0:
        return intrinsics.copy().astype(np.float64)
    scale_x = float(width) / float(src_width)
    scale_y = float(height) / float(src_height)
    scaled = intrinsics.copy().astype(np.float64)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _resolve_robot_transform_key(robot_serial: str, transforms: dict[str, Any]) -> str:
    """Resolve DROID robot serial variants used by PointWorld's transform file."""
    serial = str(robot_serial)
    serial_forms = [serial, serial.lower()]
    candidates = []
    for base in serial_forms:
        candidates.append(base)
    for prefix in ("fr3-", "franka-", "panda-"):
        if serial.lower().startswith(prefix):
            suffix = serial[len(prefix):]
            candidates.extend([suffix, suffix.lower()])
        else:
            candidates.extend([prefix + serial, prefix + serial.lower()])
    for candidate in candidates:
        if candidate in transforms:
            return candidate
    available = sorted(transforms.keys())[:12]
    raise KeyError(
        f"Could not find robot serial '{robot_serial}' in gripper2wrist transforms. "
        f"Tried {candidates}; available examples: {available}"
    )


def _load_gripper_to_wrist_transform(path: Path, robot_serial: str) -> np.ndarray:
    transforms = load_json(path)
    if not isinstance(transforms, dict):
        raise ValueError(f"Expected dict in gripper2wrist transform file: {path}")
    transform_key = _resolve_robot_transform_key(robot_serial, transforms)
    entry = transforms[transform_key]
    if isinstance(entry, dict):
        matrix = read_matrix(
            entry.get("mean_mat") or entry.get("gripper2wrist_mat") or entry.get("matrix")
        )
    else:
        matrix = read_matrix(entry)
    if matrix is None:
        raise ValueError(
            f"Failed to parse gripper2wrist matrix for robot '{transform_key}' from {path}"
        )
    return matrix.astype(np.float64)


def _load_gripper_to_base_sequence(raw_trajectory_h5: Path) -> np.ndarray:
    """Read DROID gripper cartesian pose as [T, 4, 4] gripper->base matrices."""
    with h5py.File(raw_trajectory_h5, "r") as handle:
        pose_vectors = None
        source = None
        for candidate in (
            "observation/robot_state/cartesian_position",
            "action/robot_state/cartesian_position",
            "action/cartesian_position",
        ):
            if candidate in handle:
                pose_vectors = np.asarray(handle[candidate], dtype=np.float64)
                source = candidate
                break
    if pose_vectors is None:
        raise KeyError(f"Could not find gripper cartesian pose in {raw_trajectory_h5}")
    if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6:
        raise ValueError(
            f"Expected gripper pose vectors [T, 6] from {source}, got {pose_vectors.shape}"
        )
    matrices = np.empty((pose_vectors.shape[0], 4, 4), dtype=np.float64)
    for index, pose_vector in enumerate(pose_vectors):
        matrix = read_matrix(pose_vector.tolist())
        if matrix is None:
            raise ValueError(f"Failed to parse gripper pose row {index} from {source}")
        matrices[index] = matrix
    return matrices


def load_raw_wrist_pose_sequence(
    raw_trajectory_h5: Path,
    wrist_serial: str,
    pose_source: str = "trajectory",
    raw_metadata_path: Path | None = None,
    gripper2wrist_json: Path | None = None,
    wrist_proprio_flip: str = "none",
) -> np.ndarray:
    """Return per-frame wrist pose as [T, 4, 4] c2w matrices (robot base frame).

    ``pose_source``:
      * ``"trajectory"`` — read ``observation/camera_extrinsics/{wrist_serial}_left``
        from ``raw_trajectory_h5`` (expected to be [T, 6] Euler-xyz + translation).
      * ``"metadata"`` — read ``wrist_cam_extrinsics`` from ``metadata.json`` and
        repeat it for every frame.
      * ``"proprio_gripper2wrist"`` — PointWorld-style wrist pose. Compose
        per-frame gripper cartesian pose with the robot-specific
        gripper-to-wrist-camera transform from ``gripper2wrist_json``. This
        bypasses raw DROID wrist camera extrinsics.
    """
    if pose_source not in {"trajectory", "metadata", "proprio_gripper2wrist"}:
        raise ValueError(f"Unsupported wrist pose source: {pose_source}")

    metadata_blob = None
    if raw_metadata_path is not None:
        metadata_path = Path(raw_metadata_path)
        if metadata_path.exists():
            metadata_blob = load_json(metadata_path)

    if pose_source == "proprio_gripper2wrist":
        if gripper2wrist_json is None:
            raise ValueError("gripper2wrist_json is required for proprio_gripper2wrist")
        if not isinstance(metadata_blob, dict) or not metadata_blob.get("robot_serial"):
            raise KeyError("robot_serial missing from raw metadata.json")
        gripper_to_wrist = _load_gripper_to_wrist_transform(
            Path(gripper2wrist_json),
            str(metadata_blob["robot_serial"]),
        )
        gripper_to_base = _load_gripper_to_base_sequence(raw_trajectory_h5)
        camera_to_gripper = np.linalg.inv(gripper_to_wrist)
        camera_to_base = np.einsum("nij,jk->nik", gripper_to_base, camera_to_gripper)
        if wrist_proprio_flip == "z180":
            flip = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            camera_to_base = np.einsum("nij,jk->nik", camera_to_base, flip)
        elif wrist_proprio_flip != "none":
            raise ValueError(f"Unsupported wrist_proprio_flip: {wrist_proprio_flip}")
        return camera_to_base.astype(np.float64)

    if pose_source == "metadata":
        if not isinstance(metadata_blob, dict) or metadata_blob.get("wrist_cam_extrinsics") is None:
            raise KeyError("wrist_cam_extrinsics missing from raw metadata.json")
        pose_matrix = read_matrix(metadata_blob["wrist_cam_extrinsics"])
        if pose_matrix is None:
            raise ValueError("Failed to parse metadata wrist pose")
        with h5py.File(raw_trajectory_h5, "r") as handle:
            if "observation/robot_state/joint_positions" in handle:
                frame_count = int(handle["observation/robot_state/joint_positions"].shape[0])
            elif "action/joint_position" in handle:
                frame_count = int(handle["action/joint_position"].shape[0])
            else:
                raise KeyError(f"Could not infer trajectory length from {raw_trajectory_h5}")
        return np.repeat(pose_matrix[None, ...], frame_count, axis=0)

    left_key = f"{wrist_serial}_left"
    with h5py.File(raw_trajectory_h5, "r") as handle:
        if "observation" not in handle or "camera_extrinsics" not in handle["observation"]:
            raise KeyError(f"Could not find observation/camera_extrinsics in {raw_trajectory_h5}")
        camera_extrinsics = handle["observation"]["camera_extrinsics"]
        if left_key not in camera_extrinsics:
            available = sorted(camera_extrinsics.keys())[:12]
            raise KeyError(
                f"Missing wrist pose dataset '{left_key}' in {raw_trajectory_h5}. "
                f"Available keys include: {available}"
            )
        pose_vectors = np.asarray(camera_extrinsics[left_key], dtype=np.float64)

    if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6:
        raise ValueError(f"Expected wrist pose vectors of shape [T, 6], got {pose_vectors.shape}")
    pose_matrices = np.repeat(np.eye(4, dtype=np.float64)[None, ...], pose_vectors.shape[0], axis=0)
    for index, pose_vector in enumerate(pose_vectors):
        matrix = read_matrix(pose_vector.tolist())
        if matrix is None:
            raise ValueError(f"Failed to parse wrist pose row {index} from {raw_trajectory_h5}")
        pose_matrices[index] = matrix
    return pose_matrices
