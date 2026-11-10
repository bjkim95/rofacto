#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import h5py
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from yourdfpy import URDF

try:
    from .common import (
        DEFAULT_ASSETS_DIR,
        DEFAULT_CACHE_ROOT,
        DEFAULT_OUTPUT_ROOT,
        atomic_write_bytes,
        is_nonempty_file,
        load_json,
        load_wrist_cam2base_sequence,
        write_json,
    )
    from .video_io import VideoWriter
except ImportError:
    from common import (
        DEFAULT_ASSETS_DIR,
        DEFAULT_CACHE_ROOT,
        DEFAULT_OUTPUT_ROOT,
        atomic_write_bytes,
        is_nonempty_file,
        load_json,
        load_wrist_cam2base_sequence,
        write_json,
    )
    from video_io import VideoWriter


WRIST_VIEW_NAME = "wrist"


def augment_record_with_wrist(record: dict[str, Any], *, output_root: Path) -> None:
    """In-place augment record so `views` / `video_outputs` include wrist if available.

    Conditions for augmentation:
      - record doesn't already include a wrist view
      - the episode's episode_meta.json (produced by wrist rgb export) lists wrist
        in `views_saved`

    After augmentation the render loop sees wrist as a regular view entry and
    dispatches a render alongside ext1/ext2.
    """
    views = record.get("views")
    if not isinstance(views, list):
        return
    if any(isinstance(v, dict) and v.get("view_name") == WRIST_VIEW_NAME for v in views):
        return
    episode_meta_rel = record.get("episode_meta_relative_path")
    if not episode_meta_rel:
        return
    meta_path = Path(output_root) / episode_meta_rel
    if not is_nonempty_file(meta_path):
        return
    try:
        meta_blob = load_json(meta_path)
    except Exception:
        return
    views_saved = meta_blob.get("views_saved") if isinstance(meta_blob, dict) else None
    if not isinstance(views_saved, list):
        return
    wrist_entry = next(
        (
            entry
            for entry in views_saved
            if isinstance(entry, dict) and entry.get("view_name") == WRIST_VIEW_NAME
        ),
        None,
    )
    if wrist_entry is None:
        return
    wrist_serial = wrist_entry.get("serial")
    if not wrist_serial:
        return
    relative_video_path = wrist_entry.get("relative_video_path")
    if not relative_video_path:
        return
    views.append(
        {
            "view_name": WRIST_VIEW_NAME,
            "camera_name": wrist_entry.get("camera_name"),
            "serial_key": wrist_entry.get("serial_key"),
            "serial": str(wrist_serial),
            "cam2base_source": None,
        }
    )
    video_outputs = record.setdefault("video_outputs", {})
    if isinstance(video_outputs, dict):
        video_outputs.setdefault(WRIST_VIEW_NAME, str(relative_video_path))
    # The videos script's input manifest (train_tfds_rgb.jsonl) doesn't carry
    # trajectory_indices, but episode_meta does — stash it so the wrist helper
    # can subsample the per-frame cam2base sequence to match the exported RGB.
    trajectory_indices = meta_blob.get("trajectory_indices")
    if trajectory_indices and not record.get("trajectory_indices"):
        record["trajectory_indices"] = trajectory_indices


def resolve_wrist_base_to_camera_sequence(
    *,
    episode_id: str,
    wrist_serial: str,
    trajectory_indices,
    intrinsics_blob: dict[str, Any],
    cache_root: Path,
    base_adjust: np.ndarray | None,
    wrist_pose_source: str = "trajectory_h5",
    gripper2wrist_json: Path | None = None,
    cloak_wrist_extrinsics_json: Path | None = None,
    cloak_missing_fallback: str = "proprio_gripper2wrist",
    wrist_proprio_flip: str = "none",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load per-frame base_to_camera for a wrist view.

    Wrist has no static cam2base asset, so we read pose vectors from the raw
    trajectory.h5 (cached in raw_cache_v2) keyed by `{wrist_serial}_left`, then
    subsample to the frame indices that the RGB export actually kept
    (`trajectory_indices`, produced by `export_tfds_rgb_videos.py`).

    Returns:
        intrinsics: 3x3 np.ndarray
        base_to_camera_sequence: [T, 4, 4] per-frame base_to_camera, aligned with
            the exported RGB video frames.
        source: short identifier ("trajectory_h5") for logging.
    """
    episode_intrinsics = intrinsics_blob.get(episode_id)
    if not isinstance(episode_intrinsics, dict):
        raise KeyError(f"Missing intrinsics entry for episode '{episode_id}'")
    intrinsics = read_intrinsics(episode_intrinsics.get(wrist_serial))
    if intrinsics is None:
        raise KeyError(
            f"Could not parse wrist intrinsics for episode '{episode_id}' serial '{wrist_serial}'"
        )

    trajectory_h5_path = Path(cache_root) / episode_id / "trajectory.h5"
    if not is_nonempty_file(trajectory_h5_path):
        raise FileNotFoundError(
            f"Missing cached trajectory.h5 for wrist pose: {trajectory_h5_path}"
        )
    if wrist_pose_source == "trajectory_h5":
        cam2base_sequence = load_wrist_cam2base_sequence(trajectory_h5_path, wrist_serial)

        if not trajectory_indices:
            raise RuntimeError(
                f"Missing trajectory_indices for wrist pose subsample (episode {episode_id})"
            )
        indices = np.asarray(list(trajectory_indices), dtype=np.int64)
        if indices.min() < 0 or indices.max() >= cam2base_sequence.shape[0]:
            raise RuntimeError(
                f"trajectory_indices out of range for wrist pose sequence: "
                f"indices [{indices.min()}, {indices.max()}] vs pose length {cam2base_sequence.shape[0]}"
            )
        cam2base_subsampled = cam2base_sequence[indices]
        base_to_camera_sequence = np.linalg.inv(cam2base_subsampled)
        if base_adjust is not None:
            base_to_camera_sequence = np.einsum("nij,jk->nik", base_to_camera_sequence, base_adjust)
        return intrinsics, base_to_camera_sequence.astype(np.float64), "trajectory_h5"
    if wrist_pose_source == "proprio_gripper2wrist":
        if gripper2wrist_json is None:
            gripper2wrist_json = DEFAULT_GRIPPER2WRIST_JSON
        base_to_camera_sequence, source = load_wrist_base_to_camera_sequence_from_proprio(
            episode_id=episode_id,
            trajectory_h5_path=trajectory_h5_path,
            cache_root=cache_root,
            trajectory_indices=trajectory_indices,
            gripper2wrist_json=gripper2wrist_json,
            wrist_proprio_flip=wrist_proprio_flip,
            base_adjust=base_adjust,
        )
        return intrinsics, base_to_camera_sequence, source
    if wrist_pose_source == "cloak_cam_to_gripper":
        if cloak_wrist_extrinsics_json is None:
            cloak_wrist_extrinsics_json = DEFAULT_CLOAK_WRIST_EXTRINSICS_JSON
        base_to_camera_sequence, source = load_wrist_base_to_camera_sequence_from_cloak(
            episode_id=episode_id,
            trajectory_h5_path=trajectory_h5_path,
            cache_root=cache_root,
            trajectory_indices=trajectory_indices,
            cloak_wrist_extrinsics_json=cloak_wrist_extrinsics_json,
            cloak_missing_fallback=cloak_missing_fallback,
            gripper2wrist_json=gripper2wrist_json or DEFAULT_GRIPPER2WRIST_JSON,
            wrist_proprio_flip=wrist_proprio_flip,
            base_adjust=base_adjust,
        )
        return intrinsics, base_to_camera_sequence, source
    raise ValueError(f"Unsupported wrist pose source: {wrist_pose_source}")


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_OVERLAY_DIR = REPO_ROOT / "data_processing" / "droid_urdf_overlay"
if str(LEGACY_OVERLAY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_OVERLAY_DIR))

from droid_urdf_overlay import (  # type: ignore  # noqa: E402
    materialize_tool_adjusted_urdf,
    normalize_gripper_signal,
    read_intrinsics,
    read_matrix,
    render_video,
)


# Mainline Panda + Robotiq 2F-85 URDF (meshes under third_party/robosuite). This is the
# URDF the verified render_v2 pipeline uses; the older phantom-sourced variant was removed.
DEFAULT_URDF_PATH = (
    REPO_ROOT / "data_processing" / "droid_urdf_overlay" / "assets"
    / "panda_arm_robotiq_2f85_fts300.urdf"
)
DEFAULT_INTRINSICS_JSON = DEFAULT_ASSETS_DIR / "intrinsics.json"
DEFAULT_CAM2BASE_JSON = DEFAULT_ASSETS_DIR / "cam2base_extrinsics.json"
DEFAULT_CAM2BASE_SUPERSET_JSON = DEFAULT_ASSETS_DIR / "cam2base_extrinsic_superset.json"
DEFAULT_GRIPPER2WRIST_JSON = (
    REPO_ROOT
    / "data_processing"
    / "third_party"
    / "PointWorld"
    / "real"
    / "gripper2wrist_transforms.json"
)
DEFAULT_CLOAK_WRIST_EXTRINSICS_JSON = (
    REPO_ROOT
    / "data_processing"
    / "third_party"
    / "cloak"
    / "assets"
    / "droid_wrist_extrinsics.json"
)


def load_robot_serial(cache_root: Path, episode_id: str) -> str:
    metadata_path = Path(cache_root) / episode_id / "metadata.json"
    if not is_nonempty_file(metadata_path):
        raise FileNotFoundError(f"Missing raw metadata.json for wrist proprio pose: {metadata_path}")
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata dict in {metadata_path}")
    robot_serial = metadata.get("robot_serial")
    if not isinstance(robot_serial, str) or not robot_serial.strip():
        raise KeyError(f"Missing robot_serial in {metadata_path}")
    return robot_serial.strip()


def resolve_robot_transform_key(robot_serial: str, transforms: dict[str, Any]) -> str:
    candidates = [robot_serial]
    for prefix in ("fr3-", "panda-"):
        if robot_serial.startswith(prefix):
            candidates.append(robot_serial[len(prefix) :])
        else:
            candidates.append(prefix + robot_serial)
    for candidate in candidates:
        if candidate in transforms:
            return candidate
    available = sorted(transforms.keys())[:12]
    raise KeyError(
        f"Could not find robot serial '{robot_serial}' in gripper2wrist transforms. "
        f"Tried {candidates}; available examples: {available}"
    )


def load_gripper_to_wrist_transform(path: Path, robot_serial: str) -> tuple[np.ndarray, str]:
    transforms = load_json(path)
    if not isinstance(transforms, dict):
        raise ValueError(f"Expected dict in {path}")
    transform_key = resolve_robot_transform_key(robot_serial, transforms)
    entry = transforms[transform_key]
    if isinstance(entry, dict):
        matrix = read_matrix(entry.get("mean_mat") or entry.get("gripper2wrist_mat") or entry.get("matrix"))
    else:
        matrix = read_matrix(entry)
    if matrix is None:
        raise ValueError(f"Failed to parse gripper2wrist matrix for robot '{transform_key}' from {path}")
    return matrix.astype(np.float64), transform_key


def load_gripper_to_base_sequence(trajectory_h5_path: Path) -> np.ndarray:
    with h5py.File(trajectory_h5_path, "r") as handle:
        candidates = (
            "observation/robot_state/cartesian_position",
            "action/robot_state/cartesian_position",
            "action/cartesian_position",
        )
        pose_vectors = None
        source = None
        for candidate in candidates:
            if candidate in handle:
                pose_vectors = np.asarray(handle[candidate], dtype=np.float64)
                source = candidate
                break
    if pose_vectors is None:
        raise KeyError(f"Could not find gripper cartesian pose in {trajectory_h5_path}")
    if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6:
        raise ValueError(
            f"Expected gripper pose vectors [T, 6] from {source}, got {pose_vectors.shape}"
        )
    matrices = np.empty((pose_vectors.shape[0], 4, 4), dtype=np.float64)
    for idx, pose_vector in enumerate(pose_vectors):
        matrix = read_matrix(pose_vector.tolist())
        if matrix is None:
            raise ValueError(f"Failed to parse gripper pose row {idx} from {source}")
        matrices[idx] = matrix
    return matrices


def load_wrist_base_to_camera_sequence_from_proprio(
    *,
    episode_id: str,
    trajectory_h5_path: Path,
    cache_root: Path,
    trajectory_indices: Any,
    gripper2wrist_json: Path,
    wrist_proprio_flip: str,
    base_adjust: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    robot_serial = load_robot_serial(cache_root, episode_id)
    gripper_to_wrist, transform_key = load_gripper_to_wrist_transform(
        gripper2wrist_json,
        robot_serial,
    )
    gripper_to_base_sequence = load_gripper_to_base_sequence(trajectory_h5_path)
    if not trajectory_indices:
        raise RuntimeError(
            f"Missing trajectory_indices for wrist proprio pose subsample (episode {episode_id})"
        )
    indices = np.asarray(list(trajectory_indices), dtype=np.int64)
    if indices.min() < 0 or indices.max() >= gripper_to_base_sequence.shape[0]:
        raise RuntimeError(
            f"trajectory_indices out of range for gripper pose sequence: "
            f"indices [{indices.min()}, {indices.max()}] vs pose length {gripper_to_base_sequence.shape[0]}"
        )
    base_to_gripper = np.linalg.inv(gripper_to_base_sequence[indices])
    base_to_camera_sequence = np.einsum("ij,njk->nik", gripper_to_wrist, base_to_gripper)
    if wrist_proprio_flip == "z180":
        flip_transform = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_to_camera_sequence = np.einsum("ij,njk->nik", flip_transform, base_to_camera_sequence)
    elif wrist_proprio_flip != "none":
        raise ValueError(f"Unsupported wrist proprio flip: {wrist_proprio_flip}")
    if base_adjust is not None:
        base_to_camera_sequence = np.einsum("nij,jk->nik", base_to_camera_sequence, base_adjust)
    return (
        base_to_camera_sequence.astype(np.float64),
        f"proprio_gripper2wrist:{transform_key}:{wrist_proprio_flip}",
    )


def cloak_key_from_episode_id(episode_id: str) -> str:
    parts = str(episode_id).split("+")
    if len(parts) >= 3:
        return f"{parts[0]}|{parts[-1]}"
    if "|" in str(episode_id):
        return str(episode_id)
    raise ValueError(
        f"Cannot derive Cloak wrist-extrinsics key from episode_id={episode_id!r}; "
        "expected LAB+hash+timestamp"
    )


def load_camera_to_gripper_transform_from_cloak(
    path: Path,
    episode_id: str,
) -> tuple[np.ndarray | None, str]:
    entries = load_json(path)
    if not isinstance(entries, dict):
        raise ValueError(f"Expected dict in Cloak wrist extrinsics JSON: {path}")
    key = cloak_key_from_episode_id(episode_id)
    vec6 = entries.get(key)
    if vec6 is None:
        return None, key
    gripper_to_camera = read_matrix(vec6)
    if gripper_to_camera is None:
        raise ValueError(f"Failed to parse Cloak cam_to_gripper vec for key={key!r} from {path}")
    # Cloak stores the camera pose in the end-effector/gripper frame
    # (T_gripper_camera). Our renderer expects world/base -> camera, so the
    # local mount used in world_to_camera composition is T_camera_gripper.
    camera_to_gripper = np.linalg.inv(gripper_to_camera)
    return camera_to_gripper.astype(np.float64), key


def load_wrist_base_to_camera_sequence_from_cloak(
    *,
    episode_id: str,
    trajectory_h5_path: Path,
    cache_root: Path,
    trajectory_indices: Any,
    cloak_wrist_extrinsics_json: Path,
    cloak_missing_fallback: str,
    gripper2wrist_json: Path,
    wrist_proprio_flip: str,
    base_adjust: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    camera_to_gripper, cloak_key = load_camera_to_gripper_transform_from_cloak(
        cloak_wrist_extrinsics_json,
        episode_id,
    )
    if camera_to_gripper is None:
        if cloak_missing_fallback == "proprio_gripper2wrist":
            fallback_sequence, fallback_source = load_wrist_base_to_camera_sequence_from_proprio(
                episode_id=episode_id,
                trajectory_h5_path=trajectory_h5_path,
                cache_root=cache_root,
                trajectory_indices=trajectory_indices,
                gripper2wrist_json=gripper2wrist_json,
                wrist_proprio_flip=wrist_proprio_flip,
                base_adjust=base_adjust,
            )
            return fallback_sequence, f"cloak_missing:{cloak_key}:fallback:{fallback_source}"
        if cloak_missing_fallback == "error":
            raise KeyError(
                f"Cloak wrist extrinsics missing key={cloak_key!r} in {cloak_wrist_extrinsics_json}"
            )
        raise ValueError(f"Unsupported Cloak missing fallback: {cloak_missing_fallback}")

    gripper_to_base_sequence = load_gripper_to_base_sequence(trajectory_h5_path)
    if not trajectory_indices:
        raise RuntimeError(
            f"Missing trajectory_indices for Cloak wrist pose subsample (episode {episode_id})"
        )
    indices = np.asarray(list(trajectory_indices), dtype=np.int64)
    if indices.min() < 0 or indices.max() >= gripper_to_base_sequence.shape[0]:
        raise RuntimeError(
            f"trajectory_indices out of range for gripper pose sequence: "
            f"indices [{indices.min()}, {indices.max()}] vs pose length {gripper_to_base_sequence.shape[0]}"
        )
    base_to_gripper = np.linalg.inv(gripper_to_base_sequence[indices])
    base_to_camera_sequence = np.einsum("ij,njk->nik", camera_to_gripper, base_to_gripper)
    if wrist_proprio_flip == "z180":
        flip_transform = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_to_camera_sequence = np.einsum("ij,njk->nik", flip_transform, base_to_camera_sequence)
    elif wrist_proprio_flip != "none":
        raise ValueError(f"Unsupported wrist proprio flip: {wrist_proprio_flip}")
    if base_adjust is not None:
        base_to_camera_sequence = np.einsum("nij,jk->nik", base_to_camera_sequence, base_adjust)
    return (
        base_to_camera_sequence.astype(np.float64),
        f"cloak_cam_to_gripper:{cloak_key}:{wrist_proprio_flip}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render robot_mesh_videos for DROID render_v2 outputs using the archived overlay renderer."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root used to read existing render_v2 RGB/joint/gripper inputs. Defaults to --output-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="raw_cache_v2 root used to read per-frame wrist cam2base from trajectory.h5.",
    )
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument(
        "--wrist-urdf-path",
        type=Path,
        default=None,
        help="Optional URDF used only for wrist renders. External views keep --urdf-path.",
    )
    parser.add_argument("--cam2base-json", type=Path, default=None)
    parser.add_argument("--cam2base-superset-json", type=Path, default=None)
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument(
        "--wrist-pose-source",
        choices=["trajectory_h5", "proprio_gripper2wrist", "cloak_cam_to_gripper"],
        default="trajectory_h5",
        help=(
            "Pose source for wrist renders. trajectory_h5 uses DROID camera_extrinsics; "
            "proprio_gripper2wrist composes gripper cartesian pose with a robot-specific "
            "gripper-to-wrist transform; cloak_cam_to_gripper uses Cloak's per-episode "
            "silhouette-calibrated DROID wrist extrinsics and falls back when missing."
        ),
    )
    parser.add_argument(
        "--gripper2wrist-json",
        type=Path,
        default=DEFAULT_GRIPPER2WRIST_JSON,
        help="PointWorld-style robot_serial -> gripper-to-wrist transform JSON.",
    )
    parser.add_argument(
        "--wrist-proprio-flip",
        choices=["none", "z180"],
        default="none",
        help=(
            "Optional camera-frame 180 degree Z flip for proprio_gripper2wrist. "
            "Use none for original, unrotated wrist RGB; z180 matches PointWorld's flipped-wrist VGGT convention."
        ),
    )
    parser.add_argument(
        "--cloak-wrist-extrinsics-json",
        type=Path,
        default=DEFAULT_CLOAK_WRIST_EXTRINSICS_JSON,
        help="Cloak LAB|timestamp -> cam_to_gripper JSON for silhouette-calibrated DROID wrist extrinsics.",
    )
    parser.add_argument(
        "--cloak-missing-fallback",
        choices=["proprio_gripper2wrist", "error"],
        default="proprio_gripper2wrist",
        help="Fallback when --wrist-pose-source cloak_cam_to_gripper has no episode key.",
    )
    parser.add_argument(
        "--wrist-render-mode",
        choices=["tracked_camera", "gripper_local"],
        default="tracked_camera",
        help=(
            "Wrist rendering mode. tracked_camera uses per-frame wrist camera poses; "
            "gripper_local keeps the wrist camera/arm pose fixed and renders only gripper motion."
        ),
    )
    parser.add_argument("--renderer-backend", type=str, default="pytorch3d")
    parser.add_argument(
        "--pyrender-style",
        choices=["default", "matte", "ambient", "flat", "flat_dim"],
        default="flat",
        help="Lighting/material style used when --renderer-backend is pyrender.",
    )
    parser.add_argument(
        "--robot-only-background",
        choices=["white", "gray"],
        default="white",
        help="Background color for robot-only outputs. gray means uint8 127.",
    )
    parser.add_argument("--appearance", type=str, default="shaded")
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.22, help="Fill alpha for overlay visualization.")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--write-comparison", action="store_true")
    parser.add_argument("--write-comparison-jpg", dest="write_comparison_jpg", action="store_true")
    parser.add_argument("--write-preview-jpg", dest="write_comparison_jpg", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--overlay-outline-color",
        type=int,
        nargs=3,
        default=[0, 255, 255],
        metavar=("R", "G", "B"),
        help="RGB color used for the overlay contour.",
    )
    parser.add_argument("--overlay-outline-alpha", type=float, default=0.95)
    parser.add_argument("--overlay-outline-thickness", type=int, default=2)
    parser.add_argument(
        "--overlay-background-dim",
        type=float,
        default=0.18,
        help="Darken the RGB background around the robot mask to improve visibility.",
    )
    parser.add_argument(
        "--overlay-mask-white-delta",
        type=int,
        default=12,
        help="Pixels this far from white in robot-only video are treated as robot mask.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Default keeps the already-exported render_v2 RGB video FPS.",
    )
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--robot-only-render-batch-size",
        type=int,
        default=4,
        help="Batch size used only for robot_only PyTorch3D rendering. Use 1 to disable batching.",
    )
    parser.add_argument(
        "--calibration-image-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--target-short-side",
        type=int,
        default=480,
        help="Downsample output videos so the shorter side is at most this size. Use 0 to keep RGB input size.",
    )
    parser.add_argument(
        "--joint-suffix",
        type=str,
        default="",
        help="Suffix appended to the manifest joint npy stem before loading, e.g. '_action'.",
    )
    parser.add_argument(
        "--gripper-suffix",
        type=str,
        default="",
        help="Suffix appended to the manifest gripper npy stem before loading, e.g. '_action'.",
    )
    parser.add_argument(
        "--mesh-output-dir-name",
        type=str,
        default="robot_mesh_videos",
        help="Directory name next to videos/ where robot-only mesh videos are written.",
    )
    parser.add_argument(
        "--overlay-output-dir-name",
        type=str,
        default="overlay",
        help="Directory name next to videos/ where overlay videos are written.",
    )
    parser.add_argument(
        "--comparison-output-dir-name",
        type=str,
        default="comparison",
        help="Directory name next to videos/ where comparison videos/jpgs are written.",
    )
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--base-adjust-json", type=Path, default=None)
    parser.add_argument("--tool-adjust-json", type=Path, default=None)
    parser.add_argument(
        "--ignore-view-cam2base-source",
        action="store_true",
        help=(
            "Ignore per-view cam2base_source hints embedded in older manifests "
            "and use --cam2base-json / --cam2base-superset-json order instead."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--view-name",
        action="append",
        default=None,
        help="Render only this view name. May be repeated, e.g. --view-name wrist.",
    )
    parser.add_argument("--task-tag", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing-inputs", action="store_true")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Exit zero after recording per-episode/view failures.",
    )
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(payload)
    return rows


def resolve_task_tag(args: argparse.Namespace) -> str:
    if args.task_tag is not None and args.task_tag.strip():
        return args.task_tag.strip()
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return f"{job_id}_{task_id}"
    return f"{job_id}_shard{args.shard_index}of{args.num_shards}"


def select_manifest_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards - 1}], got {args.shard_index}"
        )

    total_rows = len(rows)
    end_index = total_rows if args.end_index is None else min(args.end_index, total_rows)
    range_rows = rows[max(0, args.start_index) : end_index]
    sharded_rows = [
        row
        for idx, row in enumerate(range_rows)
        if idx % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        final_rows = sharded_rows[: max(0, args.limit)]
    else:
        final_rows = sharded_rows
    return final_rows, {
        "manifest_rows": total_rows,
        "range_rows": len(range_rows),
        "shard_rows": len(sharded_rows),
        "selected_rows": len(final_rows),
    }


def prepare_result_logs(output_root: Path, overwrite: bool, task_tag: str) -> tuple[Path, Path, Path]:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_path = manifest_dir / f"mesh_render_results_{task_tag}.jsonl"
    failures_path = manifest_dir / f"mesh_render_failures_{task_tag}.jsonl"
    summary_path = manifest_dir / f"mesh_render_summary_{task_tag}.json"
    if overwrite:
        for path in (results_path, failures_path, summary_path):
            if path.exists():
                path.unlink()
    for path in (results_path, failures_path):
        path.touch(exist_ok=True)
    return results_path, failures_path, summary_path


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_annotation_path(
    rows: list[dict[str, Any]],
    explicit_path: Path | None,
    annotation_key: str,
    default_path: Path,
    *,
    required: bool,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    for row in rows:
        annotation_assets = row.get("annotation_assets")
        if not isinstance(annotation_assets, dict):
            continue
        candidate = annotation_assets.get(annotation_key)
        if isinstance(candidate, str) and candidate.strip():
            return Path(candidate)
    if default_path.exists() or required:
        return default_path
    return None


def ensure_urdf_ready(urdf_path: Path) -> None:
    if urdf_path.exists():
        return
    raise FileNotFoundError(
        "Robot URDF not found at "
        f"{urdf_path}. Build it first with "
        "'python data_processing/droid/build_phantom_panda_arm_robotiq_2f85_urdf.py' "
        "then see data_processing/droid/URDF_ASSEMBLY_README.md."
    )


def load_named_cam2base_blobs(
    primary_path: Path | None,
    superset_path: Path | None,
) -> dict[str, dict[str, Any]]:
    named_blobs: dict[str, dict[str, Any]] = {}
    if primary_path is not None:
        blob = load_json(primary_path)
        if not isinstance(blob, dict):
            raise ValueError(f"Expected dict in {primary_path}")
        named_blobs[primary_path.name] = blob
    if superset_path is not None and superset_path != primary_path:
        blob = load_json(superset_path)
        if not isinstance(blob, dict):
            raise ValueError(f"Expected dict in {superset_path}")
        named_blobs[superset_path.name] = blob
    if not named_blobs:
        raise FileNotFoundError("No cam2base calibration JSON could be resolved for robot-mesh rendering.")
    return named_blobs


def with_stem_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def robot_mesh_relative_path(record: dict[str, Any], view_name: str, dir_name: str) -> Path:
    video_outputs = record.get("video_outputs", {})
    input_rel = Path(str(video_outputs[view_name]))
    override_dir = record.get("robot_mesh_output_relative_dir")
    if override_dir:
        return Path(str(override_dir)) / input_rel.name
    return input_rel.parent.parent / dir_name / input_rel.name


def visualization_relative_path(record: dict[str, Any], view_name: str, folder_name: str) -> Path:
    video_outputs = record.get("video_outputs", {})
    input_rel = Path(str(video_outputs[view_name]))
    return input_rel.parent.parent / folder_name / input_rel.name


def comparison_jpg_relative_path(record: dict[str, Any], view_name: str) -> Path:
    video_outputs = record.get("video_outputs", {})
    input_rel = Path(str(video_outputs[view_name]))
    return input_rel.parent.parent / "comparison" / f"{input_rel.stem}.jpg"


def visualization_jpg_relative_path(record: dict[str, Any], view_name: str, folder_name: str) -> Path:
    video_outputs = record.get("video_outputs", {})
    input_rel = Path(str(video_outputs[view_name]))
    return input_rel.parent.parent / folder_name / f"{input_rel.stem}.jpg"


def resolve_view_calibration(
    *,
    record: dict[str, Any],
    view: dict[str, Any],
    intrinsics_blob: dict[str, Any],
    cam2base_blobs: dict[str, dict[str, Any]],
    ignore_view_cam2base_source: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    episode_id = str(record["episode_id"])
    serial = str(view["serial"])
    episode_intrinsics = intrinsics_blob.get(episode_id)
    if not isinstance(episode_intrinsics, dict):
        raise KeyError(f"Missing intrinsics entry for episode '{episode_id}'")
    intrinsics = read_intrinsics(episode_intrinsics.get(serial))
    if intrinsics is None:
        raise KeyError(f"Could not parse intrinsics for episode '{episode_id}' serial '{serial}'")

    source_order: list[str] = []
    preferred_source = None if ignore_view_cam2base_source else view.get("cam2base_source")
    if isinstance(preferred_source, str) and preferred_source:
        source_order.append(preferred_source)
    for source_name in cam2base_blobs:
        if source_name not in source_order:
            source_order.append(source_name)

    for source_name in source_order:
        episode_cam2base = cam2base_blobs[source_name].get(episode_id)
        if not isinstance(episode_cam2base, dict):
            continue
        cam2base = read_matrix(episode_cam2base.get(serial))
        if cam2base is not None:
            return cam2base, intrinsics, source_name

    raise KeyError(
        f"Could not parse cam2base for episode '{episode_id}' serial '{serial}' from any configured source"
    )


def load_episode_inputs(
    record: dict[str, Any],
    input_root: Path,
    *,
    joint_suffix: str,
    gripper_suffix: str,
    invert_gripper: bool,
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    joint_path = input_root / with_stem_suffix(Path(str(record["joint_relative_path"])), joint_suffix)
    gripper_path = input_root / with_stem_suffix(Path(str(record["gripper_relative_path"])), gripper_suffix)
    if not joint_path.exists():
        raise FileNotFoundError(f"Missing joint trajectory: {joint_path}")
    if not gripper_path.exists():
        raise FileNotFoundError(f"Missing gripper trajectory: {gripper_path}")
    joints = np.asarray(np.load(joint_path), dtype=np.float64)
    if joints.ndim == 1:
        joints = joints.reshape(1, -1)
    gripper = normalize_gripper_signal(
        np.asarray(np.load(gripper_path), dtype=np.float64),
        invert=bool(invert_gripper),
    )
    return joint_path, gripper_path, joints, gripper


def load_base_to_camera_sequence(
    relative_path: str | Path | None,
    input_root: Path,
) -> np.ndarray | None:
    """Load an optional per-frame base-to-camera sequence from a manifest row."""
    if relative_path is None:
        return None
    rel = str(relative_path).strip()
    if not rel:
        return None
    path = input_root / Path(rel)
    if not path.exists():
        raise FileNotFoundError(f"Missing base_to_camera_sequence: {path}")
    sequence = np.asarray(np.load(path), dtype=np.float64)
    if sequence.ndim != 3 or sequence.shape[1:] != (4, 4):
        raise ValueError(f"Expected base_to_camera_sequence shape [T,4,4], got {sequence.shape} from {path}")
    if sequence.shape[0] < 1:
        raise ValueError(f"Empty base_to_camera_sequence: {path}")
    return sequence


def freeze_arm_joints_to_first_frame(joints: np.ndarray) -> np.ndarray:
    if joints.ndim != 2 or joints.shape[0] == 0:
        return joints
    frozen = np.asarray(joints, dtype=np.float64).copy()
    frozen[:, :] = frozen[0:1, :]
    return frozen


def infer_robot_only_background_bgr(robot_only_bgr: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            robot_only_bgr[0, :, :],
            robot_only_bgr[-1, :, :],
            robot_only_bgr[:, 0, :],
            robot_only_bgr[:, -1, :],
        ],
        axis=0,
    )
    return np.median(border.astype(np.float32), axis=0)


def build_robot_mask(robot_only_bgr: np.ndarray, white_delta: int) -> np.ndarray:
    background_bgr = infer_robot_only_background_bgr(robot_only_bgr)
    delta = np.max(np.abs(robot_only_bgr.astype(np.float32) - background_bgr[None, None, :]), axis=2)
    return delta >= max(1, int(white_delta))


def build_overlay_frame(
    *,
    rgb_bgr: np.ndarray,
    robot_only_bgr: np.ndarray,
    alpha: float,
    outline_rgb: tuple[int, int, int],
    outline_alpha: float,
    outline_thickness: int,
    background_dim: float,
    mask_white_delta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if robot_only_bgr.shape[:2] != rgb_bgr.shape[:2]:
        robot_only_bgr = cv2.resize(
            robot_only_bgr,
            (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    mask = build_robot_mask(robot_only_bgr, white_delta=mask_white_delta)
    overlay = rgb_bgr.copy()
    if not np.any(mask):
        return overlay, mask, robot_only_bgr

    mask_u8 = (mask.astype(np.uint8) * 255)
    dim_radius = max(1, int(outline_thickness) + 1)
    dim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dim_radius + 1, 2 * dim_radius + 1))
    dim_region = cv2.dilate(mask_u8, dim_kernel) > 0
    if background_dim > 0:
        overlay[dim_region] = np.clip(
            np.round(overlay[dim_region].astype(np.float32) * (1.0 - float(background_dim))),
            0.0,
            255.0,
        ).astype(np.uint8)

    blended = (
        float(alpha) * robot_only_bgr[mask].astype(np.float32)
        + (1.0 - float(alpha)) * overlay[mask].astype(np.float32)
    )
    overlay[mask] = np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)

    if outline_thickness > 0:
        outline_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(outline_thickness) + 1, 2 * int(outline_thickness) + 1),
        )
        outline_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, outline_kernel) > 0
        outline_color_bgr = np.asarray(outline_rgb[::-1], dtype=np.float32)
        overlay[outline_mask] = np.clip(
            np.round(
                float(outline_alpha) * outline_color_bgr
                + (1.0 - float(outline_alpha)) * overlay[outline_mask].astype(np.float32)
            ),
            0.0,
            255.0,
        ).astype(np.uint8)

    return overlay, mask, robot_only_bgr


def write_visualization_videos(
    *,
    input_rgb_path: Path,
    robot_only_path: Path,
    overlay_output_path: Path | None,
    comparison_output_path: Path | None,
    comparison_jpg_path: Path | None,
    fps_override: float | None,
    alpha: float,
    outline_rgb: tuple[int, int, int],
    outline_alpha: float,
    outline_thickness: int,
    background_dim: float,
    mask_white_delta: int,
) -> dict[str, Any]:
    if overlay_output_path is None and comparison_output_path is None and comparison_jpg_path is None:
        return {}

    rgb_cap = cv2.VideoCapture(str(input_rgb_path))
    robot_cap = cv2.VideoCapture(str(robot_only_path))
    if not rgb_cap.isOpened():
        raise RuntimeError(f"Failed to open RGB input video {input_rgb_path}")
    if not robot_cap.isOpened():
        raise RuntimeError(f"Failed to open robot-only video {robot_only_path}")

    overlay_writer: VideoWriter | None = None
    comparison_writer: VideoWriter | None = None
    frames_written = 0
    comparison_jpg_written = False

    try:
        fps = float(fps_override or rgb_cap.get(cv2.CAP_PROP_FPS) or 15.0)
        while True:
            ok_rgb, rgb_frame = rgb_cap.read()
            ok_robot, robot_frame = robot_cap.read()
            if not ok_robot:
                break
            if not ok_rgb:
                raise RuntimeError(
                    f"RGB/robot video length mismatch while building overlay: {input_rgb_path} vs {robot_only_path}"
                )

            overlay_frame, mask, robot_frame_vis = build_overlay_frame(
                rgb_bgr=rgb_frame,
                robot_only_bgr=robot_frame,
                alpha=alpha,
                outline_rgb=outline_rgb,
                outline_alpha=outline_alpha,
                outline_thickness=outline_thickness,
                background_dim=background_dim,
                mask_white_delta=mask_white_delta,
            )

            height, width = overlay_frame.shape[:2]
            if overlay_output_path is not None and overlay_writer is None:
                overlay_writer = VideoWriter(
                    path=overlay_output_path,
                    fps=fps,
                    frame_size=(width, height),
                    input_colorspace="bgr",
                )
            if comparison_output_path is not None and comparison_writer is None:
                comparison_writer = VideoWriter(
                    path=comparison_output_path,
                    fps=fps,
                    frame_size=(width * 3, height),
                    input_colorspace="bgr",
                )

            comparison_frame = np.concatenate([robot_frame_vis, overlay_frame, rgb_frame], axis=1)
            if comparison_jpg_path is not None and not comparison_jpg_written:
                comparison_jpg_path.parent.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", comparison_frame)
                if not ok:
                    raise RuntimeError(f"Failed to write comparison jpg {comparison_jpg_path}")
                atomic_write_bytes(comparison_jpg_path, encoded.tobytes())
                comparison_jpg_written = True

            if overlay_writer is not None:
                overlay_writer.write(overlay_frame)
            if comparison_writer is not None:
                comparison_writer.write(comparison_frame)
            frames_written += 1

            if overlay_writer is None and comparison_writer is None and comparison_jpg_written:
                break

        if frames_written == 0 and not comparison_jpg_written:
            raise RuntimeError(f"No frames were available for visualization build from {robot_only_path}")
    finally:
        rgb_cap.release()
        robot_cap.release()
        if overlay_writer is not None:
            overlay_writer.close()
        if comparison_writer is not None:
            comparison_writer.close()

    return {
        "fps": fps,
        "frames_written": frames_written,
        "overlay_output_path": str(overlay_output_path) if overlay_output_path is not None else None,
        "comparison_output_path": str(comparison_output_path) if comparison_output_path is not None else None,
        "comparison_jpg_path": str(comparison_jpg_path) if comparison_jpg_path is not None else None,
    }


def write_task_summary(
    *,
    summary_path: Path,
    task_tag: str,
    args: argparse.Namespace,
    counts: dict[str, int],
    results_path: Path,
    failures_path: Path,
    episode_rendered: int,
    episode_skipped: int,
    episode_partial_failures: int,
    episode_failures: int,
    view_rendered: int,
    view_skipped: int,
    view_failures: int,
) -> None:
    write_json(
        summary_path,
        {
            "task_tag": task_tag,
            "manifest_jsonl": str(args.manifest_jsonl),
            "output_root": str(args.output_root),
            "urdf_path": str(args.urdf_path),
            "wrist_urdf_path": str(args.wrist_urdf_path) if args.wrist_urdf_path is not None else None,
            "joint_suffix": str(args.joint_suffix),
            "gripper_suffix": str(args.gripper_suffix),
            "renderer_backend": str(args.renderer_backend),
            "pyrender_style": str(args.pyrender_style)
            if str(args.renderer_backend) in {"pyrender", "pyrender_persistent"}
            else None,
            "robot_only_background": str(args.robot_only_background),
            "mesh_output_dir_name": str(args.mesh_output_dir_name),
            "write_overlay": bool(args.write_overlay),
            "write_comparison": bool(args.write_comparison),
            "write_comparison_jpg": bool(args.write_comparison_jpg),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "counts": counts,
            "episode_rendered": int(episode_rendered),
            "episode_skipped": int(episode_skipped),
            "episode_partial_failures": int(episode_partial_failures),
            "episode_failures": int(episode_failures),
            "view_rendered": int(view_rendered),
            "view_skipped": int(view_skipped),
            "view_failures": int(view_failures),
            "results_path": str(results_path),
            "failures_path": str(failures_path),
        },
    )


def main() -> None:
    args = parse_args()
    input_root = args.output_root if args.input_root is None else args.input_root
    task_tag = resolve_task_tag(args)
    all_rows = load_manifest_rows(args.manifest_jsonl)
    rows, selection_counts = select_manifest_rows(args, all_rows)
    results_path, failures_path, summary_path = prepare_result_logs(
        args.output_root,
        overwrite=args.overwrite,
        task_tag=task_tag,
    )
    print(
        f"[mesh-render] task_tag={task_tag} | shard={args.shard_index}/{args.num_shards} | "
        f"manifest_rows={selection_counts['manifest_rows']} | "
        f"range_rows={selection_counts['range_rows']} | "
        f"shard_rows={selection_counts['shard_rows']} | "
        f"selected_rows={selection_counts['selected_rows']}"
    )

    ensure_urdf_ready(args.urdf_path)
    if args.wrist_urdf_path is not None:
        ensure_urdf_ready(args.wrist_urdf_path)
    intrinsics_path = resolve_annotation_path(
        rows=rows,
        explicit_path=args.intrinsics_json,
        annotation_key="intrinsics_json",
        default_path=DEFAULT_INTRINSICS_JSON,
        required=True,
    )
    cam2base_path = resolve_annotation_path(
        rows=rows,
        explicit_path=args.cam2base_json,
        annotation_key="cam2base_json",
        default_path=DEFAULT_CAM2BASE_JSON,
        required=True,
    )
    cam2base_superset_path = resolve_annotation_path(
        rows=rows,
        explicit_path=args.cam2base_superset_json,
        annotation_key="cam2base_superset_json",
        default_path=DEFAULT_CAM2BASE_SUPERSET_JSON,
        required=False,
    )

    intrinsics_blob = load_json(intrinsics_path)
    if not isinstance(intrinsics_blob, dict):
        raise ValueError(f"Expected dict in {intrinsics_path}")
    cam2base_blobs = load_named_cam2base_blobs(
        primary_path=cam2base_path,
        superset_path=cam2base_superset_path,
    )

    resolved_urdf_path, resolved_tool_adjust = materialize_tool_adjusted_urdf(
        urdf_path=args.urdf_path,
        tool_adjust_json=args.tool_adjust_json,
    )
    robot = URDF.load(str(resolved_urdf_path))
    wrist_robot = robot
    resolved_wrist_urdf_path = resolved_urdf_path
    resolved_wrist_tool_adjust = resolved_tool_adjust
    if args.wrist_urdf_path is not None:
        resolved_wrist_urdf_path, resolved_wrist_tool_adjust = materialize_tool_adjusted_urdf(
            urdf_path=args.wrist_urdf_path,
            tool_adjust_json=args.tool_adjust_json,
        )
        wrist_robot = URDF.load(str(resolved_wrist_urdf_path))

    base_adjust = None
    if args.base_adjust_json is not None:
        base_adjust = read_matrix(load_json(args.base_adjust_json))
        if base_adjust is None:
            raise ValueError(f"Failed to parse base-adjust matrix from {args.base_adjust_json}")

    episode_rendered = 0
    episode_skipped = 0
    episode_partial_failures = 0
    episode_failures = 0
    view_rendered = 0
    view_skipped = 0
    view_failures = 0

    if not rows:
        write_task_summary(
            summary_path=summary_path,
            task_tag=task_tag,
            args=args,
            counts=selection_counts,
            results_path=results_path,
            failures_path=failures_path,
            episode_rendered=episode_rendered,
            episode_skipped=episode_skipped,
            episode_partial_failures=episode_partial_failures,
            episode_failures=episode_failures,
            view_rendered=view_rendered,
            view_skipped=view_skipped,
            view_failures=view_failures,
        )
        print("[mesh-render] no rows assigned to this shard; exiting cleanly")
        return

    for idx, record in enumerate(rows, start=1):
        episode_id = str(record["episode_id"])
        try:
            joint_path, gripper_path, joints, gripper = load_episode_inputs(
                record,
                input_root,
                joint_suffix=str(args.joint_suffix),
                gripper_suffix=str(args.gripper_suffix),
                invert_gripper=bool(args.invert_gripper),
            )
            # The tfds rgb manifest was built with ext-only VIEW_SPECS, so wrist is
            # missing from record["views"] and record["video_outputs"]. If wrist
            # landed in episode_meta.json (post-wrist-export) and VIEW_SPECS now
            # includes wrist, augment the row in-memory so the per-view loop below
            # also dispatches a wrist render. This is a no-op when wrist isn't
            # available or was already present.
            augment_record_with_wrist(record, output_root=input_root)
            allowed_views = None
            if args.view_name:
                allowed_views = {str(view_name) for view_name in args.view_name}
            candidate_views = [
                view
                for view in record.get("views", [])
                if allowed_views is None or str(view.get("view_name")) in allowed_views
            ]
            if not candidate_views:
                episode_skipped += 1
                append_jsonl(
                    results_path,
                    [
                        {
                            "task_tag": task_tag,
                            "episode_id": episode_id,
                            "status": "skipped_no_selected_view",
                            "views": [],
                            "selected_view_names": sorted(allowed_views) if allowed_views is not None else None,
                        }
                    ],
                )
                continue
            rendered_views: list[dict[str, Any]] = []
            row_view_failures: list[dict[str, Any]] = []
            row_view_statuses: list[str] = []

            for view in candidate_views:
                view_name = str(view["view_name"])
                input_video_path = input_root / Path(str(record["video_outputs"][view_name]))
                output_path = args.output_root / robot_mesh_relative_path(
                    record,
                    view_name,
                    str(args.mesh_output_dir_name),
                )
                overlay_output_path = None
                if args.write_overlay:
                    overlay_output_path = args.output_root / visualization_relative_path(
                        record,
                        view_name,
                        str(args.overlay_output_dir_name),
                    )
                comparison_output_path = None
                if args.write_comparison:
                    comparison_output_path = args.output_root / visualization_relative_path(
                        record,
                        view_name,
                        str(args.comparison_output_dir_name),
                    )
                comparison_jpg_path = None
                if args.write_comparison_jpg:
                    comparison_jpg_path = args.output_root / visualization_jpg_relative_path(
                        record,
                        view_name,
                        str(args.comparison_output_dir_name),
                    )
                if not is_nonempty_file(input_video_path):
                    error = f"Missing render_v2 RGB input: {input_video_path}"
                    if not args.skip_missing_inputs:
                        row_view_failures.append(
                            {
                                "task_tag": task_tag,
                                "episode_id": episode_id,
                                "view": view_name,
                                "stage": "input_check",
                                "error": error,
                            }
                        )
                    rendered_views.append(
                        {
                            "view": view_name,
                            "camera_name": view.get("camera_name"),
                            "serial": view.get("serial"),
                            "status": "skipped_missing_input" if args.skip_missing_inputs else "failed",
                            "paths": {
                                "input_rgb": str(input_video_path),
                                "robot_only": str(output_path),
                                "overlay": str(overlay_output_path) if overlay_output_path is not None else None,
                                "comparison": str(comparison_output_path) if comparison_output_path is not None else None,
                                "comparison_jpg": str(comparison_jpg_path) if comparison_jpg_path is not None else None,
                            },
                            "error": error if not args.skip_missing_inputs else None,
                        }
                    )
                    row_view_statuses.append("skipped_missing_input" if args.skip_missing_inputs else "failed")
                    if args.skip_missing_inputs:
                        view_skipped += 1
                    else:
                        view_failures += 1
                    continue

                requested_outputs = [output_path]
                if overlay_output_path is not None:
                    requested_outputs.append(overlay_output_path)
                if comparison_output_path is not None:
                    requested_outputs.append(comparison_output_path)
                if comparison_jpg_path is not None:
                    requested_outputs.append(comparison_jpg_path)

                if all(is_nonempty_file(path) for path in requested_outputs) and not args.overwrite:
                    rendered_views.append(
                        {
                            "view": view_name,
                            "camera_name": view.get("camera_name"),
                            "serial": view.get("serial"),
                            "status": "skipped_existing",
                            "paths": {
                                "input_rgb": str(input_video_path),
                                "robot_only": str(output_path),
                                "overlay": str(overlay_output_path) if overlay_output_path is not None else None,
                                "comparison": str(comparison_output_path) if comparison_output_path is not None else None,
                                "comparison_jpg": str(comparison_jpg_path) if comparison_jpg_path is not None else None,
                            },
                        }
                    )
                    row_view_statuses.append("skipped_existing")
                    view_skipped += 1
                    continue

                try:
                    render_joints = joints
                    render_robot = robot
                    render_urdf_path = resolved_urdf_path
                    render_tool_adjust = resolved_tool_adjust
                    manifest_sequence = load_base_to_camera_sequence(
                        view.get("base_to_camera_sequence_relative_path")
                        or record.get("base_to_camera_sequence_relative_path"),
                        input_root,
                    )
                    if manifest_sequence is not None:
                        base_to_camera_sequence = manifest_sequence
                        base_to_camera = base_to_camera_sequence[0]
                        cam2base_source = "manifest:base_to_camera_sequence"
                    elif view_name == WRIST_VIEW_NAME:
                        render_robot = wrist_robot
                        render_urdf_path = resolved_wrist_urdf_path
                        render_tool_adjust = resolved_wrist_tool_adjust
                        intrinsics, base_to_camera_sequence, cam2base_source = resolve_wrist_base_to_camera_sequence(
                            episode_id=episode_id,
                            wrist_serial=str(view["serial"]),
                            trajectory_indices=record.get("trajectory_indices") or [],
                            intrinsics_blob=intrinsics_blob,
                            cache_root=args.cache_root,
                            base_adjust=base_adjust,
                            wrist_pose_source=str(args.wrist_pose_source),
                            gripper2wrist_json=args.gripper2wrist_json,
                            cloak_wrist_extrinsics_json=args.cloak_wrist_extrinsics_json,
                            cloak_missing_fallback=str(args.cloak_missing_fallback),
                            wrist_proprio_flip=str(args.wrist_proprio_flip),
                        )
                        base_to_camera = base_to_camera_sequence[0]
                        if str(args.wrist_render_mode) == "gripper_local":
                            render_joints = freeze_arm_joints_to_first_frame(joints)
                            base_to_camera_sequence = None
                            cam2base_source = f"{cam2base_source}:gripper_local"
                    else:
                        cam2base, intrinsics, cam2base_source = resolve_view_calibration(
                            record=record,
                            view=view,
                            intrinsics_blob=intrinsics_blob,
                            cam2base_blobs=cam2base_blobs,
                            ignore_view_cam2base_source=bool(args.ignore_view_cam2base_source),
                        )
                        base_to_camera = np.linalg.inv(cam2base)
                        if base_adjust is not None:
                            base_to_camera = base_to_camera @ base_adjust
                        base_to_camera_sequence = None
                    render_summary = None
                    if args.overwrite or not output_path.exists():
                        render_summary = render_video(
                            video_path=input_video_path,
                            output_path=None,
                            comparison_output_path=None,
                            robot_only_output_path=output_path,
                            rgb_output_path=None,
                            robot=render_robot,
                            joints=render_joints,
                            gripper=gripper,
                            base_to_camera=base_to_camera,
                            base_to_camera_sequence=base_to_camera_sequence,
                            intrinsics=intrinsics,
                            calibration_image_size=(
                                int(args.calibration_image_size[0]),
                                int(args.calibration_image_size[1]),
                            ),
                            fps_override=args.fps,
                            color=None,
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

                    visualization_summary = write_visualization_videos(
                        input_rgb_path=input_video_path,
                        robot_only_path=output_path,
                        overlay_output_path=overlay_output_path,
                        comparison_output_path=comparison_output_path,
                        comparison_jpg_path=comparison_jpg_path,
                        fps_override=args.fps,
                        alpha=float(args.alpha),
                        outline_rgb=tuple(int(value) for value in args.overlay_outline_color),
                        outline_alpha=float(args.overlay_outline_alpha),
                        outline_thickness=max(0, int(args.overlay_outline_thickness)),
                        background_dim=max(0.0, float(args.overlay_background_dim)),
                        mask_white_delta=max(1, int(args.overlay_mask_white_delta)),
                    )
                    rendered_views.append(
                        {
                            "view": view_name,
                            "camera_name": view.get("camera_name"),
                            "serial": view.get("serial"),
                            "status": "rendered",
                            "paths": {
                                "input_rgb": str(input_video_path),
                                "robot_only": str(output_path),
                                "overlay": str(overlay_output_path) if overlay_output_path is not None else None,
                                "comparison": str(comparison_output_path) if comparison_output_path is not None else None,
                                "comparison_jpg": str(comparison_jpg_path) if comparison_jpg_path is not None else None,
                            },
                            "summary": {
                                "render": render_summary,
                                "visualization": visualization_summary,
                                "cam2base_source": cam2base_source,
                                "wrist_render_mode": str(args.wrist_render_mode)
                                if view_name == WRIST_VIEW_NAME
                                else None,
                                "renderer_backend": str(args.renderer_backend),
                                "pyrender_style": str(args.pyrender_style)
                                if str(args.renderer_backend) in {"pyrender", "pyrender_persistent"}
                                else None,
                                "robot_only_background": str(args.robot_only_background),
                                "urdf_path": str(render_urdf_path),
                                "joint_path": str(joint_path),
                                "gripper_path": str(gripper_path),
                                "tool_adjust_json": render_tool_adjust,
                            },
                        }
                    )
                    row_view_statuses.append("rendered")
                    view_rendered += 1
                except Exception as exc:
                    row_view_failures.append(
                        {
                            "task_tag": task_tag,
                            "episode_id": episode_id,
                            "view": view_name,
                            "stage": "render_view",
                            "error": str(exc),
                        }
                    )
                    rendered_views.append(
                        {
                            "view": view_name,
                            "camera_name": view.get("camera_name"),
                            "serial": view.get("serial"),
                            "status": "failed",
                            "paths": {
                                "input_rgb": str(input_video_path),
                                "robot_only": str(output_path),
                                "overlay": str(overlay_output_path) if overlay_output_path is not None else None,
                                "comparison": str(comparison_output_path) if comparison_output_path is not None else None,
                                "comparison_jpg": str(comparison_jpg_path) if comparison_jpg_path is not None else None,
                            },
                            "error": str(exc),
                        }
                    )
                    row_view_statuses.append("failed")
                    view_failures += 1

            if row_view_failures:
                append_jsonl(failures_path, row_view_failures)

            if row_view_statuses and all(
                status in {"skipped_existing", "skipped_missing_input"}
                for status in row_view_statuses
            ):
                episode_status = (
                    "skipped_missing_input"
                    if "skipped_missing_input" in row_view_statuses
                    else "skipped_existing"
                )
                episode_skipped += 1
            elif "rendered" in row_view_statuses and "failed" in row_view_statuses:
                episode_status = "partial_failure"
                episode_partial_failures += 1
            elif "rendered" in row_view_statuses:
                episode_status = "rendered"
                episode_rendered += 1
            else:
                episode_status = "failed"
                episode_failures += 1

            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": episode_id,
                        "status": episode_status,
                        "views": rendered_views,
                    }
                ],
            )
        except Exception as exc:
            episode_failures += 1
            append_jsonl(
                failures_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": episode_id,
                        "stage": "prepare_episode",
                        "error": str(exc),
                    }
                ],
            )

        if idx % 10 == 0 or idx == len(rows):
            print(
                f"[mesh-render] {idx}/{len(rows)} processed | "
                f"episode_rendered={episode_rendered} | "
                f"episode_skipped={episode_skipped} | "
                f"episode_partial_failures={episode_partial_failures} | "
                f"episode_failures={episode_failures}"
            )

    write_task_summary(
        summary_path=summary_path,
        task_tag=task_tag,
        args=args,
        counts=selection_counts,
        results_path=results_path,
        failures_path=failures_path,
        episode_rendered=episode_rendered,
        episode_skipped=episode_skipped,
        episode_partial_failures=episode_partial_failures,
        episode_failures=episode_failures,
        view_rendered=view_rendered,
        view_skipped=view_skipped,
        view_failures=view_failures,
    )
    print(
        json.dumps(
            {
                "task_tag": task_tag,
                "manifest_jsonl": str(args.manifest_jsonl),
                "output_root": str(args.output_root),
                "selected_rows": len(rows),
                "episode_rendered": episode_rendered,
                "episode_skipped": episode_skipped,
                "episode_partial_failures": episode_partial_failures,
                "episode_failures": episode_failures,
                "view_rendered": view_rendered,
                "view_skipped": view_skipped,
                "view_failures": view_failures,
                "results_path": str(results_path),
                "failures_path": str(failures_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    if not args.best_effort and (episode_partial_failures or episode_failures):
        raise SystemExit(
            "mesh rendering completed with failures; rerun with --best-effort to preserve exit zero"
        )



if __name__ == "__main__":
    main()
