from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import h5py
import numpy as np


DEFAULT_DATASET_NAME = "droid"
DEFAULT_DATA_DIR = "gs://gresearch/robotics"
DEFAULT_SOURCE_STEP_RATE = 15.0
DEFAULT_OUTPUT_FPS = 15.0
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_OUTPUT_ROOT = Path("data/droid/render_v2")
DEFAULT_MANIFEST_ROOT = Path("data/droid/manifests_v2")
DEFAULT_CACHE_ROOT = Path("data/droid/raw_cache_v2")
RAW_BUCKET_ROOT = "gs://gresearch/robotics/droid_raw/1.0.1"

VIEW_SPECS = {
    "ext1": {
        "camera_name": "exterior_image_1_left",
        "serial_key": "ext1_cam_serial",
        "svo_key": "ext1_svo_path",
        "require_cam2base": True,
    },
    "ext2": {
        "camera_name": "exterior_image_2_left",
        "serial_key": "ext2_cam_serial",
        "svo_key": "ext2_svo_path",
        "require_cam2base": True,
    },
    # Wrist (ZED Mini) extrinsics are dynamic per-frame (read from trajectory.h5),
    # so the curated static cam2base JSON assets have no coverage for wrist serials.
    # Setting require_cam2base=False lets resolve_calibrated_views include wrist
    # based on intrinsics alone; robot-mesh renderers that need a static cam2base
    # should check `cam2base_source` and skip views where it is None.
    "wrist": {
        "camera_name": "wrist_image_left",
        "serial_key": "wrist_cam_serial",
        "svo_key": "wrist_svo_path",
        "require_cam2base": False,
    },
}


def load_tfds() -> Any:
    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorflow_datasets is required for the DROID TFDS RGB pipeline."
        ) from exc
    return tfds


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.partial-{uuid4().hex}"
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_numpy_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.partial-{uuid4().hex}"
    np.save(temp_path, array)
    npy_temp_path = temp_path if temp_path.suffix == ".npy" else temp_path.with_suffix(f"{temp_path.suffix}.npy")
    os.replace(npy_temp_path, path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.partial-{uuid4().hex}"
    with temp_path.open("wb") as f:
        f.write(payload)
    os.replace(temp_path, path)


def atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-",
        suffix=".npz",
        dir=str(path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_h5_write(
    path: Path,
    *,
    datasets: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.partial-{uuid4().hex}"
    try:
        with h5py.File(temp_path, "w") as handle:
            for key, value in datasets.items():
                dataset = handle.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=bool(compression),
                )
                dataset.attrs["write_complete"] = True
            if attrs:
                for key, value in attrs.items():
                    handle.attrs[key] = value
            handle.attrs["write_complete"] = True
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def normalize_path_like(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")


def candidate_keep_range_variants(candidate: str) -> list[str]:
    normalized = normalize_path_like(candidate)
    variants = [normalized]
    replacement_pairs = [
        ("nfs/kun2/datasets/", "gs://xembodiment_data/"),
        ("nfs/kun2/datasets", "gs://xembodiment_data"),
    ]
    for old_prefix, new_prefix in replacement_pairs:
        if old_prefix in normalized:
            remapped = normalized.replace(old_prefix, new_prefix)
            if remapped not in variants:
                variants.append(remapped)
    return variants


def to_python(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_python(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_episode_metadata(episode: Any) -> dict[str, Any]:
    metadata = to_python(episode.get("episode_metadata", {}))
    if not isinstance(metadata, dict):
        return {}
    return metadata


def first_nonempty_string(values: Iterable[Any]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_task_string(episode_metadata: dict[str, Any]) -> str | None:
    return first_nonempty_string(
        (
            episode_metadata.get("current_task"),
            episode_metadata.get("language_instruction"),
            episode_metadata.get("language_instruction_2"),
            episode_metadata.get("language_instruction_3"),
            episode_metadata.get("task"),
            episode_metadata.get("instruction"),
        )
    )


def extract_step_instruction(step: Any) -> str | None:
    return first_nonempty_string(
        (
            to_python(step.get("language_instruction")),
            to_python(step.get("language_instruction_2")),
            to_python(step.get("language_instruction_3")),
        )
    )


def episode_to_path_candidates(episode_metadata: dict[str, Any]) -> list[str]:
    file_path = normalize_path_like(str(episode_metadata.get("file_path") or ""))
    recording_folderpath = normalize_path_like(str(episode_metadata.get("recording_folderpath") or ""))
    candidates: list[str] = []
    if recording_folderpath and file_path:
        candidates.append(f"{recording_folderpath}--{file_path}")
        candidates.append(f"{recording_folderpath}/{Path(file_path).name}")
    if recording_folderpath.endswith("/recordings/MP4"):
        candidates.append(normalize_path_like(recording_folderpath.rsplit("/recordings/MP4", 1)[0]))
    if "r2d2-data-full/" in file_path:
        tail = file_path.split("r2d2-data-full/", 1)[1]
        candidates.append(normalize_path_like(tail.split("/trajectory", 1)[0]))
    if file_path:
        candidates.append(normalize_path_like(file_path))
        candidates.append(normalize_path_like(Path(file_path).name))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def build_episode_key(episode_metadata: dict[str, Any]) -> str | None:
    file_path = normalize_path_like(str(episode_metadata.get("file_path") or ""))
    recording_folderpath = normalize_path_like(str(episode_metadata.get("recording_folderpath") or ""))
    if not recording_folderpath or not file_path:
        return None
    return f"{recording_folderpath}--{file_path}"


def build_episode_id_reverse_lookup(blob: dict[str, Any]) -> dict[str, str]:
    return {
        normalize_path_like(str(path_like)): str(episode_id)
        for episode_id, path_like in blob.items()
        if path_like
    }


def resolve_episode_id(
    episode_metadata: dict[str, Any],
    reverse_lookup: dict[str, str],
) -> tuple[str | None, list[str]]:
    candidates = episode_to_path_candidates(episode_metadata)
    for candidate in candidates:
        if candidate in reverse_lookup:
            return reverse_lookup[candidate], candidates
    for candidate in candidates:
        for path_like, episode_id in reverse_lookup.items():
            if path_like.endswith(candidate):
                return episode_id, candidates
    return None, candidates


def serial_has_payload(entry: dict[str, Any] | None, serial: str) -> bool:
    if not isinstance(entry, dict):
        return False
    value = entry.get(str(serial))
    if value is None:
        return False
    return True


def resolve_calibrated_views(
    episode_id: str,
    camera_serials_blob: dict[str, Any],
    cam2base_sources: Iterable[tuple[str, dict[str, Any]]],
    intrinsics_blob: dict[str, Any],
) -> list[dict[str, str]]:
    episode_serials = camera_serials_blob.get(episode_id)
    episode_intrinsics = intrinsics_blob.get(episode_id)
    if not isinstance(episode_serials, dict) or not isinstance(episode_intrinsics, dict):
        return []
    normalized_sources = [
        (source_name, blob.get(episode_id))
        for source_name, blob in cam2base_sources
        if isinstance(blob, dict)
    ]
    views: list[dict[str, str]] = []
    for view_name, spec in VIEW_SPECS.items():
        serial = episode_serials.get(spec["serial_key"])
        if not serial:
            continue
        serial = str(serial)
        if not serial_has_payload(episode_intrinsics, serial):
            continue
        matched_source_name: str | None = None
        for source_name, episode_cam2base in normalized_sources:
            if serial_has_payload(episode_cam2base, serial):
                matched_source_name = source_name
                break
        if matched_source_name is None and spec.get("require_cam2base", True):
            continue
        views.append(
            {
                "view_name": view_name,
                "camera_name": spec["camera_name"],
                "serial_key": spec["serial_key"],
                "serial": serial,
                "cam2base_source": matched_source_name,
            }
        )
    return views


def normalize_keep_ranges(ranges: Any) -> list[list[int]]:
    normalized: list[list[int]] = []
    if not isinstance(ranges, list):
        return normalized
    for item in ranges:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            continue
        normalized.append([start_int, end_int])
    return normalized


def parse_keep_ranges_blob(blob: Any) -> tuple[dict[str, list[list[int]]], int]:
    keep_ranges_blob = blob
    filter_last_n = 0
    if isinstance(blob, dict) and "keep_ranges" in blob:
        keep_ranges_blob = blob.get("keep_ranges", {})
        metadata = blob.get("metadata", {})
        if isinstance(metadata, dict):
            try:
                filter_last_n = int(metadata.get("filter_last_n_in_ranges", 0) or 0)
            except (TypeError, ValueError):
                filter_last_n = 0

    keep_ranges_map: dict[str, list[list[int]]] = {}
    if isinstance(keep_ranges_blob, dict):
        for key, ranges in keep_ranges_blob.items():
            normalized = normalize_keep_ranges(ranges)
            if normalized:
                keep_ranges_map[normalize_path_like(str(key))] = normalized
    return keep_ranges_map, filter_last_n


def find_keep_ranges(
    episode_metadata: dict[str, Any],
    keep_ranges_map: dict[str, list[list[int]]],
) -> tuple[str | None, list[list[int]]]:
    candidates: list[str] = []
    episode_key = build_episode_key(episode_metadata)
    if episode_key:
        candidates.append(episode_key)
    candidates.extend(episode_to_path_candidates(episode_metadata))
    for candidate in candidates:
        for variant in candidate_keep_range_variants(candidate):
            if variant in keep_ranges_map:
                return variant, keep_ranges_map[variant]
    return episode_key, []


def estimate_kept_frames_from_ranges(
    keep_ranges: list[list[int]],
    filter_last_n: int,
) -> int:
    kept = 0
    for start, end in keep_ranges:
        effective_end = max(start, end - filter_last_n)
        kept += max(0, effective_end - start)
    return kept


def expand_keep_ranges_to_source_indices(
    keep_ranges: list[list[int]],
    filter_last_n: int,
) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for start, end in keep_ranges:
        effective_end = max(start, end - filter_last_n)
        for index in range(start, effective_end):
            if index in seen:
                continue
            indices.append(index)
            seen.add(index)
    return indices


def compute_resampled_positions(
    num_source_frames: int,
    source_step_rate_hz: float,
    output_fps: float,
) -> list[int]:
    if num_source_frames <= 0:
        return []
    if source_step_rate_hz <= 0 or output_fps <= 0:
        raise ValueError("source_step_rate_hz and output_fps must be positive")
    output_frames = max(1, int(round(num_source_frames * output_fps / source_step_rate_hz)))
    last_index = num_source_frames - 1
    return [
        min(int(frame_idx * source_step_rate_hz / output_fps), last_index)
        for frame_idx in range(output_frames)
    ]


def map_indices_between_domains(
    source_indices: list[int],
    source_total: int,
    target_total: int,
) -> list[int]:
    if not source_indices:
        return []
    if source_total <= 0 or target_total <= 0:
        raise ValueError("source_total and target_total must be positive")
    if source_total == 1:
        return [0 for _ in source_indices]
    target_last = target_total - 1
    source_last = source_total - 1
    return [
        min(target_last, max(0, int(round(index * target_last / source_last))))
        for index in source_indices
    ]


def split_episode_id_components(episode_id: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in episode_id.strip().split("+", 2)]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "unknown_hash", parts[1]
    return "unknown_lab", "unknown_hash", episode_id.strip() or "unknown_time"


def episode_output_relative_dir(episode_id: str) -> Path:
    lab_name, hash_value, _ = split_episode_id_components(episode_id)
    return Path(lab_name) / hash_value


def view_video_name(timestamp: str, view_name: str) -> str:
    return f"{timestamp}__{view_name}.mp4"


def raw_episode_root(
    relative_episode_path: str,
    raw_bucket_root: str = RAW_BUCKET_ROOT,
) -> str:
    return f"{raw_bucket_root.rstrip('/')}/{normalize_path_like(relative_episode_path)}"


def raw_trajectory_gs_path(
    relative_episode_path: str,
    raw_bucket_root: str = RAW_BUCKET_ROOT,
) -> str:
    return f"{raw_episode_root(relative_episode_path, raw_bucket_root)}/trajectory.h5"


def raw_metadata_gs_path(
    episode_id: str,
    relative_episode_path: str,
    raw_bucket_root: str = RAW_BUCKET_ROOT,
) -> str:
    return f"{raw_episode_root(relative_episode_path, raw_bucket_root)}/metadata_{episode_id}.json"


def raw_mp4_gs_path(
    relative_episode_path: str,
    serial: str,
    raw_bucket_root: str = RAW_BUCKET_ROOT,
) -> str:
    return f"{raw_episode_root(relative_episode_path, raw_bucket_root)}/recordings/MP4/{serial}.mp4"


def resolve_raw_metadata_gs_path(
    *,
    episode_id: str,
    raw_trajectory_gs_path_value: str,
) -> str:
    trajectory_path = str(raw_trajectory_gs_path_value).rstrip("/")
    if not trajectory_path.endswith("/trajectory.h5"):
        raise ValueError(f"Unexpected trajectory path: {trajectory_path}")
    episode_root = trajectory_path.rsplit("/trajectory.h5", 1)[0]
    return f"{episode_root}/metadata_{episode_id}.json"


def resolve_raw_svo_gs_path(
    *,
    raw_trajectory_gs_path_value: str,
    svo_path_value: str,
) -> str:
    svo_path_value = normalize_path_like(str(svo_path_value))
    if svo_path_value.startswith("gs://"):
        return svo_path_value
    episode_root = str(raw_trajectory_gs_path_value).rstrip("/")
    if not episode_root.endswith("/trajectory.h5"):
        raise ValueError(f"Unexpected trajectory path: {raw_trajectory_gs_path_value}")
    episode_root = episode_root.rsplit("/trajectory.h5", 1)[0]
    if "/recordings/SVO/" in svo_path_value:
        suffix = "/".join(svo_path_value.split("/")[-3:])
    else:
        suffix = svo_path_value
    return f"{episode_root}/{suffix}"


def load_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorflow is required for gs:// raw DROID access in this pipeline."
        ) from exc
    return tf


def _pose_vector_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Convert a single [tx, ty, tz, rx, ry, rz] pose (euler XYZ, R = Rz @ Ry @ Rx) to 4x4."""
    tx, ty, tz, rx, ry, rz = (float(v) for v in vec)
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot_z @ rot_y @ rot_x
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def load_wrist_cam2base_sequence(
    trajectory_h5_path: Path,
    wrist_serial: str,
) -> np.ndarray:
    """Load per-frame wrist cam2base matrices from a raw DROID trajectory.h5.

    Reads `observation/camera_extrinsics/{wrist_serial}_left` which is stored as
    [T, 6] pose vectors (tx, ty, tz, euler XYZ). Returns an [T, 4, 4] array of
    cam2base matrices (camera → robot base).

    Raises KeyError if the wrist pose dataset cannot be found and ValueError if
    the shape is unexpected.
    """
    key = f"{wrist_serial}_left"
    with h5py.File(trajectory_h5_path, "r") as handle:
        group = handle.get("observation/camera_extrinsics")
        if group is None:
            raise KeyError(
                f"Missing observation/camera_extrinsics in {trajectory_h5_path}"
            )
        if key not in group:
            available = sorted(list(group.keys()))[:12]
            raise KeyError(
                f"Missing wrist pose dataset '{key}' in {trajectory_h5_path}. "
                f"Available keys include: {available}"
            )
        pose_vectors = np.asarray(group[key], dtype=np.float64)
    if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6:
        raise ValueError(
            f"Expected wrist pose vectors [T, 6], got {pose_vectors.shape}"
        )
    matrices = np.empty((pose_vectors.shape[0], 4, 4), dtype=np.float64)
    for i in range(pose_vectors.shape[0]):
        matrices[i] = _pose_vector_to_matrix(pose_vectors[i])
    return matrices


def download_gs_file(gs_path: str, local_path: Path, overwrite: bool = False) -> Path:
    tf = load_tensorflow()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        if is_nonempty_file(local_path) and not overwrite:
            return local_path
        local_path.unlink(missing_ok=True)
    temp_path = local_path.with_name(f".{local_path.name}.partial.{os.getpid()}")
    try:
        tf.io.gfile.copy(gs_path, str(temp_path), overwrite=True)
        os.replace(temp_path, local_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return local_path


def h5_dataset_or_none(handle: h5py.File, candidates: Iterable[str]) -> np.ndarray | None:
    for key in candidates:
        if key in handle:
            return np.asarray(handle[key], dtype=np.float64)
    return None


def load_joint_gripper_from_h5(local_h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(local_h5_path, "r") as f:
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
        raise KeyError(f"Could not read joint/gripper trajectories from {local_h5_path}")
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 1:
        joints = joints.reshape(-1, 1)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1, 1)
    return joints, gripper


def sanitize_valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    return np.isfinite(depth_m) & (depth_m > 0.0)


def encode_inverse_depth_u16(
    depth_m: np.ndarray,
    *,
    near_m: float,
    far_m: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if near_m <= 0.0 or far_m <= 0.0 or far_m <= near_m:
        raise ValueError(f"Invalid near/far values: near={near_m}, far={far_m}")
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if valid_mask is None:
        valid_mask = sanitize_valid_depth_mask(depth_m)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & sanitize_valid_depth_mask(depth_m)

    encoded = np.zeros(depth_m.shape, dtype=np.uint16)
    if not np.any(valid_mask):
        return encoded, valid_mask.astype(np.uint8)

    clipped = np.clip(depth_m[valid_mask], float(near_m), float(far_m))
    inv_depth = 1.0 / clipped
    inv_near = 1.0 / float(near_m)
    inv_far = 1.0 / float(far_m)
    normalized = (inv_depth - inv_far) / (inv_near - inv_far)
    quantized = np.clip(np.round(normalized * 65534.0) + 1.0, 1.0, 65535.0).astype(np.uint16)
    encoded[valid_mask] = quantized
    return encoded, valid_mask.astype(np.uint8)


def resolve_asset_path(
    explicit_path: Path | None,
    assets_dir: Path,
    candidates: list[str],
) -> Path:
    if explicit_path is not None:
        return explicit_path
    for candidate in candidates:
        path = assets_dir / candidate
        if path.exists():
            return path
    return assets_dir / candidates[0]
