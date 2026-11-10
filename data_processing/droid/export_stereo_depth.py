#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .common import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_ASSETS_DIR,
        DEFAULT_OUTPUT_ROOT,
        VIEW_SPECS,
        atomic_h5_write,
        atomic_savez_compressed,
        download_gs_file,
        encode_inverse_depth_u16,
        is_nonempty_file,
        load_json,
        map_indices_between_domains,
        resolve_raw_svo_gs_path,
    )
except ImportError:
    from common import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_ASSETS_DIR,
        DEFAULT_OUTPUT_ROOT,
        VIEW_SPECS,
        atomic_h5_write,
        atomic_savez_compressed,
        download_gs_file,
        encode_inverse_depth_u16,
        is_nonempty_file,
        load_json,
        map_indices_between_domains,
        resolve_raw_svo_gs_path,
    )


POINTWORLD_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "PointWorld"
DEFAULT_INTRINSICS_JSON = DEFAULT_ASSETS_DIR / "intrinsics.json"
DEFAULT_ZED_CALIB_DIR = POINTWORLD_ROOT / "real" / "droid_zed_calibrations"
DEFAULT_FOUNDATION_STEREO_CKPT = os.environ.get(
    "FOUNDATION_STEREO_CKPT",
    str(
        POINTWORLD_ROOT
        / "checkpoints"
        / "foundationstereo"
        / "11-33-40"
        / "model_best_bp2.pth"
    ),
)
DEFAULT_FOUNDATION_STEREO_CFG = os.environ.get(
    "FOUNDATION_STEREO_CFG",
    str(POINTWORLD_ROOT / "checkpoints" / "foundationstereo" / "11-33-40" / "cfg.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-view DROID stereo depth aligned to render_v2 videos."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root used to read existing render_v2 RGB videos. Defaults to --output-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--zed-calibration-dir", type=Path, default=DEFAULT_ZED_CALIB_DIR)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-cache", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-tag", type=str, default=None)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--near-m", type=float, default=0.2)
    parser.add_argument("--far-m", type=float, default=3.0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Sample only every Nth eligible frame index. Use with --clip-length-frames to target clip starts.",
    )
    parser.add_argument(
        "--clip-length-frames",
        type=int,
        default=0,
        help="If >0, sample only valid clip-start frames for clips of this length. If <=0, sample all frames.",
    )
    parser.add_argument(
        "--target-frame-indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated RGB frame indices to export exactly. "
            "When set, this overrides --frame-stride/--clip-length-frames."
        ),
    )
    parser.add_argument("--foundation-stereo-ckpt", type=str, default=DEFAULT_FOUNDATION_STEREO_CKPT)
    parser.add_argument("--foundation-stereo-cfg", type=str, default=DEFAULT_FOUNDATION_STEREO_CFG)
    parser.add_argument(
        "--stereo-output-subdir",
        type=str,
        default=None,
        help="Override the stereo NPZ subdir name (default uses manifest's stereo_inverse_depth/...). "
             "Used to redirect output to e.g. stereo_inverse_depth_clips81/ for 81-aligned re-runs.",
    )
    parser.add_argument(
        "--metric-first-output-subdir",
        type=str,
        default=None,
        help="Override the metric-first depth subdir name analogously to --stereo-output-subdir.",
    )
    return parser.parse_args()


class CacheMissError(RuntimeError):
    pass


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


def resolve_intrinsics_json(rows: list[dict[str, Any]], explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    for row in rows:
        annotation_assets = row.get("annotation_assets")
        if not isinstance(annotation_assets, dict):
            continue
        candidate = annotation_assets.get("intrinsics_json")
        if isinstance(candidate, str) and candidate.strip():
            return Path(candidate)
    return DEFAULT_INTRINSICS_JSON


def parse_intrinsics_matrix(candidate: Any) -> np.ndarray | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        for key in ("K", "camera_matrix", "cameraMatrix", "intrinsics", "matrix"):
            if key in candidate:
                matrix = parse_intrinsics_matrix(candidate[key])
                if matrix is not None:
                    return matrix
        if all(k in candidate for k in ("fx", "fy", "cx", "cy")):
            return np.array(
                [
                    [float(candidate["fx"]), 0.0, float(candidate["cx"])],
                    [0.0, float(candidate["fy"]), float(candidate["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
    if isinstance(candidate, list):
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.size == 4:
            fx, cx, fy, cy = arr.reshape(4)
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return None


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
    end_index = len(rows) if args.end_index is None else min(len(rows), int(args.end_index))
    ranged = rows[max(0, int(args.start_index)) : end_index]
    sharded = [row for idx, row in enumerate(ranged) if idx % args.num_shards == args.shard_index]
    final_rows = sharded[: max(0, int(args.limit))] if args.limit is not None else sharded
    return final_rows, {
        "manifest_rows": len(rows),
        "range_rows": len(ranged),
        "shard_rows": len(sharded),
        "selected_rows": len(final_rows),
    }


def prepare_result_logs(output_root: Path, overwrite: bool, task_tag: str) -> tuple[Path, Path, Path]:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_path = manifest_dir / f"stereo_depth_results_{task_tag}.jsonl"
    failures_path = manifest_dir / f"stereo_depth_failures_{task_tag}.jsonl"
    summary_path = manifest_dir / f"stereo_depth_summary_{task_tag}.json"
    if overwrite:
        for path in (results_path, failures_path, summary_path):
            if path.exists():
                path.unlink()
    for path in (results_path, failures_path):
        path.touch(exist_ok=True)
    return results_path, failures_path, summary_path


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_target_frame_positions(
    *,
    total_frames: int,
    frame_stride: int,
    clip_length_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return []
    stride = max(1, int(frame_stride))
    clip_length = int(clip_length_frames)

    if clip_length > 0:
        max_start = max(total_frames - clip_length, 0)
        positions = list(range(0, max_start + 1, stride))
        if not positions:
            positions = [0]
        if positions[-1] != max_start:
            positions.append(max_start)
        return positions

    positions = list(range(0, total_frames, stride))
    if not positions:
        positions = [0]
    last_index = total_frames - 1
    if positions[-1] != last_index:
        positions.append(last_index)
    return positions


def parse_target_frame_indices(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out or None


def report_progress(
    *,
    processed: int,
    total: int,
    exported: int,
    skipped: int,
    failed: int,
    started_at: float,
) -> None:
    elapsed = time.monotonic() - started_at
    print(
        "[stereo-depth] "
        f"processed={processed}/{total} "
        f"exported={exported} "
        f"skipped={skipped} "
        f"failed={failed} "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )


def load_depth_estimator(ckpt_path: str, cfg_path: str | None):
    if str(POINTWORLD_ROOT) not in sys.path:
        sys.path.insert(0, str(POINTWORLD_ROOT))
    from depth_estimator import DepthEstimator  # type: ignore

    return DepthEstimator(
        ckpt_path=str(ckpt_path),
        cfg_path=None if cfg_path in (None, "", "None") else str(cfg_path),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cuda",
    )


def load_pyzed():
    try:
        import pyzed.sl as sl  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyzed is required for ZED SVO processing. Install it in the dedicated droid-depth env."
        ) from exc
    return sl


def open_svo(local_svo_path: Path):
    sl = load_pyzed()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(local_svo_path))
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO {local_svo_path}: {err}")
    return sl, zed


def scale_intrinsics(intrinsics: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    scaled = intrinsics.astype(np.float32, copy=True)
    scaled[0, 0] *= dst_w / float(src_w)
    scaled[0, 2] *= dst_w / float(src_w)
    scaled[1, 1] *= dst_h / float(src_h)
    scaled[1, 2] *= dst_h / float(src_h)
    return scaled


def extract_intrinsics_and_baseline(camera_info: Any, target_size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    calibration_params = camera_info.camera_configuration.calibration_parameters
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy
    intrinsic = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source_size = (
        int(camera_info.camera_configuration.resolution.width),
        int(camera_info.camera_configuration.resolution.height),
    )
    intrinsic = scale_intrinsics(intrinsic, source_size=source_size, target_size=target_size)
    baseline = float(calibration_params.stereo_transform.get_translation().get()[0])
    return intrinsic, baseline, source_size


def resolve_intrinsics_for_record(
    *,
    record: dict[str, Any],
    intrinsics_blob: dict[str, Any],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    episode_id = str(record["episode_id"])
    serial = str(record["serial"])
    episode_intrinsics = intrinsics_blob.get(episode_id)
    if not isinstance(episode_intrinsics, dict):
        raise KeyError(f"Missing intrinsics entry for episode '{episode_id}'")
    serial_intrinsics = episode_intrinsics.get(serial)
    if not isinstance(serial_intrinsics, dict):
        raise KeyError(f"Missing intrinsics for episode '{episode_id}' serial '{serial}'")
    source_width = int(serial_intrinsics.get("width") or 0)
    source_height = int(serial_intrinsics.get("height") or 0)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid intrinsics source size for episode '{episode_id}' serial '{serial}'")
    intrinsic = parse_intrinsics_matrix(serial_intrinsics)
    if intrinsic is None:
        raise ValueError(f"Failed to parse intrinsics for episode '{episode_id}' serial '{serial}'")
    return scale_intrinsics(intrinsic, source_size=(source_width, source_height), target_size=target_size), (
        source_width,
        source_height,
    )


def resolve_zed_calibration_conf(serial: str, calibration_dir: Path) -> Path:
    candidate = calibration_dir / f"SN{serial}.conf"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing ZED calibration conf for serial {serial}: {candidate}. "
            "Download missing files with PointWorld real/download_zed_calibrations.py."
        )
    return candidate


def load_zed_baseline_m(serial: str, calibration_dir: Path) -> float:
    conf_path = resolve_zed_calibration_conf(serial, calibration_dir)
    parser = configparser.ConfigParser()
    parser.read(conf_path)
    try:
        baseline_mm = float(parser["STEREO"]["Baseline"])
    except Exception as exc:
        raise ValueError(f"Failed to read stereo baseline from {conf_path}") from exc
    return baseline_mm / 1000.0


def bgr_or_bgra_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape {frame.shape}")
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported channel count: {frame.shape}")


def extract_svo_stereo_frames(
    *,
    zed: Any,
    target_indices: list[int],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sl = load_pyzed()
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    timestamps: list[int] = []
    frame_idx = 0
    target_ptr = 0
    target_width, target_height = target_size
    while target_ptr < len(target_indices):
        err = zed.grab(runtime_params)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed while grabbing SVO frame {frame_idx}: {err}")

        wanted_idx = int(target_indices[target_ptr])
        if frame_idx < wanted_idx:
            frame_idx += 1
            continue
        if frame_idx > wanted_idx:
            raise RuntimeError(f"Missed target SVO frame {wanted_idx} while extracting stereo frames")

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
        left_rgb = bgr_or_bgra_to_rgb(left_mat.get_data().copy())
        right_rgb = bgr_or_bgra_to_rgb(right_mat.get_data().copy())
        if left_rgb.shape[1] != target_width or left_rgb.shape[0] != target_height:
            left_rgb = cv2.resize(left_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            right_rgb = cv2.resize(right_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        timestamp_ms = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
        while target_ptr < len(target_indices) and int(target_indices[target_ptr]) == frame_idx:
            left_frames.append(left_rgb.copy())
            right_frames.append(right_rgb.copy())
            timestamps.append(timestamp_ms)
            target_ptr += 1
        frame_idx += 1

    if target_ptr != len(target_indices):
        raise RuntimeError(
            f"Stopped after {target_ptr}/{len(target_indices)} selected stereo frames"
        )
    return (
        np.stack(left_frames, axis=0),
        np.stack(right_frames, axis=0),
        np.asarray(timestamps, dtype=np.int64),
    )


def get_svo_frame_count_cv2(local_svo_path: Path) -> int:
    cap = cv2.VideoCapture(str(local_svo_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open SVO with OpenCV: {local_svo_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Could not read frame count from SVO with OpenCV: {local_svo_path}")
    return frame_count


def resolve_svo_frame_count_for_record(record: dict[str, Any], local_svo_path: Path) -> int:
    try:
        frame_count = get_svo_frame_count_cv2(local_svo_path)
        # Some OpenCV builds return garbage negative values for SVOs while still
        # being able to decode frames normally.
        if frame_count > 0 and frame_count < 10_000_000:
            return frame_count
    except RuntimeError:
        pass

    raw_frame_count = int(record.get("raw_frame_count") or 0)
    if raw_frame_count > 0:
        return raw_frame_count

    raise RuntimeError(
        f"Could not infer SVO frame count for {local_svo_path}; OpenCV metadata was invalid and "
        "raw_frame_count was missing from the depth manifest"
    )


def extract_svo_stereo_frames_cv2(
    *,
    local_svo_path: Path,
    target_indices: list[int],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    cap = cv2.VideoCapture(str(local_svo_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open SVO with OpenCV: {local_svo_path}")
    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    timestamps: list[int] = []
    target_width, target_height = target_size
    frame_idx = 0
    target_ptr = 0
    native_half_size: tuple[int, int] | None = None
    try:
        while target_ptr < len(target_indices):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            wanted_idx = int(target_indices[target_ptr])
            if frame_idx < wanted_idx:
                frame_idx += 1
                continue
            if frame_idx > wanted_idx:
                raise RuntimeError(f"Missed target SVO frame {wanted_idx} while extracting stereo frames")
            height, width = frame.shape[:2]
            if width % 2 != 0:
                raise RuntimeError(f"Unexpected side-by-side SVO width {width} for {local_svo_path}")
            half_width = width // 2
            native_half_size = (half_width, height)
            left_rgb = cv2.cvtColor(frame[:, :half_width], cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(frame[:, half_width:], cv2.COLOR_BGR2RGB)
            if half_width != target_width or height != target_height:
                left_rgb = cv2.resize(left_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                right_rgb = cv2.resize(right_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
            while target_ptr < len(target_indices) and int(target_indices[target_ptr]) == frame_idx:
                left_frames.append(left_rgb.copy())
                right_frames.append(right_rgb.copy())
                timestamps.append(timestamp_ms)
                target_ptr += 1
            frame_idx += 1
    finally:
        cap.release()
    if target_ptr != len(target_indices):
        raise RuntimeError(
            f"Stopped after {target_ptr}/{len(target_indices)} selected stereo frames while reading {local_svo_path}"
        )
    if native_half_size is None:
        raise RuntimeError(f"Could not infer native stereo size from {local_svo_path}")
    return (
        np.stack(left_frames, axis=0),
        np.stack(right_frames, axis=0),
        np.asarray(timestamps, dtype=np.int64),
        native_half_size,
    )


def get_video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RGB video {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video size from {path}")
    return width, height


def _swap_subdir(rel_path: Path, original_subdir: str, override_subdir: str | None) -> Path:
    """Replace `original_subdir` in `rel_path` with `override_subdir` if provided."""
    if not override_subdir:
        return rel_path
    parts = list(rel_path.parts)
    try:
        idx = parts.index(original_subdir)
    except ValueError:
        # Defensively fall back to last-but-one part (the dir component).
        idx = max(0, len(parts) - 2)
    parts[idx] = override_subdir
    return Path(*parts)


def resolve_view_output_paths(
    record: dict[str, Any],
    output_root: Path,
    *,
    stereo_subdir_override: str | None = None,
    metric_first_subdir_override: str | None = None,
) -> tuple[Path, Path]:
    stereo_rel = Path(str(record["stereo_inverse_depth_relative_path"]))
    metric_rel = Path(str(record["stereo_metric_first_relative_path"]))
    stereo_rel = _swap_subdir(stereo_rel, "stereo_inverse_depth", stereo_subdir_override)
    metric_rel = _swap_subdir(metric_rel, "stereo_depth_metric_first", metric_first_subdir_override)
    return output_root / stereo_rel, output_root / metric_rel


def ensure_localized_stereo_assets(
    *,
    record: dict[str, Any],
    cache_root: Path,
    overwrite_cache: bool,
    only_cache: bool,
) -> tuple[Path, Path]:
    episode_id = str(record["episode_id"])
    serial = str(record["serial"])
    metadata_path = cache_root / episode_id / "metadata.json"
    if only_cache and not is_nonempty_file(metadata_path):
        raise CacheMissError(f"Missing cached metadata: {metadata_path}")
    metadata_local = download_gs_file(
        gs_path=str(record["raw_metadata_gs_path"]),
        local_path=metadata_path,
        overwrite=overwrite_cache,
    )
    metadata = load_json(metadata_local)
    svo_key = str(VIEW_SPECS[str(record["view_name"])]["svo_key"])
    if svo_key not in metadata:
        raise KeyError(f"Missing {svo_key} in {metadata_local}")
    raw_svo_path = resolve_raw_svo_gs_path(
        raw_trajectory_gs_path_value=str(record["raw_trajectory_gs_path"]),
        svo_path_value=str(metadata[svo_key]),
    )
    local_svo_path = cache_root / episode_id / f"{serial}.svo"
    if only_cache and not is_nonempty_file(local_svo_path):
        raise CacheMissError(f"Missing cached SVO: {local_svo_path}")
    svo_local = download_gs_file(
        gs_path=raw_svo_path,
        local_path=local_svo_path,
        overwrite=overwrite_cache,
    )
    return metadata_local, svo_local


def export_stereo_depth_for_record(
    *,
    record: dict[str, Any],
    input_root: Path,
    output_root: Path,
    cache_root: Path,
    intrinsics_blob: dict[str, Any],
    zed_calibration_dir: Path,
    overwrite_cache: bool,
    only_cache: bool,
    depth_estimator: Any,
    batch_size: int,
    near_m: float,
    far_m: float,
    frame_stride: int,
    clip_length_frames: int,
    target_frame_indices: list[int] | None = None,
    stereo_subdir_override: str | None = None,
    metric_first_subdir_override: str | None = None,
) -> dict[str, Any]:
    stereo_npz_path, metric_first_path = resolve_view_output_paths(
        record,
        output_root=output_root,
        stereo_subdir_override=stereo_subdir_override,
        metric_first_subdir_override=metric_first_subdir_override,
    )
    rgb_video_path = input_root / Path(str(record["rgb_relative_path"]))
    if not is_nonempty_file(rgb_video_path):
        raise FileNotFoundError(f"Missing RGB render_v2 video: {rgb_video_path}")

    _, svo_local = ensure_localized_stereo_assets(
        record=record,
        cache_root=cache_root,
        overwrite_cache=overwrite_cache,
        only_cache=only_cache,
    )
    target_size = get_video_size(rgb_video_path)
    num_source_steps = int(record["num_source_steps"])
    source_indices = [int(idx) for idx in record.get("exported_source_step_indices", [])]
    if not source_indices:
        raise RuntimeError(f"Missing exported_source_step_indices for {record['episode_id']} {record['view_name']}")
    if target_frame_indices is None:
        target_rgb_frame_indices = select_target_frame_positions(
            total_frames=len(source_indices),
            frame_stride=frame_stride,
            clip_length_frames=clip_length_frames,
        )
    else:
        target_rgb_frame_indices = [int(i) for i in target_frame_indices]
        invalid = [i for i in target_rgb_frame_indices if i < 0 or i >= len(source_indices)]
        if invalid:
            raise ValueError(
                f"target_frame_indices out of range for {record['episode_id']} {record['view_name']}: "
                f"{invalid}; valid range is [0, {len(source_indices) - 1}]"
            )
    sampled_source_indices = [source_indices[int(i)] for i in target_rgb_frame_indices]

    try:
        sl, zed = open_svo(svo_local)
    except ModuleNotFoundError:
        zed = None
        sl = None

    if zed is not None:
        try:
            svo_frame_count = int(zed.get_svo_number_of_frames() or 0)
            if svo_frame_count <= 0:
                raise RuntimeError(f"Could not read SVO frame count from {svo_local}")
            target_svo_indices = map_indices_between_domains(
                source_indices=sampled_source_indices,
                source_total=num_source_steps,
                target_total=svo_frame_count,
            )
            left_frames, right_frames, timestamps = extract_svo_stereo_frames(
                zed=zed,
                target_indices=target_svo_indices,
                target_size=target_size,
            )
            camera_info = zed.get_camera_information()
            intrinsics, baseline_m, source_size = extract_intrinsics_and_baseline(camera_info, target_size=target_size)
        finally:
            zed.close()
    else:
        svo_frame_count = resolve_svo_frame_count_for_record(record, svo_local)
        target_svo_indices = map_indices_between_domains(
            source_indices=sampled_source_indices,
            source_total=num_source_steps,
            target_total=svo_frame_count,
        )
        left_frames, right_frames, timestamps, native_half_size = extract_svo_stereo_frames_cv2(
            local_svo_path=svo_local,
            target_indices=target_svo_indices,
            target_size=target_size,
        )
        intrinsics, source_size = resolve_intrinsics_for_record(
            record=record,
            intrinsics_blob=intrinsics_blob,
            target_size=target_size,
        )
        baseline_m = load_zed_baseline_m(str(record["serial"]), zed_calibration_dir)
        if native_half_size != source_size:
            source_size = native_half_size

    depth_estimator.set_camera_params(baseline_m, intrinsics)
    depth_m = np.asarray(
        depth_estimator.infer_depth_batch(left_frames, right_frames, batch_size=max(1, int(batch_size))),
        dtype=np.float32,
    )
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    inv_depth_u16, valid_mask_u8 = encode_inverse_depth_u16(
        depth_m,
        near_m=float(near_m),
        far_m=float(far_m),
        valid_mask=valid_mask,
    )

    atomic_savez_compressed(
        stereo_npz_path,
        inv_depth_u16=inv_depth_u16,
        valid_mask=valid_mask_u8,
        frame_indices=np.asarray(target_rgb_frame_indices, dtype=np.int32),
        near_m=np.asarray(float(near_m), dtype=np.float32),
        far_m=np.asarray(float(far_m), dtype=np.float32),
        timestamps=timestamps,
        baseline_m=np.asarray(float(baseline_m), dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        source_size=np.asarray(source_size, dtype=np.int32),
        target_size=np.asarray(target_size, dtype=np.int32),
    )
    atomic_h5_write(
        metric_first_path,
        datasets={
            "depth_m": np.asarray(depth_m[0], dtype=np.float32),
            "valid_mask": np.asarray(valid_mask_u8[0], dtype=np.uint8),
        },
        attrs={
            "episode_id": str(record["episode_id"]),
            "view_name": str(record["view_name"]),
            "frame_index": int(target_rgb_frame_indices[0]),
            "near_m": float(near_m),
            "far_m": float(far_m),
            "baseline_m": float(baseline_m),
            "encoding": "metric_depth_first_frame",
        },
    )
    return {
        "episode_id": str(record["episode_id"]),
        "view_name": str(record["view_name"]),
        "status": "exported",
        "frame_count": int(depth_m.shape[0]),
        "dense_rgb_frame_count": int(len(source_indices)),
        "sampled_rgb_frame_indices": [int(i) for i in target_rgb_frame_indices],
        "frame_stride": int(max(1, int(frame_stride))),
        "clip_length_frames": int(max(0, int(clip_length_frames))),
        "target_frame_indices": None if target_frame_indices is None else [int(i) for i in target_frame_indices],
        "stereo_inverse_depth_path": str(stereo_npz_path),
        "stereo_metric_first_path": str(metric_first_path),
        "svo_frame_count": int(svo_frame_count),
        "rgb_frame_count": int(len(source_indices)),
    }


def write_task_summary(
    *,
    summary_path: Path,
    task_tag: str,
    args: argparse.Namespace,
    counts: dict[str, int],
    results_path: Path,
    failures_path: Path,
    exported: int,
    skipped: int,
    failed: int,
) -> None:
    payload = {
        "task_tag": task_tag,
        "manifest_jsonl": str(args.manifest_jsonl),
        "output_root": str(args.output_root),
        "cache_root": str(args.cache_root),
        "counts": counts,
        "exported_records": int(exported),
        "skipped_records": int(skipped),
        "failed_records": int(failed),
        "near_m": float(args.near_m),
        "far_m": float(args.far_m),
        "batch_size": int(args.batch_size),
        "frame_stride": int(args.frame_stride),
        "clip_length_frames": int(args.clip_length_frames),
        "target_frame_indices": parse_target_frame_indices(args.target_frame_indices),
        "results_jsonl": str(results_path),
        "failures_jsonl": str(failures_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_root = args.output_root if args.input_root is None else args.input_root
    rows = load_manifest_rows(args.manifest_jsonl)
    selected_rows, selection_counts = select_manifest_rows(args, rows)
    task_tag = resolve_task_tag(args)
    results_path, failures_path, summary_path = prepare_result_logs(args.output_root, args.overwrite, task_tag)
    intrinsics_path = resolve_intrinsics_json(rows, args.intrinsics_json)
    intrinsics_blob = load_json(intrinsics_path)
    if not isinstance(intrinsics_blob, dict):
        raise ValueError(f"Expected dict in {intrinsics_path}")
    depth_estimator = load_depth_estimator(args.foundation_stereo_ckpt, args.foundation_stereo_cfg)
    target_frame_indices = parse_target_frame_indices(args.target_frame_indices)
    exported_records = 0
    skipped_records = 0
    failure_records: list[dict[str, Any]] = []
    started_at = time.monotonic()
    progress_interval = max(1, int(args.progress_interval))
    total_selected = len(selected_rows)

    for idx, row in enumerate(selected_rows, start=1):
        stereo_npz_path, metric_first_path = resolve_view_output_paths(
            row,
            output_root=args.output_root,
            stereo_subdir_override=args.stereo_output_subdir,
            metric_first_subdir_override=args.metric_first_output_subdir,
        )
        if (
            not args.overwrite
            and is_nonempty_file(stereo_npz_path)
            and is_nonempty_file(metric_first_path)
        ):
            skipped_records += 1
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": row.get("episode_id"),
                        "view_name": row.get("view_name"),
                        "status": "skipped_existing",
                    }
                ],
            )
        else:
            try:
                summary = export_stereo_depth_for_record(
                    record=row,
                    input_root=input_root,
                    output_root=args.output_root,
                    cache_root=args.cache_root,
                    intrinsics_blob=intrinsics_blob,
                    zed_calibration_dir=args.zed_calibration_dir,
                    overwrite_cache=bool(args.overwrite_cache),
                    only_cache=bool(args.only_cache),
                    depth_estimator=depth_estimator,
                    batch_size=int(args.batch_size),
                    near_m=float(args.near_m),
                    far_m=float(args.far_m),
                    frame_stride=int(args.frame_stride),
                    clip_length_frames=int(args.clip_length_frames),
                    target_frame_indices=target_frame_indices,
                    stereo_subdir_override=args.stereo_output_subdir,
                    metric_first_subdir_override=args.metric_first_output_subdir,
                )
                exported_records += 1
                append_jsonl(results_path, [{"task_tag": task_tag, **summary}])
            except Exception as exc:
                failure = {
                    "task_tag": task_tag,
                    "episode_id": row.get("episode_id"),
                    "view_name": row.get("view_name"),
                    "status": "failed",
                    "error": str(exc),
                }
                failure_records.append(failure)
                append_jsonl(failures_path, [failure])

        if idx % progress_interval == 0 or idx == total_selected:
            report_progress(
                processed=idx,
                total=total_selected,
                exported=exported_records,
                skipped=skipped_records,
                failed=len(failure_records),
                started_at=started_at,
            )

    write_task_summary(
        summary_path=summary_path,
        task_tag=task_tag,
        args=args,
        counts=selection_counts,
        results_path=results_path,
        failures_path=failures_path,
        exported=exported_records,
        skipped=skipped_records,
        failed=len(failure_records),
    )


if __name__ == "__main__":
    main()
