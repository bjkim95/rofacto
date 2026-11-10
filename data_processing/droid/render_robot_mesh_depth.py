#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from yourdfpy import URDF

try:
    from .common import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_OUTPUT_ROOT,
        atomic_h5_write,
        atomic_savez_compressed,
        encode_inverse_depth_u16,
        is_nonempty_file,
        load_json,
        load_wrist_cam2base_sequence,
    )
    from .render_robot_mesh_videos import (
        DEFAULT_CAM2BASE_JSON,
        DEFAULT_CAM2BASE_SUPERSET_JSON,
        DEFAULT_CLOAK_WRIST_EXTRINSICS_JSON,
        DEFAULT_INTRINSICS_JSON,
        DEFAULT_URDF_PATH,
        WRIST_VIEW_NAME,
        append_jsonl,
        ensure_urdf_ready,
        freeze_arm_joints_to_first_frame,
        load_episode_inputs,
        load_manifest_rows,
        load_named_cam2base_blobs,
        prepare_result_logs,
        resolve_annotation_path,
        resolve_task_tag,
        resolve_wrist_base_to_camera_sequence,
        select_manifest_rows,
    )
except ImportError:
    from common import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_OUTPUT_ROOT,
        atomic_h5_write,
        atomic_savez_compressed,
        encode_inverse_depth_u16,
        is_nonempty_file,
        load_json,
        load_wrist_cam2base_sequence,
    )
    from render_robot_mesh_videos import (
        DEFAULT_CAM2BASE_JSON,
        DEFAULT_CAM2BASE_SUPERSET_JSON,
        DEFAULT_CLOAK_WRIST_EXTRINSICS_JSON,
        DEFAULT_INTRINSICS_JSON,
        DEFAULT_URDF_PATH,
        WRIST_VIEW_NAME,
        append_jsonl,
        ensure_urdf_ready,
        freeze_arm_joints_to_first_frame,
        load_episode_inputs,
        load_manifest_rows,
        load_named_cam2base_blobs,
        prepare_result_logs,
        resolve_annotation_path,
        resolve_task_tag,
        resolve_wrist_base_to_camera_sequence,
        select_manifest_rows,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_OVERLAY_DIR = REPO_ROOT / "data_processing" / "droid_urdf_overlay"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LEGACY_OVERLAY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_OVERLAY_DIR))

from droid_urdf_overlay import (  # type: ignore  # noqa: E402
    materialize_tool_adjusted_urdf,
    read_intrinsics,
    read_matrix,
)
from data_processing.droid.depth_renderer import render_depth_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render numeric robot mesh depth aligned to DROID render_v2 videos."
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
    parser.add_argument("--cam2base-json", type=Path, default=None)
    parser.add_argument("--cam2base-superset-json", type=Path, default=None)
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument(
        "--wrist-pose-source",
        choices=["trajectory_h5", "proprio_gripper2wrist", "cloak_cam_to_gripper"],
        default="trajectory_h5",
        help=(
            "Pose source for wrist depth renders. trajectory_h5 uses DROID camera_extrinsics; "
            "proprio_gripper2wrist composes gripper cartesian pose with a robot-specific "
            "gripper-to-wrist transform; cloak_cam_to_gripper uses Cloak's per-episode "
            "silhouette-calibrated DROID wrist extrinsics and falls back when missing."
        ),
    )
    parser.add_argument(
        "--gripper2wrist-json",
        type=Path,
        default=REPO_ROOT
        / "data_processing"
        / "third_party"
        / "PointWorld"
        / "real"
        / "gripper2wrist_transforms.json",
        help="PointWorld-style robot_serial -> gripper-to-wrist transform JSON.",
    )
    parser.add_argument(
        "--wrist-proprio-flip",
        choices=["none", "z180"],
        default="none",
        help="Optional camera-frame 180 degree Z flip for proprio_gripper2wrist.",
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
        choices=["base_world", "gripper_local"],
        default="gripper_local",
        help=(
            "Mirror render_robot_mesh_videos: 'gripper_local' freezes arm joints to "
            "frame 0 and uses a static base_to_camera (camera locked to wrist link); "
            "'base_world' uses the full per-frame trajectory."
        ),
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument(
        "--robot-only-render-batch-size",
        type=int,
        default=4,
        help="Batch size used for depth rasterization.",
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
        default=0,
        help="If > 0, resize the rendered depth to this short side. Default 0 keeps the source RGB size.",
    )
    parser.add_argument(
        "--joint-suffix", type=str, default="",
        help="Suffix appended to the manifest joint npy stem (e.g. '_action').",
    )
    parser.add_argument(
        "--gripper-suffix", type=str, default="",
        help="Suffix appended to the manifest gripper npy stem (e.g. '_action').",
    )
    parser.add_argument(
        "--view-name",
        action="append",
        default=None,
        help="Render only this view name. May be repeated, e.g. --view-name wrist.",
    )
    parser.add_argument(
        "--robot-depth-output-dir-name",
        type=str,
        default=None,
        help=(
            "Optional replacement for the manifest's robot_mesh_inverse_depth "
            "directory name. Defaults to the manifest path unchanged."
        ),
    )
    parser.add_argument(
        "--robot-metric-first-output-dir-name",
        type=str,
        default=None,
        help=(
            "Optional replacement for the manifest's robot_mesh_depth_metric_first "
            "directory name. Defaults to the manifest path unchanged."
        ),
    )
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--base-adjust-json", type=Path, default=None)
    parser.add_argument("--tool-adjust-json", type=Path, default=None)
    parser.add_argument("--near-m", type=float, default=0.2)
    parser.add_argument("--far-m", type=float, default=3.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--task-tag", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_task_summary(
    *,
    summary_path: Path,
    task_tag: str,
    args: argparse.Namespace,
    counts: dict[str, Any],
    results_path: Path,
    failures_path: Path,
    rendered: int,
    skipped: int,
    failed: int,
) -> None:
    payload = {
        "task_tag": task_tag,
        "manifest_jsonl": str(args.manifest_jsonl),
        "output_root": str(args.output_root),
        "urdf_path": str(args.urdf_path),
        "counts": counts,
        "rendered_records": int(rendered),
        "skipped_records": int(skipped),
        "failed_records": int(failed),
        "near_m": float(args.near_m),
        "far_m": float(args.far_m),
        "results_jsonl": str(results_path),
        "failures_jsonl": str(failures_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_row_calibration(
    *,
    record: dict[str, Any],
    intrinsics_blob: dict[str, Any],
    cam2base_blobs: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, str]:
    episode_id = str(record["episode_id"])
    serial = str(record["serial"])
    episode_intrinsics = intrinsics_blob.get(episode_id)
    if not isinstance(episode_intrinsics, dict):
        raise KeyError(f"Missing intrinsics entry for episode '{episode_id}'")
    intrinsics = read_intrinsics(episode_intrinsics.get(serial))
    if intrinsics is None:
        raise KeyError(f"Could not parse intrinsics for episode '{episode_id}' serial '{serial}'")
    for source_name, blob in cam2base_blobs.items():
        episode_cam2base = blob.get(episode_id)
        if not isinstance(episode_cam2base, dict):
            continue
        cam2base = read_matrix(episode_cam2base.get(serial))
        if cam2base is not None:
            return cam2base, intrinsics, source_name
    raise KeyError(
        f"Could not parse cam2base for episode '{episode_id}' serial '{serial}' from any configured source"
    )


def filter_rows_by_view(
    rows: list[dict[str, Any]],
    view_names: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not view_names:
        return rows, {
            "view_filtered_rows": len(rows),
            "selected_view_names": None,
        }
    allowed = {str(name) for name in view_names}
    filtered = [row for row in rows if str(row.get("view_name")) in allowed]
    return filtered, {
        "view_filtered_rows": len(filtered),
        "selected_view_names": sorted(allowed),
    }


def replace_relative_dir_name(relative_path: Path, old_name: str, new_name: str | None) -> Path:
    if new_name is None or not new_name.strip():
        return relative_path
    parts = list(relative_path.parts)
    try:
        idx = parts.index(old_name)
    except ValueError:
        if len(parts) < 2:
            raise ValueError(f"Cannot replace directory in relative path: {relative_path}")
        idx = len(parts) - 2
    parts[idx] = new_name.strip()
    return Path(*parts)


def expected_output_paths(
    record: dict[str, Any],
    output_root: Path,
    *,
    robot_depth_output_dir_name: str | None,
    robot_metric_first_output_dir_name: str | None,
) -> tuple[Path, Path]:
    inverse_depth_rel = replace_relative_dir_name(
        Path(str(record["robot_mesh_inverse_depth_relative_path"])),
        "robot_mesh_inverse_depth",
        robot_depth_output_dir_name,
    )
    metric_first_rel = replace_relative_dir_name(
        Path(str(record["robot_mesh_metric_first_relative_path"])),
        "robot_mesh_depth_metric_first",
        robot_metric_first_output_dir_name,
    )
    return (
        output_root / inverse_depth_rel,
        output_root / metric_first_rel,
    )


def main() -> None:
    args = parse_args()
    input_root = args.output_root if args.input_root is None else args.input_root
    rows = load_manifest_rows(args.manifest_jsonl)
    rows, view_counts = filter_rows_by_view(rows, args.view_name)
    rows, selection_counts = select_manifest_rows(args, rows)
    selection_counts = {**selection_counts, **view_counts}
    task_tag = resolve_task_tag(args)
    results_path, failures_path, summary_path = prepare_result_logs(args.output_root, args.overwrite, task_tag)

    ensure_urdf_ready(args.urdf_path)
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
    resolved_urdf_path, _ = materialize_tool_adjusted_urdf(
        urdf_path=args.urdf_path,
        tool_adjust_json=args.tool_adjust_json,
    )
    robot = URDF.load(str(resolved_urdf_path))
    base_adjust = None
    if args.base_adjust_json is not None:
        base_adjust = read_matrix(load_json(args.base_adjust_json))
        if base_adjust is None:
            raise ValueError(f"Failed to parse base-adjust matrix from {args.base_adjust_json}")

    rendered_records = 0
    skipped_records = 0
    failed_records = 0
    if not rows:
        write_task_summary(
            summary_path=summary_path,
            task_tag=task_tag,
            args=args,
            counts=selection_counts,
            results_path=results_path,
            failures_path=failures_path,
            rendered=rendered_records,
            skipped=skipped_records,
            failed=failed_records,
        )
        print("[robot-depth] no rows assigned to this shard; exiting cleanly")
        return

    for record in rows:
        episode_id = str(record["episode_id"])
        view_name = str(record["view_name"])
        npz_path, metric_first_path = expected_output_paths(
            record,
            args.output_root,
            robot_depth_output_dir_name=args.robot_depth_output_dir_name,
            robot_metric_first_output_dir_name=args.robot_metric_first_output_dir_name,
        )
        if not args.overwrite and is_nonempty_file(npz_path) and is_nonempty_file(metric_first_path):
            skipped_records += 1
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": episode_id,
                        "view_name": view_name,
                        "status": "skipped_existing",
                    }
                ],
            )
            continue

        try:
            joint_path, gripper_path, joints, gripper = load_episode_inputs(
                record,
                input_root,
                joint_suffix=str(args.joint_suffix),
                gripper_suffix=str(args.gripper_suffix),
                invert_gripper=bool(args.invert_gripper),
            )
            input_video_path = input_root / Path(str(record["rgb_relative_path"]))
            if not is_nonempty_file(input_video_path):
                raise FileNotFoundError(f"Missing render_v2 RGB input: {input_video_path}")
            manifest_sequence = None
            sequence_rel = record.get("base_to_camera_sequence_relative_path")
            if sequence_rel:
                sequence_path = input_root / Path(str(sequence_rel))
                if not sequence_path.exists():
                    raise FileNotFoundError(f"Missing base_to_camera_sequence: {sequence_path}")
                manifest_sequence = np.asarray(np.load(sequence_path), dtype=np.float64)
                if manifest_sequence.ndim != 3 or manifest_sequence.shape[1:] != (4, 4):
                    raise ValueError(
                        f"Expected base_to_camera_sequence shape [T,4,4], "
                        f"got {manifest_sequence.shape} from {sequence_path}"
                    )
                if manifest_sequence.shape[0] < 1:
                    raise ValueError(f"Empty base_to_camera_sequence: {sequence_path}")

            if manifest_sequence is not None:
                base_to_camera_sequence = manifest_sequence
                base_to_camera = base_to_camera_sequence[0]
                cam2base_source = "manifest:base_to_camera_sequence"
            elif view_name == WRIST_VIEW_NAME:
                intrinsics, base_to_camera_sequence, cam2base_source = resolve_wrist_base_to_camera_sequence(
                    episode_id=episode_id,
                    wrist_serial=str(record["serial"]),
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
                    joints = freeze_arm_joints_to_first_frame(joints)
                    base_to_camera_sequence = None
                    cam2base_source = f"{cam2base_source}:gripper_local"
            else:
                cam2base, intrinsics, cam2base_source = resolve_row_calibration(
                    record=record,
                    intrinsics_blob=intrinsics_blob,
                    cam2base_blobs=cam2base_blobs,
                )
                base_to_camera = np.linalg.inv(cam2base)
                if base_adjust is not None:
                    base_to_camera = base_to_camera @ base_adjust
                base_to_camera_sequence = None
            render_summary = render_depth_sequence(
                video_path=input_video_path,
                robot=robot,
                joints=joints,
                gripper=gripper,
                base_to_camera=base_to_camera,
                base_to_camera_sequence=base_to_camera_sequence,
                intrinsics=intrinsics,
                calibration_image_size=(
                    int(args.calibration_image_size[0]),
                    int(args.calibration_image_size[1]),
                ),
                fps_override=args.fps,
                max_faces=0,
                frame_limit=args.frame_limit,
                target_short_side=max(0, int(args.target_short_side)),
                robot_only_render_batch_size=max(1, int(args.robot_only_render_batch_size)),
            )
            depth_m = np.asarray(render_summary["depth_m"], dtype=np.float32)
            valid_mask = np.asarray(render_summary["valid_mask"], dtype=bool)
            inv_depth_u16, valid_mask_u8 = encode_inverse_depth_u16(
                depth_m,
                near_m=float(args.near_m),
                far_m=float(args.far_m),
                valid_mask=valid_mask,
            )
            atomic_savez_compressed(
                npz_path,
                inv_depth_u16=inv_depth_u16,
                valid_mask=valid_mask_u8,
                near_m=np.asarray(float(args.near_m), dtype=np.float32),
                far_m=np.asarray(float(args.far_m), dtype=np.float32),
                output_fps=np.asarray(float(render_summary["output_fps"]), dtype=np.float32),
                output_frame_indices=np.asarray(render_summary["output_frame_indices"], dtype=np.int32),
                output_frame_size=np.asarray(render_summary["output_frame_size"], dtype=np.int32),
                source_frame_size=np.asarray(render_summary["source_frame_size"], dtype=np.int32),
            )
            atomic_h5_write(
                metric_first_path,
                datasets={
                    "depth_m": np.asarray(depth_m[0], dtype=np.float32),
                    "valid_mask": np.asarray(valid_mask_u8[0], dtype=np.uint8),
                },
                attrs={
                    "episode_id": episode_id,
                    "view_name": view_name,
                    "frame_index": 0,
                    "near_m": float(args.near_m),
                    "far_m": float(args.far_m),
                    "cam2base_source": str(cam2base_source),
                    "encoding": "metric_depth_first_frame",
                },
            )
            rendered_records += 1
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": episode_id,
                        "view_name": view_name,
                        "status": "rendered",
                        "joint_path": str(joint_path),
                        "gripper_path": str(gripper_path),
                        "robot_mesh_inverse_depth_path": str(npz_path),
                        "robot_mesh_metric_first_path": str(metric_first_path),
                        "frame_count": int(depth_m.shape[0]),
                    }
                ],
            )
        except Exception as exc:
            failed_records += 1
            append_jsonl(
                failures_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": episode_id,
                        "view_name": view_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                ],
            )

    write_task_summary(
        summary_path=summary_path,
        task_tag=task_tag,
        args=args,
        counts=selection_counts,
        results_path=results_path,
        failures_path=failures_path,
        rendered=rendered_records,
        skipped=skipped_records,
        failed=failed_records,
    )


if __name__ == "__main__":
    main()
