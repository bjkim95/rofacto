#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from .common import DEFAULT_ASSETS_DIR, DEFAULT_CACHE_ROOT, DEFAULT_OUTPUT_FPS, DEFAULT_OUTPUT_ROOT
except ImportError:
    from common import DEFAULT_ASSETS_DIR, DEFAULT_CACHE_ROOT, DEFAULT_OUTPUT_FPS, DEFAULT_OUTPUT_ROOT


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "data_processing" / "droid" / "export_tfds_rgb_videos.py"
RENDER_SCRIPT = REPO_ROOT / "data_processing" / "droid" / "render_robot_mesh_videos.py"
# Mainline Panda + Robotiq 2F-85 URDF (meshes under third_party/robosuite). This is the
# URDF the verified render_v2 pipeline uses; the older phantom-sourced variant was removed.
DEFAULT_URDF_PATH = (
    REPO_ROOT / "data_processing" / "droid_urdf_overlay" / "assets"
    / "panda_arm_robotiq_2f85_fts300.urdf"
)
DEFAULT_CAM2BASE_JSON = DEFAULT_ASSETS_DIR / "cam2base_extrinsics.json"
DEFAULT_CAM2BASE_SUPERSET_JSON = DEFAULT_ASSETS_DIR / "cam2base_extrinsic_superset.json"
DEFAULT_INTRINSICS_JSON = DEFAULT_ASSETS_DIR / "intrinsics.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot DROID pipeline: export render_v2 RGB videos, then render robot meshes and comparison artifacts."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root used to resolve a relative --manifest-jsonl path.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-fps", type=float, default=DEFAULT_OUTPUT_FPS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-cache", action="store_true")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--cam2base-json", type=Path, default=DEFAULT_CAM2BASE_JSON)
    parser.add_argument("--cam2base-superset-json", type=Path, default=DEFAULT_CAM2BASE_SUPERSET_JSON)
    parser.add_argument("--intrinsics-json", type=Path, default=DEFAULT_INTRINSICS_JSON)
    parser.add_argument(
        "--wrist-pose-source",
        choices=["trajectory_h5", "proprio_gripper2wrist", "cloak_cam_to_gripper"],
        default="trajectory_h5",
        help="Pose source used only for wrist robot mesh rendering.",
    )
    parser.add_argument("--gripper2wrist-json", type=Path, default=DEFAULT_GRIPPER2WRIST_JSON)
    parser.add_argument("--cloak-wrist-extrinsics-json", type=Path, default=DEFAULT_CLOAK_WRIST_EXTRINSICS_JSON)
    parser.add_argument(
        "--cloak-missing-fallback",
        choices=["proprio_gripper2wrist", "error"],
        default="proprio_gripper2wrist",
    )
    parser.add_argument(
        "--wrist-proprio-flip",
        choices=["none", "z180"],
        default="none",
        help="Use none for original wrist RGB; z180 matches PointWorld's flipped-wrist VGGT convention.",
    )
    parser.add_argument("--appearance", type=str, default="shaded")
    parser.add_argument("--target-short-side", type=int, default=480)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.22)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--robot-only-render-batch-size",
        type=int,
        default=4,
        help="Batch size used only for robot_only PyTorch3D rendering. Use 1 to disable batching.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--task-tag", type=str, default=None)
    parser.add_argument("--write-comparison-video", action="store_true")
    parser.add_argument("--write-overlay", action="store_true")
    parser.set_defaults(write_comparison_jpg=True)
    parser.add_argument("--write-comparison-jpg", dest="write_comparison_jpg", action="store_true")
    parser.add_argument("--no-write-comparison-jpg", dest="write_comparison_jpg", action="store_false")
    return parser.parse_args()


def resolve_task_tag(explicit: str | None, shard_index: int, num_shards: int) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return f"{job_id}_{task_id}"
    return f"{job_id}_shard{shard_index}of{num_shards}"


def append_common_selection_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(["--offset", str(args.offset)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    cmd.extend(["--shard-index", str(args.shard_index), "--num-shards", str(args.num_shards)])


def append_render_selection_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(["--start-index", str(args.offset)])
    if args.limit is not None:
        cmd.extend(["--end-index", str(args.offset + args.limit)])
    cmd.extend(["--shard-index", str(args.shard_index), "--num-shards", str(args.num_shards)])


def build_export_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--manifest-jsonl",
        str(args.manifest_jsonl),
        "--output-root",
        str(args.output_root),
        "--output-fps",
        str(args.output_fps),
        "--cache-root",
        str(args.cache_root),
        "--task-tag",
        resolve_task_tag(args.task_tag, args.shard_index, args.num_shards),
    ]
    append_common_selection_args(cmd, args)
    if args.overwrite_cache:
        cmd.append("--overwrite-cache")
    if args.only_cache:
        cmd.append("--only-cache")
    if not args.overwrite:
        cmd.append("--skip-existing")
    return cmd


def build_render_cmd(args: argparse.Namespace, task_tag: str) -> list[str]:
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--manifest-jsonl",
        str(args.manifest_jsonl),
        "--output-root",
        str(args.output_root),
        "--urdf-path",
        str(args.urdf_path),
        "--cam2base-json",
        str(args.cam2base_json),
        "--cam2base-superset-json",
        str(args.cam2base_superset_json),
        "--intrinsics-json",
        str(args.intrinsics_json),
        "--cache-root",
        str(args.cache_root),
        "--wrist-pose-source",
        str(args.wrist_pose_source),
        "--gripper2wrist-json",
        str(args.gripper2wrist_json),
        "--cloak-wrist-extrinsics-json",
        str(args.cloak_wrist_extrinsics_json),
        "--cloak-missing-fallback",
        str(args.cloak_missing_fallback),
        "--wrist-proprio-flip",
        str(args.wrist_proprio_flip),
        "--appearance",
        str(args.appearance),
        "--target-short-side",
        str(args.target_short_side),
        "--max-faces",
        str(args.max_faces),
        "--alpha",
        str(args.alpha),
        "--frame-step",
        str(args.frame_step),
        "--robot-only-render-batch-size",
        str(max(1, int(args.robot_only_render_batch_size))),
        "--task-tag",
        task_tag,
    ]
    append_render_selection_args(cmd, args)
    if args.frame_limit is not None:
        cmd.extend(["--frame-limit", str(args.frame_limit)])
    if args.fps is not None:
        cmd.extend(["--fps", str(args.fps)])
    if args.write_comparison_jpg:
        cmd.append("--write-comparison-jpg")
    if args.only_cache:
        cmd.append("--skip-missing-inputs")
    if args.write_comparison_video:
        cmd.append("--write-comparison")
    if args.write_overlay:
        cmd.append("--write-overlay")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_step(name: str, cmd: list[str]) -> None:
    print(f"[one-shot] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.input_root is not None and not args.manifest_jsonl.is_absolute():
        args.manifest_jsonl = args.input_root / args.manifest_jsonl
    task_tag = resolve_task_tag(args.task_tag, args.shard_index, args.num_shards)
    export_cmd = build_export_cmd(args)
    render_cmd = build_render_cmd(args, task_tag=task_tag)

    run_step("export_rgb", export_cmd)
    run_step("render_mesh", render_cmd)

    print(
        json.dumps(
            {
                "manifest_jsonl": str(args.manifest_jsonl),
                "output_root": str(args.output_root),
                "task_tag": task_tag,
                "write_comparison_jpg": bool(args.write_comparison_jpg),
                "write_comparison_video": bool(args.write_comparison_video),
                "write_overlay": bool(args.write_overlay),
                "commands": {
                    "export_rgb": export_cmd,
                    "render_mesh": render_cmd,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
