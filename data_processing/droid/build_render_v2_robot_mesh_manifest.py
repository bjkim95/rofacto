#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_MANIFEST_ROOT,
        DEFAULT_OUTPUT_ROOT,
        VIEW_SPECS,
        is_nonempty_file,
        write_jsonl,
    )
except ImportError:
    from common import (
        DEFAULT_MANIFEST_ROOT,
        DEFAULT_OUTPUT_ROOT,
        VIEW_SPECS,
        is_nonempty_file,
        write_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an episode-level manifest for render_robot_mesh_videos.py from "
            "existing DROID render_v2 episode_meta.json files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing render_v2 episode outputs used to build the manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Root for generated manifest files.",
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument(
        "--joint-suffix",
        type=str,
        default="",
        help="Only include episodes whose suffixed joint input exists.",
    )
    parser.add_argument(
        "--gripper-suffix",
        type=str,
        default="",
        help="Only include episodes whose suffixed gripper input exists.",
    )
    parser.add_argument(
        "--view-name",
        action="append",
        dest="view_names",
        help="Limit to one or more views. Defaults to ext1/ext2/wrist when present.",
    )
    parser.add_argument(
        "--require-rgb",
        action="store_true",
        help="Skip view entries whose RGB mp4 is missing or empty.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_episode_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def with_stem_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def selected_view_names(args: argparse.Namespace) -> set[str]:
    if args.view_names:
        return {str(name) for name in args.view_names}
    return set(VIEW_SPECS.keys())


def build_record(
    *,
    output_root: Path,
    episode_meta_relative_path: Path,
    episode_meta: dict[str, Any],
    selected_views: set[str],
    joint_suffix: str,
    gripper_suffix: str,
    require_rgb: bool,
) -> dict[str, Any]:
    episode_id = str(episode_meta["episode_id"])
    views_saved = episode_meta.get("views_saved")
    if not isinstance(views_saved, list):
        raise ValueError("episode_meta missing views_saved list")

    views: list[dict[str, Any]] = []
    video_outputs: dict[str, str] = {}
    for entry in views_saved:
        if not isinstance(entry, dict):
            continue
        view_name = str(entry.get("view_name") or "")
        if view_name not in selected_views:
            continue
        if view_name not in VIEW_SPECS:
            continue
        relative_video_path = entry.get("relative_video_path")
        if not isinstance(relative_video_path, str) or not relative_video_path.strip():
            continue
        rgb_path = output_root / relative_video_path
        if require_rgb and not is_nonempty_file(rgb_path):
            continue
        views.append(
            {
                "view_name": view_name,
                "camera_name": entry.get("camera_name"),
                "serial_key": entry.get("serial_key"),
                "serial": entry.get("serial"),
            }
        )
        video_outputs[view_name] = relative_video_path

    if not views:
        raise ValueError("No selected views remained after filtering")

    joint_name = episode_meta.get("joint_path")
    gripper_name = episode_meta.get("gripper_path")
    if not isinstance(joint_name, str) or not joint_name:
        raise ValueError("episode_meta missing joint_path")
    if not isinstance(gripper_name, str) or not gripper_name:
        raise ValueError("episode_meta missing gripper_path")

    relative_root = episode_meta_relative_path.parents[2]
    timestamp = episode_meta_relative_path.parent.name
    joint_relative_path = relative_root / "episodes" / timestamp / with_stem_suffix(Path(joint_name), joint_suffix).name
    gripper_relative_path = relative_root / "episodes" / timestamp / with_stem_suffix(Path(gripper_name), gripper_suffix).name
    if not is_nonempty_file(output_root / joint_relative_path):
        raise FileNotFoundError(f"Missing joint input {joint_relative_path}")
    if not is_nonempty_file(output_root / gripper_relative_path):
        raise FileNotFoundError(f"Missing gripper input {gripper_relative_path}")

    return {
        "dataset_name": episode_meta.get("dataset_name"),
        "data_dir": episode_meta.get("data_dir"),
        "split": episode_meta.get("split"),
        "episode_id": episode_id,
        "episode_index": episode_meta.get("episode_index"),
        "episode_key": episode_meta.get("episode_key"),
        "recording_folderpath": episode_meta.get("recording_folderpath"),
        "file_path": episode_meta.get("file_path"),
        "task": episode_meta.get("task"),
        "language_instruction": episode_meta.get("language_instruction"),
        "source_step_rate_hz": episode_meta.get("source_step_rate_hz"),
        "output_fps": episode_meta.get("output_fps"),
        "num_source_steps": episode_meta.get("num_source_steps"),
        "num_exported_frames": episode_meta.get("num_exported_frames"),
        "trajectory_indices": episode_meta.get("trajectory_indices", []),
        "annotation_assets": episode_meta.get("annotation_assets", {}),
        "episode_meta_relative_path": str(episode_meta_relative_path),
        "joint_relative_path": str(joint_relative_path),
        "gripper_relative_path": str(gripper_relative_path),
        "views": views,
        "video_outputs": video_outputs,
    }


def main() -> None:
    args = parse_args()
    output_jsonl = args.output_jsonl or args.output_root / "render_v2_robot_mesh.jsonl"
    failures_jsonl = args.failures_jsonl or args.output_root / "render_v2_robot_mesh_failures.jsonl"
    if not args.overwrite and (output_jsonl.exists() or failures_jsonl.exists()):
        raise FileExistsError(
            f"{output_jsonl} or {failures_jsonl} already exists. Pass --overwrite to replace them."
        )

    selected_views = selected_view_names(args)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    episode_meta_paths = sorted(args.input_root.glob("*/*/episodes/*/episode_meta.json"))
    for episode_meta_path in episode_meta_paths:
        relative_meta_path = episode_meta_path.relative_to(args.input_root)
        try:
            episode_meta = load_episode_meta(episode_meta_path)
            records.append(
                build_record(
                    output_root=args.input_root,
                    episode_meta_relative_path=relative_meta_path,
                    episode_meta=episode_meta,
                    selected_views=selected_views,
                    joint_suffix=args.joint_suffix,
                    gripper_suffix=args.gripper_suffix,
                    require_rgb=bool(args.require_rgb),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "episode_meta_json": str(episode_meta_path),
                    "error": str(exc),
                }
            )

    write_jsonl(output_jsonl, records)
    write_jsonl(failures_jsonl, failures)
    print(
        json.dumps(
            {
                "episode_meta_count": len(episode_meta_paths),
                "records": len(records),
                "failures": len(failures),
                "output_jsonl": str(output_jsonl),
                "failures_jsonl": str(failures_jsonl),
                "joint_suffix": args.joint_suffix,
                "gripper_suffix": args.gripper_suffix,
                "view_names": sorted(selected_views),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
