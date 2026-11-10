#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        DEFAULT_OUTPUT_FPS,
        DEFAULT_OUTPUT_ROOT,
        atomic_numpy_save,
        compute_resampled_positions,
        download_gs_file,
        expand_keep_ranges_to_source_indices,
        is_nonempty_file,
        load_joint_gripper_from_h5,
        map_indices_between_domains,
        split_episode_id_components,
        write_json,
    )
    from .video_io import VideoWriter
except ImportError:
    from common import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_OUTPUT_FPS,
        DEFAULT_OUTPUT_ROOT,
        atomic_numpy_save,
        compute_resampled_positions,
        download_gs_file,
        expand_keep_ranges_to_source_indices,
        is_nonempty_file,
        load_joint_gripper_from_h5,
        map_indices_between_domains,
        split_episode_id_components,
        write_json,
    )
    from video_io import VideoWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DROID render_v2 RGB videos from manifest + raw cache inputs."
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-cache", action="store_true")
    parser.add_argument("--task-tag", type=str, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Exit zero after recording per-record failures.",
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
            rows.append(json.loads(line))
    return rows


def select_manifest_rows(
    rows: list[dict[str, Any]],
    offset: int,
    limit: int | None,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    selected = rows[offset:]
    if limit is not None:
        selected = selected[:limit]
    if num_shards == 1:
        return selected
    return [row for idx, row in enumerate(selected) if idx % num_shards == shard_index]


def expected_output_paths(record: dict[str, Any], output_root: Path) -> list[Path]:
    paths = [output_root / Path(record["episode_meta_relative_path"])]
    paths.append(output_root / Path(record["joint_relative_path"]))
    paths.append(output_root / Path(record["gripper_relative_path"]))
    for relative_path in record.get("video_outputs", {}).values():
        paths.append(output_root / Path(relative_path))
    return paths


def resolve_task_tag(explicit: str | None, shard_index: int, num_shards: int) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return f"{job_id}_{task_id}"
    return f"{job_id}_shard{shard_index}of{num_shards}"


def prepare_result_logs(output_root: Path, task_tag: str) -> tuple[Path, Path, Path]:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_path = manifest_dir / f"export_rgb_results_{task_tag}.jsonl"
    failures_path = manifest_dir / f"export_rgb_failures_{task_tag}.jsonl"
    summary_path = manifest_dir / f"export_rgb_summary_{task_tag}.json"
    for path in (results_path, failures_path):
        if path.exists():
            path.unlink()
        path.touch()
    if summary_path.exists():
        summary_path.unlink()
    return results_path, failures_path, summary_path


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_localized_raw_assets(
    record: dict[str, Any],
    cache_root: Path,
    overwrite_cache: bool,
    only_cache: bool,
) -> dict[str, Path | dict[str, Path]]:
    episode_cache_dir = cache_root / str(record["episode_id"])
    trajectory_local_path = episode_cache_dir / "trajectory.h5"
    if only_cache and not is_nonempty_file(trajectory_local_path):
        raise CacheMissError(f"Missing cached trajectory: {trajectory_local_path}")
    trajectory_path = download_gs_file(
        gs_path=str(record["raw_trajectory_gs_path"]),
        local_path=trajectory_local_path,
        overwrite=overwrite_cache,
    )
    raw_video_paths: dict[str, Path] = {}
    for view in record["views"]:
        view_name = str(view["view_name"])
        serial = str(view["serial"])
        raw_video_local_path = episode_cache_dir / f"{serial}.mp4"
        if only_cache and not is_nonempty_file(raw_video_local_path):
            raise CacheMissError(f"Missing cached raw video: {raw_video_local_path}")
        raw_video_paths[view_name] = download_gs_file(
            gs_path=str(record["raw_video_gs_paths"][view_name]),
            local_path=raw_video_local_path,
            overwrite=overwrite_cache,
        )
    return {
        "trajectory": trajectory_path,
        "raw_videos": raw_video_paths,
    }


def localize_and_export_view_video(
    *,
    episode_id: str,
    view_name: str,
    raw_video_path: Path,
    output_path: Path,
    step_indices: list[int],
    total_step_count: int,
    output_fps: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open raw video {raw_video_path}")

    try:
        raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if raw_frame_count <= 0:
            raise RuntimeError(f"Failed to read frame count from {raw_video_path}")
        target_raw_frame_indices = map_indices_between_domains(
            source_indices=step_indices,
            source_total=total_step_count,
            target_total=raw_frame_count,
        )
        if not target_raw_frame_indices:
            raise RuntimeError(f"No target raw frames for {episode_id} view {view_name}")

        writer: VideoWriter | None = None
        next_target_idx = 0
        raw_frame_index = 0
        while next_target_idx < len(target_raw_frame_indices):
            ok, frame = cap.read()
            if not ok:
                break
            current_target = target_raw_frame_indices[next_target_idx]
            if raw_frame_index < current_target:
                raw_frame_index += 1
                continue
            if raw_frame_index > current_target:
                raise RuntimeError(
                    f"Missed target raw frame {current_target} while exporting {episode_id} view {view_name}"
                )
            if writer is None:
                height, width = frame.shape[:2]
                writer = VideoWriter(
                    path=output_path,
                    fps=output_fps,
                    frame_size=(width, height),
                    input_colorspace="bgr",
                )
            while (
                next_target_idx < len(target_raw_frame_indices)
                and target_raw_frame_indices[next_target_idx] == raw_frame_index
            ):
                writer.write(frame)
                next_target_idx += 1
            raw_frame_index += 1

        if writer is None:
            raise RuntimeError(f"No frames were written for {episode_id} view {view_name}")
        if next_target_idx != len(target_raw_frame_indices):
            raise RuntimeError(
                f"Stopped after {next_target_idx}/{len(target_raw_frame_indices)} mapped frames for {episode_id} view {view_name}"
            )
        writer.close()
        return {
            "raw_video_path": str(raw_video_path),
            "raw_frame_count": raw_frame_count,
            "raw_fps": raw_fps,
            "exported_raw_frame_indices": target_raw_frame_indices,
        }
    finally:
        cap.release()


def derive_total_step_count(record: dict[str, Any], joints_len: int) -> int:
    manifest_steps = record.get("num_source_steps")
    if manifest_steps is not None:
        try:
            return max(1, int(manifest_steps))
        except (TypeError, ValueError):
            pass
    # For DROID raw trajectory.h5, joint sequences are typically one longer than TFDS step count.
    return max(1, int(joints_len) - 1)


def export_episode(
    record: dict[str, Any],
    output_root: Path,
    cache_root: Path,
    output_fps: float,
    overwrite_cache: bool,
    only_cache: bool,
) -> dict[str, Any]:
    episode_id = str(record["episode_id"])
    _, _, timestamp = split_episode_id_components(episode_id)
    relative_root = Path(str(record["output_relative_dir"]))
    episode_dir = output_root / relative_root / "episodes" / timestamp
    videos_dir = output_root / relative_root / "videos"
    episode_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    keep_ranges = record.get("keep_ranges", [])
    filter_last_n = int(record.get("filter_last_n_in_ranges", 0) or 0)
    planned_source_indices = expand_keep_ranges_to_source_indices(keep_ranges, filter_last_n)
    if not planned_source_indices:
        raise RuntimeError(f"No keep-range frames available for {episode_id}")

    localized = ensure_localized_raw_assets(
        record=record,
        cache_root=cache_root,
        overwrite_cache=overwrite_cache,
        only_cache=only_cache,
    )
    joints_all, gripper_all = load_joint_gripper_from_h5(Path(localized["trajectory"]))
    total_step_count = derive_total_step_count(record, joints_len=int(joints_all.shape[0]))
    kept_source_step_indices = [index for index in planned_source_indices if index < total_step_count]
    if not kept_source_step_indices:
        raise RuntimeError(f"Episode {episode_id} produced zero kept source steps after clipping")

    source_step_rate_hz = float(record.get("source_step_rate_hz", 15.0))
    resampled_positions = compute_resampled_positions(
        num_source_frames=len(kept_source_step_indices),
        source_step_rate_hz=source_step_rate_hz,
        output_fps=output_fps,
    )
    exported_source_step_indices = [kept_source_step_indices[position] for position in resampled_positions]
    trajectory_indices = map_indices_between_domains(
        source_indices=exported_source_step_indices,
        source_total=total_step_count,
        target_total=int(joints_all.shape[0]),
    )
    joint_array = np.stack([joints_all[index] for index in trajectory_indices], axis=0)
    gripper_array = np.stack([gripper_all[index] for index in trajectory_indices], axis=0)

    joint_path = output_root / Path(record["joint_relative_path"])
    gripper_path = output_root / Path(record["gripper_relative_path"])
    atomic_numpy_save(joint_path, joint_array)
    atomic_numpy_save(gripper_path, gripper_array)

    view_export_stats: dict[str, dict[str, Any]] = {}
    for view in record["views"]:
        view_name = str(view["view_name"])
        output_path = output_root / Path(record["video_outputs"][view_name])
        view_export_stats[view_name] = localize_and_export_view_video(
            episode_id=episode_id,
            view_name=view_name,
            raw_video_path=Path(localized["raw_videos"][view_name]),
            output_path=output_path,
            step_indices=exported_source_step_indices,
            total_step_count=total_step_count,
            output_fps=output_fps,
        )

    task = record.get("task")
    episode_meta = {
        "dataset_name": record.get("dataset_name"),
        "data_dir": record.get("data_dir"),
        "split": record.get("split"),
        "episode_index": record.get("episode_index"),
        "episode_id": episode_id,
        "episode_key": record.get("episode_key"),
        "recording_folderpath": record.get("recording_folderpath"),
        "file_path": record.get("file_path"),
        "task": task,
        "language_instruction": task,
        "views_saved": [
            {
                "view_name": view["view_name"],
                "camera_name": view["camera_name"],
                "serial_key": view["serial_key"],
                "serial": view["serial"],
                "relative_video_path": record["video_outputs"][str(view["view_name"])],
                "raw_video_gs_path": record["raw_video_gs_paths"][str(view["view_name"])],
                "localized_raw_video_path": view_export_stats[str(view["view_name"])]["raw_video_path"],
                "raw_frame_count": view_export_stats[str(view["view_name"])]["raw_frame_count"],
                "raw_fps": view_export_stats[str(view["view_name"])]["raw_fps"],
            }
            for view in record["views"]
        ],
        "source_step_rate_hz": source_step_rate_hz,
        "output_fps": output_fps,
        "keep_ranges": keep_ranges,
        "filter_last_n_in_ranges": filter_last_n,
        "num_source_steps": total_step_count,
        "kept_source_step_indices": kept_source_step_indices,
        "exported_source_step_indices": exported_source_step_indices,
        "trajectory_indices": trajectory_indices,
        "num_kept_source_steps": len(kept_source_step_indices),
        "num_exported_frames": len(exported_source_step_indices),
        "joint_path": joint_path.name,
        "gripper_path": gripper_path.name,
        "raw_trajectory_gs_path": record["raw_trajectory_gs_path"],
        "localized_raw_trajectory_path": str(localized["trajectory"]),
        "annotation_assets": record.get("annotation_assets", {}),
    }
    episode_meta_path = output_root / Path(record["episode_meta_relative_path"])
    write_json(episode_meta_path, episode_meta)

    return {
        "episode_id": episode_id,
        "episode_index": int(record["episode_index"]),
        "views_saved": [view["view_name"] for view in record["views"]],
        "num_kept_source_steps": len(kept_source_step_indices),
        "num_exported_frames": len(exported_source_step_indices),
        "episode_meta_json": str(episode_meta_path),
    }


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
        "[export-rgb] "
        f"processed={processed}/{total} "
        f"exported={exported} "
        f"skipped={skipped} "
        f"failed={failed} "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    manifest_jsonl = args.manifest_jsonl
    if args.input_root is not None and not manifest_jsonl.is_absolute():
        manifest_jsonl = args.input_root / manifest_jsonl
    task_tag = resolve_task_tag(args.task_tag, args.shard_index, args.num_shards)
    rows = load_manifest_rows(manifest_jsonl)
    selected_rows = select_manifest_rows(
        rows=rows,
        offset=args.offset,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    results_path, failures_path, summary_path = prepare_result_logs(args.output_root, task_tag=task_tag)
    started_at = time.monotonic()
    progress_interval = max(1, int(args.progress_interval))

    export_summaries: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    skipped_records = 0
    total_selected = len(selected_rows)

    for idx, row in enumerate(selected_rows, start=1):
        output_paths = expected_output_paths(row, args.output_root)
        if args.skip_existing and all(is_nonempty_file(path) for path in output_paths):
            skipped_records += 1
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": row.get("episode_id"),
                        "episode_index": row.get("episode_index"),
                        "status": "skipped_existing",
                    }
                ],
            )
            if idx % progress_interval == 0 or idx == total_selected:
                report_progress(
                    processed=idx,
                    total=total_selected,
                    exported=len(export_summaries),
                    skipped=skipped_records,
                    failed=len(failure_records),
                    started_at=started_at,
                )
            continue

        try:
            summary = export_episode(
                record=row,
                output_root=args.output_root,
                cache_root=args.cache_root,
                output_fps=args.output_fps,
                overwrite_cache=args.overwrite_cache,
                only_cache=args.only_cache,
            )
            export_summaries.append(summary)
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": summary["episode_id"],
                        "episode_index": summary["episode_index"],
                        "status": "exported",
                        "summary": summary,
                    }
                ],
            )
        except CacheMissError as exc:
            skipped_records += 1
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": row.get("episode_id"),
                        "episode_index": row.get("episode_index"),
                        "status": "skipped_missing_cache",
                        "reason": str(exc),
                    }
                ],
            )
        except Exception as exc:
            failure = {
                "task_tag": task_tag,
                "episode_id": row.get("episode_id"),
                "episode_index": row.get("episode_index"),
                "error": str(exc),
            }
            failure_records.append(failure)
            append_jsonl(failures_path, [failure])

        if idx % progress_interval == 0 or idx == total_selected:
            report_progress(
                processed=idx,
                total=total_selected,
                exported=len(export_summaries),
                skipped=skipped_records,
                failed=len(failure_records),
                started_at=started_at,
            )

    summary_payload = {
        "task_tag": task_tag,
        "manifest_jsonl": str(args.manifest_jsonl),
        "output_root": str(args.output_root),
        "cache_root": str(args.cache_root),
        "selected_records": total_selected,
        "exported_records": len(export_summaries),
        "skipped_records": skipped_records,
        "failed_records": len(failure_records),
        "output_fps": args.output_fps,
        "only_cache": bool(args.only_cache),
        "results_path": str(results_path),
        "failures_path": str(failures_path),
        "summaries": export_summaries[:16],
    }
    write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, indent=2))

    if failure_records and not args.best_effort:
        raise SystemExit(
            "RGB export completed with failures; rerun with --best-effort to preserve exit zero"
        )


if __name__ == "__main__":
    main()
