#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_ASSETS_DIR,
        RAW_BUCKET_ROOT,
        DEFAULT_DATA_DIR,
        DEFAULT_DATASET_NAME,
        DEFAULT_MANIFEST_ROOT,
        DEFAULT_SOURCE_STEP_RATE,
        build_episode_id_reverse_lookup,
        build_episode_key,
        estimate_kept_frames_from_ranges,
        extract_task_string,
        find_keep_ranges,
        load_json,
        load_tfds,
        normalize_episode_metadata,
        parse_keep_ranges_blob,
        raw_mp4_gs_path,
        raw_trajectory_gs_path,
        resolve_asset_path,
        resolve_calibrated_views,
        resolve_episode_id,
        split_episode_id_components,
        write_jsonl,
    )
except ImportError:
    from common import (
        DEFAULT_ASSETS_DIR,
        RAW_BUCKET_ROOT,
        DEFAULT_DATA_DIR,
        DEFAULT_DATASET_NAME,
        DEFAULT_MANIFEST_ROOT,
        DEFAULT_SOURCE_STEP_RATE,
        build_episode_id_reverse_lookup,
        build_episode_key,
        estimate_kept_frames_from_ranges,
        extract_task_string,
        find_keep_ranges,
        load_json,
        load_tfds,
        normalize_episode_metadata,
        parse_keep_ranges_blob,
        raw_mp4_gs_path,
        raw_trajectory_gs_path,
        resolve_asset_path,
        resolve_calibrated_views,
        resolve_episode_id,
        split_episode_id_components,
        write_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TFDS/RLDS-driven manifest for DROID RGB export."
    )
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--input-root", type=Path, default=None,
                        help="Root containing assets/; defaults to this script's assets directory.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Root for generated manifest files; defaults to data/droid/manifests_v2.")
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--camera-serials-json", type=Path, default=None)
    parser.add_argument("--cam2base-json", type=Path, default=None)
    parser.add_argument("--cam2base-superset-json", type=Path, default=None)
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--episode-id-to-path-json", type=Path, default=None)
    parser.add_argument("--keep-ranges-json", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--source-step-rate", type=float, default=DEFAULT_SOURCE_STEP_RATE)
    parser.add_argument("--raw-bucket-root", type=str, default=RAW_BUCKET_ROOT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def failure_record(
    reason: str,
    episode_index: int,
    episode_metadata: dict[str, Any],
    episode_id_candidates: list[str],
    episode_id: str | None = None,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "episode_index": episode_index,
        "episode_id": episode_id,
        "episode_id_candidates": episode_id_candidates,
        "episode_key": build_episode_key(episode_metadata),
        "recording_folderpath": episode_metadata.get("recording_folderpath"),
        "file_path": episode_metadata.get("file_path"),
    }


def default_output_paths(split: str, manifest_root: Path) -> tuple[Path, Path]:
    output_jsonl = manifest_root / f"{split}_tfds_rgb.jsonl"
    failures_jsonl = manifest_root / f"{split}_tfds_rgb_failures.jsonl"
    return output_jsonl, failures_jsonl


def report_progress(
    *,
    scanned: int,
    accepted: int,
    failures: list[dict[str, Any]],
    started_at: float,
) -> None:
    elapsed = time.monotonic() - started_at
    print(
        "[manifest] "
        f"scanned={scanned} "
        f"accepted={accepted} "
        f"rejected={len(failures)} "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    assets_dir = args.assets_dir or (
        args.input_root / "assets" if args.input_root is not None else DEFAULT_ASSETS_DIR
    )
    manifest_root = args.output_root or DEFAULT_MANIFEST_ROOT
    output_jsonl, failures_jsonl = default_output_paths(args.split, manifest_root)
    if args.output_jsonl is not None:
        output_jsonl = args.output_jsonl
    if args.failures_jsonl is not None:
        failures_jsonl = args.failures_jsonl
    if not args.overwrite and (output_jsonl.exists() or failures_jsonl.exists()):
        raise FileExistsError(
            f"{output_jsonl} or {failures_jsonl} already exists. Pass --overwrite to replace them."
        )

    camera_serials_json = resolve_asset_path(
        args.camera_serials_json,
        assets_dir,
        ["camera_serials.json"],
    )
    cam2base_json = resolve_asset_path(
        args.cam2base_json,
        assets_dir,
        ["cam2base_extrinsics.json", "cam2base_extrinsic_superset.json"],
    )
    cam2base_superset_json = args.cam2base_superset_json
    if cam2base_superset_json is None:
        candidate = assets_dir / "cam2base_extrinsic_superset.json"
        if candidate.exists() and candidate != cam2base_json:
            cam2base_superset_json = candidate
    intrinsics_json = resolve_asset_path(
        args.intrinsics_json,
        assets_dir,
        ["intrinsics.json"],
    )
    episode_id_to_path_json = resolve_asset_path(
        args.episode_id_to_path_json,
        assets_dir,
        ["episode_id_to_path.json"],
    )
    keep_ranges_json = resolve_asset_path(
        args.keep_ranges_json,
        assets_dir,
        ["keep_ranges_1_0_1.json"],
    )

    camera_serials_blob = load_json(camera_serials_json)
    intrinsics_blob = load_json(intrinsics_json)
    episode_id_to_path_blob = load_json(episode_id_to_path_json)
    keep_ranges_blob = load_json(keep_ranges_json)
    cam2base_sources: list[tuple[str, dict[str, Any]]] = []
    cam2base_blob = load_json(cam2base_json)
    cam2base_sources.append((cam2base_json.name, cam2base_blob))
    if cam2base_superset_json is not None:
        cam2base_superset_blob = load_json(cam2base_superset_json)
        cam2base_sources.append((cam2base_superset_json.name, cam2base_superset_blob))

    if not isinstance(camera_serials_blob, dict):
        raise ValueError(f"Expected dict in {camera_serials_json}")
    if not all(isinstance(blob, dict) for _, blob in cam2base_sources):
        raise ValueError(f"Expected dict in {cam2base_json}")
    if not isinstance(intrinsics_blob, dict):
        raise ValueError(f"Expected dict in {intrinsics_json}")
    if not isinstance(episode_id_to_path_blob, dict):
        raise ValueError(f"Expected dict in {episode_id_to_path_json}")

    reverse_lookup = build_episode_id_reverse_lookup(episode_id_to_path_blob)
    keep_ranges_map, filter_last_n = parse_keep_ranges_blob(keep_ranges_blob)

    tfds = load_tfds()
    ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=args.split, shuffle_files=False)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scanned = 0
    accepted = 0
    started_at = time.monotonic()
    progress_interval = max(1, int(args.progress_interval))
    for episode_index, episode in enumerate(ds):
        if episode_index < args.offset:
            continue
        if args.limit is not None and scanned >= args.limit:
            break
        scanned += 1

        episode_metadata = normalize_episode_metadata(episode)
        resolved_episode_id, episode_id_candidates = resolve_episode_id(
            episode_metadata=episode_metadata,
            reverse_lookup=reverse_lookup,
        )
        if resolved_episode_id is None:
            failures.append(
                failure_record(
                    reason="missing_episode_id",
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    episode_id_candidates=episode_id_candidates,
                )
            )
            if scanned % progress_interval == 0:
                report_progress(scanned=scanned, accepted=accepted, failures=failures, started_at=started_at)
            continue

        matched_keep_key, keep_ranges = find_keep_ranges(
            episode_metadata=episode_metadata,
            keep_ranges_map=keep_ranges_map,
        )
        if not keep_ranges:
            failures.append(
                failure_record(
                    reason="missing_keep_ranges",
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    episode_id_candidates=episode_id_candidates,
                    episode_id=resolved_episode_id,
                )
            )
            if scanned % progress_interval == 0:
                report_progress(scanned=scanned, accepted=accepted, failures=failures, started_at=started_at)
            continue

        estimated_kept_frames = estimate_kept_frames_from_ranges(
            keep_ranges=keep_ranges,
            filter_last_n=filter_last_n,
        )
        if estimated_kept_frames <= 0:
            failures.append(
                failure_record(
                    reason="empty_keep_ranges",
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    episode_id_candidates=episode_id_candidates,
                    episode_id=resolved_episode_id,
                )
            )
            if scanned % progress_interval == 0:
                report_progress(scanned=scanned, accepted=accepted, failures=failures, started_at=started_at)
            continue

        views = resolve_calibrated_views(
            episode_id=resolved_episode_id,
            camera_serials_blob=camera_serials_blob,
            cam2base_sources=cam2base_sources,
            intrinsics_blob=intrinsics_blob,
        )
        if not views:
            failures.append(
                failure_record(
                    reason="no_calibrated_exterior_view",
                    episode_index=episode_index,
                    episode_metadata=episode_metadata,
                    episode_id_candidates=episode_id_candidates,
                    episode_id=resolved_episode_id,
                )
            )
            if scanned % progress_interval == 0:
                report_progress(scanned=scanned, accepted=accepted, failures=failures, started_at=started_at)
            continue

        lab_name, hash_value, timestamp = split_episode_id_components(resolved_episode_id)
        relative_episode_path = str(episode_id_to_path_blob[resolved_episode_id])
        relative_root = Path(lab_name) / hash_value
        task = extract_task_string(episode_metadata)
        video_outputs = {
            view["view_name"]: str(relative_root / "videos" / f"{timestamp}__{view['view_name']}.mp4")
            for view in views
        }
        raw_video_gs_paths = {
            view["view_name"]: raw_mp4_gs_path(
                relative_episode_path=relative_episode_path,
                serial=str(view["serial"]),
                raw_bucket_root=args.raw_bucket_root,
            )
            for view in views
        }
        record = {
            "dataset_name": args.dataset_name,
            "data_dir": args.data_dir,
            "split": args.split,
            "episode_index": episode_index,
            "episode_id": resolved_episode_id,
            "relative_episode_path": relative_episode_path,
            "episode_id_candidates": episode_id_candidates,
            "episode_key": matched_keep_key,
            "recording_folderpath": episode_metadata.get("recording_folderpath"),
            "file_path": episode_metadata.get("file_path"),
            "task": task,
            "source_step_rate_hz": args.source_step_rate,
            "filter_last_n_in_ranges": filter_last_n,
            "keep_ranges": keep_ranges,
            "estimated_kept_source_frames": estimated_kept_frames,
            "views": views,
            "lab_name": lab_name,
            "hash_value": hash_value,
            "timestamp": timestamp,
            "output_relative_dir": str(relative_root),
            "video_outputs": video_outputs,
            "raw_trajectory_gs_path": raw_trajectory_gs_path(
                relative_episode_path=relative_episode_path,
                raw_bucket_root=args.raw_bucket_root,
            ),
            "raw_video_gs_paths": raw_video_gs_paths,
            "episode_meta_relative_path": str(relative_root / "episodes" / timestamp / "episode_meta.json"),
            "joint_relative_path": str(relative_root / "episodes" / timestamp / "joint_position.npy"),
            "gripper_relative_path": str(relative_root / "episodes" / timestamp / "gripper_position.npy"),
            "annotation_assets": {
                "camera_serials_json": str(camera_serials_json),
                "cam2base_json": str(cam2base_json),
                "cam2base_superset_json": (
                    str(cam2base_superset_json) if cam2base_superset_json is not None else None
                ),
                "intrinsics_json": str(intrinsics_json),
                "episode_id_to_path_json": str(episode_id_to_path_json),
                "keep_ranges_json": str(keep_ranges_json),
                "raw_bucket_root": args.raw_bucket_root,
            },
        }
        records.append(record)
        accepted += 1
        if scanned % progress_interval == 0:
            report_progress(scanned=scanned, accepted=accepted, failures=failures, started_at=started_at)

    write_jsonl(output_jsonl, records)
    write_jsonl(failures_jsonl, failures)
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "split": args.split,
                "scanned": scanned,
                "accepted": accepted,
                "rejected": len(failures),
                "output_jsonl": str(output_jsonl),
                "failures_jsonl": str(failures_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
