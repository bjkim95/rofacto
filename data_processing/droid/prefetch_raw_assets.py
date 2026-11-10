#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .common import DEFAULT_CACHE_ROOT, download_gs_file
except ImportError:
    from common import DEFAULT_CACHE_ROOT, download_gs_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch raw DROID trajectory.h5 and raw mp4 files into the local cache."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root used to resolve a relative --manifest-jsonl path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination raw-cache root; overrides --cache-root.",
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--task-tag", type=str, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
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


def resolve_task_tag(explicit: str | None, shard_index: int, num_shards: int) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return f"{job_id}_{task_id}"
    return f"{job_id}_shard{shard_index}of{num_shards}"


def prepare_result_logs(cache_root: Path, task_tag: str) -> tuple[Path, Path, Path]:
    manifest_dir = cache_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_path = manifest_dir / f"prefetch_results_{task_tag}.jsonl"
    failures_path = manifest_dir / f"prefetch_failures_{task_tag}.jsonl"
    summary_path = manifest_dir / f"prefetch_summary_{task_tag}.json"
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def localized_paths_for_record(record: dict[str, Any], cache_root: Path) -> tuple[Path, dict[str, Path]]:
    episode_cache_dir = cache_root / str(record["episode_id"])
    trajectory_path = episode_cache_dir / "trajectory.h5"
    video_paths: dict[str, Path] = {}
    for view in record.get("views", []):
        if not isinstance(view, dict):
            continue
        view_name = str(view["view_name"])
        serial = str(view["serial"])
        video_paths[view_name] = episode_cache_dir / f"{serial}.mp4"
    return trajectory_path, video_paths


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def fetch_single_asset(
    *,
    asset_name: str,
    gs_path: str,
    local_path: Path,
    overwrite_cache: bool,
) -> dict[str, Any]:
    existed_before = is_nonempty_file(local_path)
    if existed_before and not overwrite_cache:
        return {
            "asset_name": asset_name,
            "gs_path": gs_path,
            "local_path": str(local_path),
            "status": "existing",
            "available": True,
        }

    try:
        downloaded_path = download_gs_file(
            gs_path=gs_path,
            local_path=local_path,
            overwrite=overwrite_cache,
        )
    except Exception as exc:
        return {
            "asset_name": asset_name,
            "gs_path": gs_path,
            "local_path": str(local_path),
            "status": "failed",
            "available": is_nonempty_file(local_path),
            "error": str(exc),
        }

    return {
        "asset_name": asset_name,
        "gs_path": gs_path,
        "local_path": str(downloaded_path),
        "status": "downloaded",
        "available": is_nonempty_file(downloaded_path),
    }


def prefetch_record(record: dict[str, Any], cache_root: Path, overwrite_cache: bool) -> dict[str, Any]:
    episode_id = str(record["episode_id"])
    trajectory_local_path, video_local_paths = localized_paths_for_record(record, cache_root)
    if (
        not overwrite_cache
        and is_nonempty_file(trajectory_local_path)
        and all(is_nonempty_file(path) for path in video_local_paths.values())
        ):
        return {
            "episode_id": episode_id,
            "episode_index": record.get("episode_index"),
            "status": "skipped_existing",
            "trajectory_local_path": str(trajectory_local_path),
            "video_local_paths": {k: str(v) for k, v in video_local_paths.items()},
        }

    trajectory_result = fetch_single_asset(
        asset_name="trajectory.h5",
        gs_path=str(record["raw_trajectory_gs_path"]),
        local_path=trajectory_local_path,
        overwrite_cache=overwrite_cache,
    )
    video_results: dict[str, dict[str, Any]] = {}
    raw_video_gs_paths = record.get("raw_video_gs_paths", {})
    for view in record.get("views", []):
        view_name = str(view["view_name"])
        local_path = video_local_paths[view_name]
        video_results[view_name] = fetch_single_asset(
            asset_name=f"{view_name}.mp4",
            gs_path=str(raw_video_gs_paths[view_name]),
            local_path=local_path,
            overwrite_cache=overwrite_cache,
        )

    asset_results = [trajectory_result, *video_results.values()]
    available_assets = [asset["asset_name"] for asset in asset_results if asset["available"]]
    missing_assets = [asset["asset_name"] for asset in asset_results if not asset["available"]]
    downloaded_assets = [asset["asset_name"] for asset in asset_results if asset["status"] == "downloaded"]
    failed_assets = [
        {
            "asset_name": asset["asset_name"],
            "gs_path": asset["gs_path"],
            "local_path": asset["local_path"],
            "error": asset.get("error"),
        }
        for asset in asset_results
        if asset["status"] == "failed"
    ]

    if all(asset["available"] for asset in asset_results):
        row_status = "prefetched" if downloaded_assets else "skipped_existing"
    elif any(asset["available"] for asset in asset_results):
        row_status = "prefetched_partial"
    else:
        row_status = "failed"

    return {
        "episode_id": episode_id,
        "episode_index": record.get("episode_index"),
        "status": row_status,
        "trajectory_local_path": trajectory_result["local_path"],
        "video_local_paths": {k: v["local_path"] for k, v in video_results.items()},
        "asset_results": asset_results,
        "available_assets": available_assets,
        "missing_assets": missing_assets,
        "downloaded_assets": downloaded_assets,
        "failed_assets": failed_assets,
    }


def report_progress(
    *,
    processed: int,
    total: int,
    prefetched: int,
    partial: int,
    skipped: int,
    failed: int,
    started_at: float,
) -> None:
    elapsed = time.monotonic() - started_at
    print(
        "[prefetch] "
        f"processed={processed}/{total} "
        f"prefetched={prefetched} "
        f"partial={partial} "
        f"skipped={skipped} "
        f"failed={failed} "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.output_root is not None:
        args.cache_root = args.output_root
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

    results_path, failures_path, summary_path = prepare_result_logs(args.cache_root, task_tag)
    started_at = time.monotonic()
    progress_interval = max(1, int(args.progress_interval))

    prefetched_records = 0
    partial_records = 0
    skipped_records = 0
    failed_records = 0

    for idx, row in enumerate(selected_rows, start=1):
        try:
            summary = prefetch_record(
                record=row,
                cache_root=args.cache_root,
                overwrite_cache=args.overwrite_cache,
            )
            if summary["status"] == "prefetched":
                prefetched_records += 1
            elif summary["status"] == "prefetched_partial":
                partial_records += 1
            elif summary["status"] == "failed":
                failed_records += 1
                append_jsonl(
                    failures_path,
                    [
                        {
                            "task_tag": task_tag,
                            "episode_id": summary["episode_id"],
                            "episode_index": summary["episode_index"],
                            "status": summary["status"],
                            "missing_assets": summary.get("missing_assets", []),
                            "failed_assets": summary.get("failed_assets", []),
                        }
                    ],
                )
            else:
                skipped_records += 1
            if summary["status"] == "prefetched_partial":
                append_jsonl(
                    failures_path,
                    [
                        {
                            "task_tag": task_tag,
                            "episode_id": summary["episode_id"],
                            "episode_index": summary["episode_index"],
                            "status": summary["status"],
                            "missing_assets": summary.get("missing_assets", []),
                            "failed_assets": summary.get("failed_assets", []),
                        }
                    ],
                )
            append_jsonl(
                results_path,
                [
                    {
                        "task_tag": task_tag,
                        "episode_id": summary["episode_id"],
                        "episode_index": summary["episode_index"],
                        "status": summary["status"],
                        "summary": summary,
                    }
                ],
            )
        except Exception as exc:
            failure = {
                "task_tag": task_tag,
                "episode_id": row.get("episode_id"),
                "episode_index": row.get("episode_index"),
                "status": "exception",
                "error": str(exc),
            }
            failed_records += 1
            append_jsonl(failures_path, [failure])

        if idx % progress_interval == 0 or idx == len(selected_rows):
            report_progress(
                processed=idx,
                total=len(selected_rows),
                prefetched=prefetched_records,
                partial=partial_records,
                skipped=skipped_records,
                failed=failed_records,
                started_at=started_at,
            )

    summary_payload = {
        "task_tag": task_tag,
        "manifest_jsonl": str(args.manifest_jsonl),
        "cache_root": str(args.cache_root),
        "selected_records": len(selected_rows),
        "prefetched_records": prefetched_records,
        "partial_records": partial_records,
        "skipped_records": skipped_records,
        "failed_records": failed_records,
        "results_path": str(results_path),
        "failures_path": str(failures_path),
    }
    write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, indent=2))

    if prefetched_records == 0 and partial_records == 0 and skipped_records == 0 and failed_records:
        sys.exit(1)


if __name__ == "__main__":
    main()
