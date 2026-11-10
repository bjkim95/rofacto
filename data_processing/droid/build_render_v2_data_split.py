#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a final render_v2 data split by matching render_v1 rewrite-ready "
            "train/test membership and keeping only rows with video, robot mesh, "
            "and prompts_structured_rewrite assets."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root used to resolve relative dataset-file inputs.",
    )
    parser.add_argument(
        "--v1-train",
        type=Path,
        default=Path("data/dataset_files/droid_render_v1_date_split/train_rewrite_ready.txt"),
    )
    parser.add_argument(
        "--v1-test",
        type=Path,
        default=Path("data/dataset_files/droid_render_v1_date_split/test_rewrite_ready.txt"),
    )
    parser.add_argument(
        "--v2-train",
        type=Path,
        default=Path("data/dataset_files/droid_render_v2_date_split_legacy/train_rewrite_ready.txt"),
    )
    parser.add_argument(
        "--v2-test",
        type=Path,
        default=Path("data/dataset_files/droid_render_v2_date_split_legacy/test_rewrite_ready.txt"),
    )
    parser.add_argument(
        "--complete-inventory",
        type=Path,
        default=Path(
            "data/dataset_files/droid_render_v2_inventory/video_robot_prompt_rewrite_complete.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/droid/dataset_files/droid_render_v2_data_split"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination directory for generated split files; overrides --output-dir.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def split_key(rel_video_path: str) -> tuple[str, str, str]:
    video_path = Path(rel_video_path)
    return video_path.parts[2], video_path.parts[3], video_path.stem


def robot_mesh_relpath(rel_video_path: str) -> str:
    video_path = Path(rel_video_path)
    return str(video_path.parent.parent / "robot_mesh_videos" / video_path.name)


def prompt_relpath(rel_video_path: str) -> str:
    video_path = Path(rel_video_path)
    return str(
        video_path.parent.parent / "prompts_structured_rewrite" / f"{video_path.stem}.txt"
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    args = parse_args()
    if args.input_root is not None:
        for name in ("v1_train", "v1_test", "v2_train", "v2_test", "complete_inventory"):
            path = getattr(args, name)
            if not path.is_absolute():
                setattr(args, name, args.input_root / path)
    if args.output_root is not None:
        args.output_dir = args.output_root

    v1_train = read_lines(args.v1_train)
    v1_test = read_lines(args.v1_test)
    v2_train = read_lines(args.v2_train)
    v2_test = read_lines(args.v2_test)
    complete_inventory = set(read_lines(args.complete_inventory))

    v1_membership: dict[tuple[str, str, str], str] = {}
    for rel in v1_train:
        v1_membership[split_key(rel)] = "train"
    for rel in v1_test:
        v1_membership[split_key(rel)] = "test"

    final_train: list[str] = []
    final_test: list[str] = []
    train_rsync_files: list[str] = []
    test_rsync_files: list[str] = []
    seen_rows: set[str] = set()
    dropped_missing_assets: list[str] = []

    matched_v1_train = 0
    matched_v1_test = 0
    unmatched_to_train = 0
    duplicate_rows = 0

    for rel_video_path in v2_train + v2_test:
        if rel_video_path in seen_rows:
            duplicate_rows += 1
            continue
        seen_rows.add(rel_video_path)

        if rel_video_path not in complete_inventory:
            dropped_missing_assets.append(rel_video_path)
            continue

        membership = v1_membership.get(split_key(rel_video_path))
        rsync_triplet = [
            rel_video_path,
            robot_mesh_relpath(rel_video_path),
            prompt_relpath(rel_video_path),
        ]
        if membership == "test":
            final_test.append(rel_video_path)
            test_rsync_files.extend(rsync_triplet)
            matched_v1_test += 1
        else:
            final_train.append(rel_video_path)
            train_rsync_files.extend(rsync_triplet)
            if membership == "train":
                matched_v1_train += 1
            else:
                unmatched_to_train += 1

    all_rsync_files = sorted(set(train_rsync_files) | set(test_rsync_files))

    output_dir = args.output_dir
    write_lines(output_dir / "train.txt", final_train)
    write_lines(output_dir / "test.txt", final_test)
    write_lines(output_dir / "train_rsync_files.txt", sorted(set(train_rsync_files)))
    write_lines(output_dir / "test_rsync_files.txt", sorted(set(test_rsync_files)))
    write_lines(output_dir / "all_rsync_files.txt", all_rsync_files)

    summary = {
        "inputs": {
            "v1_train": str(args.v1_train),
            "v1_test": str(args.v1_test),
            "v2_train": str(args.v2_train),
            "v2_test": str(args.v2_test),
            "complete_inventory": str(args.complete_inventory),
        },
        "counts": {
            "v1_train_source": len(v1_train),
            "v1_test_source": len(v1_test),
            "v2_train_source": len(v2_train),
            "v2_test_source": len(v2_test),
            "complete_inventory": len(complete_inventory),
            "matched_v1_train": matched_v1_train,
            "matched_v1_test": matched_v1_test,
            "unmatched_assigned_train": unmatched_to_train,
            "dropped_missing_assets": len(dropped_missing_assets),
            "duplicate_rows_skipped": duplicate_rows,
            "final_train": len(final_train),
            "final_test": len(final_test),
            "train_rsync_files": len(sorted(set(train_rsync_files))),
            "test_rsync_files": len(sorted(set(test_rsync_files))),
            "all_rsync_files": len(all_rsync_files),
        },
        "first_dropped_missing_assets": dropped_missing_assets[:20],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"train={len(final_train)} test={len(final_test)}")
    print(
        "matched_v1_train="
        f"{matched_v1_train} matched_v1_test={matched_v1_test} "
        f"unmatched_assigned_train={unmatched_to_train}"
    )
    print(f"dropped_missing_assets={len(dropped_missing_assets)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
