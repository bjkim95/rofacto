#!/usr/bin/env python3
"""Build task-balanced RoboCasa DiT4DiT clip-start dataset_file splits.

The random split is episode-level, but a tiny random test ratio can miss many
tasks. This builder keeps the same dataset_file convention while selecting test
episodes per task, preferably one successful and one failed rollout per task.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from build_dit4dit_clipstart_dataset_file import (
    DATA_ROOT,
    DEFAULT_REQUIRED_MP4,
    DEFAULT_ROOT_DIR,
    REPO_ROOT,
    build_rsync_files,
    episode_key,
    rel_to_data_root,
    task_key,
    validate_candidate,
    write_lines,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "dataset_files"
    / "robocasa_dit4dit_clipstart_480x832_balanced24_task_success_balanced_split"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-mp4-modalities", nargs="+", default=DEFAULT_REQUIRED_MP4)
    parser.add_argument("--require-prompts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-dir", default="videos_clips81_prompts_qwen3vl")
    parser.add_argument("--probe-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-width", type=int, default=832)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-frames", type=int, default=81)
    parser.add_argument("--expected-fps", type=float, default=16.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.05)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--test-success-per-task", type=int, default=1)
    parser.add_argument("--test-failure-per-task", type=int, default=1)
    parser.add_argument(
        "--min-test-episodes-per-task",
        type=int,
        default=2,
        help="Backfill from available statuses so every task has at least this many test episodes.",
    )
    return parser.parse_args()
def validate_output_dir(output_dir: Path, data_root: Path) -> Path:
    """Reject outputs under the input data tree or the source tree behind its symlink."""
    raw_data_root = data_root.expanduser()
    resolved_output_dir = output_dir.expanduser().resolve()
    forbidden_roots = [raw_data_root.resolve()]
    for candidate_data_root in (raw_data_root, DATA_ROOT.expanduser()):
        if candidate_data_root.is_symlink():
            forbidden_roots.append(candidate_data_root.resolve().parent)

    for forbidden_root in forbidden_roots:
        try:
            resolved_output_dir.relative_to(forbidden_root)
        except ValueError:
            continue
        raise ValueError(
            f"Output directory must not be under data root or its symlink target: {resolved_output_dir}"
        )
    return resolved_output_dir




def read_episode_status(root_dir: Path, task: str, episode: str) -> str:
    status_path = root_dir / task / episode / "rollout_actions_and_success.npz"
    if not status_path.is_file():
        return "unknown"
    try:
        with np.load(status_path) as payload:
            successes = np.asarray(payload["successes"])
    except Exception:
        return "unknown"
    return "success" if bool(successes.any()) else "failure"


def shuffled(values: Iterable[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    items = sorted(values)
    rng.shuffle(items)
    return items


def select_test_episodes(
    episodes: list[tuple[str, str]],
    episode_status: dict[tuple[str, str], str],
    args: argparse.Namespace,
) -> tuple[set[tuple[str, str]], dict[str, object]]:
    rng = random.Random(args.seed)
    by_task_status: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for episode in sorted(episodes):
        task, _episode_name = episode
        by_task_status[task][episode_status[episode]].append(episode)

    selected: set[tuple[str, str]] = set()
    shortages: list[dict[str, object]] = []

    for task in sorted(by_task_status):
        selected_for_task: set[tuple[str, str]] = set()
        quotas = {
            "success": args.test_success_per_task,
            "failure": args.test_failure_per_task,
        }
        for status, quota in quotas.items():
            candidates = shuffled(by_task_status[task].get(status, []), rng)
            picked = candidates[:quota]
            selected_for_task.update(picked)
            if len(picked) < quota:
                shortages.append(
                    {
                        "task": task,
                        "status": status,
                        "requested": quota,
                        "available": len(candidates),
                        "picked": len(picked),
                    }
                )

        if len(selected_for_task) < args.min_test_episodes_per_task:
            remaining = [
                episode
                for status in ("success", "failure", "unknown")
                for episode in shuffled(by_task_status[task].get(status, []), rng)
                if episode not in selected_for_task
            ]
            needed = args.min_test_episodes_per_task - len(selected_for_task)
            selected_for_task.update(remaining[:needed])
            if len(remaining) < needed:
                shortages.append(
                    {
                        "task": task,
                        "status": "any",
                        "requested": args.min_test_episodes_per_task,
                        "available": len(selected_for_task) + len(remaining),
                        "picked": len(selected_for_task),
                    }
                )

        selected.update(selected_for_task)

    debug = {
        "shortages": shortages,
        "test_success_per_task": args.test_success_per_task,
        "test_failure_per_task": args.test_failure_per_task,
        "min_test_episodes_per_task": args.min_test_episodes_per_task,
    }
    return selected, debug


def counter_to_dict(counter: Counter[object]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def nested_counter_to_dict(mapping: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: counter_to_dict(value) for key, value in sorted(mapping.items())}


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = validate_output_dir(args.output_dir, args.data_root)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    dynamic_paths = sorted(root_dir.glob("*/*/videos_clips81/*.mp4"))
    if args.max_clips > 0:
        dynamic_paths = dynamic_paths[: args.max_clips]

    stats: Counter[str] = Counter()
    episode_to_clips: dict[tuple[str, str], list[str]] = defaultdict(list)
    task_counts: Counter[str] = Counter()
    for path in dynamic_paths:
        stats["candidate"] += 1
        rel = rel_to_data_root(path, data_root)
        kept, reason = validate_candidate(rel, args, data_root)
        stats[reason] += 1
        if kept is None:
            continue
        key = episode_key(kept)
        episode_to_clips[key].append(kept)
        task_counts[task_key(kept)] += 1

    episodes = sorted(episode_to_clips)
    episode_status = {
        episode: read_episode_status(root_dir, episode[0], episode[1]) for episode in episodes
    }
    test_episodes, selection_debug = select_test_episodes(episodes, episode_status, args)
    train_episodes = set(episodes) - test_episodes

    train = [clip for episode in sorted(train_episodes) for clip in episode_to_clips[episode]]
    test = [clip for episode in sorted(test_episodes) for clip in episode_to_clips[episode]]
    all_kept = sorted(train + test)

    suffix = "with_prompts" if args.require_prompts else "no_prompt"
    all_path = output_dir / f"all_videos_clips81_{suffix}.txt"
    train_path = output_dir / f"train_videos_clips81_{suffix}.txt"
    test_path = output_dir / f"test_videos_clips81_{suffix}.txt"
    all_rsync_path = output_dir / f"all_rsync_files_{suffix}.txt"
    train_rsync_path = output_dir / f"train_rsync_files_{suffix}.txt"
    test_rsync_path = output_dir / f"test_rsync_files_{suffix}.txt"
    summary_path = output_dir / f"summary_{suffix}.json"

    train_rsync = build_rsync_files(train, args, data_root)
    test_rsync = build_rsync_files(test, args, data_root)
    dataset_file_rels = []
    for path in (all_path, train_path, test_path, all_rsync_path, train_rsync_path, test_rsync_path, summary_path):
        try:
            dataset_file_rels.append(rel_to_data_root(path, data_root))
        except ValueError:
            # Output overrides may intentionally be outside the data symlink.
            pass
    all_rsync = sorted(set(train_rsync) | set(test_rsync) | set(dataset_file_rels))

    write_lines(all_path, all_kept)
    write_lines(train_path, train)
    write_lines(test_path, test)
    write_lines(train_rsync_path, sorted(set(train_rsync) | set(dataset_file_rels)))
    write_lines(test_rsync_path, sorted(set(test_rsync) | set(dataset_file_rels)))
    write_lines(all_rsync_path, all_rsync)

    status_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    test_status_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    train_status_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    test_clip_counts_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    train_clip_counts_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        task = episode[0]
        status = episode_status[episode]
        status_by_task[task][status] += 1
        if episode in test_episodes:
            test_status_by_task[task][status] += 1
            test_clip_counts_by_task[task][status] += len(episode_to_clips[episode])
        else:
            train_status_by_task[task][status] += 1
            train_clip_counts_by_task[task][status] += len(episode_to_clips[episode])

    summary = {
        "root_dir": str(root_dir),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "require_prompts": args.require_prompts,
        "seed": args.seed,
        "stats": dict(sorted(stats.items())),
        "episodes": {
            "all": len(episodes),
            "train": len(train_episodes),
            "test": len(test_episodes),
        },
        "clips": {
            "all": len(all_kept),
            "train": len(train),
            "test": len(test),
        },
        "tasks": dict(sorted(task_counts.items())),
        "episode_status_by_task": nested_counter_to_dict(status_by_task),
        "train_episode_status_by_task": nested_counter_to_dict(train_status_by_task),
        "test_episode_status_by_task": nested_counter_to_dict(test_status_by_task),
        "train_clip_status_by_task": nested_counter_to_dict(train_clip_counts_by_task),
        "test_clip_status_by_task": nested_counter_to_dict(test_clip_counts_by_task),
        "selection": selection_debug,
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "test": str(test_path),
            "all_rsync": str(all_rsync_path),
            "train_rsync": str(train_rsync_path),
            "test_rsync": str(test_rsync_path),
            "summary": str(summary_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
