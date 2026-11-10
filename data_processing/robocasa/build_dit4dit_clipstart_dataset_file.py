#!/usr/bin/env python3
"""Build dataset_file splits for DiT4DiT clip-start RoboCasa rollouts.

Expected layout:

    <root>/<task>/<episode>/videos_clips81/<stem>.mp4
    <root>/<task>/<episode>/videos_static_clips81/<stem>.mp4
    <root>/<task>/<episode>/stereo_depth_videos_clips81/<stem>.mp4
    <root>/<task>/<episode>/robot_mesh_videos_nominal_clips81/<stem>.mp4
    <root>/<task>/<episode>/robot_eef_nominal_depth_videos_clips81/<stem>.mp4

The output lines are relative to the repo data root, matching the DROID and
RoboCasa render_v2 dataset_file convention.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__import__("os").environ.get("ROFACTO_DATA_ROOT", REPO_ROOT / "data"))
DEFAULT_ROOT_DIR = DATA_ROOT / "robocasa" / "dit4dit_clipstart_480x832_balanced24"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "dataset_files" / "robocasa_dit4dit_clipstart_480x832_balanced24_random_split"
)
DEFAULT_REQUIRED_MP4 = [
    "videos_clips81",
    "static_videos_clips81",
    "stereo_depth_videos_clips81",
    "robot_mesh_videos_nominal_clips81",
    "robot_eef_nominal_depth_videos_clips81",
]


def read_ratio(value: str) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1)")
    return ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=read_ratio, default=0.02)
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
    return parser.parse_args()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    values = list(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def rel_to_data_root(path: Path, data_root: Path) -> str:
    return path.resolve().relative_to(data_root.resolve()).as_posix()


def replace_modality(rel: str, src_dir: str, dst_dir: str, suffix: str) -> str:
    parts = rel.split("/")
    if len(parts) < 2 or parts[-2] != src_dir:
        raise ValueError(f"Expected path under {src_dir}: {rel}")
    parts[-2] = dst_dir
    parts[-1] = Path(parts[-1]).with_suffix(suffix).name
    return "/".join(parts)


def parse_rate(raw: str | None) -> float | None:
    if not raw or raw == "0/0":
        return None
    if "/" in raw:
        num, den = raw.split("/", 1)
        den_value = float(den)
        if den_value == 0:
            return None
        return float(num) / den_value
    return float(raw)


def ffprobe_video(path: Path) -> dict[str, float | int | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError("ffprobe found no video stream")
    stream = streams[0]
    frames_raw = stream.get("nb_read_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(frames_raw) if frames_raw not in (None, "N/A") else None,
        "avg_fps": parse_rate(stream.get("avg_frame_rate")),
        "r_fps": parse_rate(stream.get("r_frame_rate")),
    }


def validate_video(path: Path, args: argparse.Namespace) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    if not args.probe_video:
        return True, ""
    try:
        info = ffprobe_video(path)
    except Exception as exc:
        return False, f"ffprobe_error:{type(exc).__name__}"
    if info["width"] != args.expected_width or info["height"] != args.expected_height:
        return False, f"bad_resolution:{info['width']}x{info['height']}"
    if info["frames"] != args.expected_frames:
        return False, f"bad_frames:{info['frames']}"
    fps = info["avg_fps"] or info["r_fps"]
    if fps is None or abs(float(fps) - args.expected_fps) > args.fps_tolerance:
        return False, f"bad_fps:{fps}"
    return True, ""


def has_valid_json(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    general = payload.get("general")
    return isinstance(general, dict) and bool(general.get("text"))


def required_rels(rel: str, args: argparse.Namespace) -> list[tuple[str, str]]:
    dynamic_dir = "videos_clips81"
    required = [("mp4", replace_modality(rel, dynamic_dir, subdir, ".mp4")) for subdir in args.required_mp4_modalities]
    if args.require_prompts:
        required.append(("json", replace_modality(rel, dynamic_dir, args.prompt_dir, ".json")))
    return required


def episode_key(rel: str) -> tuple[str, str]:
    parts = rel.split("/")
    # <root-rel>/<task>/<episode>/videos_clips81/<stem>.mp4
    if len(parts) < 5:
        raise ValueError(f"Unexpected clip path: {rel}")
    return parts[-4], parts[-3]


def task_key(rel: str) -> str:
    return episode_key(rel)[0]


def validate_candidate(rel: str, args: argparse.Namespace, data_root: Path) -> tuple[str | None, str]:
    try:
        required = required_rels(rel, args)
    except ValueError:
        return None, "bad_path"
    for kind, required_rel in required:
        path = data_root / required_rel
        if kind == "json":
            if not has_valid_json(path):
                return None, f"missing_json:{Path(required_rel).parent.name}"
            continue
        ok, reason = validate_video(path, args)
        if not ok:
            return None, f"{reason}:{Path(required_rel).parent.name}"
    return rel, "kept"


def build_rsync_files(clips: list[str], args: argparse.Namespace, data_root: Path) -> list[str]:
    files: set[str] = set()
    for rel in clips:
        for _kind, required_rel in required_rels(rel, args):
            if (data_root / required_rel).is_file():
                files.add(required_rel)
    return sorted(files)


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
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
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    n_test = max(1, round(len(episodes) * args.test_ratio)) if episodes else 0
    test_episodes = set(episodes[:n_test])
    train_episodes = set(episodes[n_test:])

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

    summary = {
        "root_dir": str(root_dir),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "require_prompts": args.require_prompts,
        "test_ratio": args.test_ratio,
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
