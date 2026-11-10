#!/usr/bin/env python3
"""
Cut DROID render_v2 mp4 modalities into 81-frame clips.

Walks `<root_dir>/<lab>/<episode>/<modality>/<stem>__<cam>.mp4` for every
`(modality, camera)` in the given set, computes non-overlap 81-frame clip
ranges with a tail-aligned final clip (see `clip_utils.list_clip_ranges`),
and runs ffmpeg to write
`<root_dir>/<lab>/<episode>/<modality>_clips81/<stem>__<cam>__c<idx>_<a>-<b>.mp4`.

CPU only (ffmpeg). Designed for SLURM array sharding; idempotent with
`--skip_existing`.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from decord import VideoReader, cpu

# Allow running as a script: add parent dir so `clip_utils` resolves.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_utils import clip_stem, cut_clip_ffmpeg, list_clip_ranges


@dataclass
class Job:
    lab: str
    episode: str
    modality: str
    video_path: Path


def parse_camera_suffix(stem: str) -> str | None:
    sep = "__"
    if sep not in stem:
        return None
    return stem.rsplit(sep, 1)[-1]


def gather_jobs(
    root_dir: Path,
    modalities: list[str],
    cameras: set[str],
) -> list[Job]:
    jobs: list[Job] = []
    for lab_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for ep_dir in sorted(p for p in lab_dir.iterdir() if p.is_dir()):
            for modality in modalities:
                mod_dir = ep_dir / modality
                if not mod_dir.exists():
                    continue
                for video_path in sorted(mod_dir.glob("*.mp4")):
                    if video_path.name.startswith("."):
                        continue
                    cam = parse_camera_suffix(video_path.stem)
                    if cam is None or cam not in cameras:
                        continue
                    jobs.append(Job(
                        lab=lab_dir.name, episode=ep_dir.name,
                        modality=modality, video_path=video_path,
                    ))
    return jobs


def split_for_array(items, array_index: int, num_splits: int):
    if num_splits <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_splits == array_index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=None,
                   help="Legacy input/output root; superseded by --input-root and --output-root.")
    p.add_argument("--input-root", type=Path, default=None,
                   help="Root containing render_v2 modality directories.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Root for generated clips; defaults to --input-root.")
    p.add_argument(
        "--modalities", nargs="+", required=True,
        help="Modality dirs to process, e.g. robot_mesh_videos robot_mesh_videos_pyrender",
    )
    p.add_argument("--cameras", nargs="+", default=["ext1", "ext2", "wrist"])
    p.add_argument("--clip_len", type=int, default=81)
    p.add_argument(
        "--clip_stride",
        type=int,
        default=None,
        help="Clip start stride. Defaults to --clip_len for historical non-overlap clips.",
    )
    p.add_argument("--clip_dir_suffix", default="_clips81",
                   help="Output dir for each modality is <modality><suffix>.")
    p.add_argument("--array_index", type=int, default=0)
    p.add_argument("--num_splits", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--best-effort",
        action="store_true",
        help="Exit zero after recording decode or clip-write failures.",
    )
    p.add_argument("--limit_jobs", type=int, default=0,
                   help="If >0, only process the first N jobs (debug).")
    p.add_argument("--print_every", type=int, default=50)
    args = p.parse_args()

    if args.input_root is None and args.root_dir is None:
        p.error("one of --input-root or --root_dir is required")
    input_root = Path(args.input_root or args.root_dir).resolve()
    output_root = Path(args.output_root or input_root).resolve()
    cameras = set(args.cameras)

    print(f"[INFO] input_root={input_root} output_root={output_root}")
    print(f"[INFO] modalities={args.modalities} cameras={sorted(cameras)} clip_len={args.clip_len}")

    all_jobs = gather_jobs(input_root, args.modalities, cameras)
    print(f"[INFO] discovered_jobs={len(all_jobs)}")

    my_jobs = split_for_array(all_jobs, args.array_index, args.num_splits)
    if args.limit_jobs > 0:
        my_jobs = my_jobs[: args.limit_jobs]
    print(f"[INFO] this_array={args.array_index}/{args.num_splits} my_jobs={len(my_jobs)}")

    n_clips_done = 0
    n_clips_skipped = 0
    n_videos_failed = 0
    n_clip_failures = 0
    t0 = time.perf_counter()

    for job_i, job in enumerate(my_jobs):
        try:
            vr = VideoReader(str(job.video_path), ctx=cpu(0))
            n_orig = len(vr)
            fps = float(vr.get_avg_fps())
        except Exception as exc:
            print(f"[WARN] decord_failed {job.video_path}: {exc}")
            n_videos_failed += 1
            continue

        ranges = list_clip_ranges(n_orig, args.clip_len, args.clip_stride)
        if not ranges:
            continue

        out_dir = output_root / job.lab / job.episode / f"{job.modality}{args.clip_dir_suffix}"

        for clip_idx, (a, b) in enumerate(ranges):
            stem = clip_stem(job.video_path.stem, clip_idx, a, b)
            out_path = out_dir / f"{stem}.mp4"
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                n_clips_skipped += 1
                continue
            try:
                cut_clip_ffmpeg(job.video_path, out_path, a, b, fps)
                n_clips_done += 1
            except Exception as exc:
                print(f"[ERROR] ffmpeg_failed {out_path}: {exc}")
                n_clip_failures += 1

        if (job_i + 1) % args.print_every == 0:
            dt = time.perf_counter() - t0
            print(
                f"[PROGRESS] videos={job_i+1}/{len(my_jobs)} "
                f"clips_done={n_clips_done} skipped={n_clips_skipped} "
                f"video_failures={n_videos_failed} clip_failures={n_clip_failures} elapsed={dt:.1f}s"
            )

    dt = time.perf_counter() - t0
    print(
        f"[SUMMARY] videos={len(my_jobs)} clips_done={n_clips_done} "
        f"skipped={n_clips_skipped} video_failures={n_videos_failed} "
        f"clip_failures={n_clip_failures} elapsed={dt:.1f}s"
    )
    if not args.best_effort and (n_videos_failed or n_clip_failures):
        raise SystemExit(
            "clip cutting completed with failures; rerun with --best-effort to preserve exit zero"
        )



if __name__ == "__main__":
    main()
