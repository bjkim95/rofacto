#!/usr/bin/env python3
"""
Qwen3-VL clip captioning for DROID render_v2.

For each source video under `<root_dir>/<lab>/<episode>/<video_folder_name>/<stem>__<cam>.mp4`:
  1. Compute non-overlap 81-frame clip ranges with tail aligned to the last 81 frames.
  2. Cut each clip with ffmpeg -> `<root_dir>/<lab>/<episode>/<clip_folder_name>/<stem>__c<idx>_<a>-<b>.mp4`.
  3. Sample `--num_sampled_frames` frames uniformly from each clip.
  4. Run Qwen3-VL on the sampled frames with two prompts (loaded from
     `--input_prompt_files key=path`) and write
     `<root_dir>/<lab>/<episode>/<caption_folder_name>/<stem>__c<idx>_<a>-<b>.json`
     with `{detailed: {...}, general: {...}, meta...}` keys.

Array job: split videos across `--array_index` / `--num_splits`. Skips clips that
already have a caption JSON when `--skip_existing` is set.

Source attribution: structure mirrors `internvl2_video_recaptioning.py` from
this repo (job discovery, array splitting, atomic writes, manifest pattern).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


# --------------------------------------------------------------------- types

@dataclass
class Job:
    lab: str
    episode: str
    video_path: Path


@dataclass
class ClipTask:
    job: Job
    clip_idx: int
    a: int  # inclusive start frame in original video
    b: int  # inclusive end frame in original video
    fps: float
    n_orig: int
    clip_path: Path
    caption_path: Path
    missing_keys: frozenset  # prompt keys that still need to be generated


# --------------------------------------------------------------------- discovery

CAMERA_ORDER = {"ext1": 0, "ext2": 1, "wrist": 2}


def parse_camera_suffix(stem: str) -> str | None:
    """Return 'ext1', 'ext2', 'wrist', etc. from a stem like 'YYYY-...__ext1'."""
    sep = "__"
    if sep not in stem:
        return None
    return stem.rsplit(sep, 1)[-1]


def gather_droid_jobs(
    root_dir: Path,
    video_folder_name: str,
    cameras: set[str],
) -> list[Job]:
    jobs: list[Job] = []
    for lab_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for ep_dir in sorted(p for p in lab_dir.iterdir() if p.is_dir()):
            video_dir = ep_dir / video_folder_name
            if not video_dir.exists():
                continue
            for video_path in sorted(video_dir.glob("*.mp4")):
                if video_path.name.startswith("."):
                    continue  # partial / temp downloads
                cam = parse_camera_suffix(video_path.stem)
                if cam is None or cam not in cameras:
                    continue
                jobs.append(Job(lab=lab_dir.name, episode=ep_dir.name, video_path=video_path))
    return jobs


def split_items(items, array_index: int, num_splits: int):
    if num_splits <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_splits == array_index]


# --------------------------------------------------------------------- clip layout

def list_clip_ranges(
    n_orig: int,
    clip_len: int,
    clip_stride: int | None = None,
) -> list[tuple[int, int]]:
    if n_orig < clip_len:
        return []
    stride = int(clip_len if clip_stride is None else clip_stride)
    if stride <= 0:
        raise ValueError(f"clip_stride must be positive, got {clip_stride}")
    starts = list(range(0, n_orig - clip_len + 1, stride))
    last_start = n_orig - clip_len
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(s, s + clip_len - 1) for s in starts]


def clip_paths(
    job: Job,
    clip_idx: int,
    a: int,
    b: int,
    clip_folder_name: str,
    caption_folder_name: str,
) -> tuple[Path, Path]:
    ep_dir = job.video_path.parent.parent
    clip_dir = ep_dir / clip_folder_name
    cap_dir = ep_dir / caption_folder_name
    name = f"{job.video_path.stem}__c{clip_idx:02d}_{a:04d}-{b:04d}"
    return clip_dir / f"{name}.mp4", cap_dir / f"{name}.json"


def video_meta(video_path: Path) -> tuple[int, float]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    return len(vr), float(vr.get_avg_fps())


# --------------------------------------------------------------------- ffmpeg cut

def cut_clip_ffmpeg(video_path: Path, out_path: Path, a: int, b: int, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expr = f"between(n\\,{a}\\,{b})"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"select='{expr}',setpts=N/FRAME_RATE/TB",
        "-an",
        "-r", f"{fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def sample_clip_frames(clip_path: Path, num: int) -> list[Image.Image]:
    vr = VideoReader(str(clip_path), ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty clip: {clip_path}")
    idx = (
        np.linspace(0, n - 1, num).round().astype(int).tolist() if num > 1 else [0]
    )
    arr = vr.get_batch(idx).asnumpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]


# --------------------------------------------------------------------- prompts / requests

def parse_prompt_args(values: list[str]) -> dict[str, Path]:
    prompts: dict[str, Path] = {}
    for v in values:
        if "=" not in v:
            raise ValueError(f"--input_prompt_files expects key=path, got {v!r}")
        key, path = v.split("=", 1)
        key = key.strip()
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Prompt file not found: {path_obj}")
        prompts[key] = path_obj
    return prompts


def build_request(processor, frames: list[Image.Image], base_prompt: str) -> dict:
    content = []
    for i, _ in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": base_prompt})
    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": chat_text, "multi_modal_data": {"image": frames}}


# --------------------------------------------------------------------- IO helpers

def atomic_write_json(save_path: Path, payload: dict) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if not text.strip():
        raise RuntimeError(f"Refusing to save empty JSON: {save_path}")
    tmp = save_path.with_name(f".{save_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as h:
            h.write(text)
            h.flush()
            os.fsync(h.fileno())
        os.replace(tmp, save_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def missing_prompt_keys(p: Path, all_keys: set[str]) -> frozenset[str]:
    """Return the subset of all_keys not yet present in the caption JSON at p.

    If the file is absent or corrupt, all keys are considered missing.
    If the file exists and contains all keys, returns an empty frozenset.
    """
    if not p.exists():
        return frozenset(all_keys)
    try:
        if p.stat().st_size == 0:
            p.unlink()
            return frozenset(all_keys)
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return frozenset(k for k in all_keys if k not in data)
    except (OSError, json.JSONDecodeError):
        return frozenset(all_keys)


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--video_folder_name", default="videos")
    parser.add_argument("--clip_folder_name", default="clips81")
    parser.add_argument("--caption_folder_name", default="clips81_prompts_qwen3vl")
    parser.add_argument(
        "--cameras", nargs="+", default=["ext1", "ext2"],
        help="Camera suffixes to include (after __). Default: ext1 ext2.",
    )
    parser.add_argument("--clip_len", type=int, default=81)
    parser.add_argument(
        "--clip_stride",
        type=int,
        default=None,
        help="Clip start stride. Defaults to --clip_len for historical non-overlap clips.",
    )
    parser.add_argument("--num_sampled_frames", type=int, default=27)
    parser.add_argument(
        "--input_prompt_files", nargs="+", required=True,
        help="key=path entries; each file becomes one prompt key in the output JSON. "
             "e.g. detailed=prompt/clip_detailed.txt general=prompt/clip_general.txt",
    )
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model_path", default="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_pixels", type=int, default=480 * 640)

    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of clips per llm.generate call. "
             "Effective in-flight requests = batch_size * len(prompts).",
    )

    parser.add_argument("--array_index", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument(
        "--limit_jobs", type=int, default=0,
        help="If >0, only process the first N jobs (debug).",
    )
    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    root_dir = Path(args.root_dir).resolve()
    cameras = set(args.cameras)
    prompts = parse_prompt_args(args.input_prompt_files)
    prompt_texts = {k: p.read_text(encoding="utf-8").strip() for k, p in prompts.items()}
    if not prompt_texts:
        raise ValueError("at least one --input_prompt_files entry required")

    print(f"[INFO] root_dir={root_dir}")
    print(f"[INFO] cameras={sorted(cameras)} clip_len={args.clip_len} "
          f"num_sampled_frames={args.num_sampled_frames}")
    print(f"[INFO] prompts={list(prompt_texts.keys())} "
          f"clip_folder={args.clip_folder_name} caption_folder={args.caption_folder_name}")

    all_jobs = gather_droid_jobs(root_dir, args.video_folder_name, cameras)
    print(f"[INFO] discovered_jobs={len(all_jobs)}")

    my_jobs = split_items(all_jobs, args.array_index, args.num_splits)
    if args.limit_jobs > 0:
        my_jobs = my_jobs[: args.limit_jobs]
    print(f"[INFO] this_array={args.array_index}/{args.num_splits} my_jobs={len(my_jobs)}")

    # Lazy-load model only after job list is ready.
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    t_load0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": max(1, args.num_sampled_frames)},
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        quantization=args.quantization,
        dtype="auto",
        mm_processor_kwargs={"max_pixels": args.max_pixels},
    )
    load_dt = time.perf_counter() - t_load0
    print(f"[INFO] model_load_seconds={load_dt:.1f}")

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=True,
        seed=args.seed,
    )

    # ---- enumerate ClipTasks for this array ----
    tasks: list[ClipTask] = []
    for job in my_jobs:
        try:
            n_orig, fps = video_meta(job.video_path)
        except Exception as exc:
            print(f"[WARN] video_meta_failed {job.video_path}: {exc}")
            continue
        ranges = list_clip_ranges(n_orig, args.clip_len, args.clip_stride)
        for clip_idx, (a, b) in enumerate(ranges):
            clip_path, cap_path = clip_paths(
                job, clip_idx, a, b,
                args.clip_folder_name, args.caption_folder_name,
            )
            if args.skip_existing:
                missing = missing_prompt_keys(cap_path, set(prompt_texts.keys()))
                if not missing:
                    continue
            else:
                missing = frozenset(prompt_texts.keys())
            tasks.append(ClipTask(
                job=job, clip_idx=clip_idx, a=a, b=b, fps=fps,
                n_orig=n_orig, clip_path=clip_path, caption_path=cap_path,
                missing_keys=missing,
            ))
    print(f"[INFO] clip_tasks={len(tasks)}")

    # ---- batched processing ----
    n_done = 0
    n_failed = 0
    t_overall0 = time.perf_counter()
    for batch_start in range(0, len(tasks), args.batch_size):
        batch = tasks[batch_start : batch_start + args.batch_size]
        # 1) cut + sample frames sequentially (CPU-bound; fast)
        prepped: list[tuple[ClipTask, list[Image.Image]] | None] = []
        for task in batch:
            try:
                if not task.clip_path.exists():
                    cut_clip_ffmpeg(task.job.video_path, task.clip_path, task.a, task.b, task.fps)
                frames = sample_clip_frames(task.clip_path, args.num_sampled_frames)
                prepped.append((task, frames))
            except Exception as exc:
                print(f"[ERROR] prep_failed {task.clip_path}: {exc}")
                n_failed += 1
                prepped.append(None)

        # 2) build requests only for missing keys per clip
        requests: list[dict] = []
        request_meta: list[tuple[int, str]] = []  # (batch_idx, prompt_key)
        for i, item in enumerate(prepped):
            if item is None:
                continue
            task, frames = item
            for key, ptext in prompt_texts.items():
                if key not in task.missing_keys:
                    continue
                requests.append(build_request(processor, frames, ptext))
                request_meta.append((i, key))

        if not requests:
            continue

        # 3) one llm.generate call (vLLM batches/continuous-batches internally)
        t0 = time.perf_counter()
        outputs = llm.generate(requests, sampling_params=sp)
        dt = time.perf_counter() - t0

        # 4) collect outputs by clip
        per_clip: dict[int, dict[str, dict]] = {}
        for (i, key), out in zip(request_meta, outputs):
            text = out.outputs[0].text.strip()
            n_tok = len(out.outputs[0].token_ids)
            per_clip.setdefault(i, {})[key] = {"text": text, "tokens": n_tok}

        # 5) save per-clip JSON atomically (merge into existing if present)
        for i, item in enumerate(prepped):
            if item is None:
                continue
            task, _ = item
            captions = per_clip.get(i)
            if not captions:
                n_failed += 1
                continue
            # Load existing data to merge into (handles partial JSONs)
            existing: dict = {}
            if task.caption_path.exists():
                try:
                    with task.caption_path.open(encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    existing = {}
            base = {
                "source_video": str(task.job.video_path),
                "lab": task.job.lab,
                "episode": task.job.episode,
                "camera": parse_camera_suffix(task.job.video_path.stem),
                "clip_index": task.clip_idx,
                "orig_start": task.a,
                "orig_end": task.b,
                "n_orig_frames": task.n_orig,
                "fps": task.fps,
                "clip_len": args.clip_len,
                "num_sampled_frames": args.num_sampled_frames,
                "clip_path": str(task.clip_path),
                "model_path": args.model_path,
                "quantization": args.quantization,
                "max_pixels": args.max_pixels,
            }
            payload = {**base, **existing, **captions}  # new captions win on conflict
            try:
                atomic_write_json(task.caption_path, payload)
                n_done += 1
            except Exception as exc:
                print(f"[ERROR] write_failed {task.caption_path}: {exc}")
                n_failed += 1

        elapsed_total = time.perf_counter() - t_overall0
        avg_per_clip = elapsed_total / max(n_done, 1)
        print(
            f"[BATCH] reqs={len(requests)} dt={dt:.2f}s "
            f"n_done={n_done}/{len(tasks)} n_failed={n_failed} "
            f"avg_per_clip={avg_per_clip:.2f}s"
        )

    overall_dt = time.perf_counter() - t_overall0
    print(
        f"[SUMMARY] processed={n_done} failed={n_failed} "
        f"total={overall_dt:.1f}s avg_per_clip={overall_dt/max(n_done,1):.2f}s"
    )
    gc.collect()


if __name__ == "__main__":
    main()
