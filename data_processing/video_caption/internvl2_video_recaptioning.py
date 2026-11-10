#!/usr/bin/env python3
"""
InternVL2 video recaptioning for supported datasets.

Source attribution:
- This script is based on ideas and implementation patterns from:
  `VideoX-Fun/videox_fun/video_caption/internvl2_video_recaptioning_my.py`

Refactor notes:
- Path conventions simplified for supported dataset layouts.
- Prompt text is loaded from `--input_prompt_file` (no hardcoded prompt in runner scripts).
- Third-person mode defaults to JSON output for `prompts_aux`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


@dataclass
class Job:
    scene: str
    action: str
    video_path: Path


@dataclass
class Stats:
    jobs_total: int = 0
    jobs_selected: int = 0
    clips_seen: int = 0
    clips_generated: int = 0
    clips_skipped_existing: int = 0
    clips_failed: int = 0


def natural_sort_key(path: Path):
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def load_scene_filter(scene_filter_file: Path | None) -> set[str] | None:
    if scene_filter_file is None:
        return None
    if not scene_filter_file.exists():
        raise FileNotFoundError(f"Scene filter file not found: {scene_filter_file}")
    scenes = set()
    for line in scene_filter_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.add(line)
    return scenes


def load_path_entries_file(path: Path) -> set[str]:
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def normalize_text_for_dedup(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _normalize_exclude_video_files(exclude_videos_file: Path | list[Path] | tuple[Path, ...] | None) -> list[Path]:
    if exclude_videos_file is None:
        return []
    if isinstance(exclude_videos_file, Path):
        return [exclude_videos_file]
    return [Path(path) for path in exclude_videos_file]


def resolve_excluded_video_entries(
    root_dir: Path,
    exclude_videos_file: Path | list[Path] | tuple[Path, ...] | None,
) -> set[str]:
    exclude_paths = _normalize_exclude_video_files(exclude_videos_file)
    if not exclude_paths:
        return set()
    entries: set[str] = set()
    for exclude_path in exclude_paths:
        if not exclude_path.exists():
            raise FileNotFoundError(f"exclude_videos_file not found: {exclude_path}")
        path_entries = load_path_entries_file(exclude_path)
        entries.update(path_entries)
        print(f"[INFO] loaded_excluded_videos={len(path_entries)} from {exclude_path}")
    print(f"[INFO] loaded_excluded_videos_total={len(entries)} from {len(exclude_paths)} file(s)")
    return entries


def split_items(items: list, array_index: int | None, num_splits: int) -> list:
    if array_index is None:
        return items
    if array_index < 0 or array_index >= num_splits:
        raise ValueError(f"array_index out of range: {array_index} for num_splits={num_splits}")
    chunk = (len(items) + num_splits - 1) // num_splits
    start = array_index * chunk
    end = min(start + chunk, len(items))
    return items[start:end]


def read_prompt_file(input_prompt_file: Path) -> str:
    if not input_prompt_file.exists():
        raise FileNotFoundError(f"input_prompt_file not found: {input_prompt_file}")
    prompt = input_prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"input_prompt_file is empty: {input_prompt_file}")
    return prompt


def sample_frames(video_path: Path, num_sampled_frames: int, sample_method: str) -> np.ndarray:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    if sample_method == "mid":
        mid = n // 2
        indices = [mid]
    elif sample_method == "image":
        indices = [0]
    else:
        if num_sampled_frames <= 1:
            indices = [0]
        else:
            indices = np.linspace(0, n - 1, num_sampled_frames).round().astype(int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    return frames


def normalize_frames_for_processor(frames: np.ndarray) -> list[Image.Image]:
    """
    Normalize video frames for multimodal processors.
    Ensures each frame is HWC RGB-like and converted to PIL.Image (RGB).
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray for frames, got {type(frames)}")
    if frames.ndim < 3:
        raise ValueError(f"Invalid frame tensor rank: {frames.ndim}, shape={getattr(frames, 'shape', None)}")

    normalized: list[np.ndarray] = []
    for idx in range(frames.shape[0]):
        frame = frames[idx]

        # Handle CHW input if it appears.
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))

        # Grayscale -> RGB
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        # Drop alpha channel if present.
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]

        if frame.ndim != 3 or frame.shape[-1] != 3:
            continue

        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.size > 0 and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif frame.size > 0 and frame.max() <= 1:
            # Some pipelines accidentally keep uint8 in {0,1}; expand to 0..255.
            frame = (frame.astype(np.uint16) * 255).astype(np.uint8)

        frame = np.ascontiguousarray(frame)
        normalized.append(Image.fromarray(frame, mode="RGB"))

    if not normalized:
        raise RuntimeError(
            "No valid frames after normalization "
            f"(input shape={frames.shape}, dtype={frames.dtype})"
        )
    return normalized


def build_chat_prompt(tokenizer, base_prompt: str, num_frames: int) -> str:
    placeholders = "".join(f"Frame{i}: <image>\n" for i in range(1, num_frames + 1))
    messages = [{"role": "user", "content": f"{placeholders}{base_prompt}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_third_context_for_ego(action_dir: Path, clip_stem: str, third_prompt_dirname: str):
    prompt_dir = action_dir / third_prompt_dirname

    # Prefer JSON schema: {"prompt": "...", "action_hints": [...]}
    context_json_path = prompt_dir / f"{clip_stem}.json"
    if context_json_path.exists():
        try:
            data = json.loads(context_json_path.read_text(encoding="utf-8"))
        except Exception:
            return None, []
        prompt = data.get("prompt", None)
        hints = data.get("action_hints", [])
        if not isinstance(hints, list):
            hints = []
        hints = [str(x).strip() for x in hints if str(x).strip()]
        return prompt, hints

    # Fallback for datasets where prompts_aux is plain text (e.g., taste_rob).
    context_txt_path = prompt_dir / f"{clip_stem}.txt"
    if context_txt_path.exists():
        try:
            hint_text = context_txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            return None, []
        if hint_text:
            return None, [hint_text]
    return None, []


def build_prompt_with_context(
    tokenizer,
    base_prompt: str,
    num_frames: int,
    third_prompt: str | None,
    action_hints: list[str],
) -> str:
    prompt_text = base_prompt.strip()
    chunks = []
    if third_prompt:
        chunks.append(third_prompt.strip())
    if action_hints:
        chunks.append(" ; ".join(action_hints))
    if chunks:
        context_rules = [
            "- Use the references only as hints for visible actions, object relations, and state changes.",
            "- Follow the viewpoint and style requested in the main instructions.",
            "- Do not mention references.",
        ]
        prompt_text += (
            "\n\nReference (use for action accuracy):\n"
            + "\n".join(context_rules)
            + "\n\n"
            + "\n".join(chunks).strip()
        )
    return build_chat_prompt(tokenizer, prompt_text, num_frames=num_frames)


def load_droid_language_annotations(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"droid_language_annotations_file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"droid_language_annotations_file must contain a JSON object: {path}")

    result: dict[str, list[str]] = {}
    for episode_id, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        ordered_instructions: list[str] = []
        seen_norm: set[str] = set()
        for idx in (1, 2, 3):
            value = entry.get(f"language_instruction{idx}", None)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if not value:
                continue
            norm = normalize_text_for_dedup(value)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            ordered_instructions.append(value)
        if ordered_instructions:
            result[str(episode_id)] = ordered_instructions
    print(f"[INFO] loaded_droid_language_annotations={len(result)} from {path}")
    return result


def join_instruction_sentences(instructions: list[str]) -> str | None:
    parts: list[str] = []
    for instruction in instructions:
        text = instruction.strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text = f"{text}."
        parts.append(text)
    if not parts:
        return None
    return " ".join(parts)


def iter_action_dirs(root_dir: Path, scene_filter: set[str] | None):
    for scene_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if scene_filter is not None and scene_dir.name not in scene_filter:
            continue
        for action_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            yield scene_dir.name, action_dir.name, action_dir


def gather_jobs_trumans(root_dir: Path, video_folder_name: str, scene_filter: set[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for scene, action, action_dir in iter_action_dirs(root_dir, scene_filter):
        video_dir = action_dir / video_folder_name
        if not video_dir.exists():
            continue
        for video_path in sorted(video_dir.glob("*.mp4"), key=natural_sort_key):
            jobs.append(Job(scene=scene, action=action, video_path=video_path))
    return jobs


def gather_jobs_taste_rob(root_dir: Path, video_folder_name: str, scene_filter: set[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for split_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            if scene_filter is not None and scene_dir.name not in scene_filter:
                continue
            video_dir = scene_dir / video_folder_name
            if not video_dir.exists():
                continue
            for video_path in sorted(video_dir.glob("*.mp4"), key=natural_sort_key):
                jobs.append(Job(scene=split_dir.name, action=scene_dir.name, video_path=video_path))
    return jobs


def gather_jobs_droid(root_dir: Path, video_folder_name: str, scene_filter: set[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for lab_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if scene_filter is not None and lab_dir.name not in scene_filter:
            continue
        for hash_dir in sorted([p for p in lab_dir.iterdir() if p.is_dir()]):
            video_dir = hash_dir / video_folder_name
            if not video_dir.exists():
                continue
            for video_path in sorted(video_dir.glob("*.mp4"), key=natural_sort_key):
                jobs.append(Job(scene=lab_dir.name, action=hash_dir.name, video_path=video_path))
    return jobs


def output_path_for_video(
    video_path: Path,
    output_folder_name: str,
    save_format: str,
) -> Path:
    action_dir = video_path.parent.parent
    output_dir = action_dir / output_folder_name
    ext = ".json" if save_format == "json" else ".txt"
    return output_dir / f"{video_path.stem}{ext}"


def relative_video_path_from_root(video_path: Path, root_dir: Path) -> str:
    return video_path.relative_to(root_dir).as_posix()


def candidate_video_keys(video_path: Path, root_dir: Path) -> set[str]:
    keys = {str(video_path)}
    try:
        keys.add(video_path.relative_to(root_dir).as_posix())
    except ValueError:
        pass
    try:
        data_root = root_dir.parent.parent
        keys.add(video_path.relative_to(data_root).as_posix())
    except ValueError:
        pass
    return keys


def job_is_excluded(job: Job, root_dir: Path, excluded_entries: set[str]) -> bool:
    if not excluded_entries:
        return False
    return bool(candidate_video_keys(job.video_path, root_dir) & excluded_entries)


def has_valid_existing_output(save_path: Path) -> bool:
    if not save_path.exists():
        return False
    try:
        if save_path.stat().st_size == 0:
            print(f"[WARN] Removing empty output before retry: {save_path}")
            save_path.unlink()
            return False
    except OSError as exc:
        print(f"[WARN] Failed to inspect existing output {save_path}: {exc}")
        return False
    return True


def atomic_write_text(save_path: Path, content: str, *, require_nonempty: bool = True) -> None:
    if require_nonempty and not content.strip():
        raise RuntimeError(f"Refusing to save empty text output: {save_path}")

    tmp_path = save_path.with_name(f".{save_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, save_path)
        dir_fd = os.open(str(save_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        if save_path.stat().st_size == 0:
            save_path.unlink(missing_ok=True)
            raise RuntimeError(f"zero-byte output after save: {save_path}")
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def append_line_atomic(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (line.rstrip("\n") + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def build_runnable_jobs(
    all_jobs: list[Job],
    root_dir: Path,
    output_folder_name: str,
    save_format: str,
    skip_existing: bool,
    excluded_entries: set[str],
) -> tuple[list[Job], int, int]:
    pre_skipped_existing = 0
    excluded_bad = 0
    runnable_jobs: list[Job] = []
    for job in all_jobs:
        if job_is_excluded(job, root_dir, excluded_entries):
            excluded_bad += 1
            continue
        save_path = output_path_for_video(job.video_path, output_folder_name, save_format)
        if skip_existing and has_valid_existing_output(save_path):
            pre_skipped_existing += 1
            continue
        runnable_jobs.append(job)
    return runnable_jobs, pre_skipped_existing, excluded_bad


def load_manifest_jobs(manifest_path: Path, all_jobs: list[Job], root_dir: Path) -> tuple[list[Job], int, int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relpaths = payload.get("runnable_jobs", [])
    job_map = {
        relative_video_path_from_root(job.video_path, root_dir): job
        for job in all_jobs
    }
    jobs: list[Job] = []
    missing_relpaths: list[str] = []
    for relpath in relpaths:
        job = job_map.get(relpath)
        if job is None:
            missing_relpaths.append(relpath)
            continue
        jobs.append(job)
    if missing_relpaths:
        print(
            f"[WARN] manifest_missing_jobs={len(missing_relpaths)} "
            f"first_missing={missing_relpaths[:5]}"
        )
    return (
        jobs,
        int(payload.get("pre_skipped_existing", 0)),
        int(payload.get("excluded_bad", 0)),
    )


def build_or_load_coordinated_manifest(
    manifest_path: Path,
    all_jobs: list[Job],
    root_dir: Path,
    output_folder_name: str,
    save_format: str,
    skip_existing: bool,
    excluded_entries: set[str],
    timeout_sec: int,
) -> tuple[list[Job], int, int]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    waited_sec = 0

    while True:
        if manifest_path.exists():
            print(f"[INFO] using_existing_manifest={manifest_path}")
            return load_manifest_jobs(manifest_path, all_jobs, root_dir)
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if waited_sec >= timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for coordinated manifest {manifest_path} "
                    f"after {timeout_sec} seconds"
                )
            time.sleep(1.0)
            waited_sec += 1

    try:
        runnable_jobs, pre_skipped_existing, excluded_bad = build_runnable_jobs(
            all_jobs=all_jobs,
            root_dir=root_dir,
            output_folder_name=output_folder_name,
            save_format=save_format,
            skip_existing=skip_existing,
            excluded_entries=excluded_entries,
        )
        payload = {
            "root_dir": str(root_dir),
            "jobs_total": len(all_jobs),
            "pre_skipped_existing": pre_skipped_existing,
            "excluded_bad": excluded_bad,
            "runnable_jobs": [
                relative_video_path_from_root(job.video_path, root_dir)
                for job in runnable_jobs
            ],
        }
        atomic_write_text(
            manifest_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            require_nonempty=False,
        )
        print(
            f"[INFO] created_manifest={manifest_path} jobs_total={len(all_jobs)} "
            f"excluded_bad={excluded_bad} pre_skipped_existing={pre_skipped_existing} "
            f"jobs_runnable={len(runnable_jobs)}"
        )
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass

    return load_manifest_jobs(manifest_path, all_jobs, root_dir)


def droid_base_clip_stem(video_stem: str) -> str:
    prefix, sep, suffix = video_stem.rpartition("__ext")
    if sep and suffix.isdigit():
        return prefix
    return video_stem


def load_droid_context_for_ego(
    video_path: Path,
    droid_language_annotations: dict[str, list[str]] | None = None,
) -> tuple[str | None, list[str]]:
    clip_stem = droid_base_clip_stem(video_path.stem)
    meta_path = video_path.parent.parent / "episodes" / clip_stem / "episode_meta.json"
    if not meta_path.exists():
        print(f"[WARN] Missing DROID episode metadata for {video_path}: {meta_path}")
        return None, []

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read DROID metadata for {video_path}: {exc}")
        return None, []

    episode_id = data.get("episode_id", None)
    if isinstance(episode_id, str) and episode_id.strip() and droid_language_annotations:
        joined_instruction = join_instruction_sentences(
            droid_language_annotations.get(episode_id.strip(), [])
        )
        if joined_instruction:
            return joined_instruction, []

    task = data.get("task", None)
    if isinstance(task, str) and task.strip():
        return task.strip(), []

    print(f"[WARN] Empty DROID task for {video_path}: {meta_path}")
    return None, []


def extract_json_dict(text: str) -> dict:
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            json_slice = stripped[start : end + 1]
            if json_slice != stripped:
                candidates.append(json_slice)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload, idx = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if candidate[idx:].strip():
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError("Model output does not contain a valid JSON object")


def parse_structured_environment_action_output(text: str) -> dict[str, str]:
    payload = extract_json_dict(text)
    environment = payload.get("environment", None)
    action = payload.get("action", None)
    if not isinstance(environment, str) or not environment.strip():
        raise ValueError("Structured JSON output is missing non-empty 'environment'")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("Structured JSON output is missing non-empty 'action'")
    return {
        "environment": environment.strip(),
        "action": action.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored InternVL2 recaptioning for supported datasets.")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--video_type", type=str, choices=["third_person", "egocentric"], required=True)
    parser.add_argument("--video_folder_name", type=str, default=None)
    parser.add_argument("--output_folder_name", type=str, default=None)
    parser.add_argument("--input_prompt_file", type=Path, required=True)
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL2-40B-AWQ")
    parser.add_argument("--num_sampled_frames", type=int, default=16)
    parser.add_argument("--frame_sample_method", type=str, choices=["uniform", "mid", "image"], default="uniform")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--array_index", type=int, default=None)
    parser.add_argument("--num_splits", type=int, default=8)
    parser.add_argument("--split_by", type=str, choices=["action", "video"], default="action")
    parser.add_argument("--scene_filter_file", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--exclude_videos_file", type=Path, nargs="+", default=None)
    parser.add_argument("--droid_language_annotations_file", type=Path, default=None)
    parser.add_argument("--coordinated_jobs_manifest", type=Path, default=None)
    parser.add_argument("--coordinated_jobs_manifest_timeout_sec", type=int, default=1800)
    parser.add_argument("--structured_parse_retry_count", type=int, default=3)
    parser.add_argument("--structured_parse_failed_videos_file", type=Path, default=None)
    parser.add_argument("--dataset_type", type=str, choices=["trumans", "taste_rob", "droid"], default="trumans")
    parser.add_argument("--save_format", type=str, choices=["auto", "json", "txt"], default="auto")
    parser.add_argument("--json_output_mode", type=str, choices=["prompt", "environment_action"], default="prompt")
    parser.add_argument("--use_third_person_context", action="store_true")
    parser.add_argument("--third_prompt_dirname", type=str, default="prompts_aux")
    return parser.parse_args()


def ensure_vllm_transformers_compat() -> None:
    """Patch missing Transformers symbols expected by newer vLLM builds."""
    import sys
    import types

    import transformers
    import transformers.configuration_utils as configuration_utils
    from transformers.processing_utils import ProcessorMixin

    if hasattr(configuration_utils, "ALLOWED_LAYER_TYPES"):
        pass
    else:
        # vLLM expects this constant from newer Transformers releases when checking
        # whether rope_parameters is keyed by layer type. A conservative superset is
        # sufficient for our captioning workloads.
        configuration_utils.ALLOWED_LAYER_TYPES = {
            "",
            "attention",
            "chunked_attention",
            "full_attention",
            "linear_attention",
            "local_attention",
            "mamba",
            "sliding_attention",
        }

    # Newer vLLM also imports the video processor APIs introduced in newer
    # Transformers releases. The InternVL2 caption path only relies on the
    # generic processor interface, so aliasing these names is sufficient.
    if not hasattr(transformers, "AutoVideoProcessor"):
        auto_video_processor = transformers.AutoProcessor
        transformers.__dict__["AutoVideoProcessor"] = auto_video_processor
        sys.modules["transformers"].__dict__["AutoVideoProcessor"] = auto_video_processor

    if "transformers.video_processing_utils" not in sys.modules:
        video_processing_utils = types.ModuleType("transformers.video_processing_utils")
        video_processing_utils.BaseVideoProcessor = ProcessorMixin
        sys.modules["transformers.video_processing_utils"] = video_processing_utils

    if "DeepseekV2Config" not in transformers.__dict__ and hasattr(transformers, "DeepseekV3Config"):
        transformers.__dict__["DeepseekV2Config"] = transformers.DeepseekV3Config
        sys.modules["transformers"].__dict__["DeepseekV2Config"] = transformers.DeepseekV3Config


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    ensure_vllm_transformers_compat()
    from vllm import LLM, SamplingParams

    if not args.root_dir.exists():
        raise SystemExit(f"root_dir not found: {args.root_dir}")

    if args.video_folder_name is None:
        args.video_folder_name = "videos_third" if args.video_type == "third_person" else "videos"
    if args.output_folder_name is None:
        args.output_folder_name = "prompts_aux" if args.video_type == "third_person" else "prompts"
    if args.save_format == "auto":
        args.save_format = "json" if args.video_type == "third_person" else "txt"
    if args.json_output_mode != "prompt" and args.save_format != "json":
        raise SystemExit("json_output_mode requires --save_format json")

    base_prompt = read_prompt_file(args.input_prompt_file)
    scene_filter = load_scene_filter(args.scene_filter_file)
    excluded_entries = resolve_excluded_video_entries(args.root_dir, args.exclude_videos_file)
    droid_language_annotations = (
        load_droid_language_annotations(args.droid_language_annotations_file)
        if args.dataset_type == "droid" and args.use_third_person_context
        else {}
    )
    if args.dataset_type == "trumans":
        all_jobs = gather_jobs_trumans(args.root_dir, args.video_folder_name, scene_filter)
    elif args.dataset_type == "taste_rob":
        all_jobs = gather_jobs_taste_rob(args.root_dir, args.video_folder_name, scene_filter)
    else:
        all_jobs = gather_jobs_droid(args.root_dir, args.video_folder_name, scene_filter)

    stats = Stats(jobs_total=len(all_jobs))
    if not all_jobs:
        if args.dataset_type == "droid":
            raise SystemExit(f"No videos found under {args.root_dir}/*/*/{args.video_folder_name}")
        raise SystemExit(f"No videos found under {args.root_dir}/*/*/{args.video_folder_name}")

    if args.coordinated_jobs_manifest is not None:
        runnable_jobs, pre_skipped_existing, excluded_bad = build_or_load_coordinated_manifest(
            manifest_path=args.coordinated_jobs_manifest,
            all_jobs=all_jobs,
            root_dir=args.root_dir,
            output_folder_name=args.output_folder_name,
            save_format=args.save_format,
            skip_existing=args.skip_existing,
            excluded_entries=excluded_entries,
            timeout_sec=args.coordinated_jobs_manifest_timeout_sec,
        )
    else:
        runnable_jobs, pre_skipped_existing, excluded_bad = build_runnable_jobs(
            all_jobs=all_jobs,
            root_dir=args.root_dir,
            output_folder_name=args.output_folder_name,
            save_format=args.save_format,
            skip_existing=args.skip_existing,
            excluded_entries=excluded_entries,
        )

    if args.split_by == "action":
        grouped: dict[tuple[str, str], list[Job]] = {}
        for j in runnable_jobs:
            grouped.setdefault((j.scene, j.action), []).append(j)
        action_keys = sorted(grouped.keys())
        selected_keys = split_items(action_keys, args.array_index, args.num_splits)
        jobs = []
        for key in selected_keys:
            jobs.extend(grouped[key])
    else:
        jobs = split_items(runnable_jobs, args.array_index, args.num_splits)

    stats.jobs_selected = len(jobs)
    print(
        f"[INFO] jobs_total={stats.jobs_total}, pre_skipped_existing={pre_skipped_existing}, "
        f"excluded_bad={excluded_bad}, jobs_runnable={len(runnable_jobs)}, jobs_selected={stats.jobs_selected}"
    )

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
    quantization = "awq" if "awq" in args.model_path.lower() else None
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": max(1, args.num_sampled_frames)},
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp,
        quantization=quantization,
        dtype="float16",
        mm_processor_kwargs={"max_dynamic_patch": 1},
    )

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = []
    for token in stop_tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None:
            stop_token_ids.append(tid)
    def build_sampling_params(seed: int | None):
        return SamplingParams(
            temperature=0.1,
            max_tokens=512,
            stop_token_ids=stop_token_ids,
            repetition_penalty=1.1,
            skip_special_tokens=True,
            seed=seed,
        )

    structured_failed_videos_recorded: set[str] = set()

    for job in jobs:
        stats.clips_seen += 1
        save_path = output_path_for_video(job.video_path, args.output_folder_name, args.save_format)
        structured_parse_exhausted = False

        try:
            raw_frames = sample_frames(
                video_path=job.video_path,
                num_sampled_frames=args.num_sampled_frames,
                sample_method=args.frame_sample_method,
            )
            frames = normalize_frames_for_processor(raw_frames)
            if args.video_type == "egocentric" and args.use_third_person_context:
                if args.dataset_type == "droid":
                    third_prompt, action_hints = load_droid_context_for_ego(
                        job.video_path,
                        droid_language_annotations=droid_language_annotations,
                    )
                else:
                    third_prompt, action_hints = load_third_context_for_ego(
                        action_dir=job.video_path.parent.parent,
                        clip_stem=job.video_path.stem,
                        third_prompt_dirname=args.third_prompt_dirname,
                    )
                if third_prompt or action_hints:
                    print(f"[INFO] Loaded third context for {job.video_path.stem}")
                prompt = build_prompt_with_context(
                    tokenizer=tokenizer,
                    base_prompt=base_prompt,
                    num_frames=len(frames),
                    third_prompt=third_prompt,
                    action_hints=action_hints,
                )
            else:
                prompt = build_chat_prompt(
                    tokenizer=tokenizer,
                    base_prompt=base_prompt,
                    num_frames=len(frames),
                )

            payload = None
            model_output = ""
            requires_structured_parse = args.save_format == "json" and args.json_output_mode == "environment_action"
            max_attempts = max(1, args.structured_parse_retry_count) if requires_structured_parse else 1
            last_parse_error: Exception | None = None
            for attempt_idx in range(max_attempts):
                sampling_seed = uuid.uuid4().int % (2**31 - 1)
                outputs = llm.generate(
                    [
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": frames},
                        }
                    ],
                    sampling_params=build_sampling_params(sampling_seed),
                )
                model_output = outputs[0].outputs[0].text.strip()
                if not model_output:
                    raise RuntimeError(f"Empty caption generated for {job.video_path}")

                if requires_structured_parse:
                    try:
                        payload = parse_structured_environment_action_output(model_output)
                        if attempt_idx > 0:
                            print(
                                f"[INFO] Structured parse recovered for {job.video_path} "
                                f"on attempt {attempt_idx + 1}/{max_attempts} seed={sampling_seed}"
                            )
                        break
                    except Exception as parse_exc:
                        last_parse_error = parse_exc
                        if attempt_idx + 1 < max_attempts:
                            print(
                                f"[WARN] Structured parse failed for {job.video_path} "
                                f"attempt {attempt_idx + 1}/{max_attempts} seed={sampling_seed}: {parse_exc}"
                            )
                            continue
                        structured_parse_exhausted = True
                        raise last_parse_error
                else:
                    break

            save_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_format == "json":
                if args.json_output_mode == "environment_action":
                    if payload is None:
                        raise RuntimeError(f"Structured payload missing after generation for {job.video_path}")
                else:
                    payload = {"prompt": model_output}
                    if args.video_type == "third_person":
                        payload["action_hints"] = []
                atomic_write_text(
                    save_path,
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                )
            else:
                atomic_write_text(save_path, model_output + "\n")
            stats.clips_generated += 1
            print(f"[OK] {job.video_path} -> {save_path}")
        except Exception as e:
            stats.clips_failed += 1
            if structured_parse_exhausted and args.structured_parse_failed_videos_file is not None:
                rel_key = relative_video_path_from_root(job.video_path, args.root_dir)
                if rel_key not in structured_failed_videos_recorded:
                    append_line_atomic(args.structured_parse_failed_videos_file, rel_key)
                    structured_failed_videos_recorded.add(rel_key)
                    print(
                        f"[INFO] recorded_structured_parse_failure={rel_key} "
                        f"-> {args.structured_parse_failed_videos_file}"
                    )
            print(f"[FAIL] {job.video_path}: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print("\n=== Summary ===")
    print(f"jobs_total={stats.jobs_total}")
    print(f"jobs_selected={stats.jobs_selected}")
    print(f"clips_seen={stats.clips_seen}")
    print(f"clips_generated={stats.clips_generated}")
    print(f"clips_skipped_existing={pre_skipped_existing + stats.clips_skipped_existing}")
    print(f"clips_failed={stats.clips_failed}")


if __name__ == "__main__":
    main()
