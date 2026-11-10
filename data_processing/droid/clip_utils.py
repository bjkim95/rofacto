"""
Shared helpers for splitting render_v2 outputs into 81-frame clips.

Used by:
- data_processing/droid/make_render_v2_clips81.py
- data_processing/droid/split_inverse_depth_to_clip81.py
- (also referenced by data_processing/video_caption/qwen3vl_clip_caption.py)

Naming convention: every clip output uses the stem
    <source_stem>__c<clip_idx:02d>_<orig_start:04d>-<orig_end:04d>
so the same stem can be reused across mp4 / npz / json variants and the
training dataset can derive condition paths via simple folder swapping.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

CLIP_STEM_RE = re.compile(r"^(?P<base>.+)__c(?P<idx>\d{2})_(?P<a>\d{4})-(?P<b>\d{4})$")


def list_clip_ranges(
    n_orig: int,
    clip_len: int,
    clip_stride: int | None = None,
) -> list[tuple[int, int]]:
    """Clip starts with tail alignment.

    Defaults to the historical non-overlap stride=clip_len behavior. Passing
    clip_stride enables overlapping clips while preserving the tail-aligned
    final clip convention.

    For n_orig < clip_len, returns an empty list (clip cannot be formed).
    For n_orig == clip_len, returns a single range covering the whole video.
    """
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


def clip_stem(source_stem: str, clip_idx: int, a: int, b: int) -> str:
    return f"{source_stem}__c{clip_idx:02d}_{a:04d}-{b:04d}"


def parse_clip_stem(stem: str) -> tuple[str, int, int, int] | None:
    """Inverse of clip_stem(): returns (source_stem, idx, a, b) or None."""
    m = CLIP_STEM_RE.match(stem)
    if not m:
        return None
    return m.group("base"), int(m.group("idx")), int(m.group("a")), int(m.group("b"))


def cut_clip_ffmpeg(
    src_video: Path,
    out_path: Path,
    a: int,
    b: int,
    fps: float,
    crf: int = 18,
    preset: str = "veryfast",
) -> None:
    """Frame-accurate cut [a, b] inclusive via ffmpeg select filter (re-encodes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expr = f"between(n\\,{a}\\,{b})"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_video),
        "-vf", f"select='{expr}',setpts=N/FRAME_RATE/TB",
        "-an",
        "-r", f"{fps}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
