# Copyright 2024 The HuggingFace Team.
# Modified for WAN-based training with VideoX-Fun style.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
WAN-based training script for DWM (Dexterous World Model).

Supports:
- LoRA training (VideoX-Fun lora_utils style)
- SFT training (full or partial fine-tuning)
- Flow Matching loss
- WanTransformer3DModelWithConcat for conditional generation
"""

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import time
import pickle
import random
import re
import shutil
import sys
import warnings
from contextlib import nullcontext
from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from safetensors.torch import load_file, save_file

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DummyScheduler,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, T5Tokenizer, UMT5EncoderModel
from transformers.utils import ContextManagers
import imageio.v3 as iio
import cv2

# WAN models and utilities from training.wan
from training.wan.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
)
from training.wan.models.wan_transformer3d_with_conditions import (
    WanTransformer3DModelWithConcat,
)
from training.wan.models.wan_transformer3d_adaln import (
    WanTransformer3DModelWithStateAdaLN,
)
from training.wan.models.wan2_2_fun_control_transformer3d_with_conditions import (
    Wan2_2FunControlTransformer3DModelWithConcat,
)
from training.wan.models.wan_fun_control_transformer3d_with_conditions import (
    WanFunControlTransformer3DModelWithConcat,
)
from training.wan.models.wan_transformer3d_multiview import (
    MultiViewWanTransformer3DModelWithConcat,
)
from training.wan.models.wan_transformer3d_vace import (
    WanTransformer3DVace,
)
from training.wan.pipeline.pipeline_wan_fun_inpaint import (
    WanFunInpaintPipeline,
)
from training.wan.pipeline.pipeline_wan_fun_inpaint_hand_concat import (
    WanFunInpaintHandConcatPipeline,
)
from training.wan.robot_state_conditioning import (
    align_state_adaln_to_latent_frames,
    load_robot_state_condition,
)
from training.wan.pipeline.pipeline_wan2_2_fun_inpaint_hand_concat import (
    Wan2_2FunInpaintHandConcatPipeline,
)
from training.wan.pipeline.pipeline_wan_fun_inpaint_hand_vace import (
    WanFunInpaintHandVacePipeline,
)
from training.wan.utils.lora_utils import (
    create_network,
    merge_lora,
    unmerge_lora,
)
from training.wan.utils.lightx2v_compat import (
    load_lightx2v_with_fallback,
)
from training.wan.utils.checkpoint_utils import (
    extract_checkpoint_step,
    list_checkpoint_dirs,
    save_training_checkpoint,
)
from training.wan.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)
from training.wan.utils.discrete_sampler import DiscreteSampling
from training.wan.config_loader import load_experiment_config

# Dataset
from training.wan.dataset import (
    BucketSampler,
    MultiViewCompositeVideoDataset,
    MultiViewPropeDataset,
    MultiViewWidthConcatVideoDataset,
    VideoDatasetWithConditionsAndResizing,
    MULTIVIEW_EXT_PATTERN,
    MULTIVIEW_SUFFIX_PATTERN,
    _COMPOSITION_CONDITION_DEFAULT_FILL,
    _FILL_NAME_TO_UINT8,
    _validate_composition_layout,
)

# Wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = get_logger(__name__)

FUN_CONTROL_TRANSFORMER_TYPES = {
    "WanFunControlTransformer3DModelWithConcat",
    "Wan2_2FunControlTransformer3DModelWithConcat",
}


def _is_fun_control_transformer_type(transformer_type: str) -> bool:
    return transformer_type in FUN_CONTROL_TRANSFORMER_TYPES


def _require_wan2_2_fun_control_pipeline_class():
    repo_root = Path(__file__).resolve().parents[2]
    videox_fun_root = repo_root / "VideoX-Fun"
    if videox_fun_root.is_dir() and str(videox_fun_root) not in sys.path:
        sys.path.insert(0, str(videox_fun_root))
    try:
        from videox_fun.pipeline.pipeline_wan2_2_fun_control import (
            Wan2_2FunControlPipeline,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Wan2.2 Fun-Control validation requires the local VideoX-Fun package "
            f"at {videox_fun_root}."
        ) from e
    return Wan2_2FunControlPipeline


PROMPT_MODE_NORMAL = "normal"
PROMPT_MODE_EMPTY = "empty"
VALIDATION_PROMPT_MODE_ORIGINAL = "original"
VALIDATION_PROMPT_MODE_BOTH = "both"


def resolve_prompt_mode(training_config: Dict[str, Any]) -> str:
    """Resolve training prompt mode with limited backward compatibility."""
    prompt_mode = str(training_config.get("prompt_mode", "")).strip().lower()
    if prompt_mode:
        if prompt_mode not in {PROMPT_MODE_NORMAL, PROMPT_MODE_EMPTY}:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        return prompt_mode

    if "prompt_dropout_prob" in training_config:
        legacy_prob = float(training_config["prompt_dropout_prob"])
        if legacy_prob == 0.0:
            warnings.warn(
                "training.prompt_dropout_prob is deprecated. Use training.prompt_mode=normal instead.",
                stacklevel=2,
            )
            return PROMPT_MODE_NORMAL
        if legacy_prob == 1.0:
            warnings.warn(
                "training.prompt_dropout_prob is deprecated. Use training.prompt_mode=empty instead.",
                stacklevel=2,
            )
            return PROMPT_MODE_EMPTY
        raise ValueError(
            "training.prompt_dropout_prob no longer supports fractional values. "
            "Use training.prompt_mode with one of: normal, empty."
        )

    return PROMPT_MODE_NORMAL


def resolve_text_dropout_prob(training_config: Dict[str, Any]) -> float:
    """Return per-batch text dropout probability (0.0 = disabled)."""
    prob = float(training_config.get("text_dropout_prob", 0.0))
    if not (0.0 <= prob < 1.0):
        raise ValueError(f"text_dropout_prob must be in [0, 1), got {prob}")
    return prob


def resolve_validation_prompt_mode(training_config: Dict[str, Any]) -> str:
    """Resolve validation prompt mode."""
    validation_prompt_mode = str(
        training_config.get("validation_prompt_mode", VALIDATION_PROMPT_MODE_ORIGINAL)
    ).strip().lower()
    valid_modes = {
        VALIDATION_PROMPT_MODE_ORIGINAL,
        PROMPT_MODE_EMPTY,
        VALIDATION_PROMPT_MODE_BOTH,
    }
    if validation_prompt_mode not in valid_modes:
        raise ValueError(f"Unsupported validation_prompt_mode: {validation_prompt_mode}")
    return validation_prompt_mode


def validate_i2v_condition_latent_type(
    pipeline_type: str,
    conditioning_mode: str,
    i2v_condition_latent_type: str,
    use_diffusers_i2v_backend: bool = False,
) -> None:
    """Sanity-check I2V conditioning setups.

    Two valid `i2v_condition_latent_type` values exist:
      - "image"   : first-frame latent + zero-filled tail. This is what the
                    Wan2.1 official I2V checkpoint was trained against, and
                    is the default for `WanTransformer3DModelWithConcat` with
                    `model_type='i2v'` (self-impl loader, regardless of which
                    pipeline class is used to host it at validation time).
      - "fun_inp" : VAE-encoded masked video. This is what the Fun-InP
                    checkpoint family was trained against.

    The previous version of this check raised whenever pipeline.type
    contained "fun", which was correct for the legacy diffusers-only i2v
    backend but wrong now: the unified self-impl path runs `image` latents
    through `WanFunInpaintHandConcatPipeline`, and that combination is
    intentional. Keep the check informational instead of fatal.
    """
    if conditioning_mode != "i2v":
        return

    normalized_pipeline_type = str(pipeline_type).strip().lower()
    legacy_fun_inp_only = normalized_pipeline_type == "wan2.1_fun_inp"
    if legacy_fun_inp_only and i2v_condition_latent_type == "image":
        raise ValueError(
            "training.i2v_condition_latent_type='image' is unsupported for the bare "
            f"Fun-InP pipeline (pipeline.type='{pipeline_type}'). Use "
            "training.i2v_condition_latent_type='fun_inp' there because the Fun-InP "
            "checkpoint expects masked-video latents."
        )


def get_validation_prompt_variants(prompt_text: str, validation_prompt_mode: str) -> List[tuple[str, str]]:
    """Return validation prompt variants as (label, prompt_text)."""
    if validation_prompt_mode == VALIDATION_PROMPT_MODE_ORIGINAL:
        return [(VALIDATION_PROMPT_MODE_ORIGINAL, prompt_text)]
    if validation_prompt_mode == PROMPT_MODE_EMPTY:
        return [(PROMPT_MODE_EMPTY, "")]
    if validation_prompt_mode == VALIDATION_PROMPT_MODE_BOTH:
        return [
            (VALIDATION_PROMPT_MODE_ORIGINAL, prompt_text),
            (PROMPT_MODE_EMPTY, ""),
        ]
    raise ValueError(f"Unsupported validation_prompt_mode: {validation_prompt_mode}")


def _load_path_entries_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.lstrip().startswith("#")]


def _load_named_validation_sets(validation_entries: Any, data_root: str) -> List[tuple[str, List[str]]]:
    if isinstance(validation_entries, (list, tuple)):
        normalized_entries = [str(entry) for entry in validation_entries if entry]
    else:
        normalized_entries = [str(validation_entries)]

    validation_groups: List[tuple[str, List[str]]] = []
    for entry in normalized_entries:
        validation_set_path = os.path.join(data_root, entry)
        if os.path.exists(validation_set_path):
            validation_groups.append((entry, _load_path_entries_file(Path(validation_set_path))))
        else:
            logger.warning("Validation set file not found: %s", validation_set_path)
    return validation_groups


def _interleave_validation_groups(validation_groups: List[tuple[str, List[str]]]) -> List[str]:
    interleaved: List[str] = []
    max_group_len = max((len(paths) for _, paths in validation_groups), default=0)
    for index in range(max_group_len):
        for _, paths in validation_groups:
            if index < len(paths):
                interleaved.append(paths[index])
    return interleaved


def _normalize_exclude_videos_files(exclude_videos_file: Optional[Any]) -> List[str]:
    if not exclude_videos_file:
        return []
    if isinstance(exclude_videos_file, (str, Path)):
        return [str(exclude_videos_file)]
    try:
        return [str(path) for path in exclude_videos_file if path]
    except TypeError:
        return [str(exclude_videos_file)]


def load_excluded_video_entries(data_root: str, exclude_videos_file: Optional[Any]) -> set[str]:
    exclude_paths_raw = _normalize_exclude_videos_files(exclude_videos_file)
    if not exclude_paths_raw:
        return set()

    entries: set[str] = set()
    for exclude_path_raw in exclude_paths_raw:
        exclude_path = Path(exclude_path_raw)
        if not exclude_path.is_absolute():
            exclude_path = Path(data_root) / exclude_path
        if not exclude_path.exists():
            raise FileNotFoundError(f"Exclude videos file not found: {exclude_path}")

        path_entries = set(_load_path_entries_file(exclude_path))
        entries.update(path_entries)
        logger.info("Loaded %d excluded videos from %s", len(path_entries), exclude_path)

    logger.info("Loaded %d unique excluded videos from %d file(s)", len(entries), len(exclude_paths_raw))
    return entries


def _safe_read_validation_video(path: Path, role: str) -> Optional[np.ndarray]:
    try:
        return iio.imread(str(path)).astype(np.float32) / 255.0
    except Exception as e:
        logger.warning("Failed to decode %s video %s: %s. Skipping.", role, path, e)
        return None


def _safe_read_validation_disparity_npz(
    mp4_path: Path, role: str, mode: str, num_target_frames: Optional[int] = None
) -> Optional[np.ndarray]:
    """Load a disparity NPZ at the same logical location as `mp4_path` (with
    .mp4 → .npz suffix swap), turbo-colormap it, and return (T, H, W, 3)
    float in [0, 1] matching `_safe_read_validation_video`'s convention.

    mode:
      - "first_frame_repeat": render NPZ frame 0 once and tile to num_target_frames.
      - "per_frame": render every NPZ frame in sequence.
    """
    npz_path = mp4_path.with_suffix(".npz")
    if not npz_path.exists():
        logger.warning("%s NPZ not found: %s. Skipping.", role, npz_path)
        return None
    try:
        # Lazy import to avoid pulling repo modules at top-level when validation
        # isn't running.
        import sys as _sys
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parents[2]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from data_processing.droid.depth_colormap import depth_frame_to_turbo_rgb

        with np.load(str(npz_path)) as payload:
            inv = np.asarray(payload["inv_depth_u16"])  # (T, H, W) uint16
            valid = np.asarray(payload["valid_mask"])    # (T, H, W) uint8
            near_m = float(payload["near_m"])
            far_m = float(payload["far_m"])
        t_src = int(inv.shape[0])
        if mode == "first_frame_repeat":
            rgb0 = depth_frame_to_turbo_rgb(inv[0], valid[0], near_m=near_m, far_m=far_m)
            n = max(1, int(num_target_frames or 1))
            stack = np.repeat(rgb0[None, ...], n, axis=0)
        elif mode == "per_frame":
            stack = np.empty((t_src, inv.shape[1], inv.shape[2], 3), dtype=np.uint8)
            for i in range(t_src):
                stack[i] = depth_frame_to_turbo_rgb(inv[i], valid[i], near_m=near_m, far_m=far_m)
        else:
            logger.warning("Unsupported disparity npz mode %r for %s; skipping.", mode, npz_path)
            return None
        return stack.astype(np.float32) / 255.0
    except Exception as e:
        logger.warning("Failed to decode %s NPZ %s: %s. Skipping.", role, npz_path, e)
        return None


def _build_validation_video_save_id(video_path: str | Path) -> str:
    """Build a readable, collision-resistant validation video id from a relative dataset path."""
    path_obj = Path(video_path)
    path_no_suffix = path_obj.with_suffix("")
    parts = list(path_no_suffix.parts)
    if len(parts) >= 2 and parts[-2] in {
        "videos",
        "video",
        "robot_mesh_videos",
        "comparison",
        "rgb",
    }:
        parts = parts[:-2] + [parts[-1]]

    sanitized_parts: List[str] = []
    for part in parts:
        cleaned = re.sub(r"[^A-Za-z0-9._@+-]+", "-", str(part)).strip("-")
        if cleaned:
            sanitized_parts.append(cleaned)

    if not sanitized_parts:
        fallback = re.sub(r"[^A-Za-z0-9._@+-]+", "-", path_no_suffix.name).strip("-")
        save_id = fallback or "validation_video"
    else:
        save_id = "__".join(sanitized_parts)

    # Long RoboCasa task/episode names can exceed the filesystem's per-file
    # component limit once step/prompt/GPU suffixes are added.
    max_id_bytes = 176
    if len(save_id.encode("utf-8")) <= max_id_bytes:
        return save_id

    digest = hashlib.sha1(save_id.encode("utf-8")).hexdigest()[:12]
    suffix = f"__{digest}"
    budget = max_id_bytes - len(suffix.encode("utf-8"))
    truncated = save_id.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip("._-")
    return f"{truncated}{suffix}" if truncated else f"validation_video{suffix}"


def _build_single_validation_samples(validation_paths: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "kind": "single",
            "video_path": str(video_path),
            "save_key": str(video_path),
        }
        for video_path in validation_paths
    ]


def _build_multiview_width_concat_validation_samples(
    validation_paths: List[str],
    expected_view_ids: List[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[int, str]] = {}
    invalid_groups: set[str] = set()
    skipped_missing_suffix = 0
    skipped_unexpected_view = 0
    duplicate_groups = 0
    incomplete_groups = 0

    for raw_path in validation_paths:
        video_path = Path(raw_path)
        match = MULTIVIEW_EXT_PATTERN.match(video_path.stem)
        if match is None:
            skipped_missing_suffix += 1
            continue

        view_id = int(match.group("view"))
        if view_id not in expected_view_ids:
            skipped_unexpected_view += 1
            continue

        base_name = match.group("base")
        rel_parent = video_path.parent.as_posix()
        pair_key = base_name if rel_parent in {"", "."} else f"{rel_parent}/{base_name}"
        views = grouped.setdefault(pair_key, {})
        if view_id in views:
            invalid_groups.add(pair_key)
            duplicate_groups += 1
            continue
        views[view_id] = raw_path

    samples: List[Dict[str, Any]] = []
    for pair_key, views in grouped.items():
        if pair_key in invalid_groups:
            continue
        missing_view_ids = [view_id for view_id in expected_view_ids if view_id not in views]
        if missing_view_ids:
            incomplete_groups += 1
            continue
        ordered_views = {view_id: views[view_id] for view_id in expected_view_ids}
        view_suffix = "-".join(f"ext{view_id}" for view_id in expected_view_ids)
        samples.append(
            {
                "kind": "multiview_width_concat",
                "pair_key": pair_key,
                "views": ordered_views,
                "view_ids": list(expected_view_ids),
                "save_key": f"{pair_key}__{view_suffix}",
            }
        )

    logger.info(
        "Built %d width-concat validation groups from %d flat samples "
        "(missing_suffix=%d, unexpected_view=%d, duplicate_groups=%d, incomplete_groups=%d)",
        len(samples),
        len(validation_paths),
        skipped_missing_suffix,
        skipped_unexpected_view,
        duplicate_groups,
        incomplete_groups,
    )
    return samples


def _build_multiview_composition_validation_samples(
    validation_paths: List[str],
    layout_spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    required_suffixes: set[str] = set()
    optional_suffixes: set[str] = set()
    for cell in layout_spec["cells"]:
        if "suffix" in cell:
            required_suffixes.add(cell["suffix"])
        elif "suffix_pool" in cell:
            optional_suffixes.update(cell["suffix_pool"])
    all_suffixes = required_suffixes | optional_suffixes

    grouped: Dict[str, Dict[str, str]] = {}
    invalid_groups: set[str] = set()
    skipped_unknown_suffix = 0
    duplicate_suffix = 0
    incomplete_required = 0
    missing_pool = 0

    for raw_path in validation_paths:
        video_path = Path(raw_path)
        match = MULTIVIEW_SUFFIX_PATTERN.match(video_path.stem)
        if match is None:
            skipped_unknown_suffix += 1
            continue

        suffix = match.group("suffix")
        if suffix not in all_suffixes:
            skipped_unknown_suffix += 1
            continue

        base_name = match.group("base")
        rel_parent = video_path.parent.as_posix()
        pair_key = base_name if rel_parent in {"", "."} else f"{rel_parent}/{base_name}"
        views = grouped.setdefault(pair_key, {})
        if suffix in views:
            invalid_groups.add(pair_key)
            duplicate_suffix += 1
            continue
        views[suffix] = raw_path

    samples: List[Dict[str, Any]] = []
    for pair_key, views in grouped.items():
        if pair_key in invalid_groups:
            continue
        missing_required = [suffix for suffix in required_suffixes if suffix not in views]
        if missing_required:
            incomplete_required += 1
            continue

        pool_candidates: List[List[str]] = []
        pool_ok = True
        for cell in layout_spec["cells"]:
            if "suffix_pool" not in cell:
                continue
            present = [suffix for suffix in cell["suffix_pool"] if suffix in views]
            if not present:
                pool_ok = False
                break
            pool_candidates.append(present)
        if not pool_ok:
            missing_pool += 1
            continue

        samples.append(
            {
                "kind": "multiview_composition",
                "pair_key": pair_key,
                "views": views,
                "pool_candidates": pool_candidates,
                "save_key": pair_key,
            }
        )

    logger.info(
        "Built %d composition validation groups from %d flat samples "
        "(unknown_suffix=%d, duplicate_suffix=%d, incomplete_required=%d, missing_pool=%d)",
        len(samples),
        len(validation_paths),
        skipped_unknown_suffix,
        duplicate_suffix,
        incomplete_required,
        missing_pool,
    )
    return samples


def _build_validation_samples_for_mode(
    validation_paths: List[str],
    *,
    multiview_composition: bool,
    composition_layout_spec: Optional[Dict[str, Any]],
    multiview_width_concat: bool,
    width_concat_view_ids: Optional[List[int]],
) -> List[Dict[str, Any]]:
    if multiview_composition:
        if composition_layout_spec is None:
            raise ValueError("composition_layout_spec is required for multiview composition validation")
        return _build_multiview_composition_validation_samples(validation_paths, composition_layout_spec)
    if multiview_width_concat:
        if not width_concat_view_ids:
            raise ValueError("width_concat_view_ids is required for width-concat validation")
        return _build_multiview_width_concat_validation_samples(validation_paths, width_concat_view_ids)
    return _build_single_validation_samples(validation_paths)


def _stable_validation_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:16], 16)


def _pick_validation_composition_cell_suffixes(
    sample: Dict[str, Any],
    layout_spec: Dict[str, Any],
    global_step: int,
) -> Dict[int, str]:
    picks: Dict[int, str] = {}
    pool_index = 0
    for cell_idx, cell in enumerate(layout_spec["cells"]):
        if "suffix" in cell:
            picks[cell_idx] = cell["suffix"]
        elif "suffix_pool" in cell:
            present = sample["pool_candidates"][pool_index]
            pool_index += 1
            seed = _stable_validation_seed("validation_pool_pick", global_step, sample["pair_key"], cell_idx)
            picks[cell_idx] = random.Random(seed).choice(present)
    return picks


def _resize_validation_video_exact(
    video_np: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).contiguous()
    resized = F.interpolate(
        tensor,
        size=(int(target_h), int(target_w)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).cpu().numpy()


def _derive_validation_condition_path(
    video_rel_path: str | Path,
    data_root: str | Path,
    condition_folder: Optional[str],
) -> Optional[Path]:
    if not condition_folder:
        return None

    full_video_path = Path(data_root) / str(video_rel_path)
    parent_dir = full_video_path.parent
    condition_path = parent_dir.parent / condition_folder / full_video_path.name
    if not condition_path.exists() and parent_dir.parent.name == "processed2":
        fallback_condition_path = parent_dir.parent.parent / condition_folder / full_video_path.name
        if fallback_condition_path.exists():
            condition_path = fallback_condition_path
    return condition_path


def _resolve_validation_latent_path(condition_path: Path, latent_subdir: Optional[str]) -> Optional[Path]:
    if not latent_subdir:
        return None
    filename = f"{condition_path.stem}.pt"
    action_dir = condition_path.parent.parent
    candidates = [action_dir / latent_subdir / filename]
    if action_dir.name == "processed2":
        candidates.append(action_dir.parent / latent_subdir / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_validation_condition_latents(
    condition_path: Optional[Path],
    latent_subdir: Optional[str],
    role: str,
) -> Optional[torch.Tensor]:
    if condition_path is None or not latent_subdir:
        return None
    latent_path = _resolve_validation_latent_path(condition_path, latent_subdir)
    if latent_path is None or not latent_path.is_file():
        logger.warning("%s latent not found: %s", role, latent_path)
        return None
    latents = torch.load(latent_path, map_location="cpu", weights_only=True)
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5:
        raise ValueError(f"Expected {role} latents [B,C,F,H,W] or [C,F,H,W], got {tuple(latents.shape)}")
    logger.info("Loaded %s latent for validation: %s", role, latent_path)
    return latents


def _build_validation_prompt_path(
    video_rel_path: str | Path,
    data_root: str | Path,
    prompt_subdir: str,
    prompt_ext: str,
) -> Path:
    full_video_path = Path(data_root) / str(video_rel_path)
    return full_video_path.parent.parent / prompt_subdir / f"{full_video_path.stem}{prompt_ext}"


def _load_validation_prompt_text(
    prompt_path: Path,
    prompt_json_fields: Optional[List[str]],
) -> str:
    def extract_json_field(data: Dict[str, Any], field: str) -> str:
        value: Any = data
        for key in field.split("."):
            if not isinstance(value, dict):
                return ""
            value = value.get(key, "")
        if isinstance(value, dict) and "text" in value:
            value = value.get("text", "")
        return str(value).strip()

    if prompt_json_fields:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        parts = [extract_json_field(prompt_data, field) for field in prompt_json_fields]
        return " ".join(part for part in parts if part)

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _resolve_prompt_json_fields(
    prompt_json_fields: Optional[List[str]],
    prompt_subdir: str,
) -> Optional[List[str]]:
    if prompt_json_fields:
        return [str(field) for field in prompt_json_fields]
    if "qwen3vl" in str(prompt_subdir).lower():
        return ["general.text"]
    return None
def build_dataset_init_kwargs(
    data_config: Dict[str, Any],
    *,
    load_tensors: bool,
    conditioning_mode: str,
    use_preencoded_prompt_embeds: bool,
    align_width_to_32: bool,
) -> Dict[str, Any]:
    """Build common dataset constructor arguments from normalized config."""
    static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
    return {
        "data_root": data_config["data_root"],
        "dataset_file": data_config["dataset_file"],
        "max_num_frames": data_config.get("max_num_frames", 49),
        "load_tensors": load_tensors,
        "random_flip": data_config.get("random_flip", False),
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 720),
        "frame_buckets": data_config.get("frame_buckets", 49),
        "clip_start_frame_stride": data_config.get("clip_start_frame_stride", 1),
        "temporal_frame_skip_choices": data_config.get("temporal_frame_skip_choices"),
        "temporal_frame_skip_probs": data_config.get("temporal_frame_skip_probs"),
        "action_density_index": data_config.get("action_density_index"),
        "clip_sampling": data_config.get("clip_sampling", "uniform"),
        "action_sampling_alpha": data_config.get("action_sampling_alpha", 0.5),
        "action_sampling_eps": data_config.get("action_sampling_eps", 1e-2),
        "state_adaln_source": data_config.get("state_adaln_source"),
        "state_adaln_joint_filename": data_config.get("state_adaln_joint_filename", "joint_position_isaac_nominal.npy"),
        "state_adaln_gripper_filename": data_config.get("state_adaln_gripper_filename", "gripper_position_isaac_nominal.npy"),
        "state_adaln_action_sidecar_subdir": data_config.get("state_adaln_action_sidecar_subdir", "action_values_clips81"),
        "state_adaln_union_droid_dim": data_config.get("state_adaln_union_droid_dim", 7),
        "state_adaln_union_robocasa_dim": data_config.get("state_adaln_union_robocasa_dim", 24),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir", "prompt_embeds_fun_rewrite_wan"),
        "use_preencoded_prompt_embeds": use_preencoded_prompt_embeds,
        "prompt_json_fields": _resolve_prompt_json_fields(
            data_config.get("prompt_json_fields"),
            data_config.get("prompt_subdir", "prompts"),
        ),
        "static_video_subdir": data_config.get("static_video_subdir", "videos_static"),
        "static_video_source_mode": static_video_source_mode,
        "robot_mesh_video_subdir": data_config.get("robot_mesh_video_subdir", "videos_hands"),
        "robot_mesh_video_latents_subdir": data_config.get("robot_mesh_video_latents_subdir", "hand_video_latents_wan"),
        "static_disparity_subdir": data_config.get("static_disparity_subdir"),
        "static_disparity_latents_subdir": data_config.get("static_disparity_latents_subdir"),
        "static_disparity_npz_mode": data_config.get("static_disparity_npz_mode"),
        "eef_depth_subdir": data_config.get("eef_depth_subdir"),
        "eef_depth_latents_subdir": data_config.get("eef_depth_latents_subdir"),
        "eef_depth_npz_mode": data_config.get("eef_depth_npz_mode"),
        "video_latents_subdir": data_config.get("video_latents_subdir", "video_latents_wan"),
        "static_video_latents_subdir": data_config.get("static_video_latents_subdir", "static_video_latents_wan"),
        "image_latents_subdir": data_config.get("image_latents_subdir", "image_latents_wan"),
        "fun_inp_i2v_latents_subdir": data_config.get("fun_inp_i2v_latents_subdir", "fun_inp_i2v_latents_wan"),
        "clip_image_embeds_subdir": data_config.get("clip_image_embeds_subdir", "clip_image_embeds"),
        "exclude_videos_file": data_config.get("exclude_videos_file"),
        "validate_condition_paths": data_config.get("validate_condition_paths", True),
        "eager_load_prompts": data_config.get("eager_load_prompts", True),
        "require_static_videos": conditioning_mode == "dwm",
        "align_width_to_32": align_width_to_32,
        "image_to_video": conditioning_mode == "i2v",
    }




def _resolve_validation_prompt_text(
    candidate_video_paths: List[str],
    *,
    data_root: str,
    prompt_subdir: str,
    prompt_ext: str,
    prompt_json_fields: Optional[List[str]],
    validation_missing_prompt_mode: str,
) -> Optional[str]:
    seen_paths: set[Path] = set()
    prompt_candidates: List[Path] = []
    for video_rel_path in candidate_video_paths:
        prompt_path = _build_validation_prompt_path(
            video_rel_path,
            data_root=data_root,
            prompt_subdir=prompt_subdir,
            prompt_ext=prompt_ext,
        )
        if prompt_path in seen_paths:
            continue
        seen_paths.add(prompt_path)
        prompt_candidates.append(prompt_path)

    for prompt_path in prompt_candidates:
        if not prompt_path.exists():
            continue
        try:
            return _load_validation_prompt_text(prompt_path, prompt_json_fields)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse structured prompt %s: %s", prompt_path, exc)
            return ""

    if validation_missing_prompt_mode == "empty":
        if prompt_candidates:
            logger.warning(
                "Prompt not found for validation sample. Using empty prompt. Tried: %s",
                prompt_candidates[0],
            )
        return ""

    if prompt_candidates:
        logger.warning("Prompt not found: %s. Skipping.", prompt_candidates[0])
    else:
        logger.warning("No prompt candidates could be built for validation sample. Skipping.")
    return None


def _compose_validation_fill_tile(
    cell: Dict[str, Any],
    condition_key: str,
    num_frames: int,
) -> np.ndarray:
    fill_map = cell.get("fill", {})
    color_name = fill_map.get(
        condition_key,
        _COMPOSITION_CONDITION_DEFAULT_FILL.get(condition_key, "black"),
    )
    uint8_value = _FILL_NAME_TO_UINT8[str(color_name).lower()]
    _, _, h, w = cell["bbox"]
    fill_value = float(uint8_value) / 255.0
    return np.full((int(num_frames), int(h), int(w), 3), fill_value, dtype=np.float32)


def _compose_validation_multiview_canvas(
    *,
    sample: Dict[str, Any],
    layout_spec: Dict[str, Any],
    cell_suffix_picks: Dict[int, str],
    data_root: str,
    condition_key: str,
    condition_folder: Optional[str],
    role_name: str,
    is_target: bool,
) -> Optional[np.ndarray]:
    cells = layout_spec["cells"]
    source_tiles: Dict[int, np.ndarray] = {}
    temporal_lengths: List[int] = []

    for cell_idx, cell in enumerate(cells):
        if "fill" in cell:
            continue

        suffix = cell_suffix_picks[cell_idx]
        video_rel_path = sample["views"].get(suffix)
        if video_rel_path is None:
            logger.warning(
                "Missing %s source for pair=%s suffix=%s",
                role_name,
                sample["pair_key"],
                suffix,
            )
            return None

        if is_target:
            src_path = Path(data_root) / video_rel_path
        else:
            src_path = _derive_validation_condition_path(video_rel_path, data_root, condition_folder)
            if src_path is None or not src_path.is_file():
                logger.warning(
                    "Missing %s for pair=%s suffix=%s: %s",
                    role_name,
                    sample["pair_key"],
                    suffix,
                    src_path,
                )
                return None

        tile = _safe_read_validation_video(src_path, role_name)
        if tile is None:
            return None
        _, _, h, w = cell["bbox"]
        tile = _resize_validation_video_exact(tile, target_h=h, target_w=w)
        source_tiles[cell_idx] = tile
        temporal_lengths.append(int(tile.shape[0]))

    if not temporal_lengths:
        return None

    common_len = min(temporal_lengths)
    canvas = np.zeros(
        (
            common_len,
            int(layout_spec["canvas_height"]),
            int(layout_spec["canvas_width"]),
            3,
        ),
        dtype=np.float32,
    )
    for cell_idx, cell in enumerate(cells):
        y, x, h, w = cell["bbox"]
        if "fill" in cell:
            tile = _compose_validation_fill_tile(cell, condition_key, common_len)
        else:
            tile = source_tiles[cell_idx][:common_len]
        canvas[:, y:y + h, x:x + w, :] = tile
    return canvas


_VALIDATION_PCD_CACHE = None  # PcdCalibrationCache lazily initialized per process


def _render_wrist_pcd_validation_canvas(
    *,
    sample: Dict[str, Any],
    layout_spec: Dict[str, Any],
    cell_suffix_picks: Dict[int, str],
    data_root: str,
    gt_video: np.ndarray,
    wrist_pcd_config: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Validation analogue of MultiViewCompositeVideoDataset._compose_static_with_wrist_pcd.

    Starts from gt_video[0] repeated across T frames, then renders the wrist cell
    via the pcd renderer (frame 0 = wrist GT at start_idx=0, frames 1+ = ext-pcd
    splat) and pastes it into the canvas. Returns float32 (T, H, W, 3) in [0, 255]
    matching gt_video's dtype/range. Falls back to the copy_first_frame canvas
    on any error so validation never blocks training.
    """
    global _VALIDATION_PCD_CACHE
    from training.wan.utils import wrist_pcd_rendering as pcd_mod

    canvas = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()

    wrist_cells: List[Tuple[int, Dict[str, Any]]] = []
    picked_ext: Optional[str] = None
    for cell_idx, cell in enumerate(layout_spec["cells"]):
        if "fill" in cell:
            continue
        suffix = cell_suffix_picks.get(cell_idx)
        if suffix == "wrist":
            wrist_cells.append((cell_idx, cell))
        elif suffix in {"ext1", "ext2"}:
            picked_ext = suffix
    if not wrist_cells or picked_ext is None:
        return canvas

    wrist_rel = sample["views"].get("wrist")
    if wrist_rel is None:
        return canvas
    wrist_path = Path(data_root) / wrist_rel
    clip_dir = wrist_path.parent.parent
    timestamp = wrist_path.stem.rsplit("__", 1)[0]
    episode_meta_path = clip_dir / "episodes" / timestamp / "episode_meta.json"
    if not episode_meta_path.exists():
        logger.warning(
            "validation pcd: missing episode_meta %s; falling back to copy_first_frame",
            episode_meta_path,
        )
        return canvas

    if _VALIDATION_PCD_CACHE is None:
        _VALIDATION_PCD_CACHE = pcd_mod.PcdCalibrationCache()
    try:
        calib = _VALIDATION_PCD_CACHE.get(episode_meta_path)
    except pcd_mod.WristPcdRenderError as exc:
        logger.warning("validation pcd: calib load failed (%s); falling back", exc)
        return canvas

    cfg = wrist_pcd_config or {}
    num_frames = int(gt_video.shape[0])
    for cell_idx, cell in wrist_cells:
        y, x, h, w = cell["bbox"]
        try:
            wrist_tile = pcd_mod.render_wrist_cell_static(
                calibration=calib,
                clip_dir=clip_dir,
                timestamp=timestamp,
                start_idx=0,
                num_frames=num_frames,
                frame_skip=1,
                cell_hw=(int(h), int(w)),
                preferred_ext_suffix=str(picked_ext),
                point_radius=float(cfg.get("point_radius", 0.012)),
                points_per_pixel=int(cfg.get("points_per_pixel", 8)),
                unproject_stride=int(cfg.get("unproject_stride", 1)),
                ext_max_depth_m=float(cfg.get("ext_max_depth_m", 1.5)),
                ext_depth_subdir=str(cfg.get("ext_depth_subdir", "stereo_inverse_depth")),
                robot_mesh_depth_subdir=str(
                    cfg.get("robot_mesh_depth_subdir", "robot_mesh_depth_videos")
                ),
                videos_subdir=str(cfg.get("videos_subdir", "videos")),
            )
        except pcd_mod.WristPcdRenderError as exc:
            logger.warning(
                "validation pcd: render failed for pair=%s (%s); skipping wrist cell",
                sample.get("pair_key"),
                exc,
            )
            continue
        # wrist_tile: (T, 3, h, w) float32 in [-1, 1] → (T, h, w, 3) float32 in [0, 1].
        # The other cells in `gt_video` come from `_safe_read_validation_video`
        # which returns float32 / 255 → [0, 1], so we MUST match that range here.
        tile = wrist_tile.detach().cpu().numpy()
        tile = np.transpose(tile, (0, 2, 3, 1))
        tile = (tile * 0.5 + 0.5)
        tile = np.clip(tile, 0.0, 1.0).astype(np.float32)
        if tile.shape[0] != canvas.shape[0]:
            logger.warning(
                "validation pcd: tile frame count %d != canvas %d; skipping wrist cell",
                tile.shape[0],
                canvas.shape[0],
            )
            continue
        canvas[:, y:y + h, x:x + w, :] = tile
    return canvas


def _concat_validation_multiview_views(
    *,
    sample: Dict[str, Any],
    view_ids: List[int],
    data_root: str,
    target_height: int,
    per_view_width: int,
    condition_folder: Optional[str],
    role_name: str,
    is_target: bool,
) -> Optional[np.ndarray]:
    view_tiles: List[np.ndarray] = []
    for view_id in view_ids:
        video_rel_path = sample["views"].get(view_id)
        if video_rel_path is None:
            logger.warning(
                "Missing %s source for pair=%s view=%s",
                role_name,
                sample["pair_key"],
                view_id,
            )
            return None

        if is_target:
            src_path = Path(data_root) / video_rel_path
        else:
            src_path = _derive_validation_condition_path(video_rel_path, data_root, condition_folder)
            if src_path is None or not src_path.is_file():
                logger.warning(
                    "Missing %s for pair=%s view=%s: %s",
                    role_name,
                    sample["pair_key"],
                    view_id,
                    src_path,
                )
                return None

        tile = _safe_read_validation_video(src_path, role_name)
        if tile is None:
            return None
        tile = _resize_validation_video_exact(tile, target_h=target_height, target_w=per_view_width)
        view_tiles.append(tile)

    if not view_tiles:
        return None

    common_len = min(int(tile.shape[0]) for tile in view_tiles)
    return np.concatenate([tile[:common_len] for tile in view_tiles], axis=2)


def resize_mask(mask, latent, process_first_frame_only=True):
    """Resize mask to match latent dimensions."""
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    """Get sigmas for flow matching."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _build_prope_condition_video(
    video_bcfhw: torch.Tensor,
    conditioning_mode: str,
) -> torch.Tensor:
    """Build the per-view video that is VAE-encoded into PRoPE inpaint latents."""
    mode = str(conditioning_mode).strip().lower()
    if mode == "dwm":
        return video_bcfhw[:, :, 0:1].repeat(1, 1, video_bcfhw.shape[2], 1, 1)
    if mode == "i2v":
        conditioned = torch.zeros_like(video_bcfhw)
        conditioned[:, :, 0:1, :, :] = video_bcfhw[:, :, 0:1, :, :]
        return conditioned
    raise ValueError(f"Unsupported PRoPE conditioning_mode: {conditioning_mode}")


def _build_prope_mask_latents(
    *,
    batch_size: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    conditioning_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build DWM or I2V inpaint masks in latent cadence for PRoPE."""
    mode = str(conditioning_mode).strip().lower()
    if mode == "dwm":
        return torch.ones(
            batch_size,
            4,
            latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )
    if mode == "i2v":
        mask = torch.zeros(
            batch_size,
            4,
            latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )
        mask[:, :, 0:1, :, :] = 1.0
        return mask
    raise ValueError(f"Unsupported PRoPE conditioning_mode: {conditioning_mode}")


def _prope_training_step(
    *,
    batch,
    transformer3d,
    noise_scheduler,
    weight_dtype,
    accelerator,
    batch_encode_vae,
    idx_sampling,
    torch_rng,
    empty_prompt_embeds,
    tokenizer,
    text_encoder,
    training_config,
    args,
    conditioning_mode,
    prompt_mode,
    text_dropout_prob: float,
    load_tensors: bool,
    uses_fun_control_y: bool,
    is_wan2_2_fun_control: bool,
    i2v_use_hand_condition: bool,
) -> torch.Tensor:
    """One training step for the per-view token-concat multi-view path.

    This helper is still named after the original PRoPE dataset route, but it is
    also used by ``multiview_mode="full_rope_offset"``. It builds per-view
    inpaint/control bundles and calls the transformer with per-view latent lists.
    For Wan2.1 Fun-Control, the pretrained ``y`` layout is control latents +
    static latents. For Wan2.2 Fun-Control, it remains control latents +
    inpaint latents.
    """
    device = accelerator.device
    dtype = weight_dtype
    conditioning_mode = str(conditioning_mode).strip().lower()

    video_views = batch["video_views"]
    hand_views = batch.get("hand_video_views")
    static_views = batch.get("static_video_views")
    static_disparity_views = batch.get("static_disparity_video_views")
    eef_depth_views = batch.get("eef_depth_video_views")
    if batch.get("camera_poses") is None or batch.get("intrinsics") is None:
        raise RuntimeError("PRoPE batch missing camera_poses/intrinsics")
    camera_poses = batch["camera_poses"].to(device=device, dtype=torch.float32)  # [B,V,F,4,4]
    intrinsics = batch["intrinsics"].to(device=device, dtype=torch.float32)        # [B,V,3,3]
    view_sizes = batch.get("view_sizes")
    if view_sizes is not None:
        view_sizes = view_sizes.to(device=device, dtype=torch.float32)             # [B,V,2] as (H,W)
    prompts = batch["prompts"]

    num_views = len(video_views)
    bsz = int(video_views[0].shape[0])
    frames_video = int(camera_poses.shape[2])

    # VAE-encode each view + its static (scene-anchor) counterpart. Per-view
    # static_video preference order:
    #   1. `static_views[v_idx]` from the dataset — DWM mode, the dataset built
    #      this from disk (wrist view = pre-rendered PCD splat) or as
    #      copy_first_frame (ext views).
    #   2. Runtime build via `_build_prope_condition_video` — I2V mode (dataset
    #      passed None) or any view without a dataset-provided static.
    def as_latents(tensor: torch.Tensor, label: str) -> torch.Tensor:
        latents = tensor.to(device=device, dtype=dtype)
        if latents.ndim != 5:
            raise ValueError(f"{label} expected [B,C,F,H,W] latent tensor, got {tuple(latents.shape)}")
        return latents

    def encode_raw_video(tensor: torch.Tensor, label: str) -> torch.Tensor:
        if load_tensors:
            return as_latents(tensor, label)
        video_bcfhw = rearrange(tensor.to(device=device, dtype=dtype), "b f c h w -> b c f h w")
        with torch.no_grad():
            return batch_encode_vae(video_bcfhw)

    def build_latent_condition_from_target(latents: torch.Tensor) -> torch.Tensor:
        if conditioning_mode == "dwm":
            return latents[:, :, 0:1, :, :].repeat(1, 1, latents.shape[2], 1, 1).clone()
        if conditioning_mode == "i2v":
            conditioned = torch.zeros_like(latents)
            conditioned[:, :, 0:1, :, :] = latents[:, :, 0:1, :, :]
            return conditioned
        raise ValueError(f"Unsupported PRoPE conditioning_mode: {conditioning_mode}")

    latents_views: list = []
    static_latents_views: list = []
    for v_idx, vid in enumerate(video_views):
        lat = encode_raw_video(vid, f"video_views[{v_idx}]")
        latents_views.append(lat)
        if static_views is not None and v_idx < len(static_views) and static_views[v_idx] is not None:
            if load_tensors:
                static_latents_views.append(as_latents(static_views[v_idx], f"static_video_views[{v_idx}]"))
                continue
            static_bcfhw = rearrange(static_views[v_idx].to(device=device, dtype=dtype), "b f c h w -> b c f h w")
        else:
            if load_tensors:
                static_latents_views.append(build_latent_condition_from_target(lat))
                continue
            v_bcfhw = rearrange(vid.to(device=device, dtype=dtype), "b f c h w -> b c f h w")
            static_bcfhw = _build_prope_condition_video(v_bcfhw, conditioning_mode)
        with torch.no_grad():
            static_latents_views.append(batch_encode_vae(static_bcfhw))

    # Optional hand/control video condition. Fun-Control consumes this stream
    # through the pretrained y lane; non-control concat paths keep using
    # condition_latents for DWM and optional I2V hand conditioning.
    hand_latents_views: list = [None] * num_views
    use_hand_views = (
        hand_views is not None
        and (uses_fun_control_y or is_wan2_2_fun_control or conditioning_mode == "dwm" or i2v_use_hand_condition)
    )
    if uses_fun_control_y and hand_views is None:
        raise ValueError("Wan2.1 Fun-Control PRoPE requires per-view hand/control videos.")
    if is_wan2_2_fun_control and hand_views is None:
        raise ValueError("Wan2.2 Fun-Control PRoPE requires per-view hand/control videos.")
    if use_hand_views:
        hand_latents_views = []
        for v_idx, h in enumerate(hand_views):
            hand_latents_views.append(encode_raw_video(h, f"hand_video_views[{v_idx}]"))
        robot_dropout_prob = float(training_config.get("robot_dropout_prob", 0.0) or 0.0)
        if robot_dropout_prob > 0:
            dropout_mask = torch.rand(bsz, device=device) < robot_dropout_prob
            if dropout_mask.any():
                for tensor in hand_latents_views:
                    tensor[dropout_mask] = torch.zeros_like(tensor[dropout_mask])

    # Optional auxiliary conditioning streams (depth maps). VAE-encoded per view and
    # routed through the patch_embedding's extra `condition_channels` lane.
    static_disparity_latents_views: List[Optional[torch.Tensor]] = [None] * num_views
    eef_depth_latents_views: List[Optional[torch.Tensor]] = [None] * num_views
    if static_disparity_views is not None:
        static_disparity_latents_views = []
        for v_idx, sd in enumerate(static_disparity_views):
            static_disparity_latents_views.append(encode_raw_video(sd, f"static_disparity_video_views[{v_idx}]"))
    if eef_depth_views is not None:
        eef_depth_latents_views = []
        for v_idx, hd in enumerate(eef_depth_views):
            eef_depth_latents_views.append(encode_raw_video(hd, f"eef_depth_video_views[{v_idx}]"))
        eef_depth_dropout_prob = float(training_config.get("eef_depth_dropout_prob", 0.0) or 0.0)
        if eef_depth_dropout_prob > 0:
            dropout_mask = torch.rand(bsz, device=device) < eef_depth_dropout_prob
            if dropout_mask.any():
                for tensor in eef_depth_latents_views:
                    tensor[dropout_mask] = torch.zeros_like(tensor[dropout_mask])

    first_lat = latents_views[0]
    frames_lat = int(first_lat.shape[2])

    # Shared timesteps + sigmas across views.
    indices = idx_sampling(bsz, generator=torch_rng, device=first_lat.device).long().cpu()
    timesteps = noise_scheduler.timesteps[indices].to(device=device)
    sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=first_lat.ndim, dtype=first_lat.dtype, device=device)

    noisy_views, target_views, inpaint_views, condition_views = [], [], [], []
    for v_idx, (lat, stat) in enumerate(zip(latents_views, static_latents_views)):
        noise = torch.randn_like(lat)
        noisy = (1.0 - sigmas) * lat + sigmas * noise
        target = noise - lat
        mask_latents = _build_prope_mask_latents(
            batch_size=bsz,
            latent_frames=frames_lat,
            latent_height=lat.shape[3],
            latent_width=lat.shape[4],
            conditioning_mode=conditioning_mode,
            device=device, dtype=dtype,
        )
        inpaint = torch.cat([mask_latents, stat], dim=1)
        y_view = inpaint
        condition_view = hand_latents_views[v_idx]
        if uses_fun_control_y:
            if not isinstance(condition_view, torch.Tensor):
                raise ValueError("Wan2.1 Fun-Control PRoPE requires control latents for every view.")
            if condition_view.shape[2:] != stat.shape[2:]:
                raise ValueError(
                    "PRoPE control latent shape mismatch: "
                    f"control={tuple(condition_view.shape)}, static={tuple(stat.shape)}"
                )
            y_view = torch.cat([condition_view, stat], dim=1)
            cond_parts = []
            if static_disparity_latents_views[v_idx] is not None:
                cond_parts.append(static_disparity_latents_views[v_idx])
            if eef_depth_latents_views[v_idx] is not None:
                cond_parts.append(eef_depth_latents_views[v_idx])
            condition_view = torch.cat(cond_parts, dim=1) if cond_parts else None
        elif is_wan2_2_fun_control:
            if not isinstance(condition_view, torch.Tensor):
                raise ValueError("Wan2.2 Fun-Control PRoPE requires control latents for every view.")
            if condition_view.shape[2:] != inpaint.shape[2:]:
                raise ValueError(
                    "PRoPE control latent shape mismatch: "
                    f"control={tuple(condition_view.shape)}, inpaint={tuple(inpaint.shape)}"
                )
            y_view = torch.cat([condition_view, inpaint], dim=1)
            # Reuse the condition_latents lane for auxiliary depth streams. Matches
            # the single-view fun-control path: condition_latents = cat(static_disparity, eef_depth).
            cond_parts = []
            if static_disparity_latents_views[v_idx] is not None:
                cond_parts.append(static_disparity_latents_views[v_idx])
            if eef_depth_latents_views[v_idx] is not None:
                cond_parts.append(eef_depth_latents_views[v_idx])
            condition_view = torch.cat(cond_parts, dim=1) if cond_parts else None
        noisy_views.append(noisy)
        target_views.append(target)
        inpaint_views.append(y_view)
        condition_views.append(condition_view)

    if not any(isinstance(c, torch.Tensor) for c in condition_views):
        condition_views_for_model = None
    else:
        condition_views_for_model = condition_views

    # Subsample per-frame poses to latent cadence (nearest).
    # Keep viewmats/Ks in fp32 — PRoPE camera math (1/fx, K @ viewmat, SE3 inverse)
    # quantizes badly in bf16 because fx/fy are in image-pixel units, so 1/fx falls
    # in ~1e-3 where bf16 has very few representable values. HY-WorldPlay's runners
    # explicitly cast both to fp32 before calling the transformer for the same reason.
    lat_indices = torch.linspace(0, frames_video - 1, frames_lat).round().long()
    poses_lat = camera_poses[:, :, lat_indices]  # [B, V, F_lat, 4, 4]
    viewmats = poses_lat.reshape(bsz, num_views * frames_lat, 4, 4).to(torch.float32)
    Ks_flat = (
        intrinsics.unsqueeze(2)
        .expand(-1, -1, frames_lat, -1, -1)
        .reshape(bsz, num_views * frames_lat, 3, 3)
        .to(torch.float32)
    )

    # Prompt embeds. Force empty when prompt_mode=empty, or stochastic text dropout.
    _use_empty_prompt = prompt_mode == PROMPT_MODE_EMPTY
    if not _use_empty_prompt and text_dropout_prob > 0.0:
        _use_empty_prompt = torch.rand(1).item() < text_dropout_prob
    if _use_empty_prompt:
        if empty_prompt_embeds is None:
            raise RuntimeError("prompt_mode=empty but empty_prompt_embeds is not initialized.")
        empty_dev = empty_prompt_embeds.to(device=device, dtype=dtype)
        prompt_embeds = [empty_dev.clone() for _ in range(bsz)]
    elif isinstance(prompts, torch.Tensor):
        prompt_embeds = prompts.to(device=device, dtype=dtype)
    elif isinstance(prompts, list) and prompts and isinstance(prompts[0], torch.Tensor):
        prompt_embeds = [p.to(device=device, dtype=dtype) for p in prompts]
    else:
        if tokenizer is None or text_encoder is None:
            raise RuntimeError(
                "PRoPE path received string prompts but tokenizer/text_encoder is unavailable."
            )
        if not isinstance(prompts, list):
            prompts = [str(prompts)] * bsz
        prompt_embeds = _encode_prompt_list_for_prope(
            tokenizer,
            text_encoder,
            [str(prompt) for prompt in prompts],
            device,
            dtype,
            max_length=args.tokenizer_max_length,
        )

    # seq_len: rough upper bound = sum over views of F_lat * (H_tok * W_tok).
    # Patch size is (1, 2, 2) so each view's token count = F_lat * (H_lat/2) * (W_lat/2).
    seq_len = sum(
        int(lat.shape[2] * (lat.shape[3] // 2) * (lat.shape[4] // 2)) for lat in latents_views
    )
    seq_len = max(seq_len, 128)

    with torch.amp.autocast(device_type="cuda", dtype=weight_dtype):
        noise_pred_views = transformer3d(
            x=noisy_views,
            t=timesteps,
            context=prompt_embeds,
            seq_len=seq_len,
            y=inpaint_views,
            condition_latents=condition_views_for_model,
            viewmats=viewmats,
            Ks=Ks_flat,
            image_sizes=view_sizes,
        )

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=args.weighting_scheme, sigmas=sigmas
    )
    per_view_losses = []
    for pred, tgt in zip(noise_pred_views, target_views):
        per_view_losses.append(
            custom_mse_loss(pred.float(), tgt.float(), weighting.float()).mean()
        )
    loss = sum(per_view_losses) / len(per_view_losses)

    # NaN survival path: dump per-batch-element diagnostics (which sample, its
    # viewmats/Ks magnitudes, which view's noise_pred is non-finite), log to a
    # rank-local file, then return a zero loss with no graph so the optimizer
    # update is a no-op for this step. This trades fail-fast for forward
    # progress — root cause analysis happens by replaying the dumped sample
    # offline. The rest of the training run continues with finite weights.
    if not torch.isfinite(loss):
        pair_keys = batch.get("pair_key") or batch.get("pair_keys")
        if isinstance(pair_keys, str):
            pair_keys = [pair_keys]
        elif pair_keys is None:
            pair_keys = [None] * bsz

        per_sample_lines = []
        for b_i in range(bsz):
            pk = pair_keys[b_i] if b_i < len(pair_keys) else None
            t_b = float(timesteps[b_i]) if timesteps.numel() > b_i else float("nan")
            sig_b = float(sigmas.flatten()[b_i]) if sigmas.numel() > b_i else float("nan")
            view_pred_summary = []
            for v_idx, p in enumerate(noise_pred_views):
                p_b = p[b_i:b_i + 1]
                n_nonfinite = int((~torch.isfinite(p_b)).sum())
                view_pred_summary.append((v_idx, n_nonfinite, int(p_b.numel())))
            view_lat_summary = []
            for v_idx, lat in enumerate(latents_views):
                lat_b = lat[b_i:b_i + 1]
                view_lat_summary.append((v_idx, int((~torch.isfinite(lat_b)).sum())))
            vm_b = viewmats[b_i] if viewmats.dim() == 4 else viewmats
            ks_b = Ks_flat[b_i] if Ks_flat.dim() == 4 else Ks_flat
            per_sample_lines.append(
                f"  batch_idx={b_i} pair_key={pk} t={t_b:.4g} sigma={sig_b:.4g} "
                f"viewmats_minmax=({float(vm_b.min()):.4g},{float(vm_b.max()):.4g}) "
                f"viewmats_finite={bool(torch.isfinite(vm_b).all())} "
                f"Ks_minmax=({float(ks_b.min()):.4g},{float(ks_b.max()):.4g}) "
                f"Ks_finite={bool(torch.isfinite(ks_b).all())} "
                f"latents_nonfinite_per_view={view_lat_summary} "
                f"noise_pred_nonfinite_per_view={view_pred_summary}"
            )
        msg = (
            f"PRoPE training: NON-FINITE loss SKIPPED at rank={accelerator.process_index}. "
            f"loss={loss.item()} per_view_losses={[float(l) for l in per_view_losses]}\n"
            + "\n".join(per_sample_lines)
        )
        logger.warning(msg)
        print(msg, flush=True)

        # Also append the offending pair_keys to a per-rank file under output_dir
        # so we can collect bad samples across the full run.
        try:
            out_dir = getattr(args, "output_dir", None)
            if out_dir:
                bad_path = os.path.join(out_dir, f"prope_nan_samples_rank{accelerator.process_index}.txt")
                os.makedirs(out_dir, exist_ok=True)
                with open(bad_path, "a") as fh:
                    for b_i in range(bsz):
                        pk = pair_keys[b_i] if b_i < len(pair_keys) else None
                        # Only log samples whose noise_pred is actually non-finite for any view.
                        any_nonfinite = any(
                            (~torch.isfinite(p[b_i:b_i + 1])).any().item()
                            for p in noise_pred_views
                        )
                        if any_nonfinite:
                            t_b = float(timesteps[b_i]) if timesteps.numel() > b_i else float("nan")
                            sig_b = float(sigmas.flatten()[b_i]) if sigmas.numel() > b_i else float("nan")
                            fh.write(f"{pk}\tt={t_b:.4f}\tsigma={sig_b:.4f}\n")
        except Exception as _e:
            logger.warning("Could not append bad-sample log: %s", _e)

        # Recompute loss with non-finite predictions zeroed out. Returning a
        # different graph (e.g. ``noise_pred.sum() * 0``) breaks DeepSpeed Zero-2
        # AllReduce because only the rank that hit NaN takes the alternate path
        # — the others stay on the standard MSE graph and the grad shapes /
        # collective ordering diverge, deadlocking the run inside
        # ``ProcessGroupNCCL`` until the 600 s watchdog fires (260428 run, 4×
        # H200, step 19). ``torch.nan_to_num`` keeps the same graph: NaN/Inf
        # spots become 0 in the forward output, gradient through replaced
        # positions is also 0, and DDP/Zero-2 AllReduce sees the same shapes
        # everywhere.
        sanitized_preds = [
            torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            for p in noise_pred_views
        ]
        per_view_losses_safe = [
            custom_mse_loss(pred.float(), tgt.float(), weighting.float()).mean()
            for pred, tgt in zip(sanitized_preds, target_views)
        ]
        return sum(per_view_losses_safe) / len(per_view_losses_safe)

    return loss


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50, spatial_mask=None):
    """Custom MSE loss with optional weighting, thresholding, and spatial masking."""
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    valid_ratio = None
    if spatial_mask is not None:
        spatial_mask = spatial_mask.float()
        mask = mask * spatial_mask
        valid_ratio = spatial_mask.mean().clamp_min(1e-8)
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    if valid_ratio is not None:
        final_loss = final_loss / valid_ratio
    return final_loss


def sanitize_nonfinite_gradients(
    params,
    *,
    accelerator: Accelerator,
    label: str,
) -> Tuple[int, int]:
    """Zero NaN/Inf gradient entries before clipping/optimizer step.

    PRoPE camera transforms can produce finite forward losses with non-finite
    backward entries when the projection branch is ill-conditioned. Letting one
    such gradient reach Adam poisons the weights and makes the next few steps
    collapse to NaN. Keep the graph/collectives unchanged, but sanitize the
    gradient tensor before clipping.
    """
    local_bad_tensors = 0
    local_bad_elems = 0
    seen = set()

    for param in params:
        if param is None:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all().item()):
            continue

        local_bad_tensors += 1
        local_bad_elems += int((~finite_mask).sum().detach().cpu().item())
        grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)

    stats = torch.tensor(
        [local_bad_tensors, local_bad_elems],
        device=accelerator.device,
        dtype=torch.long,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    total_bad_tensors = int(stats[0].item())
    total_bad_elems = int(stats[1].item())
    if total_bad_tensors and accelerator.is_main_process:
        logger.warning(
            "%s: sanitized non-finite gradients before optimizer step "
            "(tensors=%d, elements=%d)",
            label,
            total_bad_tensors,
            total_bad_elems,
        )
    return total_bad_tensors, total_bad_elems


def encode_clip_context_batch(
    clip_image_encoder: Optional[CLIPVisionModel],
    clip_images: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Encode first-frame images into CLIP hidden states for WAN i2v training."""
    if clip_image_encoder is None or clip_images is None:
        return None

    if clip_images.ndim == 5:
        # Raw i2v samples carry first-frame images as [B, 1, C, H, W].
        clip_images = clip_images[:, 0]
    elif clip_images.ndim == 3:
        clip_images = clip_images.unsqueeze(0)
    elif clip_images.ndim != 4:
        raise ValueError(
            f"Unexpected clip_images shape for CLIP encoding: {tuple(clip_images.shape)}. "
            "Expected [B, C, H, W] or [B, 1, C, H, W]."
        )

    # Legacy WAN CLIPModel expects a list of [C, T, H, W] tensors, one per
    # sample. This is the self-impl WAN 2.1 path used by the current configs.
    if hasattr(clip_image_encoder, "model") and not hasattr(clip_image_encoder, "vision_model"):
        clip_inputs = [image[:, None, :, :].to(dtype=weight_dtype) for image in clip_images]
        return clip_image_encoder(clip_inputs)

    # HF CLIPVisionModel path used by the removed/legacy diffusers i2v backend.
    image_size = int(getattr(clip_image_encoder.config, "image_size", 224))
    pixel_values = F.interpolate(
        clip_images,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )
    pixel_values = pixel_values.mul(0.5).add(0.5).clamp(0, 1)
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    ).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std

    outputs = clip_image_encoder(pixel_values=pixel_values.to(dtype=weight_dtype))
    return outputs.last_hidden_state


def build_i2v_image_condition_latents_for_validation(
    pipeline: WanFunInpaintHandConcatPipeline,
    condition_video: torch.Tensor,
    height: int,
    width: int,
    weight_dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the self-impl WAN i2v image condition used by the training loop.

    `i2v_condition_latent_type="image"` means: encode only the first RGB frame
    with the VAE, place it in latent frame 0, and zero-fill the latent tail.
    The inpaint mask latent is 1 only at latent frame 0.
    """
    if condition_video.ndim != 5:
        raise ValueError(
            f"Expected condition_video [B, C, F, H, W], got {tuple(condition_video.shape)}"
        )

    batch_size = condition_video.shape[0]
    latent_frames = (
        (condition_video.shape[2] - 1) // pipeline.vae_temporal_compression_ratio + 1
    )
    latent_height = height // pipeline.vae_scale_factor
    latent_width = width // pipeline.vae_scale_factor
    latent_channels = int(pipeline.vae.config.latent_channels)

    first_frame = condition_video[:, :, 0:1, :, :]
    first_frame_processed = pipeline.image_processor.preprocess(
        rearrange(first_frame, "b c f h w -> (b f) c h w"),
        height=height,
        width=width,
    )
    first_frame_processed = first_frame_processed.to(dtype=torch.float32)
    first_frame_processed = rearrange(first_frame_processed, "(b f) c h w -> b c f h w", f=1)

    vae_dtype = next(pipeline.vae.parameters()).dtype
    first_frame_processed = first_frame_processed.to(device=device, dtype=vae_dtype)
    first_frame_latents = pipeline.vae.encode(first_frame_processed)[0].sample(generator)
    first_frame_latents = first_frame_latents.to(device=device, dtype=weight_dtype)

    if first_frame_latents.shape[-2:] != (latent_height, latent_width):
        first_frame_latents = F.interpolate(
            first_frame_latents.squeeze(2),
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(2)

    static_video_latents = torch.zeros(
        batch_size,
        latent_channels,
        latent_frames,
        latent_height,
        latent_width,
        device=device,
        dtype=weight_dtype,
    )
    static_video_latents[:, :, 0:1, :, :] = first_frame_latents[:, :, 0:1, :, :]

    mask_latents = torch.zeros(
        batch_size,
        4,
        latent_frames,
        latent_height,
        latent_width,
        device=device,
        dtype=weight_dtype,
    )
    mask_latents[:, :, 0:1, :, :] = 1.0
    return static_video_latents, mask_latents


def _resolve_component_path(base_path: str, subpath: Optional[str], default_subpath: str) -> str:
    """
    Resolve model/tokenizer component path.
    - If local `<base_path>/<subpath>` exists, use it.
    - Otherwise treat `subpath` as a direct model id/path (e.g., `google/umt5-xxl`).
    """
    resolved_subpath = subpath or default_subpath
    if os.path.isabs(resolved_subpath):
        return resolved_subpath
    local_candidate = os.path.join(base_path, resolved_subpath)
    if os.path.exists(local_candidate):
        return local_candidate
    return resolved_subpath


def _read_video_fps(video_path: str) -> Optional[float]:
    """Read FPS metadata from a video file with OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return None
    return float(fps)


def _build_time_resample_indices(src_len: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Build time-aligned frame indices for FPS resampling."""
    if src_len <= 1 or src_fps <= 0 or tgt_fps <= 0:
        return np.array([0], dtype=np.int64)

    target_len = max(1, int(round(src_len * float(tgt_fps) / float(src_fps))))
    raw = np.linspace(0, src_len - 1, target_len)
    indices = np.clip(np.round(raw), 0, src_len - 1).astype(np.int64)
    # Use monotonic unique indices to avoid duplicated neighboring frames.
    indices = np.unique(indices)
    if indices.size == 0:
        return np.array([0], dtype=np.int64)
    return indices


def _apply_frame_indices(video_np_or_tensor, indices: np.ndarray, layout_spec: str):
    """Apply temporal frame indexing to video in supported layouts."""
    if layout_spec == "FHWC":
        return video_np_or_tensor[indices]
    if layout_spec == "BCFHW":
        return video_np_or_tensor[:, :, indices, :, :]
    raise ValueError(f"Unsupported layout_spec for frame indexing: {layout_spec}")


def _clip_or_pad_validation_video(video_np: np.ndarray, target_num_frames: int) -> np.ndarray:
    """Deterministically trim or tail-pad a validation video to a fixed length."""
    if video_np.shape[0] <= 0:
        raise ValueError("Validation video has no frames")
    if target_num_frames <= 0:
        raise ValueError(f"Invalid target_num_frames={target_num_frames}")

    if video_np.shape[0] > target_num_frames:
        return video_np[:target_num_frames]
    if video_np.shape[0] < target_num_frames:
        pad = np.repeat(video_np[-1:], target_num_frames - video_np.shape[0], axis=0)
        return np.concatenate([video_np, pad], axis=0)
    return video_np


def _build_validation_frame_skip_indices(src_len: int, frame_skip: int) -> np.ndarray:
    """Build deterministic temporal decimation indices for validation."""
    if src_len <= 0:
        raise ValueError(f"Invalid src_len={src_len}")
    frame_skip = int(frame_skip)
    if frame_skip <= 0:
        raise ValueError(f"validation frame skip must be positive, got {frame_skip}")
    return np.arange(0, src_len, frame_skip, dtype=np.int64)


def _align_and_normalize_validation_videos(
    gt_video: np.ndarray,
    condition_video_np: np.ndarray,
    hand_video_np: Optional[np.ndarray] = None,
    static_disparity_video_np: Optional[np.ndarray] = None,
    eef_depth_video_np: Optional[np.ndarray] = None,
    target_num_frames: Optional[int] = None,
    frame_skip: int = 1,
):
    """Align validation modalities to a common length, then optionally normalize to a fixed length."""
    temporal_lengths = [gt_video.shape[0], condition_video_np.shape[0]]
    if hand_video_np is not None:
        temporal_lengths.append(hand_video_np.shape[0])
    if static_disparity_video_np is not None:
        temporal_lengths.append(static_disparity_video_np.shape[0])
    if eef_depth_video_np is not None:
        temporal_lengths.append(eef_depth_video_np.shape[0])

    common_len = min(temporal_lengths)
    if common_len <= 0:
        raise ValueError("Validation sample has non-positive temporal length")

    if gt_video.shape[0] != common_len:
        gt_video = gt_video[:common_len]
    if condition_video_np.shape[0] != common_len:
        condition_video_np = condition_video_np[:common_len]
    if hand_video_np is not None and hand_video_np.shape[0] != common_len:
        hand_video_np = hand_video_np[:common_len]
    if static_disparity_video_np is not None and static_disparity_video_np.shape[0] != common_len:
        static_disparity_video_np = static_disparity_video_np[:common_len]
    if eef_depth_video_np is not None and eef_depth_video_np.shape[0] != common_len:
        eef_depth_video_np = eef_depth_video_np[:common_len]

    if int(frame_skip) > 1:
        frame_indices = _build_validation_frame_skip_indices(common_len, frame_skip)
        gt_video = _apply_frame_indices(gt_video, frame_indices, layout_spec="FHWC")
        condition_video_np = _apply_frame_indices(condition_video_np, frame_indices, layout_spec="FHWC")
        if hand_video_np is not None:
            hand_video_np = _apply_frame_indices(hand_video_np, frame_indices, layout_spec="FHWC")
        if static_disparity_video_np is not None:
            static_disparity_video_np = _apply_frame_indices(
                static_disparity_video_np, frame_indices, layout_spec="FHWC"
            )
        if eef_depth_video_np is not None:
            eef_depth_video_np = _apply_frame_indices(
                eef_depth_video_np, frame_indices, layout_spec="FHWC"
            )

    if target_num_frames is not None:
        gt_video = _clip_or_pad_validation_video(gt_video, target_num_frames)
        condition_video_np = _clip_or_pad_validation_video(condition_video_np, target_num_frames)
        if hand_video_np is not None:
            hand_video_np = _clip_or_pad_validation_video(hand_video_np, target_num_frames)
        if static_disparity_video_np is not None:
            static_disparity_video_np = _clip_or_pad_validation_video(static_disparity_video_np, target_num_frames)
        if eef_depth_video_np is not None:
            eef_depth_video_np = _clip_or_pad_validation_video(eef_depth_video_np, target_num_frames)

    return (
        gt_video,
        condition_video_np,
        hand_video_np,
        static_disparity_video_np,
        eef_depth_video_np,
    )


def resize_state_adaln_to_latent_frames(
    state_adaln: Optional[torch.Tensor],
    latent_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    temporal_mode: str = "sample4",
) -> Optional[torch.Tensor]:
    return align_state_adaln_to_latent_frames(
        state_adaln,
        latent_frames,
        device=device,
        dtype=dtype,
        temporal_mode=temporal_mode,
    )


def _resize_and_pad_video(
    video,
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
    short_side_mode: bool = True,
    center_pad: bool = True,
    layout_spec: str = "FHWC",
):
    """Resize video preserving aspect ratio, then pad to exact target size."""
    if layout_spec == "FHWC":
        # [F, H, W, C] -> [F, C, H, W]
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
    elif layout_spec == "BCFHW":
        # [B, C, F, H, W] -> [B*F, C, H, W]
        b, c, f, h, w = video.shape
        tensor = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).contiguous()
    else:
        raise ValueError(f"Unsupported layout_spec for resize/pad: {layout_spec}")

    _, _, src_h, src_w = tensor.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid video shape for resize/pad: H={src_h}, W={src_w}")

    if short_side_mode:
        src_short = min(src_h, src_w)
        tgt_short = min(target_h, target_w)
        scale = float(tgt_short) / float(src_short)
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))
        # Safety clamp in case rounded long side exceeds target.
        if new_h > target_h or new_w > target_w:
            scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
            new_h = max(1, int(round(src_h * scale)))
            new_w = max(1, int(round(src_w * scale)))
    else:
        scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))

    resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = max(0, target_h - new_h)
    pad_w = max(0, target_w - new_w)
    if center_pad:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top, pad_left = 0, 0
        pad_bottom, pad_right = pad_h, pad_w

    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=float(pad_value),
    )

    if layout_spec == "FHWC":
        return padded.permute(0, 2, 3, 1).cpu().numpy().astype(video.dtype, copy=False)

    # BCFHW path: [B*F, C, H, W] -> [B, C, F, H, W]
    b, c, f, _, _ = video.shape
    return padded.reshape(b, f, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous().to(video.dtype)


def _normalize_bucket_values(bucket_value: Optional[Any]) -> List[int]:
    if bucket_value is None:
        return []
    if isinstance(bucket_value, (list, tuple)):
        values = bucket_value
    else:
        values = [bucket_value]
    return [int(value) for value in values]


def _encode_prompt_list_for_prope(
    tokenizer,
    text_encoder,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Mimic ``WanFunInpaintPipeline._get_t5_prompt_embeds`` → list-of-[seq_len, D]."""
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    ids = tokenized.input_ids.to(device)
    mask = tokenized.attention_mask.to(device)
    with torch.no_grad():
        out = text_encoder(ids, attention_mask=mask)[0]  # [B, L, D]
    seq_lens = mask.gt(0).sum(dim=1).long()
    return [out[i, : int(seq_lens[i])].to(dtype=dtype) for i in range(out.shape[0])]


def _run_prope_validation(
    *,
    vae,
    text_encoder,
    tokenizer,
    transformer3d,
    config: Dict[str, Any],
    args,
    accelerator,
    weight_dtype: torch.dtype,
    global_step: int,
) -> None:
    """Validation for ``MultiViewWanTransformer3DModelWithConcat`` multi-view modes.

    Mirrors the composite mv-dwm validation flow but:
      - per-view latents instead of a canvas-composite tensor,
      - per-view VAE encode/decode,
      - transformer called with ``x=[list of per-view latents]`` and the flattened
        ``viewmats`` / ``Ks`` the training path already produces.
    """
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    custom_settings = training_config.get("custom_settings", {})
    scheduler_kwargs = config.get("scheduler_kwargs", {})

    prope_config = data_config.get("multiview_prope_config") or {}
    views_spec = prope_config.get("views")
    if not views_spec:
        logger.warning(
            "multiview_prope=true but multiview_prope_config.views missing. Skipping PRoPE validation."
        )
        return

    validation_set = data_config.get("validation_set")
    if validation_set is None:
        logger.warning("No validation_set specified for PRoPE validation. Skipping.")
        return
    dataset_files = [validation_set] if isinstance(validation_set, str) else list(validation_set)

    conditioning_mode = str(training_config.get("conditioning_mode", "dwm")).strip().lower()
    i2v_use_hand_condition = bool(training_config.get("i2v_use_hand_condition", False))
    transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
    pipeline_type = str(pipeline_config.get("type", "")).strip()
    uses_fun_control_y = bool(transformer_config.get("uses_fun_control_y", False))
    is_wan2_2_fun_control = (
        pipeline_type == "wan2.2_fun_control_i2v_mesh"
        or transformer_type == "Wan2_2FunControlTransformer3DModelWithConcat"
    )
    val_kwargs: Dict[str, Any] = {
        "data_root": data_config["data_root"],
        "dataset_file": dataset_files,
        "max_num_frames": int(data_config.get("max_num_frames", 81)),
        "load_tensors": False,
        "random_flip": False,
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 832),
        "frame_buckets": data_config.get("frame_buckets", 81),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir"),
        "prompt_json_fields": _resolve_prompt_json_fields(
            data_config.get("prompt_json_fields"),
            data_config.get("prompt_subdir", "prompts"),
        ),
        "static_video_subdir": data_config.get("static_video_subdir", "videos_static"),
        "static_video_source_mode": str(
            data_config.get("static_video_source_mode", "directory")
        ).strip().lower(),
        "robot_mesh_video_subdir": data_config.get("robot_mesh_video_subdir", "videos_hands"),
        "static_disparity_subdir": data_config.get("static_disparity_subdir"),
        "eef_depth_subdir": data_config.get("eef_depth_subdir"),
        "exclude_videos_file": data_config.get("exclude_videos_file"),
        "require_static_videos": conditioning_mode == "dwm",
        "align_width_to_32": False,
        "image_to_video": conditioning_mode == "i2v",
        "views_spec": views_spec,
    }
    for key in (
        "camera_serials_json",
        "cam2base_json",
        "cam2base_superset_json",
        "intrinsics_json",
        "raw_cache_root",
        "wrist_pose_source",
        "gripper2wrist_json",
        "wrist_proprio_flip",
    ):
        if key in prope_config and prope_config[key] is not None:
            val_kwargs[key] = prope_config[key]

    try:
        val_dataset = MultiViewPropeDataset(**val_kwargs)
    except Exception as e:
        logger.error("Failed to build PRoPE validation dataset: %s", e)
        import traceback; traceback.print_exc()
        return

    num_groups = len(val_dataset.sample_groups)
    if num_groups == 0:
        logger.warning("PRoPE validation dataset yielded 0 groups. Skipping.")
        return

    max_validation_videos = int(data_config.get("max_validation_videos", 2) or 0)
    if max_validation_videos <= 0:
        return
    total_validation_videos = max_validation_videos * accelerator.num_processes
    random_validation = bool(data_config.get("random_validation", False))

    if num_groups > total_validation_videos:
        if random_validation:
            rng = random.Random(int(global_step))
            indices = rng.sample(range(num_groups), total_validation_videos)
        else:
            indices = list(range(total_validation_videos))
    else:
        indices = list(range(num_groups))

    videos_per_gpu = max(1, len(indices) // accelerator.num_processes)
    start = accelerator.process_index * videos_per_gpu
    end = start + videos_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(indices)
    local_indices = indices[start:end]
    if not local_indices:
        return

    validation_prompt_mode = resolve_validation_prompt_mode(training_config)
    use_empty = validation_prompt_mode == PROMPT_MODE_EMPTY

    # Empty-prompt embeds (if requested / fallback when text_encoder is absent).
    empty_prompt_embeds = None
    empty_prompt_embeds_path = data_config.get("empty_prompt_embeds_path")
    if empty_prompt_embeds_path and os.path.exists(empty_prompt_embeds_path):
        try:
            empty_prompt_embeds = torch.load(empty_prompt_embeds_path, map_location="cpu")
        except Exception as e:
            logger.warning("Failed to load empty_prompt_embeds from %s: %s", empty_prompt_embeds_path, e)

    num_inference_steps = int(custom_settings.get("validation_num_inference_steps", 50))
    guidance_scale = float(custom_settings.get("validation_guidance_scale", 6.0))
    validation_save_fps = int(custom_settings.get("validation_save_fps", 16))

    transformer = accelerator.unwrap_model(transformer3d)
    was_training = transformer.training
    transformer.eval()
    torch.cuda.empty_cache()

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    device = accelerator.device

    logger.info(
        "🎯 PRoPE validation: num_groups=%d selected=%d gpu=%d/%d local=%d steps=%d guidance=%.2f",
        num_groups,
        len(indices),
        accelerator.process_index,
        accelerator.num_processes,
        len(local_indices),
        num_inference_steps,
        guidance_scale,
    )

    try:
        for sample_i, dataset_idx in enumerate(local_indices):
            try:
                sample = val_dataset[dataset_idx]
            except Exception as e:
                logger.warning("PRoPE validation: failed to load sample[%d]: %s", dataset_idx, e)
                continue

            video_views = sample["video_views"]              # List[[F, C, H_v, W_v]] in [-1, 1]
            hand_views = sample.get("hand_video_views")      # same shape or None
            static_views = sample.get("static_video_views")  # DWM per-view static; None in I2V
            static_disparity_views = sample.get("static_disparity_video_views")
            eef_depth_views = sample.get("eef_depth_video_views")
            camera_poses = sample["camera_poses"]            # [V, F, 4, 4]
            intrinsics = sample["intrinsics"]                # [V, 3, 3]
            view_sizes = sample.get("view_sizes")            # [V, 2] as (H, W)
            prompt_text = sample.get("prompt", "") if not use_empty else ""
            try:
                pair_key = str(val_dataset.sample_groups[dataset_idx]["pair_key"])
            except Exception:
                pair_key = f"sample_{dataset_idx}"
            video_save_id = _build_validation_video_save_id(pair_key.replace("/", "__"))

            num_views = len(video_views)
            frames_video = int(video_views[0].shape[0])

            # Add batch dim + move to device. Each view: [1, F, C, H_v, W_v] in [-1, 1].
            video_views_b = [
                v.unsqueeze(0).to(device=device, dtype=weight_dtype) for v in video_views
            ]
            hand_views_b: Optional[List[torch.Tensor]] = None
            use_hand_views = (
                hand_views is not None
                and (uses_fun_control_y or is_wan2_2_fun_control or conditioning_mode == "dwm" or i2v_use_hand_condition)
            )
            if uses_fun_control_y and hand_views is None:
                logger.warning(
                    "PRoPE validation: Wan2.1 Fun-Control sample %s has no hand/control views. Skipping.",
                    pair_key,
                )
                continue
            if is_wan2_2_fun_control and hand_views is None:
                logger.warning(
                    "PRoPE validation: Wan2.2 Fun-Control sample %s has no hand/control views. Skipping.",
                    pair_key,
                )
                continue
            if use_hand_views:
                hand_views_b = [
                    h.unsqueeze(0).to(device=device, dtype=weight_dtype) for h in hand_views
                ]

            static_views_b: Optional[List[torch.Tensor]] = None
            if static_views is not None:
                static_views_b = [
                    sv.unsqueeze(0).to(device=device, dtype=weight_dtype) for sv in static_views
                ]

            static_disparity_views_b: Optional[List[torch.Tensor]] = None
            if static_disparity_views is not None:
                static_disparity_views_b = [
                    sd.unsqueeze(0).to(device=device, dtype=weight_dtype)
                    for sd in static_disparity_views
                ]
            eef_depth_views_b: Optional[List[torch.Tensor]] = None
            if eef_depth_views is not None:
                eef_depth_views_b = [
                    hd.unsqueeze(0).to(device=device, dtype=weight_dtype)
                    for hd in eef_depth_views
                ]

            # VAE-encode per view + static counterpart + optional hand.
            latents_views: List[torch.Tensor] = []
            static_latents_views: List[torch.Tensor] = []
            hand_latents_views: List[Optional[torch.Tensor]] = [None] * num_views
            static_disparity_latents_views: List[Optional[torch.Tensor]] = [None] * num_views
            eef_depth_latents_views: List[Optional[torch.Tensor]] = [None] * num_views
            with torch.no_grad():
                for v_idx, v_bfchw in enumerate(video_views_b):
                    v_bcfhw = rearrange(v_bfchw, "b f c h w -> b c f h w")
                    latents_views.append(vae.encode(v_bcfhw)[0].sample().to(weight_dtype))
                    # Prefer dataset-provided per-view static (DWM with disk-
                    # backed wrist PCD anchor); fall back to runtime build for
                    # I2V or views without a dataset static.
                    if static_views_b is not None and v_idx < len(static_views_b) and static_views_b[v_idx] is not None:
                        static_bcfhw = rearrange(static_views_b[v_idx], "b f c h w -> b c f h w")
                    else:
                        static_bcfhw = _build_prope_condition_video(v_bcfhw, conditioning_mode)
                    static_latents_views.append(vae.encode(static_bcfhw)[0].sample().to(weight_dtype))
                if hand_views_b is not None:
                    hand_latents_views = []
                    for h_bfchw in hand_views_b:
                        h_bcfhw = rearrange(h_bfchw, "b f c h w -> b c f h w")
                        hand_latents_views.append(vae.encode(h_bcfhw)[0].sample().to(weight_dtype))
                if static_disparity_views_b is not None:
                    static_disparity_latents_views = []
                    for sd_bfchw in static_disparity_views_b:
                        sd_bcfhw = rearrange(sd_bfchw, "b f c h w -> b c f h w")
                        static_disparity_latents_views.append(
                            vae.encode(sd_bcfhw)[0].sample().to(weight_dtype)
                        )
                if eef_depth_views_b is not None:
                    eef_depth_latents_views = []
                    for hd_bfchw in eef_depth_views_b:
                        hd_bcfhw = rearrange(hd_bfchw, "b f c h w -> b c f h w")
                        eef_depth_latents_views.append(
                            vae.encode(hd_bcfhw)[0].sample().to(weight_dtype)
                        )

            frames_lat = int(latents_views[0].shape[2])

            # Fixed inpaint (mask=all-unknown + static latents) and hand condition, per view.
            inpaint_views: List[torch.Tensor] = []
            skip_sample = False
            for v_idx, (lat, stat) in enumerate(zip(latents_views, static_latents_views)):
                mask_latents = _build_prope_mask_latents(
                    batch_size=lat.shape[0],
                    latent_frames=frames_lat,
                    latent_height=lat.shape[3],
                    latent_width=lat.shape[4],
                    conditioning_mode=conditioning_mode,
                    device=device, dtype=weight_dtype,
                )
                inpaint = torch.cat([mask_latents, stat], dim=1)
                if uses_fun_control_y:
                    control = hand_latents_views[v_idx]
                    if not isinstance(control, torch.Tensor):
                        logger.warning(
                            "PRoPE validation: missing control latent for sample %s view=%d. Skipping.",
                            pair_key,
                            v_idx,
                        )
                        skip_sample = True
                        break
                    if control.shape[2:] != stat.shape[2:]:
                        logger.warning(
                            "PRoPE validation control/static shape mismatch: "
                            "control=%s, static=%s. Skipping sample %s.",
                            tuple(control.shape),
                            tuple(stat.shape),
                            pair_key,
                        )
                        skip_sample = True
                        break
                    inpaint = torch.cat([control, stat], dim=1)
                elif is_wan2_2_fun_control:
                    control = hand_latents_views[v_idx]
                    if not isinstance(control, torch.Tensor):
                        logger.warning(
                            "PRoPE validation: missing control latent for sample %s view=%d. Skipping.",
                            pair_key,
                            v_idx,
                        )
                        skip_sample = True
                        break
                    if control.shape[2:] != inpaint.shape[2:]:
                        logger.warning(
                            "PRoPE validation control/inpaint shape mismatch: "
                            "control=%s, inpaint=%s. Skipping sample %s.",
                            tuple(control.shape),
                            tuple(inpaint.shape),
                            pair_key,
                        )
                        skip_sample = True
                        break
                    inpaint = torch.cat([control, inpaint], dim=1)
                inpaint_views.append(inpaint)
            if skip_sample:
                continue

            # If no hand views (i2v w/o hand), pass None rather than [None, None, ...]
            # so ``_forward_multiview_token_concat`` skips the cat.
            if uses_fun_control_y or is_wan2_2_fun_control:
                # Build per-view condition_latents from auxiliary depth streams.
                # Mirrors the single-view fun-control path:
                #   condition_latents = cat(static_disparity, eef_depth).
                cond_views_built: List[Optional[torch.Tensor]] = []
                any_cond = False
                for v_idx in range(num_views):
                    cond_parts: List[torch.Tensor] = []
                    if static_disparity_latents_views[v_idx] is not None:
                        cond_parts.append(static_disparity_latents_views[v_idx])
                    if eef_depth_latents_views[v_idx] is not None:
                        cond_parts.append(eef_depth_latents_views[v_idx])
                    if cond_parts:
                        cond_views_built.append(torch.cat(cond_parts, dim=1))
                        any_cond = True
                    else:
                        cond_views_built.append(None)
                condition_views: Optional[List[Optional[torch.Tensor]]] = (
                    cond_views_built if any_cond else None
                )
            elif any(isinstance(c, torch.Tensor) for c in hand_latents_views):
                condition_views: Optional[List[Optional[torch.Tensor]]] = list(hand_latents_views)
            else:
                condition_views = None

            # Viewmats / Ks at latent cadence, matching ``_prope_training_step``.
            # Kept in fp32 (HY-WorldPlay pattern) — see comment in `_prope_training_step`.
            lat_frame_indices = torch.linspace(0, frames_video - 1, frames_lat).round().long()
            poses_lat = camera_poses[:, lat_frame_indices].to(device=device, dtype=torch.float32)
            viewmats = poses_lat.reshape(num_views * frames_lat, 4, 4).unsqueeze(0)
            Ks_flat = (
                intrinsics.to(device=device, dtype=torch.float32)
                .unsqueeze(1)
                .expand(num_views, frames_lat, 3, 3)
                .reshape(num_views * frames_lat, 3, 3)
                .unsqueeze(0)
            )  # [1, V*F, 3, 3]
            if view_sizes is None:
                view_sizes_b = torch.tensor(
                    [[int(v.shape[-2]), int(v.shape[-1])] for v in video_views],
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
            else:
                view_sizes_b = view_sizes.to(device=device, dtype=torch.float32).unsqueeze(0)

            # Prompt embeddings. Only enable CFG when we have an actual prompt + text encoder.
            do_cfg = (
                guidance_scale > 1.0
                and (not use_empty)
                and text_encoder is not None
                and tokenizer is not None
                and bool(prompt_text.strip())
            )

            if use_empty or (text_encoder is None or tokenizer is None) or (not prompt_text.strip()):
                if empty_prompt_embeds is None:
                    logger.warning(
                        "PRoPE validation: empty prompt path active but empty_prompt_embeds cache not loaded. "
                        "Skipping sample %s.",
                        pair_key,
                    )
                    continue
                empty_dev = empty_prompt_embeds.to(device=device, dtype=weight_dtype)
                prompt_embeds_pos = [empty_dev]
                prompt_embeds_neg = [empty_dev]
            else:
                try:
                    prompt_embeds_pos = _encode_prompt_list_for_prope(
                        tokenizer, text_encoder, [prompt_text], device, weight_dtype
                    )
                    if do_cfg:
                        prompt_embeds_neg = _encode_prompt_list_for_prope(
                            tokenizer, text_encoder, ["bad detailed"], device, weight_dtype
                        )
                    else:
                        prompt_embeds_neg = None
                except Exception as e:
                    logger.warning(
                        "PRoPE validation: text_encoder failed (%s) — falling back to empty prompt.", e
                    )
                    if empty_prompt_embeds is None:
                        continue
                    empty_dev = empty_prompt_embeds.to(device=device, dtype=weight_dtype)
                    prompt_embeds_pos = [empty_dev]
                    prompt_embeds_neg = [empty_dev]
                    do_cfg = False

            # Estimate total token count (used to pad/gate seq_len inside the transformer).
            seq_len = sum(
                int(lat.shape[2] * (lat.shape[3] // 2) * (lat.shape[4] // 2)) for lat in latents_views
            )
            seq_len = max(seq_len, 128)

            scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = scheduler.timesteps
            # FlowMatchEulerDiscreteScheduler.step() advances an internal ``_step_index``
            # counter per call, which would overflow when we call it V times per timestep
            # (once per view). We replicate the Euler update manually using the precomputed
            # sigma schedule. Sigmas length is ``num_inference_steps + 1``. Keep sigmas in
            # float32 so the per-step ``lat + dsigma * pred`` upcast matches the diffusers
            # scheduler.step (it casts ``sample`` to fp32 before the Euler update); doing it
            # in bf16 accumulates enough precision loss over 50 steps to make decoded frames
            # appear uniformly dark.
            sigmas_device = scheduler.sigmas.to(device=device, dtype=torch.float32)

            generator = torch.Generator(device=device).manual_seed(
                int(args.seed or 0) + sample_i + accelerator.process_index * 10007
            )
            # Start from Gaussian noise per view. Multiplying by ``init_noise_sigma`` (a fp32
            # tensor on device) silently promotes the tensor to fp32, which then drives the
            # rest of the loop in fp32 and confuses the bf16 autocast inside the transformer.
            # Cast back to ``weight_dtype`` explicitly.
            init_sigma = float(scheduler.init_noise_sigma) if hasattr(scheduler, "init_noise_sigma") else 1.0
            latents_views_run = [
                (
                    randn_tensor(lat.shape, generator=generator, device=device, dtype=weight_dtype)
                    * init_sigma
                ).to(dtype=weight_dtype)
                for lat in latents_views
            ]

            with torch.no_grad():
                for step_idx, t in enumerate(timesteps):
                    if do_cfg:
                        lat_in_views = [torch.cat([l, l], dim=0) for l in latents_views_run]
                        y_in_views = [torch.cat([y, y], dim=0) for y in inpaint_views]
                        if condition_views is None:
                            cond_in_views = None
                        else:
                            cond_in_views = [
                                torch.cat([c, c], dim=0) if isinstance(c, torch.Tensor) else None
                                for c in condition_views
                            ]
                        viewmats_in = torch.cat([viewmats, viewmats], dim=0)
                        Ks_in = torch.cat([Ks_flat, Ks_flat], dim=0)
                        image_sizes_in = torch.cat([view_sizes_b, view_sizes_b], dim=0)
                        prompt_in = list(prompt_embeds_neg) + list(prompt_embeds_pos)
                    else:
                        lat_in_views = list(latents_views_run)
                        y_in_views = list(inpaint_views)
                        cond_in_views = list(condition_views) if condition_views is not None else None
                        viewmats_in = viewmats
                        Ks_in = Ks_flat
                        image_sizes_in = view_sizes_b
                        prompt_in = list(prompt_embeds_pos)

                    timestep = t.expand(lat_in_views[0].shape[0])
                    with torch.amp.autocast(device_type="cuda", dtype=weight_dtype):
                        noise_pred_views = transformer(
                            x=lat_in_views,
                            t=timestep,
                            context=prompt_in,
                            seq_len=seq_len,
                            y=y_in_views,
                            condition_latents=cond_in_views,
                            viewmats=viewmats_in,
                            Ks=Ks_in,
                            image_sizes=image_sizes_in,
                        )

                    # Diagnostic: catch NaN/Inf in raw model output before CFG mixes it.
                    for v_idx, pred in enumerate(noise_pred_views):
                        if not torch.isfinite(pred).all():
                            n_nan = int(torch.isnan(pred).sum())
                            n_inf = int(torch.isinf(pred).sum())
                            logger.warning(
                                "PRoPE val: noise_pred[view=%d] step=%d t=%s nan=%d inf=%d "
                                "min=%.3g max=%.3g (clamping)",
                                v_idx, step_idx, t.item() if t.numel() == 1 else t,
                                n_nan, n_inf,
                                float(pred[torch.isfinite(pred)].min()) if n_nan + n_inf < pred.numel() else float("nan"),
                                float(pred[torch.isfinite(pred)].max()) if n_nan + n_inf < pred.numel() else float("nan"),
                            )
                            noise_pred_views[v_idx] = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)

                    if do_cfg:
                        guided = []
                        for pred in noise_pred_views:
                            p_uncond, p_text = pred.chunk(2, dim=0)
                            guided.append(p_uncond + guidance_scale * (p_text - p_uncond))
                        noise_pred_views = guided

                    # Manual Euler update shared across views (avoids scheduler._step_index
                    # drift — see comment near sigmas_device definition). Mirrors
                    # ``FlowMatchEulerDiscreteScheduler.step``: upcast to fp32 for the
                    # Euler step, then cast back to model dtype.
                    sigma = sigmas_device[step_idx]
                    sigma_next = sigmas_device[step_idx + 1]
                    dsigma = (sigma_next - sigma)
                    new_latents: List[torch.Tensor] = []
                    for pred, lat in zip(noise_pred_views, latents_views_run):
                        out_dtype = lat.dtype
                        prev = lat.to(torch.float32) + dsigma * pred.to(torch.float32)
                        if not torch.isfinite(prev).all():
                            logger.warning(
                                "PRoPE val: latents step=%d went non-finite after Euler — clamping",
                                step_idx,
                            )
                            prev = torch.nan_to_num(prev, nan=0.0, posinf=1e3, neginf=-1e3)
                        new_latents.append(prev.to(out_dtype))
                    latents_views_run = new_latents

            # Decode each view's latents → [1, 3, F, H_v, W_v] in [0, 1].
            sample_tensors: List[torch.Tensor] = []
            gt_tensors: List[torch.Tensor] = []
            static_tensors: List[torch.Tensor] = []
            hand_tensors_list: List[torch.Tensor] = []
            static_disparity_tensors_list: List[torch.Tensor] = []
            eef_depth_tensors_list: List[torch.Tensor] = []
            with torch.no_grad():
                for v_idx, (lat, v_bfchw) in enumerate(zip(latents_views_run, video_views_b)):
                    if not torch.isfinite(lat).all():
                        logger.warning(
                            "PRoPE val: final latents[view=%d] non-finite before VAE decode — clamping",
                            v_idx,
                        )
                        lat = torch.nan_to_num(lat, nan=0.0, posinf=1e3, neginf=-1e3)
                    decoded = vae.decode(lat.to(vae.dtype)).sample  # [1, 3, F, H, W]
                    if not torch.isfinite(decoded).all():
                        logger.warning(
                            "PRoPE val: VAE decode[view=%d] produced non-finite — clamping",
                            v_idx,
                        )
                        decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=-1.0)
                    decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                    sample_tensors.append(decoded)
                    v_bcfhw = rearrange(v_bfchw.cpu().float(), "b f c h w -> b c f h w")
                    gt_tensors.append((v_bcfhw / 2 + 0.5).clamp(0, 1))
                    # Match the latent-encoding lane (above): prefer dataset-
                    # provided per-view static (DWM wrist PCD anchor) so the
                    # comparison MP4 displays exactly what the model conditioned
                    # on. Falls back to runtime build (DWM copy_first_frame /
                    # I2V [frame0, 0...]) when the dataset doesn't supply one.
                    if static_views_b is not None and v_idx < len(static_views_b) and static_views_b[v_idx] is not None:
                        cond_bcfhw = rearrange(
                            static_views_b[v_idx].cpu().float(),
                            "b f c h w -> b c f h w",
                        )
                    else:
                        cond_bcfhw = _build_prope_condition_video(v_bcfhw, conditioning_mode)
                    static_tensors.append((cond_bcfhw / 2 + 0.5).clamp(0, 1))
                    if hand_views_b is not None:
                        h_bcfhw = rearrange(
                            hand_views_b[v_idx].cpu().float(), "b f c h w -> b c f h w"
                        )
                        hand_tensors_list.append((h_bcfhw / 2 + 0.5).clamp(0, 1))
                    if static_disparity_views_b is not None:
                        sd_bcfhw = rearrange(
                            static_disparity_views_b[v_idx].cpu().float(), "b f c h w -> b c f h w"
                        )
                        static_disparity_tensors_list.append((sd_bcfhw / 2 + 0.5).clamp(0, 1))
                    if eef_depth_views_b is not None:
                        hd_bcfhw = rearrange(
                            eef_depth_views_b[v_idx].cpu().float(), "b f c h w -> b c f h w"
                        )
                        eef_depth_tensors_list.append((hd_bcfhw / 2 + 0.5).clamp(0, 1))

            # Unify temporal length across all role tensors for this sample.
            all_tensors = (
                sample_tensors
                + gt_tensors
                + static_tensors
                + hand_tensors_list
                + static_disparity_tensors_list
                + eef_depth_tensors_list
            )
            num_frames_actual = min(int(t.shape[2]) for t in all_tensors)
            sample_tensors = [t[:, :, :num_frames_actual] for t in sample_tensors]
            gt_tensors = [t[:, :, :num_frames_actual] for t in gt_tensors]
            static_tensors = [t[:, :, :num_frames_actual] for t in static_tensors]
            if hand_tensors_list:
                hand_tensors_list = [t[:, :, :num_frames_actual] for t in hand_tensors_list]
            if static_disparity_tensors_list:
                static_disparity_tensors_list = [
                    t[:, :, :num_frames_actual] for t in static_disparity_tensors_list
                ]
            if eef_depth_tensors_list:
                eef_depth_tensors_list = [
                    t[:, :, :num_frames_actual] for t in eef_depth_tensors_list
                ]

            # Per view: (condition | scene_depth | hand | eef_depth | generated | gt) concat
            # along W at the view's native resolution (no upscale — preserves the visual
            # size difference between ext and wrist crops). Streams that aren't loaded skip
            # their pane.
            per_view_rows: List[torch.Tensor] = []
            comparison_labels: List[str] = []
            for v_idx in range(num_views):
                parts = [static_tensors[v_idx]]
                labels = ["condition"]
                if static_disparity_tensors_list:
                    parts.append(static_disparity_tensors_list[v_idx])
                    labels.append("scene_depth")
                if hand_tensors_list:
                    parts.append(hand_tensors_list[v_idx])
                    labels.append("hand")
                if eef_depth_tensors_list:
                    parts.append(eef_depth_tensors_list[v_idx])
                    labels.append("eef_depth")
                parts.append(sample_tensors[v_idx])
                parts.append(gt_tensors[v_idx])
                labels.extend(["generated", "gt"])
                per_view_rows.append(torch.cat(parts, dim=4))
                if v_idx == 0:
                    comparison_labels = labels

            # Stack views vertically (along H). Pad each row's W with zeros (black) on the
            # right so all rows share the same width without upscaling smaller views.
            max_row_w = max(int(r.shape[4]) for r in per_view_rows)

            def _pad_w(t: torch.Tensor, target_w: int) -> torch.Tensor:
                w = int(t.shape[4])
                if w == target_w:
                    return t
                pad = t.new_zeros(t.shape[0], t.shape[1], t.shape[2], t.shape[3], target_w - w)
                return torch.cat([t, pad], dim=4)

            per_view_rows_padded = [_pad_w(r, max_row_w) for r in per_view_rows]
            comparison = torch.cat(per_view_rows_padded, dim=3)
            comparison_caption = "(" + "|".join(comparison_labels) + ")"

            # Generated-only stacking: views laid out side-by-side along W. Pad H so
            # rows of differently-sized views share the same height.
            sample_max_h = max(int(s.shape[3]) for s in sample_tensors)

            def _pad_h(t: torch.Tensor, target_h: int) -> torch.Tensor:
                h = int(t.shape[3])
                if h == target_h:
                    return t
                pad = t.new_zeros(t.shape[0], t.shape[1], t.shape[2], target_h - h, t.shape[4])
                return torch.cat([t, pad], dim=3)

            sample_tensors_padded = [_pad_h(s, sample_max_h) for s in sample_tensors]
            generated_stacked = torch.cat(sample_tensors_padded, dim=4)

            save_name = f"{video_save_id}_{PROMPT_MODE_EMPTY if use_empty else 'normal'}"
            gpu_suffix = f"_gpu{accelerator.process_index}" if accelerator.num_processes > 1 else ""
            os.makedirs(args.output_dir, exist_ok=True)
            generated_filename = os.path.join(
                args.output_dir, f"step_{global_step}_generated_{save_name}{gpu_suffix}.mp4"
            )
            comparison_filename = os.path.join(
                args.output_dir, f"step_{global_step}_comparison_{save_name}{gpu_suffix}.mp4"
            )
            save_videos_grid(generated_stacked, generated_filename, fps=validation_save_fps)
            save_videos_grid(comparison, comparison_filename, fps=validation_save_fps)
            logger.info(
                "📹 GPU %d: PRoPE validation saved pair_key=%s (save_id=%s, views=%d, frames=%d)",
                accelerator.process_index,
                pair_key,
                video_save_id,
                num_views,
                num_frames_actual,
            )
            logger.info("   Generated: %s", generated_filename)
            logger.info("   Comparison %s: %s", comparison_caption, comparison_filename)

            if accelerator.is_main_process:
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb" and HAS_WANDB:
                        caption = (
                            f"step_{global_step}_{video_save_id}_"
                            f"{PROMPT_MODE_EMPTY if use_empty else 'normal'}_comparison "
                            f"{comparison_caption}"
                        )
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Video(
                                        comparison_filename,
                                        caption=caption,
                                        fps=validation_save_fps,
                                    )
                                ]
                            },
                            step=global_step,
                        )
    except Exception as e:
        logger.error("PRoPE validation error: %s", e)
        import traceback; traceback.print_exc()
    finally:
        if was_training:
            transformer.train()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    transformer3d,
    network,
    config,
    args,
    accelerator,
    weight_dtype,
    global_step,
    load_tensors: bool = False,
    vae_path: str = None,
    vae_kwargs: dict = None,
    text_encoder_path: str = None,
    text_encoder_kwargs: dict = None,
    tokenizer_path: str = None,
):
    """Run validation during training using dataset-based validation samples."""
    validation_stage = "initialization"
    transformer_for_val = None
    transformer_state_owner = None
    transformer_for_val_was_training = None
    restore_text_encoder_to_cpu = False
    validation_clip_image_encoder = None
    try:
        logger.info("Running validation...")
        training_config = config.get('training', {})
        custom_settings = training_config.get("custom_settings", {})
        conditioning_mode_for_reload = training_config.get("conditioning_mode", "dwm")
        use_diffusers_i2v_backend_for_reload = bool(custom_settings.get("use_diffusers_i2v_backend", False))
        use_diffusers_i2v_backend_for_reload = (
            use_diffusers_i2v_backend_for_reload and conditioning_mode_for_reload == "i2v"
        )
        transformer_config_for_val = config.get("transformer", {})
        transformer_type_for_val = transformer_config_for_val.get("class", "WanTransformer3DModelWithConcat")

        # If load_tensors=True, VAE/text_encoder were deleted during training. Reload them for validation.
        models_reloaded = False
        if load_tensors and vae is None:
            if vae_path is None:
                logger.warning("load_tensors=True but vae_path not provided. Skipping validation.")
                return
            logger.info(f"📦 Reloading models for validation...")

            # Reload VAE
            logger.info(f"   Loading VAE from {vae_path}...")
            if use_diffusers_i2v_backend_for_reload:
                vae = diffusers.AutoencoderKLWan.from_pretrained(
                    vae_path,
                    torch_dtype=weight_dtype,
                ).eval()
            else:
                Choosen_AutoencoderKL = {
                    "AutoencoderKLWan": AutoencoderKLWan,
                    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
                }[vae_kwargs.get('vae_type', 'AutoencoderKLWan') if vae_kwargs else 'AutoencoderKLWan']
                vae = Choosen_AutoencoderKL.from_pretrained(
                    vae_path,
                    additional_kwargs=vae_kwargs or {},
                ).eval()
            vae = vae.to(accelerator.device, dtype=weight_dtype)

            # Reload text_encoder and tokenizer
            if text_encoder_path is not None and tokenizer_path is not None:
                logger.info(f"   Loading text encoder from {text_encoder_path}...")
                if use_diffusers_i2v_backend_for_reload:
                    text_encoder = UMT5EncoderModel.from_pretrained(
                        text_encoder_path,
                        torch_dtype=weight_dtype,
                    ).eval()
                else:
                    text_encoder = WanT5EncoderModel.from_pretrained(
                        text_encoder_path,
                        additional_kwargs=text_encoder_kwargs or {},
                        torch_dtype=weight_dtype,
                    ).eval()
                text_encoder = text_encoder.to(accelerator.device)

                logger.info(f"   Loading tokenizer from {tokenizer_path}...")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            models_reloaded = True
            logger.info("   Models reloaded successfully.")
        elif text_encoder is not None:
            text_encoder_device = next(text_encoder.parameters()).device
            if text_encoder_device.type == "cpu":
                logger.info("📦 Moving text encoder to GPU for validation...")
                text_encoder = text_encoder.to(accelerator.device)
                restore_text_encoder_to_cpu = True

        # PRoPE multi-view dispatch: the composite pipeline builders below do not know how to
        # handle list-of-per-view latents, so branch early but *after* the VAE reload above.
        if transformer_type_for_val == "MultiViewWanTransformer3DModelWithConcat":
            try:
                _run_prope_validation(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer3d=transformer3d,
                    config=config,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    global_step=global_step,
                )
            finally:
                if models_reloaded:
                    logger.info("📦 Cleaning up reloaded models after PRoPE validation...")
                    del vae
                    if text_encoder is not None:
                        del text_encoder
                    if tokenizer is not None:
                        del tokenizer
                elif restore_text_encoder_to_cpu and text_encoder is not None:
                    logger.info("📦 Moving text encoder back to CPU after PRoPE validation...")
                    text_encoder = text_encoder.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return

        # Extract config values
        pipeline_config = config.get('pipeline', {})
        training_config = config.get('training', {})
        custom_settings = training_config.get("custom_settings", {})
        transformer_config = config.get('transformer', {})
        data_config = config.get('data', {})
        scheduler_kwargs = config.get('scheduler_kwargs', {})
        transformer_additional_kwargs = config.get('transformer_additional_kwargs', {})
        validation_prompt_mode = resolve_validation_prompt_mode(training_config)
        validation_lightx2v = bool(custom_settings.get("validation_lightx2v", False))
        validation_lightx2v_lora_path = custom_settings.get(
            "validation_lightx2v_lora_path", "ckpts/loras/lightx2v_rank64.safetensors"
        )
        validation_lightx2v_lora_scale = float(custom_settings.get("validation_lightx2v_lora_scale", 1.0))
        validation_lightx2v_compat_mode = str(custom_settings.get("lightx2v_compat_mode", "auto")).lower()
        validation_lightx2v_compat_strict = bool(custom_settings.get("lightx2v_compat_strict", True))
        validation_lightx2v_apply_non_block_diff = bool(custom_settings.get("lightx2v_apply_non_block_diff", False))

        # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
        if pipeline_config.get("fps") is not None:
            transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
        
        # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
        is_wan2_2 = "2.2" in pipeline_config.get("type", "")
        pipeline_type = pipeline_config.get("type", "")

        # Validation runtime adaptation (default off for backward compatibility)
        validation_runtime_adapt = bool(data_config.get("validation_runtime_adapt", False))
        validation_inference_size = data_config.get("validation_inference_size", None)
        validation_resize_mode = data_config.get("validation_resize_mode", "short_side_pad")
        validation_pad_value = float(data_config.get("validation_pad_value", 0))
        validation_center_pad = bool(data_config.get("validation_center_pad", True))
        validation_target_fps = data_config.get("validation_target_fps", None)
        validation_temporal_frame_skip = data_config.get("validation_temporal_frame_skip", 1)
        validation_fps_mode = data_config.get("validation_fps_mode", "time_resample")
        validation_resize_to_target = False
        validation_target_h: Optional[int] = None
        validation_target_w: Optional[int] = None
        validation_target_num_frames: Optional[int] = None

        if validation_target_fps is not None:
            try:
                validation_target_fps = float(validation_target_fps)
                if validation_target_fps <= 0:
                    logger.warning(
                        f"Invalid validation_target_fps={validation_target_fps}. Disabling FPS adaptation."
                    )
                    validation_target_fps = None
            except Exception:
                logger.warning(
                    f"Failed to parse validation_target_fps={validation_target_fps}. Disabling FPS adaptation."
                )
                validation_target_fps = None
        try:
            validation_temporal_frame_skip = int(validation_temporal_frame_skip)
            if validation_temporal_frame_skip <= 0:
                raise ValueError
        except Exception:
            logger.warning(
                "Invalid validation_temporal_frame_skip=%s. Falling back to 1.",
                validation_temporal_frame_skip,
            )
            validation_temporal_frame_skip = 1

        inferred_validation_height_buckets = sorted(set(_normalize_bucket_values(data_config.get("height_buckets"))))
        inferred_validation_width_buckets = sorted(set(_normalize_bucket_values(data_config.get("width_buckets"))))
        inferred_validation_frame_buckets = sorted(set(_normalize_bucket_values(data_config.get("frame_buckets"))))

        raw_validation_target_num_frames = data_config.get("max_num_frames", custom_settings.get("max_num_frames"))
        if raw_validation_target_num_frames is not None:
            try:
                validation_target_num_frames = int(raw_validation_target_num_frames)
                if validation_target_num_frames <= 0:
                    raise ValueError
            except Exception:
                logger.warning(
                    f"Invalid validation max_num_frames={raw_validation_target_num_frames}. "
                    "Disabling temporal length normalization."
                )
                validation_target_num_frames = None
        elif len(inferred_validation_frame_buckets) == 1:
            validation_target_num_frames = inferred_validation_frame_buckets[0]
            logger.info(
                "Validation frame-count fallback: using single training frame bucket %d.",
                validation_target_num_frames,
            )

        if validation_target_fps is None:
            if "2.1" in pipeline_type:
                validation_target_fps = 16.0
            elif is_wan2_2 and pipeline_config.get("fps") is not None:
                validation_target_fps = float(pipeline_config.get("fps"))

        if validation_runtime_adapt:
            if (
                not isinstance(validation_inference_size, (list, tuple))
                or len(validation_inference_size) != 2
            ):
                logger.warning(
                    "validation_runtime_adapt=True but validation_inference_size is missing/invalid. "
                    "Falling back to legacy validation behavior."
                )
                validation_runtime_adapt = False
            else:
                try:
                    validation_target_h = int(validation_inference_size[0])
                    validation_target_w = int(validation_inference_size[1])
                    if validation_target_h <= 0 or validation_target_w <= 0:
                        raise ValueError
                except Exception:
                    logger.warning(
                        f"Invalid validation_inference_size={validation_inference_size}. "
                        "Falling back to legacy validation behavior."
                    )
                    validation_runtime_adapt = False
                else:
                    validation_resize_to_target = True

            if validation_resize_mode != "short_side_pad":
                logger.warning(
                    f"Unsupported validation_resize_mode={validation_resize_mode}. "
                    "Using short_side_pad."
                )
                validation_resize_mode = "short_side_pad"

            if validation_fps_mode != "time_resample":
                logger.warning(
                    f"Unsupported validation_fps_mode={validation_fps_mode}. "
                    "Disabling FPS adaptation."
                )
                validation_target_fps = None

            if validation_runtime_adapt and is_wan2_2 and (validation_target_w % 32 != 0):
                logger.warning(
                    f"WAN 2.2 with validation_inference_size width={validation_target_w} is not divisible by 32. "
                    "This may degrade compatibility."
                )
        if (
            not validation_resize_to_target
            and len(inferred_validation_height_buckets) == 1
            and len(inferred_validation_width_buckets) == 1
        ):
            validation_target_h = inferred_validation_height_buckets[0]
            validation_target_w = inferred_validation_width_buckets[0]
            validation_resize_to_target = True
            logger.info(
                "Validation size fallback: using single training bucket %dx%d.",
                validation_target_h,
                validation_target_w,
            )
        elif (
            not validation_resize_to_target
            and (len(inferred_validation_height_buckets) > 1 or len(inferred_validation_width_buckets) > 1)
        ):
            logger.warning(
                "Validation uses multiple spatial buckets and validation_runtime_adapt is disabled. "
                "Skipping automatic validation resize; set validation_inference_size explicitly if needed."
            )

        validation_save_fps = 16
        if validation_runtime_adapt and validation_target_fps is not None:
            validation_save_fps = max(1, int(round(validation_target_fps)))
            logger.info(
                f"🎯 Validation runtime adaptation enabled: size={validation_inference_size}, "
                f"target_fps={validation_target_fps}, save_fps={validation_save_fps}"
            )
        elif validation_temporal_frame_skip > 1:
            logger.info(
                "Validation temporal frame skip enabled: frame_skip=%d",
                validation_temporal_frame_skip,
            )
        
        # Load validation set from config
        validation_stage = "validation_set_loading"
        validation_entries = data_config.get("validation_set")
        if validation_entries is None:
            logger.warning("No validation_set specified in config. Skipping validation.")
            return
        
        data_root = data_config.get("data_root", args.train_data_dir)
        static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
        exclude_videos_file = data_config.get("exclude_videos_file")
        excluded_videos = load_excluded_video_entries(data_root, exclude_videos_file)
        
        validation_sampling_mode = str(data_config.get("validation_sampling_mode", "combined")).strip().lower()
        if validation_sampling_mode not in {"combined", "per_file"}:
            raise ValueError(
                f"Unsupported validation_sampling_mode: {validation_sampling_mode}. "
                "Use one of: combined, per_file."
            )
        validation_missing_prompt_mode = str(
            data_config.get("validation_missing_prompt_mode", "skip")
        ).strip().lower()
        if validation_missing_prompt_mode not in {"skip", "empty"}:
            raise ValueError(
                f"Unsupported validation_missing_prompt_mode: {validation_missing_prompt_mode}. "
                "Use one of: skip, empty."
            )

        raw_validation_groups = _load_named_validation_sets(validation_entries, data_root)
        if not raw_validation_groups:
            logger.warning("No validation set files could be loaded. Skipping validation.")
            return

        if excluded_videos:
            original_validation_count = sum(len(paths) for _, paths in raw_validation_groups)
            raw_validation_groups = [
                (entry_name, [video for video in videos if video not in excluded_videos])
                for entry_name, videos in raw_validation_groups
            ]
            filtered_validation_count = sum(len(paths) for _, paths in raw_validation_groups)
            excluded_count = original_validation_count - filtered_validation_count
            if excluded_count:
                logger.info("Excluded %d validation videos using %s", excluded_count, exclude_videos_file)

        multiview_composition = bool(data_config.get("multiview_composition", False))
        multiview_width_concat = bool(data_config.get("multiview_width_concat", False))
        if multiview_composition and multiview_width_concat:
            raise ValueError("validation does not support enabling both multiview_composition and multiview_width_concat")

        composition_layout_spec = None
        width_concat_view_ids: Optional[List[int]] = None
        width_concat_target_height: Optional[int] = None
        width_concat_per_view_width: Optional[int] = None

        if multiview_composition:
            raw_layout = data_config.get("multiview_layout")
            if raw_layout is None:
                raise ValueError("multiview_composition=true requires multiview_layout for validation")
            composition_layout_spec = _validate_composition_layout(raw_layout)
        elif multiview_width_concat:
            multiview_views = data_config.get("multiview_views", ["ext1", "ext2"])
            parsed_view_ids: List[int] = []
            for view in multiview_views:
                if isinstance(view, int):
                    parsed_view_ids.append(int(view))
                    continue
                view_str = str(view).strip().lower()
                if view_str.startswith("ext") and view_str[3:].isdigit():
                    parsed_view_ids.append(int(view_str[3:]))
                else:
                    raise ValueError(
                        f"multiview_views entries must be of the form 'extN' for validation, got: {view}"
                    )
            inferred_heights = sorted(set(_normalize_bucket_values(data_config.get("height_buckets"))))
            if len(inferred_heights) != 1:
                raise ValueError(
                    "multiview_width_concat validation requires a single height_buckets value "
                    f"to match training preprocessing, got: {inferred_heights}"
                )
            width_concat_view_ids = parsed_view_ids
            width_concat_target_height = inferred_heights[0]
            width_concat_per_view_width = int(data_config.get("multiview_per_view_width", 832))

        validation_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
        for entry_name, videos in raw_validation_groups:
            grouped_samples = _build_validation_samples_for_mode(
                videos,
                multiview_composition=multiview_composition,
                composition_layout_spec=composition_layout_spec,
                multiview_width_concat=multiview_width_concat,
                width_concat_view_ids=width_concat_view_ids,
            )
            validation_groups.append((entry_name, grouped_samples))

        validation_pool_size = sum(len(paths) for _, paths in validation_groups)
        if validation_pool_size <= 0:
            logger.warning("No validation samples remained after grouping/filtering. Skipping validation.")
            return
        
        fsdp_validation = bool(getattr(args, "use_fsdp", False))

        # Limit validation samples
        max_validation_videos = data_config.get("max_validation_videos", 2)

        # DDP validates independent samples per rank. FSDP ranks must run the
        # same forward sequence, so do not multiply the validation pool.
        validation_rank_multiplier = 1 if fsdp_validation else accelerator.num_processes
        total_validation_videos = max_validation_videos * validation_rank_multiplier
        random_validation = bool(data_config.get("random_validation", False))

        if validation_sampling_mode == "per_file" and len(validation_groups) > 1:
            selected_validation_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
            for group_index, (entry_name, videos) in enumerate(validation_groups):
                entry_pool_size = len(videos)
                if entry_pool_size > total_validation_videos:
                    if random_validation:
                        entry_seed = int(global_step) + group_index * 1000003
                        entry_rng = random.Random(entry_seed)
                        selected_videos = entry_rng.sample(videos, total_validation_videos)
                        logger.info(
                            "Validation selection [%s]: per-file random sample enabled "
                            "(seed=%d, selected=%d/%d)",
                            entry_name,
                            entry_seed,
                            total_validation_videos,
                            entry_pool_size,
                        )
                    else:
                        selected_videos = videos[:total_validation_videos]
                        logger.info(
                            "Validation selection [%s]: per-file sequential prefix (selected=%d/%d)",
                            entry_name,
                            total_validation_videos,
                            entry_pool_size,
                        )
                else:
                    selected_videos = videos[:total_validation_videos]
                    logger.info(
                        "Validation selection [%s]: using full available set (selected=%d/%d)",
                        entry_name,
                        len(selected_videos),
                        entry_pool_size,
                    )
                selected_validation_groups.append((entry_name, selected_videos))
            validation_set = _interleave_validation_groups(selected_validation_groups)
            logger.info(
                "Validation selection: per-file mode enabled across %d validation files "
                "(combined selected=%d/%d)",
                len(selected_validation_groups),
                len(validation_set),
                validation_pool_size,
            )
        else:
            validation_set = [video for _, videos in validation_groups for video in videos]
            # Select validation videos (randomized or sequential)
            if len(validation_set) > total_validation_videos:
                if random_validation:
                    validation_seed = int(global_step)
                    validation_rng = random.Random(validation_seed)
                    validation_set = validation_rng.sample(validation_set, total_validation_videos)
                    logger.info(
                        "Validation selection: random sample enabled (seed=%d, selected=%d/%d)",
                        validation_seed,
                        total_validation_videos,
                        validation_pool_size,
                    )
                else:
                    validation_set = validation_set[:total_validation_videos]
                    logger.info(
                        "Validation selection: sequential prefix (selected=%d/%d)",
                        total_validation_videos,
                        validation_pool_size,
                    )
            else:
                validation_set = validation_set[:total_validation_videos]
                logger.info(
                    "Validation selection: using full available set (selected=%d/%d)",
                    len(validation_set),
                    validation_pool_size,
                )
        
        if fsdp_validation:
            videos_per_gpu = len(validation_set)
            start_idx = 0
            end_idx = len(validation_set)
            logger.info(
                "🎯 FSDP validation strategy: all %d ranks process the same %d sample(s); "
                "only the main rank writes files.",
                accelerator.num_processes,
                len(validation_set),
            )
        else:
            # Distribute videos across GPUs - each GPU gets different videos
            videos_per_gpu = len(validation_set) // accelerator.num_processes
            start_idx = accelerator.process_index * videos_per_gpu
            end_idx = start_idx + videos_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(validation_set)

            # Each GPU gets its own subset of videos
            validation_set = validation_set[start_idx:end_idx]
        
        if not validation_set:
            logger.warning("No validation videos found. Skipping validation.")
            return
        
        if not fsdp_validation:
            logger.info(f"🎯 Multi-GPU Validation Strategy:")
            logger.info(f"   - Total GPUs: {accelerator.num_processes}")
            logger.info(f"   - Videos per GPU: {videos_per_gpu}")
            logger.info(f"   - GPU {accelerator.process_index}: Processing {len(validation_set)} validation videos (indices {start_idx}-{end_idx-1})")

        # Determine conditioning mode
        conditioning_mode = training_config.get("conditioning_mode", "dwm")
        i2v_use_hand_condition = bool(training_config.get("i2v_use_hand_condition", False))
        i2v_condition_latent_type = str(
            training_config.get("i2v_condition_latent_type", "image")
        ).strip().lower()
        use_diffusers_i2v_backend = bool(custom_settings.get("use_diffusers_i2v_backend", False))
        transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
        is_wan2_2 = "2.2" in pipeline_config.get("type", "")
        is_fun_control_transformer_for_val = _is_fun_control_transformer_type(transformer_type)
        is_wan2_2_fun_control_i2v = (
            pipeline_type == "wan2.2_fun_control_i2v_mesh"
            or transformer_type == "Wan2_2FunControlTransformer3DModelWithConcat"
        )
        lightx2v_allowed = conditioning_mode == "i2v" and use_diffusers_i2v_backend

        # Reuse the training transformer for validation to avoid duplicating 14B weights on GPU.
        validation_stage = "model_build"
        transformer_state_owner = accelerator.unwrap_model(transformer3d)
        transformer_for_val_was_training = transformer_state_owner.training
        transformer_state_owner.eval()
        transformer_for_val = transformer3d if getattr(args, "use_fsdp", False) else transformer_state_owner
        if transformer_for_val is not transformer_state_owner:
            transformer_for_val.eval()
        torch.cuda.empty_cache()
        logger.info(
            "Validation transformer reuse: allocated=%.2f GiB, reserved=%.2f GiB",
            torch.cuda.memory_allocated() / (1024 ** 3),
            torch.cuda.memory_reserved() / (1024 ** 3),
        )

        validation_stage = "pipeline_build"
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
        )

        if is_wan2_2_fun_control_i2v:
            logger.info("🔧 Using WAN 2.2 Fun-Control pipeline for validation")
            Wan2_2FunControlPipeline = _require_wan2_2_fun_control_pipeline_class()
            pipeline = Wan2_2FunControlPipeline(
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer_for_val,
                transformer_2=None,
                scheduler=scheduler,
            )
        elif is_wan2_2:
            # WAN 2.2: No CLIP encoder needed
            logger.info("🔧 Using WAN 2.2 pipeline for validation")
            _transformer_2_for_val = None
            _t2_subpath = getattr(args, 'transformer_2_subpath', None)
            if _t2_subpath:
                _t2_path = os.path.join(args.pretrained_model_name_or_path, _t2_subpath)
                if os.path.isdir(_t2_path):
                    try:
                        logger.info("🔧 Loading WAN 2.2 high-noise transformer for validation: %s", _t2_path)
                        _transformer_2_for_val = WanTransformer3DModelWithConcat.from_pretrained(
                            _t2_path,
                            transformer_additional_kwargs={
                                "condition_channels": 0,
                                "is_wan2_2": True,
                                "model_type": "i2v",
                            },
                            torch_dtype=weight_dtype,
                        ).eval()
                        _transformer_2_for_val.requires_grad_(False)
                        _transformer_2_for_val = _transformer_2_for_val.cpu()
                        logger.info("✅ High-noise transformer loaded on CPU for per-step expert offloading")
                    except Exception as _e:
                        logger.warning("⚠️ Failed to load high-noise transformer, falling back to low-noise only: %s", _e)
                        _transformer_2_for_val = None
                else:
                    logger.warning("⚠️ transformer_2_subpath '%s' not found at %s", _t2_subpath, _t2_path)
            pipeline = Wan2_2FunInpaintHandConcatPipeline(
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer_for_val,
                transformer_2=_transformer_2_for_val,
                scheduler=scheduler,
            )
            if _transformer_2_for_val is not None:
                pipeline._expert_offload = True
        else:
            # WAN 2.1: CLIP encoder required
            clip_image_encoder_for_pipeline = clip_image_encoder
            if clip_image_encoder_for_pipeline is None:
                logger.warning("⚠️  Creating dummy CLIP encoder for pipeline compatibility")
                # Create a minimal dummy CLIP encoder
                clip_image_encoder_for_pipeline = CLIPModel()
                clip_image_encoder_for_pipeline.eval()
                clip_image_encoder_for_pipeline.requires_grad_(False)
            
            if pipeline_type == "wan2.1_fun_inp":
                logger.info("🔧 Using WAN 2.1 base pipeline for validation")
                pipeline = WanFunInpaintPipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer_for_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            elif pipeline_type == "wan2.1_fun_inp_hand_concat":
                logger.info("🔧 Using WAN 2.1 hand concat pipeline for validation")
                pipeline = WanFunInpaintHandConcatPipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer_for_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            elif pipeline_type == "wan2.1_fun_inp_hand_vace":
                logger.info("🔧 Using WAN 2.1 VACE pipeline for validation")
                pipeline = WanFunInpaintHandVacePipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer_for_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            else:
                raise ValueError(f"Unsupported WAN 2.1 pipeline type: {pipeline_type}")
        # All neural pipeline modules are shared with the training graph and
        # already carry the dtype chosen in main():
        #   - transformer: FSDP-wrapped (or DeepSpeed-prepared); calling
        #     `pipeline.to(..., dtype=bf16)` on the wrapper can corrupt its
        #     internal flat-param state.
        #   - vae / image_encoder (i2v backend): kept in fp32 in main(); a
        #     bf16 round-trip would in-place truncate the mantissa to 7 bits.
        # Don't recast dtype here. But we still need to make sure each module
        # is on the validation device -- when prompt_mode=empty or various
        # offload paths, text_encoder/vae can sit on CPU even though
        # `text_encoder` was reactivated for validation a few lines above.
        for module_name in ("vae", "text_encoder", "image_encoder"):
            module = getattr(pipeline, module_name, None)
            if module is None:
                continue
            try:
                current_device = next(module.parameters()).device
            except StopIteration:
                continue
            if current_device != accelerator.device:
                module.to(accelerator.device)
        if getattr(args, "use_fsdp", False):
            # The pipeline must dispatch through the FSDP wrapper so it can
            # all-gather sharded Conv3d/Linear weights during sampling.
            pipeline.register_modules(transformer=transformer3d)
            logger.info("Validation transformer call path: using FSDP wrapper")
        validation_guidance_scale = float(custom_settings.get("validation_guidance_scale", 5.0))
        validation_num_inference_steps_raw = custom_settings.get("validation_num_inference_steps")
        validation_num_inference_steps = (
            int(validation_num_inference_steps_raw)
            if validation_num_inference_steps_raw is not None
            else None
        )
        validation_negative_prompt = str(custom_settings.get("validation_negative_prompt", ""))
        validation_attention_kwargs = None
        if validation_lightx2v:
            if not validation_lightx2v_lora_path:
                raise ValueError(
                    "validation_lightx2v=true requires a non-empty validation_lightx2v_lora_path."
                )

            peft_capable = hasattr(transformer_for_val, "load_lora_adapter") or hasattr(transformer_for_val, "add_adapter")
            logger.info(
                "Validation transformer PEFT capability check (i2v diffusers backend): %s (%s)",
                peft_capable,
                type(transformer_for_val).__name__,
            )
            if lightx2v_allowed and not peft_capable:
                raise TypeError(
                    f"Expected a PEFT-capable transformer for i2v diffusers backend, got {type(transformer_for_val).__name__}"
                )

            logger.info("Loading LightX2V LoRA for validation: %s", validation_lightx2v_lora_path)
            logger.info(
                "LightX2V compat settings: mode=%s, strict=%s",
                validation_lightx2v_compat_mode,
                validation_lightx2v_compat_strict,
            )
            lightx2v_method, lightx2v_stats = load_lightx2v_with_fallback(
                pipeline=pipeline,
                lora_path=validation_lightx2v_lora_path,
                lora_scale=validation_lightx2v_lora_scale,
                compat_mode=validation_lightx2v_compat_mode,
                compat_strict=validation_lightx2v_compat_strict,
                expected_blocks=None,
                apply_non_block_diff=validation_lightx2v_apply_non_block_diff,
                logger=logger,
            )
            logger.info("LightX2V load method: %s", lightx2v_method)
            if lightx2v_method == "compat":
                logger.info(
                    "LightX2V compat stats: lora_pairs=%s diff=%s diff_b=%s",
                    lightx2v_stats.get("applied_lora_pairs"),
                    lightx2v_stats.get("applied_diff"),
                    lightx2v_stats.get("applied_diff_b"),
                )
            validation_attention_kwargs = {"scale": validation_lightx2v_lora_scale}
            logger.info(
                "LightX2V scale will be applied via attention_kwargs: %s",
                validation_attention_kwargs,
            )

            if hasattr(pipeline, "get_active_adapters"):
                logger.info("Validation pipeline active adapters: %s", pipeline.get_active_adapters())

            logger.info(
                "LightX2V validation enabled. Overriding guidance_scale %.1f -> 1.0 and num_inference_steps -> 4",
                validation_guidance_scale,
            )
            validation_guidance_scale = 1.0
            validation_num_inference_steps = 4

        if validation_lightx2v and validation_attention_kwargs is not None:
            logger.info("LightX2V attention scale path: attention_kwargs.")

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        
        # Get subdirectory names from config
        prompt_subdir = data_config.get("prompt_subdir", "prompts")
        prompt_json_fields = _resolve_prompt_json_fields(
            data_config.get("prompt_json_fields"),
            prompt_subdir,
        )
        robot_mesh_video_subdir = data_config.get("robot_mesh_video_subdir", "videos_hands")
        static_video_subdir = data_config.get("static_video_subdir", "videos_static")
        static_disparity_subdir = data_config.get("static_disparity_subdir")
        eef_depth_subdir = data_config.get("eef_depth_subdir")
        robot_mesh_video_latents_subdir = data_config.get("robot_mesh_video_latents_subdir")
        static_disparity_npz_mode = data_config.get("static_disparity_npz_mode")
        eef_depth_npz_mode = data_config.get("eef_depth_npz_mode")
        state_adaln_source = data_config.get("state_adaln_source")
        state_adaln_temporal_mode = data_config.get("state_adaln_temporal_mode", "sample4")
        state_adaln_joint_filename = data_config.get("state_adaln_joint_filename", "joint_position_isaac_nominal.npy")
        state_adaln_gripper_filename = data_config.get("state_adaln_gripper_filename", "gripper_position_isaac_nominal.npy")
        static_disparity_i2v_first_frame_only = (
            bool(data_config.get("static_disparity_i2v_first_frame_only", False))
            and conditioning_mode == "i2v"
        )
        prompt_ext = ".json" if prompt_json_fields else ".txt"

        # Run validation for each sample
        logged_attention_kwargs_passthrough = False
        for i, validation_sample in enumerate(validation_set):
            validation_stage = f"sampling:{i+1}/{len(validation_set)}"
            sample_kind = str(validation_sample.get("kind", "single"))
            prompt_candidate_video_paths: List[str] = []
            source_fps_path: Optional[Path] = None
            hand_latents = None

            if sample_kind == "single":
                video_rel_path = str(validation_sample["video_path"])
                video_path_obj = Path(video_rel_path)
                video_name = video_path_obj.stem
                video_save_id = _build_validation_video_save_id(validation_sample["save_key"])
                full_video_path = Path(data_root) / video_rel_path
                source_fps_path = full_video_path
                prompt_candidate_video_paths = [video_rel_path]

                hand_video_path = _derive_validation_condition_path(
                    video_rel_path, data_root, robot_mesh_video_subdir
                )
                static_video_path = _derive_validation_condition_path(
                    video_rel_path, data_root, static_video_subdir
                )
                static_disparity_video_path = _derive_validation_condition_path(
                    video_rel_path, data_root, static_disparity_subdir
                )
                eef_depth_video_path = _derive_validation_condition_path(
                    video_rel_path, data_root, eef_depth_subdir
                )

                gt_video = _safe_read_validation_video(full_video_path, "ground-truth")
                if gt_video is None:
                    continue

                if conditioning_mode == "dwm":
                    # NOTE: pcd_from_ext_first_frame validation uses the same
                    # copy_first_frame fallback. Matching the training signal
                    # would require running the pcd renderer here too;
                    # revisit if validation quality is needed.
                    if static_video_source_mode in {"copy_first_frame", "pcd_from_ext_first_frame"}:
                        condition_video_np = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()
                    else:
                        if static_video_path is None or not static_video_path.exists():
                            logger.warning(f"Static video not found: {static_video_path}. Skipping.")
                            continue
                        condition_video_np = _safe_read_validation_video(static_video_path, "static condition")
                        if condition_video_np is None:
                            continue
                elif conditioning_mode == "i2v":
                    condition_video_np = gt_video.copy()
                    condition_video_np[1:] = 0.0
                else:
                    raise ValueError(f"Unsupported conditioning_mode for validation: {conditioning_mode}")

                use_hand_for_validation = (conditioning_mode == "dwm") or i2v_use_hand_condition
                if use_hand_for_validation and hand_video_path is not None and hand_video_path.exists():
                    hand_video_np = _safe_read_validation_video(hand_video_path, "hand condition")
                else:
                    hand_video_np = None
                    if use_hand_for_validation:
                        hand_latents = _load_validation_condition_latents(
                            hand_video_path,
                            robot_mesh_video_latents_subdir,
                            "hand condition",
                        )

                if static_disparity_video_path is not None:
                    if static_disparity_npz_mode:
                        # NPZ runtime decode: derive .npz path from the .mp4
                        # path returned by _derive_validation_condition_path.
                        static_disparity_video_np = _safe_read_validation_disparity_npz(
                            static_disparity_video_path, "static disparity",
                            mode=str(static_disparity_npz_mode),
                            num_target_frames=int(gt_video.shape[0]),
                        )
                        if static_disparity_video_np is None:
                            continue
                    elif static_disparity_video_path.exists():
                        static_disparity_video_np = _safe_read_validation_video(
                            static_disparity_video_path, "static disparity"
                        )
                        if static_disparity_video_np is None:
                            continue
                    else:
                        logger.warning(
                            f"Static disparity video not found: {static_disparity_video_path}. Skipping."
                        )
                        continue
                else:
                    static_disparity_video_np = None

                if eef_depth_video_path is not None:
                    if eef_depth_npz_mode:
                        eef_depth_video_np = _safe_read_validation_disparity_npz(
                            eef_depth_video_path, "hand disparity",
                            mode=str(eef_depth_npz_mode),
                            num_target_frames=int(gt_video.shape[0]),
                        )
                        if eef_depth_video_np is None:
                            continue
                    elif eef_depth_video_path.exists():
                        eef_depth_video_np = _safe_read_validation_video(
                            eef_depth_video_path, "hand disparity"
                        )
                        if eef_depth_video_np is None:
                            continue
                    else:
                        logger.warning(
                            f"Hand disparity video not found: {eef_depth_video_path}. Skipping."
                        )
                        continue
                else:
                    eef_depth_video_np = None

            elif sample_kind == "multiview_width_concat":
                pair_key = str(validation_sample["pair_key"])
                video_path_obj = Path(pair_key)
                video_name = video_path_obj.name
                video_save_id = _build_validation_video_save_id(validation_sample["save_key"])
                prompt_candidate_video_paths = [
                    str(validation_sample["views"][view_id])
                    for view_id in validation_sample["view_ids"]
                ]
                if prompt_candidate_video_paths:
                    source_fps_path = Path(data_root) / prompt_candidate_video_paths[0]

                gt_video = _concat_validation_multiview_views(
                    sample=validation_sample,
                    view_ids=validation_sample["view_ids"],
                    data_root=data_root,
                    target_height=width_concat_target_height,
                    per_view_width=width_concat_per_view_width,
                    condition_folder=None,
                    role_name="ground-truth",
                    is_target=True,
                )
                if gt_video is None:
                    continue

                if conditioning_mode == "dwm":
                    # NOTE: pcd_from_ext_first_frame validation uses the same
                    # copy_first_frame fallback. Matching the training signal
                    # would require running the pcd renderer here too;
                    # revisit if validation quality is needed.
                    if static_video_source_mode in {"copy_first_frame", "pcd_from_ext_first_frame"}:
                        condition_video_np = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()
                    else:
                        condition_video_np = _concat_validation_multiview_views(
                            sample=validation_sample,
                            view_ids=validation_sample["view_ids"],
                            data_root=data_root,
                            target_height=width_concat_target_height,
                            per_view_width=width_concat_per_view_width,
                            condition_folder=static_video_subdir,
                            role_name="static condition",
                            is_target=False,
                        )
                        if condition_video_np is None:
                            continue
                elif conditioning_mode == "i2v":
                    condition_video_np = gt_video.copy()
                    condition_video_np[1:] = 0.0
                else:
                    raise ValueError(f"Unsupported conditioning_mode for validation: {conditioning_mode}")

                use_hand_for_validation = (conditioning_mode == "dwm") or i2v_use_hand_condition
                if use_hand_for_validation and robot_mesh_video_subdir:
                    hand_video_np = _concat_validation_multiview_views(
                        sample=validation_sample,
                        view_ids=validation_sample["view_ids"],
                        data_root=data_root,
                        target_height=width_concat_target_height,
                        per_view_width=width_concat_per_view_width,
                        condition_folder=robot_mesh_video_subdir,
                        role_name="hand condition",
                        is_target=False,
                    )
                else:
                    hand_video_np = None

                static_disparity_video_np = None
                if static_disparity_subdir:
                    static_disparity_video_np = _concat_validation_multiview_views(
                        sample=validation_sample,
                        view_ids=validation_sample["view_ids"],
                        data_root=data_root,
                        target_height=width_concat_target_height,
                        per_view_width=width_concat_per_view_width,
                        condition_folder=static_disparity_subdir,
                        role_name="static disparity",
                        is_target=False,
                    )
                    if static_disparity_video_np is None:
                        continue

                eef_depth_video_np = None
                if eef_depth_subdir:
                    eef_depth_video_np = _concat_validation_multiview_views(
                        sample=validation_sample,
                        view_ids=validation_sample["view_ids"],
                        data_root=data_root,
                        target_height=width_concat_target_height,
                        per_view_width=width_concat_per_view_width,
                        condition_folder=eef_depth_subdir,
                        role_name="hand disparity",
                        is_target=False,
                    )
                    if eef_depth_video_np is None:
                        continue

            elif sample_kind == "multiview_composition":
                pair_key = str(validation_sample["pair_key"])
                video_path_obj = Path(pair_key)
                video_name = video_path_obj.name
                cell_suffix_picks = _pick_validation_composition_cell_suffixes(
                    validation_sample,
                    composition_layout_spec,
                    global_step,
                )
                selected_source_suffixes: List[str] = []
                for cell_idx, cell in enumerate(composition_layout_spec["cells"]):
                    if "fill" in cell:
                        continue
                    suffix = cell_suffix_picks[cell_idx]
                    if suffix not in selected_source_suffixes:
                        selected_source_suffixes.append(suffix)
                        prompt_candidate_video_paths.append(str(validation_sample["views"][suffix]))
                if prompt_candidate_video_paths:
                    source_fps_path = Path(data_root) / prompt_candidate_video_paths[0]
                if selected_source_suffixes:
                    video_save_key = f"{validation_sample['save_key']}__{'-'.join(selected_source_suffixes)}"
                else:
                    video_save_key = str(validation_sample["save_key"])
                video_save_id = _build_validation_video_save_id(video_save_key)

                gt_video = _compose_validation_multiview_canvas(
                    sample=validation_sample,
                    layout_spec=composition_layout_spec,
                    cell_suffix_picks=cell_suffix_picks,
                    data_root=data_root,
                    condition_key="video",
                    condition_folder=None,
                    role_name="ground-truth",
                    is_target=True,
                )
                if gt_video is None:
                    continue

                if conditioning_mode == "dwm":
                    if static_video_source_mode == "copy_first_frame":
                        condition_video_np = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()
                    elif static_video_source_mode == "pcd_from_ext_first_frame":
                        condition_video_np = _render_wrist_pcd_validation_canvas(
                            sample=validation_sample,
                            layout_spec=composition_layout_spec,
                            cell_suffix_picks=cell_suffix_picks,
                            data_root=data_root,
                            gt_video=gt_video,
                            wrist_pcd_config=data_config.get("wrist_pcd_config"),
                        )
                    else:
                        condition_video_np = _compose_validation_multiview_canvas(
                            sample=validation_sample,
                            layout_spec=composition_layout_spec,
                            cell_suffix_picks=cell_suffix_picks,
                            data_root=data_root,
                            condition_key="static_video",
                            condition_folder=static_video_subdir,
                            role_name="static condition",
                            is_target=False,
                        )
                        if condition_video_np is None:
                            continue
                elif conditioning_mode == "i2v":
                    condition_video_np = gt_video.copy()
                    condition_video_np[1:] = 0.0
                else:
                    raise ValueError(f"Unsupported conditioning_mode for validation: {conditioning_mode}")

                use_hand_for_validation = (conditioning_mode == "dwm") or i2v_use_hand_condition
                if use_hand_for_validation and robot_mesh_video_subdir:
                    hand_video_np = _compose_validation_multiview_canvas(
                        sample=validation_sample,
                        layout_spec=composition_layout_spec,
                        cell_suffix_picks=cell_suffix_picks,
                        data_root=data_root,
                        condition_key="hand_video",
                        condition_folder=robot_mesh_video_subdir,
                        role_name="hand condition",
                        is_target=False,
                    )
                else:
                    hand_video_np = None

                static_disparity_video_np = None
                if static_disparity_subdir:
                    static_disparity_video_np = _compose_validation_multiview_canvas(
                        sample=validation_sample,
                        layout_spec=composition_layout_spec,
                        cell_suffix_picks=cell_suffix_picks,
                        data_root=data_root,
                        condition_key="static_disparity",
                        condition_folder=static_disparity_subdir,
                        role_name="static disparity",
                        is_target=False,
                    )
                    if static_disparity_video_np is None:
                        continue

                eef_depth_video_np = None
                if eef_depth_subdir:
                    eef_depth_video_np = _compose_validation_multiview_canvas(
                        sample=validation_sample,
                        layout_spec=composition_layout_spec,
                        cell_suffix_picks=cell_suffix_picks,
                        data_root=data_root,
                        condition_key="eef_depth",
                        condition_folder=eef_depth_subdir,
                        role_name="hand disparity",
                        is_target=False,
                    )
                    if eef_depth_video_np is None:
                        continue

            else:
                raise ValueError(f"Unsupported validation sample kind: {sample_kind}")

            if validation_prompt_mode == PROMPT_MODE_EMPTY:
                prompt_text = ""
            else:
                prompt_text = _resolve_validation_prompt_text(
                    prompt_candidate_video_paths,
                    data_root=data_root,
                    prompt_subdir=prompt_subdir,
                    prompt_ext=prompt_ext,
                    prompt_json_fields=prompt_json_fields,
                    validation_missing_prompt_mode=validation_missing_prompt_mode,
                )
                if prompt_text is None:
                    continue

            if validation_runtime_adapt:
                source_fps = _read_video_fps(str(source_fps_path)) if source_fps_path is not None else None
                if source_fps is None and validation_target_fps is not None:
                    logger.warning(
                        f"Could not read FPS from {source_fps_path}. Skipping temporal FPS adaptation for this sample."
                    )

                if (
                    validation_target_fps is not None
                    and source_fps is not None
                    and validation_fps_mode == "time_resample"
                ):
                    temporal_lengths = [gt_video.shape[0], condition_video_np.shape[0]]
                    if hand_video_np is not None:
                        temporal_lengths.append(hand_video_np.shape[0])
                    if static_disparity_video_np is not None:
                        temporal_lengths.append(static_disparity_video_np.shape[0])
                    if eef_depth_video_np is not None:
                        temporal_lengths.append(eef_depth_video_np.shape[0])
                    common_len = min(temporal_lengths)
                    if common_len <= 0:
                        logger.warning(
                            f"Invalid frame length detected for {video_name}. Skipping temporal FPS adaptation."
                        )
                    else:
                        if gt_video.shape[0] != common_len:
                            gt_video = gt_video[:common_len]
                        if condition_video_np.shape[0] != common_len:
                            condition_video_np = condition_video_np[:common_len]
                        if hand_video_np is not None and hand_video_np.shape[0] != common_len:
                            hand_video_np = hand_video_np[:common_len]
                        if static_disparity_video_np is not None and static_disparity_video_np.shape[0] != common_len:
                            static_disparity_video_np = static_disparity_video_np[:common_len]
                        if eef_depth_video_np is not None and eef_depth_video_np.shape[0] != common_len:
                            eef_depth_video_np = eef_depth_video_np[:common_len]

                        frame_indices = _build_time_resample_indices(
                            src_len=common_len,
                            src_fps=source_fps,
                            tgt_fps=validation_target_fps,
                        )
                        gt_video = _apply_frame_indices(gt_video, frame_indices, layout_spec="FHWC")
                        condition_video_np = _apply_frame_indices(condition_video_np, frame_indices, layout_spec="FHWC")
                        if hand_video_np is not None:
                            hand_video_np = _apply_frame_indices(hand_video_np, frame_indices, layout_spec="FHWC")
                        if static_disparity_video_np is not None:
                            static_disparity_video_np = _apply_frame_indices(
                                static_disparity_video_np, frame_indices, layout_spec="FHWC"
                            )
                        if eef_depth_video_np is not None:
                            eef_depth_video_np = _apply_frame_indices(
                                eef_depth_video_np, frame_indices, layout_spec="FHWC"
                            )

            try:
                (
                    gt_video,
                    condition_video_np,
                    hand_video_np,
                    static_disparity_video_np,
                    eef_depth_video_np,
                ) = _align_and_normalize_validation_videos(
                    gt_video=gt_video,
                    condition_video_np=condition_video_np,
                    hand_video_np=hand_video_np,
                    static_disparity_video_np=static_disparity_video_np,
                    eef_depth_video_np=eef_depth_video_np,
                    target_num_frames=validation_target_num_frames,
                    frame_skip=validation_temporal_frame_skip,
                )
            except ValueError as e:
                logger.warning(
                    "Failed to normalize validation sample %s temporally: %s. Skipping.",
                    video_name,
                    e,
                )
                continue

            if validation_resize_to_target:
                gt_video = _resize_and_pad_video(
                    gt_video,
                    target_h=validation_target_h,
                    target_w=validation_target_w,
                    pad_value=validation_pad_value,
                    short_side_mode=(validation_resize_mode == "short_side_pad"),
                    center_pad=validation_center_pad,
                    layout_spec="FHWC",
                )
                condition_video_np = _resize_and_pad_video(
                    condition_video_np,
                    target_h=validation_target_h,
                    target_w=validation_target_w,
                    pad_value=validation_pad_value,
                    short_side_mode=(validation_resize_mode == "short_side_pad"),
                    center_pad=validation_center_pad,
                    layout_spec="FHWC",
                )
                if hand_video_np is not None:
                    hand_video_np = _resize_and_pad_video(
                        hand_video_np,
                        target_h=validation_target_h,
                        target_w=validation_target_w,
                        pad_value=validation_pad_value,
                        short_side_mode=(validation_resize_mode == "short_side_pad"),
                        center_pad=validation_center_pad,
                        layout_spec="FHWC",
                    )
                if static_disparity_video_np is not None:
                    static_disparity_video_np = _resize_and_pad_video(
                        static_disparity_video_np,
                        target_h=validation_target_h,
                        target_w=validation_target_w,
                        pad_value=validation_pad_value,
                        short_side_mode=(validation_resize_mode == "short_side_pad"),
                        center_pad=validation_center_pad,
                        layout_spec="FHWC",
                    )
                if eef_depth_video_np is not None:
                    eef_depth_video_np = _resize_and_pad_video(
                        eef_depth_video_np,
                        target_h=validation_target_h,
                        target_w=validation_target_w,
                        pad_value=validation_pad_value,
                        short_side_mode=(validation_resize_mode == "short_side_pad"),
                        center_pad=validation_center_pad,
                        layout_spec="FHWC",
                    )

            # Rebuild first-frame-only condition after all runtime adaptations.
            if conditioning_mode == "i2v":
                i2v_condition = np.zeros_like(gt_video)
                i2v_condition[0] = gt_video[0]
                condition_video_np = i2v_condition
                if static_disparity_i2v_first_frame_only and static_disparity_video_np is not None:
                    static_disparity_i2v = np.zeros_like(static_disparity_video_np)
                    static_disparity_i2v[0] = static_disparity_video_np[0]
                    static_disparity_video_np = static_disparity_i2v

            condition_video = torch.from_numpy(condition_video_np).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
            if hand_video_np is not None:
                hand_video = torch.from_numpy(hand_video_np).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
            else:
                hand_video = None
            static_disparity_video = (
                torch.from_numpy(static_disparity_video_np).permute(3, 0, 1, 2).unsqueeze(0)
                if static_disparity_video_np is not None else None
            )
            eef_depth_video = (
                torch.from_numpy(eef_depth_video_np).permute(3, 0, 1, 2).unsqueeze(0)
                if eef_depth_video_np is not None else None
            )

            # Align width to 32 for WAN 2.2 compatibility (legacy path only)
            if is_wan2_2 and not validation_resize_to_target:
                # Get original dimensions
                num_frames = gt_video.shape[0]
                orig_height, orig_width = gt_video.shape[1], gt_video.shape[2]
                
                # Calculate aligned width (round to nearest multiple of 32)
                aligned_width = round(orig_width / 32) * 32
                
                if aligned_width != orig_width:
                    # Resize GT video: [f, h, w, c] -> resize -> [f, h, w, c]
                    gt_video_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2)  # [f, c, h, w]
                    gt_video_tensor = F.interpolate(
                        gt_video_tensor, 
                        size=(orig_height, aligned_width), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    gt_video = gt_video_tensor.permute(0, 2, 3, 1).numpy()  # [f, h, w, c]
                    
                    # Resize condition video: [1, c, f, h, w] -> [1, c, f, h, w]
                    condition_video = F.interpolate(
                        condition_video.squeeze(0),  # [c, f, h, w]
                        size=(orig_height, aligned_width),
                        mode='bilinear',
                        align_corners=False
                    ).unsqueeze(0)  # [1, c, f, h, w]
                    
                    # Resize hand video if exists: [1, c, f, h, w] -> [1, c, f, h, w]
                    if hand_video is not None:
                        hand_video = F.interpolate(
                            hand_video.squeeze(0),  # [c, f, h, w]
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)  # [1, c, f, h, w]
                    if static_disparity_video is not None:
                        static_disparity_video = F.interpolate(
                            static_disparity_video.squeeze(0),
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)
                    if eef_depth_video is not None:
                        eef_depth_video = F.interpolate(
                            eef_depth_video.squeeze(0),
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)
                    
                    height, width = orig_height, aligned_width
                else:
                    height, width = orig_height, orig_width
            else:
                num_frames = gt_video.shape[0]
                height, width = gt_video.shape[1], gt_video.shape[2]

            state_adaln_tensor = None
            if state_adaln_source:
                if sample_kind != "single":
                    raise ValueError("state_adaln validation currently supports single-view samples only")
                state_adaln_tensor = load_robot_state_condition(
                    full_video_path,
                    source=state_adaln_source,
                    clip_plan={
                        "target_num_frames": int(num_frames),
                        "start_idx": 0,
                        "frame_skip": int(validation_temporal_frame_skip),
                    },
                    joint_filename=state_adaln_joint_filename,
                    gripper_filename=state_adaln_gripper_filename,
                    action_sidecar_subdir=data_config.get("state_adaln_action_sidecar_subdir", "action_values_clips81"),
                    union_droid_dim=data_config.get("state_adaln_union_droid_dim", 7),
                    union_robocasa_dim=data_config.get("state_adaln_union_robocasa_dim", 24),
                )
            
            prompt_variants = get_validation_prompt_variants(prompt_text, validation_prompt_mode)
            i2v_static_video_latents = None
            i2v_mask_latents = None
            if (
                conditioning_mode == "i2v"
                and i2v_condition_latent_type == "image"
                and isinstance(pipeline, WanFunInpaintHandConcatPipeline)
            ):
                with torch.no_grad():
                    i2v_static_video_latents, i2v_mask_latents = build_i2v_image_condition_latents_for_validation(
                        pipeline=pipeline,
                        condition_video=condition_video.to(accelerator.device, dtype=weight_dtype),
                        height=height,
                        width=width,
                        weight_dtype=weight_dtype,
                        device=accelerator.device,
                        generator=generator,
                    )
            with torch.no_grad():
                validation_autocast = (
                    nullcontext()
                    if use_diffusers_i2v_backend
                    else torch.autocast("cuda", dtype=weight_dtype)
                )
                with validation_autocast:
                    for prompt_variant_label, validation_prompt_text in prompt_variants:
                        mask_video = torch.zeros(1, 1, num_frames, height, width)
                        clip_image_for_i2v = None
                        if conditioning_mode == "i2v":
                            # Match original VideoX-Fun behavior: always build CLIP conditioning from
                            # the first frame for I2V validation when a real first frame is available.
                            first_frame = condition_video[0, :, 0].permute(1, 2, 0).cpu().numpy()
                            first_frame = np.clip(first_frame * 255.0, 0, 255).astype(np.uint8)
                            clip_image_for_i2v = Image.fromarray(first_frame)
                        common_sampling_kwargs = {
                            "num_frames": num_frames,
                            "negative_prompt": validation_negative_prompt,
                            "height": height,
                            "width": width,
                            "guidance_scale": validation_guidance_scale,
                            "generator": generator,
                        }
                        if validation_num_inference_steps is not None:
                            common_sampling_kwargs["num_inference_steps"] = validation_num_inference_steps
                        if validation_attention_kwargs is not None:
                            common_sampling_kwargs["attention_kwargs"] = validation_attention_kwargs
                            if not logged_attention_kwargs_passthrough and lightx2v_allowed:
                                logger.info(
                                    "Validation sampling uses attention_kwargs passthrough for LightX2V: %s",
                                    validation_attention_kwargs,
                                )
                                logged_attention_kwargs_passthrough = True

                        # Pre-encode text and offload T5 to CPU before VAE encoding.
                        # WAN T5 (11 GB) + transformer (31 GB) + misc already use ~66 GB on a
                        # single 80 GB GPU.  VAE encoding 81-frame 480×832 conditioning video
                        # then needs ~8 GB for a Conv3d intermediate → OOM at prepare_mask_latents.
                        # Encode text here, move T5 to CPU, pass prompt_embeds to pipeline so the
                        # text encoder is never on GPU during VAE ops.
                        _call_prompt = validation_prompt_text
                        if getattr(pipeline, "text_encoder", None) is not None:
                            try:
                                _do_cfg_pre = validation_guidance_scale > 1.0
                                # The validation loop offloads T5 after each sample to keep
                                # 80GB GPUs below the OOM line. Bring it back before the next
                                # pre-encode; otherwise token ids land on CUDA while embedding
                                # weights remain on CPU.
                                pipeline.text_encoder.to(accelerator.device)
                                with torch.no_grad():
                                    _pre_pos, _pre_neg = pipeline.encode_prompt(
                                        prompt=validation_prompt_text,
                                        negative_prompt=validation_negative_prompt if _do_cfg_pre else None,
                                        do_classifier_free_guidance=_do_cfg_pre,
                                        num_videos_per_prompt=1,
                                        max_sequence_length=512,
                                        device=accelerator.device,
                                    )
                                pipeline.text_encoder.to("cpu")
                                gc.collect()
                                torch.cuda.empty_cache()
                                _call_prompt = None
                                common_sampling_kwargs = dict(common_sampling_kwargs)
                                common_sampling_kwargs.pop("negative_prompt", None)
                                common_sampling_kwargs["prompt_embeds"] = _pre_pos
                                common_sampling_kwargs["negative_prompt_embeds"] = _pre_neg
                                logger.info(
                                    "Validation pre-encode done; T5 → CPU. "
                                    "GPU: allocated=%.2f GiB, reserved=%.2f GiB",
                                    torch.cuda.memory_allocated() / (1024 ** 3),
                                    torch.cuda.memory_reserved() / (1024 ** 3),
                                )
                            except Exception as _pre_err:
                                logger.warning(
                                    "Validation text pre-encode failed (%s); "
                                    "falling back to in-pipeline encode.",
                                    _pre_err,
                                )
                                _call_prompt = validation_prompt_text

                        if transformer_type == "WanTransformer3DModel":
                            # Base pipeline uses 'video' parameter instead of 'static_video'
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            sample = pipeline(
                                _call_prompt,
                                video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                clip_image=clip_image_for_i2v,
                                **common_sampling_kwargs,
                            ).videos
                        elif transformer_type == "Wan2_2FunControlTransformer3DModelWithConcat":
                            # Validation mask: for I2V frame 0 = anchor (mask=0),
                            # frames 1+ = predict (mask=255). For DWM every frame
                            # uses the static input video as its anchor — i.e.
                            # mask=0 everywhere (training-time `mask_latents=1`
                            # at every position). The model conditions on
                            # static_video (first-frame-replicated) + control
                            # for the whole 81f span.
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            else:
                                mask_video[...] = 0
                            if hand_video is None:
                                logger.warning(
                                    "Wan2.2 Fun-Control validation has no hand/control video for %s; "
                                    "using a zero control signal.",
                                    video_path_obj,
                                )
                            sample = pipeline(
                                _call_prompt,
                                video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                control_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                                static_disparity_video=(
                                    static_disparity_video.to(accelerator.device, dtype=weight_dtype)
                                    if static_disparity_video is not None else None
                                ),
                                eef_depth_video=(
                                    eef_depth_video.to(accelerator.device, dtype=weight_dtype)
                                    if eef_depth_video is not None else None
                                ),
                                ref_image=None,
                                **common_sampling_kwargs,
                            ).videos
                        elif transformer_type in (
                            "WanTransformer3DModelWithConcat",
                            "WanFunControlTransformer3DModelWithConcat",
                            "WanTransformer3DModelWithStateAdaLN",
                        ):
                            # Concat pipeline uses 'static_video' and 'hand_video'
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            elif is_fun_control_transformer_for_val:
                                mask_video[...] = 0
                            concat_sampling_kwargs = {}
                            if isinstance(pipeline, WanFunInpaintHandConcatPipeline):
                                concat_sampling_kwargs["static_disparity_video"] = (
                                    static_disparity_video.to(accelerator.device, dtype=weight_dtype)
                                    if static_disparity_video is not None else None
                                )
                                concat_sampling_kwargs["static_disparity_i2v_first_frame_only"] = (
                                    static_disparity_i2v_first_frame_only
                                )
                                concat_sampling_kwargs["eef_depth_video"] = (
                                    eef_depth_video.to(accelerator.device, dtype=weight_dtype)
                                    if eef_depth_video is not None else None
                                )
                                if state_adaln_tensor is not None:
                                    concat_sampling_kwargs["state_adaln"] = state_adaln_tensor
                                    concat_sampling_kwargs["state_adaln_temporal_mode"] = state_adaln_temporal_mode
                                if hand_latents is not None:
                                    concat_sampling_kwargs["hand_latents"] = hand_latents
                            sample = pipeline(
                                _call_prompt,
                                static_video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                hand_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                                **concat_sampling_kwargs,
                                **(
                                    {
                                        "static_video_latents": i2v_static_video_latents,
                                        "mask_latents": i2v_mask_latents,
                                        "clip_image": clip_image_for_i2v,
                                    }
                                    if isinstance(pipeline, WanFunInpaintHandConcatPipeline)
                                    else {}
                                ),
                                **common_sampling_kwargs,
                            ).videos
                        elif transformer_type == "WanTransformer3DVace":
                            # VACE pipeline uses 'static_video', 'hand_video', and 'vace_context_scale'
                            vace_context_scale = training_config.get("vace_context_scale", 1.0)
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            sample = pipeline(
                                _call_prompt,
                                static_video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                hand_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                                vace_context_scale=vace_context_scale,
                                clip_image=clip_image_for_i2v,
                                **common_sampling_kwargs,
                            ).videos
                        else:
                            raise ValueError(f"Unsupported transformer type: {transformer_type}")

                        # Prepare videos for comparison: condition | hand | generated | gt
                        # All videos should be [1, c, f, h, w] format
                        static_tensor = condition_video.cpu()  # Already [1, c, f, h, w]

                        if hand_video is not None:
                            hand_tensor = hand_video.cpu()  # Already [1, c, f, h, w]
                        else:
                            hand_tensor = None
                        static_disparity_tensor = static_disparity_video.cpu() if static_disparity_video is not None else None
                        eef_depth_tensor = eef_depth_video.cpu() if eef_depth_video is not None else None

                        sample_tensor = sample.cpu()  # Already [1, c, f, h, w]
                        gt_tensor = torch.from_numpy(gt_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]

                        # Ensure all videos have the same number of frames
                        num_frames_actual = min(
                            static_tensor.shape[2],
                            sample_tensor.shape[2],
                            gt_tensor.shape[2]
                        )
                        if hand_tensor is not None:
                            num_frames_actual = min(num_frames_actual, hand_tensor.shape[2])
                        if static_disparity_tensor is not None:
                            num_frames_actual = min(num_frames_actual, static_disparity_tensor.shape[2])
                        if eef_depth_tensor is not None:
                            num_frames_actual = min(num_frames_actual, eef_depth_tensor.shape[2])
                        static_tensor = static_tensor[:, :, :num_frames_actual, :, :]
                        if hand_tensor is not None:
                            hand_tensor = hand_tensor[:, :, :num_frames_actual, :, :]
                        if static_disparity_tensor is not None:
                            static_disparity_tensor = static_disparity_tensor[:, :, :num_frames_actual, :, :]
                        if eef_depth_tensor is not None:
                            eef_depth_tensor = eef_depth_tensor[:, :, :num_frames_actual, :, :]
                        sample_tensor = sample_tensor[:, :, :num_frames_actual, :, :]
                        gt_tensor = gt_tensor[:, :, :num_frames_actual, :, :]

                        comparison_parts = [static_tensor]
                        comparison_labels = ["condition"]
                        if static_disparity_tensor is not None:
                            comparison_parts.append(static_disparity_tensor)
                            comparison_labels.append("static_disparity")
                        if hand_tensor is not None:
                            comparison_parts.append(hand_tensor)
                            comparison_labels.append("hand")
                        if eef_depth_tensor is not None:
                            comparison_parts.append(eef_depth_tensor)
                            comparison_labels.append("eef_depth")
                        comparison_parts.extend([sample_tensor, gt_tensor])
                        comparison_labels.extend(["generated", "gt"])
                        comparison_caption = "(" + "|".join(comparison_labels) + ")"
                        comparison = torch.cat(comparison_parts, dim=4)

                        save_name = f"{video_save_id}_{prompt_variant_label}"
                        fsdp_validation = bool(getattr(args, "use_fsdp", False))
                        should_save_validation = (not fsdp_validation) or accelerator.is_main_process
                        gpu_suffix = (
                            f"_gpu{accelerator.process_index}"
                            if accelerator.num_processes > 1 and not fsdp_validation
                            else ""
                        )

                        os.makedirs(args.output_dir, exist_ok=True)

                        generated_filename = os.path.join(
                            args.output_dir,
                            f"step_{global_step}_generated_{save_name}{gpu_suffix}.mp4",
                        )

                        comparison_filename = os.path.join(
                            args.output_dir,
                            f"step_{global_step}_comparison_{save_name}{gpu_suffix}.mp4",
                        )
                        metadata_filename = os.path.join(
                            args.output_dir,
                            f"step_{global_step}_validation_meta_{save_name}{gpu_suffix}.json",
                        )
                        if should_save_validation:
                            save_videos_grid(sample_tensor, generated_filename, fps=validation_save_fps)
                            save_videos_grid(comparison, comparison_filename, fps=validation_save_fps)
                            metadata = {
                                "global_step": int(global_step),
                                "source_video_path": str(video_path_obj),
                                "source_fps_path": str(source_fps_path) if source_fps_path is not None else None,
                                "prompt_candidate_video_paths": prompt_candidate_video_paths,
                                "save_id": video_save_id,
                                "prompt_mode": prompt_variant_label,
                                "gpu_index": int(accelerator.process_index),
                                "num_processes": int(accelerator.num_processes),
                                "generated_filename": os.path.basename(generated_filename),
                                "comparison_filename": os.path.basename(comparison_filename),
                                "comparison_labels": comparison_labels,
                                "validation_sample": validation_sample,
                            }
                            with open(metadata_filename, "w", encoding="utf-8") as f:
                                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

                            logger.info(
                                f"📹 GPU {accelerator.process_index}: Saved validation videos for {video_path_obj} "
                                f"(save_id={video_save_id}, prompt_mode={prompt_variant_label})"
                            )
                            logger.info(f"   Generated: {generated_filename}")
                            logger.info(f"   Comparison {comparison_caption}: {comparison_filename}")
                            logger.info(f"   Metadata: {metadata_filename}")

                        if should_save_validation and accelerator.is_main_process:
                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb" and HAS_WANDB:
                                    caption = (
                                        f"step_{global_step}_{video_save_id}_{prompt_variant_label}_comparison "
                                        f"{comparison_caption}"
                                    )
                                    tracker.log({
                                        "validation": [
                                            wandb.Video(comparison_filename, caption=caption, fps=validation_save_fps)
                                        ]
                                    }, step=global_step)

                        if should_save_validation:
                            logger.info(
                                f"Validation {i+1}/{len(validation_set)}: {video_path_obj} saved "
                                f"(save_id={video_save_id}, prompt_mode={prompt_variant_label})"
                            )

        # Explicitly delete all validation-specific objects
        del pipeline
        del scheduler
        # Clean up reloaded models if they were loaded for this validation run
        if models_reloaded:
            logger.info("📦 Cleaning up reloaded models after validation...")
            del vae
            if text_encoder is not None:
                del text_encoder
            if tokenizer is not None:
                del tokenizer
            if validation_clip_image_encoder is not None:
                del validation_clip_image_encoder
        elif restore_text_encoder_to_cpu and text_encoder is not None:
            logger.info("📦 Moving text encoder back to CPU after validation...")
            text_encoder = text_encoder.to("cpu")
            if validation_clip_image_encoder is not None:
                del validation_clip_image_encoder
        elif validation_clip_image_encoder is not None:
            del validation_clip_image_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        if restore_text_encoder_to_cpu and text_encoder is not None:
            text_encoder = text_encoder.to("cpu")
        if validation_clip_image_encoder is not None:
            del validation_clip_image_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.error(f"Validation error during stage '{validation_stage}': {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        restore_transformer = transformer_state_owner if transformer_state_owner is not None else transformer_for_val
        if restore_transformer is not None and transformer_for_val_was_training:
            restore_transformer.train()


def save_model_lora(output_dir, network, weight_dtype, accelerator, transformer3d=None, trainable_parameter_patterns=None):
    """Save LoRA weights and non-LoRA trainable weights."""
    if accelerator.is_main_process:
        # Save LoRA weights
        safetensor_save_path = os.path.join(output_dir, "lora_diffusion_pytorch_model.safetensors")
        accelerator.unwrap_model(network).save_weights(safetensor_save_path, weight_dtype, None)
        logger.info(f"Saved LoRA weights to {safetensor_save_path}")
        
        # Save non-LoRA trainable weights (e.g., patch_embedding)
        if transformer3d is not None and trainable_parameter_patterns:
            non_lora_state_dict = {}
            model = accelerator.unwrap_model(transformer3d)
            for name, param in model.named_parameters():
                if name.startswith("network."):
                    continue
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        non_lora_state_dict[name] = param.data.clone()
                        break
            
            if non_lora_state_dict:
                non_lora_path = os.path.join(output_dir, "non_lora_weights.safetensors")
                save_file(non_lora_state_dict, non_lora_path)
                logger.info(f"Saved non-LoRA weights ({len(non_lora_state_dict)} params) to {non_lora_path}")


def save_model_sft(output_dir, transformer3d, accelerator):
    """Save SFT model weights."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(transformer3d)
        model.save_pretrained(os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB")
        logger.info(f"Saved transformer to {output_dir}/transformer")


def save_model_vace(output_dir, transformer3d, accelerator):
    """Save VACE weights (vace_blocks + vace_patch_embedding only)."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(transformer3d)

        # Extract only VACE-related weights
        vace_state_dict = {}
        for name, param in model.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                vace_state_dict[name] = param.data.clone()

        if vace_state_dict:
            vace_path = os.path.join(output_dir, "vace_weights.safetensors")
            save_file(vace_state_dict, vace_path)
            logger.info(f"Saved VACE weights ({len(vace_state_dict)} params) to {vace_path}")
        else:
            logger.warning("No VACE weights found to save!")


def save_train_mode_weights(
    save_path,
    args,
    network,
    weight_dtype,
    accelerator,
    transformer3d,
    trainable_parameter_patterns,
):
    """Save evaluation-friendly model weights for the active training mode."""
    os.makedirs(save_path, exist_ok=True)

    if args.train_mode == "lora":
        save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns)
    elif args.train_mode == "vace_lora":
        save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns=None)
        save_model_vace(save_path, transformer3d, accelerator)
    elif args.train_mode == "vace":
        save_model_vace(save_path, transformer3d, accelerator)
    else:
        save_model_sft(save_path, transformer3d, accelerator)

    logger.info(f"Saved checkpoint to {save_path}")


def load_or_create_empty_prompt_embeds(
    empty_prompt_embeds_path,
    tokenizer,
    text_encoder,
    text_encoder_kwargs,
    accelerator,
    weight_dtype,
):
    if empty_prompt_embeds_path and os.path.exists(empty_prompt_embeds_path):
        logger.info("Loading cached empty prompt embeds from %s", empty_prompt_embeds_path)
        return torch.load(empty_prompt_embeds_path, map_location="cpu")

    if tokenizer is None or text_encoder is None:
        raise ValueError(
            "prompt_mode=empty requires tokenizer/text_encoder when empty_prompt_embeds cache is missing."
        )

    logger.info("Cached empty prompt embeds not found. Building from text encoder.")
    max_length = int((text_encoder_kwargs or {}).get("text_length", 512))
    text_encoder_was_training = text_encoder.training
    text_encoder.eval()

    text_encoder_device = next(text_encoder.parameters()).device
    moved_to_device = False
    if text_encoder_device.type == "cpu":
        text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
        moved_to_device = True

    tokenized = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = tokenized.input_ids.to(accelerator.device)
    prompt_attention_mask = tokenized.attention_mask.to(accelerator.device)
    seq_len = int(prompt_attention_mask[0].gt(0).sum().item())

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]

    empty_prompt_embeds = prompt_embeds[0, :seq_len].detach().to("cpu").contiguous()

    if text_encoder_was_training:
        text_encoder.train()
    if moved_to_device:
        text_encoder.to("cpu")
        torch.cuda.empty_cache()

    if empty_prompt_embeds_path and accelerator.is_main_process:
        os.makedirs(os.path.dirname(empty_prompt_embeds_path), exist_ok=True)
        torch.save(empty_prompt_embeds, empty_prompt_embeds_path)
        logger.info("Saved empty prompt embeds cache to %s", empty_prompt_embeds_path)

    accelerator.wait_for_everyone()
    return empty_prompt_embeds


def main():
    from training.wan.args import add_all_args
    
    parser = argparse.ArgumentParser(description="WAN-based DWM Training Script")
    # Core arguments
    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment YAML config file")
    parser.add_argument("--override", type=str, nargs="*",
                        help="Override config values (key=value format)")
    parser.add_argument("--mode", type=str, choices=["debug", "slurm_test", "slurm", "batch"], default="slurm",
                        help="Training mode: debug, slurm_test, slurm, or batch")
    # # Add all other arguments from args.py
    add_all_args(parser)
    args = parser.parse_args()
    
    # ========== Load Configuration ==========
    print("🚀 Loading experiment configuration...")
    config = load_experiment_config(args.experiment_config, args.override)
    
    # Extract configuration sections
    experiment_config = config.get("experiment", {})
    training_config = config.get("training", {})
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    data_config = config.get("data", {})
    custom_settings = training_config.get("custom_settings", {})
    prompt_mode = resolve_prompt_mode(training_config)
    text_dropout_prob = resolve_text_dropout_prob(training_config)
    validation_prompt_mode = resolve_validation_prompt_mode(training_config)
    i2v_condition_latent_type = str(training_config.get("i2v_condition_latent_type", "image")).strip().lower()
    if i2v_condition_latent_type not in {"image", "fun_inp"}:
        raise ValueError(
            f"Unsupported i2v_condition_latent_type: {i2v_condition_latent_type}. "
            "Use one of: image, fun_inp."
        )
    empty_prompt_embeds_path = data_config.get(
        "empty_prompt_embeds_path",
        "data/empty_prompt_embeds_wan.pt",
    )

    # Print training FPS (used for RoPE scaling in WanTransformer3DModel)
    if pipeline_config.get("fps") is not None:
        print(f"🎞️ Training FPS: {int(pipeline_config['fps'])} (from pipeline_config)")
    else:
        print("🎞️ Training FPS: 16 (default)")
    
    # Model-related configs from pipeline config (merged from configs/pipelines/*.yaml)
    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    vae_kwargs = config.get("vae_kwargs", {})
    scheduler_kwargs = config.get("scheduler_kwargs", {})
    transformer_additional_kwargs = config.get("transformer_additional_kwargs", {})
    # `transformer_subpath` is consumed by the loader (where to look for the
    # checkpoint relative to base_model_name_or_path); it is NOT a transformer
    # __init__ argument, so make sure we never forward it into from_config.
    transformer_subpath = transformer_additional_kwargs.pop("transformer_subpath", "transformer")
    transformer_2_subpath = transformer_additional_kwargs.pop("transformer_2_subpath", None)
    args.transformer_2_subpath = transformer_2_subpath  # expose to run_validation via args
    image_encoder_kwargs = config.get("image_encoder_kwargs", {})

    # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
    if pipeline_config.get("fps") is not None:
        transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
    
    # Override args with config values (config values take priority over command line args)
    # Use "is not None" check to ensure config values (including 0, False, empty string) override args
    if training_config.get("learning_rate") is not None:
        args.learning_rate = float(training_config["learning_rate"])
    if training_config.get("batch_size") is not None:
        args.train_batch_size = int(training_config["batch_size"])
    if training_config.get("max_train_steps") is not None:
        args.max_train_steps = int(training_config["max_train_steps"])
    if training_config.get("gradient_accumulation_steps") is not None:
        args.gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    # Note: args.pretrained_model_name_or_path is NOT overridden by pipeline_config
    # to allow specifying SFT checkpoint path via CLI (--pretrained_model_name_or_path)
    # pipeline_config["base_model_name_or_path"] is used for model structure definition
    if args.pretrained_model_name_or_path is None and pipeline_config.get("base_model_name_or_path") is not None:
        args.pretrained_model_name_or_path = pipeline_config["base_model_name_or_path"]
    if data_config.get("data_root") is not None:
        args.train_data_dir = data_config["data_root"]
    if data_config.get("dataset_file") is not None:
        args.train_data_meta = data_config["dataset_file"]
    # Determine transformer type
    transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    is_wan_fun_control = transformer_type == "WanFunControlTransformer3DModelWithConcat"
    is_fun_control_transformer = _is_fun_control_transformer_type(transformer_type)
    uses_fun_control_y = bool(transformer_config.get("uses_fun_control_y", False))
    is_wan2_2_fun_control_i2v = (
        pipeline_config.get("type", "") == "wan2.2_fun_control_i2v_mesh"
        or transformer_type == "Wan2_2FunControlTransformer3DModelWithConcat"
    )
    custom_settings = training_config.get("custom_settings", {})
    if custom_settings.get("use_diffusers_i2v_backend", False):
        print(
            "⚠️ `use_diffusers_i2v_backend` was removed. The Wan I2V path now reuses the "
            "self-impl WanTransformer3DModelWithConcat / WanFunInpaintHandConcatPipeline "
            "(set transformer.model_type='i2v' via transformer_additional_kwargs). "
            "Ignoring the legacy flag."
        )
    use_diffusers_i2v_backend = False
    keep_full_state_checkpoints = int(
        custom_settings.get("keep_full_state_checkpoints", args.keep_full_state_checkpoints)
    )
    if keep_full_state_checkpoints < 0:
        raise ValueError("training.custom_settings.keep_full_state_checkpoints must be >= 0")
    args.keep_full_state_checkpoints = keep_full_state_checkpoints
    validate_i2v_condition_latent_type(
        pipeline_config.get("type", ""),
        conditioning_mode,
        i2v_condition_latent_type,
        use_diffusers_i2v_backend=use_diffusers_i2v_backend,
    )
    if transformer_config.get("condition_channels") is not None:
        args.condition_channels = int(transformer_config["condition_channels"])
    if training_config.get("mode") is not None:
        args.train_mode = training_config["mode"]
    if training_config.get("lora_rank") is not None:
        args.rank = int(training_config["lora_rank"])
    if training_config.get("lora_alpha") is not None:
        args.network_alpha = int(training_config["lora_alpha"])
    if training_config.get("lora_skip_name") is not None:
        args.lora_skip_name = str(training_config["lora_skip_name"])
    
    # ========== Setup Output Directory with SLURM Support ==========
    # Check for resume configuration
    resume_from_checkpoint = training_config.get("resume_from_checkpoint") or args.resume_from_checkpoint
    if resume_from_checkpoint:
        print(f"🔄 Resume mode enabled: resume_from_checkpoint = {resume_from_checkpoint}")
    
    # Setup output directory based on mode
    exp_name = experiment_config.get("name", "wan_training")
    exp_date = experiment_config.get("date", "unknown")
    
    # Extract date suffix
    if exp_date != "unknown":
        try:
            parsed_date = dt.strptime(exp_date, "%Y-%m-%d")
            date_suffix = parsed_date.strftime("%y%m%d")
        except:
            date_suffix = "unknown"
    else:
        date_suffix = "unknown"
    
    # Construct output directory based on mode
    base_output_dir = f"outputs/{date_suffix}/{exp_name}"
    if args.mode == "debug":
        args.output_dir = f"{base_output_dir}_debug"
        print(f"🔧 Debug mode: Output directory set to {args.output_dir}")
    elif args.mode == "slurm_test":
        slurm_job_id = training_config.get("slurm_job_id")
        if resume_from_checkpoint and slurm_job_id:
            print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            args.output_dir = f"{base_output_dir}_slurm_{slurm_job_id}"
        else:
            args.output_dir = f"{base_output_dir}_slurm_test"
        print(f"🧪 SLURM test mode: Output directory set to {args.output_dir}")
    elif args.mode == "slurm":
        if resume_from_checkpoint:
            slurm_job_id = training_config.get("slurm_job_id")
            if slurm_job_id:
                print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            else:
                slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
                print(f"🔄 Resume mode: Using current SLURM Job ID: {slurm_job_id}")
        else:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
            print(f"🚀 New training: Using current SLURM Job ID: {slurm_job_id}")
        args.output_dir = f"{base_output_dir}_slurm_{slurm_job_id}"
        print(f"📁 SLURM mode: Output directory set to {args.output_dir}")
    elif args.mode == "batch":
        # Batch mode: add timestamp to output directory (use env var so all processes get same dir)
        timestamp = os.environ.get("BATCH_TIMESTAMP")
        if not timestamp:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{base_output_dir}_batch_{timestamp}"
        print(f"🚀 Batch mode: Output directory set to {args.output_dir}")
    
    print(f"📁 Final output directory: {args.output_dir}")
    print(f"   - Base: {base_output_dir}")
    print(f"   - Experiment: {exp_name}")
    print(f"   - Date: {exp_date} -> {date_suffix}")
    print(f"   - Mode: {args.mode}")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", args.mixed_precision),
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
    # ========== Detect DeepSpeed / FSDP ==========
    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"🚀 Using DeepSpeed Zero stage: {zero_stage}")
        args.use_deepspeed = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"🚀 Using FSDP stage: {fsdp_stage}")
        args.use_fsdp = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("🚀 DeepSpeed/FSDP is not enabled.")

    # Create output directory and copy experiment config
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Copy experiment config YAML to output directory
        if args.experiment_config:
            config_filename = os.path.basename(args.experiment_config)
            output_config_path = os.path.join(args.output_dir, config_filename)
            shutil.copy2(args.experiment_config, output_config_path)
            print(f"📋 Copied experiment config to: {output_config_path}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ========== Load Models ==========
    print("🔧 Loading models...")
    
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    # Load tokenizer / text encoder / VAE
    use_preencoded_prompt_embeds = bool(data_config.get("use_preencoded_prompt_embeds", False))
    skip_text_encoder_load = bool(custom_settings.get("skip_text_encoder_load", False))
    load_tensors_config = bool(custom_settings.get("load_tensors", False))
    if skip_text_encoder_load and not (load_tensors_config or use_preencoded_prompt_embeds or prompt_mode == PROMPT_MODE_EMPTY):
        raise ValueError(
            "training.custom_settings.skip_text_encoder_load=true requires "
            "load_tensors=true, data.use_preencoded_prompt_embeds=true, or prompt_mode=empty."
        )

    if conditioning_mode == "i2v" and use_diffusers_i2v_backend:
        # Video-As-Prompt style: load from diffusers base checkpoint with subfolders.
        base_ckpt = args.pretrained_model_name_or_path
        tokenizer_path = os.path.join(base_ckpt, "tokenizer")
        text_encoder_path = os.path.join(base_ckpt, "text_encoder")
        if skip_text_encoder_load:
            logger.info("📦 Skipping tokenizer/text_encoder load; using preencoded prompt embeds.")
            tokenizer = None
            text_encoder = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_ckpt, subfolder="tokenizer")

            text_encoder = UMT5EncoderModel.from_pretrained(
                base_ckpt,
                subfolder="text_encoder",
                torch_dtype=weight_dtype,
            ).eval()

        vae = diffusers.AutoencoderKLWan.from_pretrained(
            base_ckpt,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).eval()

        # Keep paths for validation reload.
        vae_path = os.path.join(base_ckpt, "vae")
    else:
        tokenizer_path = _resolve_component_path(
            args.pretrained_model_name_or_path,
            text_encoder_kwargs.get('tokenizer_subpath'),
            'tokenizer',
        )
        text_encoder_path = _resolve_component_path(
            args.pretrained_model_name_or_path,
            text_encoder_kwargs.get('text_encoder_subpath'),
            'text_encoder',
        )
        if skip_text_encoder_load:
            logger.info("📦 Skipping tokenizer/text_encoder load; using preencoded prompt embeds.")
            tokenizer = None
            text_encoder = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            text_encoder = WanT5EncoderModel.from_pretrained(
                text_encoder_path,
                additional_kwargs=text_encoder_kwargs,
                torch_dtype=weight_dtype,
            ).eval()

        vae_path = os.path.join(args.pretrained_model_name_or_path, vae_kwargs.get('vae_subpath', 'vae'))
        Choosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[vae_kwargs.get('vae_type', 'AutoencoderKLWan')]
        vae = Choosen_AutoencoderKL.from_pretrained(
            vae_path,
            additional_kwargs=vae_kwargs,
        ).eval()

    # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
    is_wan2_2 = "2.2" in pipeline_config.get("type", "")
    
    # Load CLIP image encoder (skip only for WAN 2.2)
    if is_wan2_2:
        logger.info("⚠️  Skipping CLIP image encoder loading for WAN 2.2 backend")
        clip_image_encoder = None
    else:
        image_encoder_subpath = image_encoder_kwargs.get("image_encoder_subpath", "image_encoder")
        image_encoder_path = os.path.join(args.pretrained_model_name_or_path, image_encoder_subpath)
        if conditioning_mode == "i2v" and use_diffusers_i2v_backend:
            logger.info(
                "🔧 Loading HF CLIPVision image encoder for i2v-diffusers backend "
                "(base=%s, subfolder=%s)",
                args.pretrained_model_name_or_path,
                image_encoder_subpath,
            )
            try:
                # Video-As-Prompt style: load from base checkpoint + subfolder.
                clip_image_encoder = CLIPVisionModel.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder=image_encoder_subpath,
                    torch_dtype=torch.float32,
                ).eval()
            except Exception as e:
                if os.path.isdir(image_encoder_path):
                    logger.warning(
                        "Failed subfolder loading, fallback to direct directory loading: %s (%s)",
                        image_encoder_path,
                        e,
                    )
                    clip_image_encoder = CLIPVisionModel.from_pretrained(
                        image_encoder_path,
                        torch_dtype=torch.float32,
                    ).eval()
                else:
                    raise RuntimeError(
                        f"Failed to load CLIPVision image encoder for i2v-diffusers backend. "
                        f"base={args.pretrained_model_name_or_path}, subfolder={image_encoder_subpath}, "
                        f"resolved_path={image_encoder_path}"
                    ) from e
        else:
            clip_image_encoder = CLIPModel.from_pretrained(image_encoder_path).eval()

    # Load transformer
    if transformer_type == "WanTransformer3DModel":
        print(f"🔧 Loading WanTransformer3DModel (base model)...")
        transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_subpath),
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanTransformer3DModelWithConcat":
        # i2v and dwm both go through the same self-impl WanTransformer3DModel.
        # `model_type` ('i2v' vs 't2v') comes through transformer_additional_kwargs
        # from the YAML and switches the cross-attention class + img_emb.
        print(f"🔧 Loading WanTransformer3DModelWithConcat with {args.condition_channels} condition channels...")
        transformer3d = WanTransformer3DModelWithConcat.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_subpath),
            transformer_additional_kwargs={
                "condition_channels": args.condition_channels,
                "is_wan2_2": is_wan2_2,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanTransformer3DModelWithStateAdaLN":
        print("🔧 Loading WanTransformer3DModelWithStateAdaLN...")
        state_adaln_kwargs = {
            key: transformer_config[key]
            for key in (
                "state_adaln_dim",
                "state_adaln_hidden_dim",
                "state_adaln_zero_init",
                "state_adaln_scale",
                "state_adaln_zero_first_frame",
            )
            if key in transformer_config
        }
        transformer3d = WanTransformer3DModelWithStateAdaLN.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_subpath),
            transformer_additional_kwargs={
                "condition_channels": 0,
                "is_wan2_2": is_wan2_2,
                **state_adaln_kwargs,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanFunControlTransformer3DModelWithConcat":
        extra_condition_channels = int(
            transformer_config.get(
                "extra_condition_channels",
                transformer_config.get("condition_channels", 0),
            )
        )
        args.condition_channels = extra_condition_channels
        print(
            "🔧 Loading WanFunControlTransformer3DModelWithConcat "
            f"with {extra_condition_channels} extra condition channels..."
        )
        transformer3d = WanFunControlTransformer3DModelWithConcat.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_subpath),
            transformer_additional_kwargs={
                "condition_channels": extra_condition_channels,
                "is_wan2_2": False,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "Wan2_2FunControlTransformer3DModelWithConcat":
        extra_condition_channels = int(
            transformer_config.get(
                "extra_condition_channels",
                transformer_config.get("condition_channels", 0),
            )
        )
        print(
            "🔧 Loading Wan2_2FunControlTransformer3DModelWithConcat "
            f"with {extra_condition_channels} extra condition channels..."
        )
        transformer3d = Wan2_2FunControlTransformer3DModelWithConcat.from_pretrained(
            os.path.join(
                args.pretrained_model_name_or_path,
                transformer_subpath,
            ),
            transformer_additional_kwargs={
                "condition_channels": extra_condition_channels,
                "is_wan2_2": True,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "MultiViewWanTransformer3DModelWithConcat":
        multiview_mode = transformer_config.get("multiview_mode", "full_prope")
        prope_attention_variant = transformer_config.get("prope_attention_variant", "projection_only")
        print(
            f"🔧 Loading MultiViewWanTransformer3DModelWithConcat "
            f"(mode={multiview_mode}, prope_variant={prope_attention_variant}) "
            f"with {args.condition_channels} condition channels..."
        )
        transformer3d = MultiViewWanTransformer3DModelWithConcat.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_subpath),
            transformer_additional_kwargs={
                "condition_channels": args.condition_channels,
                "is_wan2_2": is_wan2_2,
                "multiview_mode": multiview_mode,
                "prope_attention_variant": prope_attention_variant,
                "uses_fun_control_y": uses_fun_control_y,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanTransformer3DVace":
        # Get VACE-specific configuration
        vace_layers = transformer_config.get('vace_layers', None)  # e.g., [0, 2, 4, 6, ...] or None for default
        vace_in_dim = transformer_config.get('vace_in_dim', 16)  # Hand latent channels

        print(f"🔧 Loading WanTransformer3DVace with VACE conditioning...")
        print(f"   vace_layers: {vace_layers if vace_layers else 'default (every 2nd layer)'}")
        print(f"   vace_in_dim: {vace_in_dim}")

        transformer3d = WanTransformer3DVace.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_subpath),
            transformer_additional_kwargs={
                "vace_layers": vace_layers,
                "vace_in_dim": vace_in_dim,
                "is_wan2_2": is_wan2_2,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    else:
        raise ValueError(f"Unsupported transformer type: {transformer_type}")

    # Ensure VACE modules are in the correct dtype (they're not loaded from checkpoint)
    if transformer_type == "WanTransformer3DVace":
        if hasattr(transformer3d, 'vace_blocks'):
            for block in transformer3d.vace_blocks:
                block.to(weight_dtype)
            print(f"✅ Converted vace_blocks to {weight_dtype}")
        if hasattr(transformer3d, 'vace_patch_embedding'):
            transformer3d.vace_patch_embedding.to(weight_dtype)
            print(f"✅ Converted vace_patch_embedding to {weight_dtype}")

    # Load custom weights if provided
    if args.transformer_path is not None:
        print(f"Loading transformer from: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    if args.vae_path is not None:
        print(f"Loading VAE from: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    # ========== Setup Training Mode ==========
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder:
        raise ValueError("Cannot train text encoder when skip_text_encoder_load=true.")
    if clip_image_encoder is not None:
        clip_image_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    network = None
    trainable_params = None
    vace_trainable_params = []

    # Get trainable_parameter_patterns from config (for non-LoRA trainable params like patch_embedding)
    trainable_parameter_patterns = training_config.get("trainable_parameter_patterns", [])
    
    if args.train_mode == "lora":
        print("🔧 Setting up LoRA training...")
        
        # Create LoRA network using VideoX-Fun style
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            skip_name=args.lora_skip_name,
        )
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)
        num_lora_modules = len(getattr(network, "text_encoder_loras", [])) + len(getattr(network, "unet_loras", []))
        allow_empty_lora = training_config.get(
            "allow_empty_lora",
            custom_settings.get("allow_empty_lora", False),
        )
        if num_lora_modules == 0 and not allow_empty_lora:
            raise ValueError(
                "LoRA training created 0 LoRA modules. "
                f"transformer class={transformer3d.__class__.__name__}, "
                f"skip_name={args.lora_skip_name!r}. "
                "Add this transformer class to LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE "
                "or set training.custom_settings.allow_empty_lora=true for intentional patch-only training."
            )
        
        # Enable additional trainable parameters (e.g., patch_embedding for channel expansion)
        non_lora_trainable_params = []
        if trainable_parameter_patterns:
            print(f"📋 Trainable parameter patterns: {trainable_parameter_patterns}")
            for name, param in transformer3d.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        param.requires_grad = True
                        non_lora_trainable_params.append(param)
                        print(f"   ✓ Enabled: {name}")
                        break
        
        lora_trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params = list(lora_trainable_params)
        trainable_params_optim = network.prepare_optimizer_params(
            args.learning_rate / 2, args.learning_rate, args.learning_rate
        )
        
        # Add non-LoRA trainable params to optimizer
        if non_lora_trainable_params:
            trainable_params_optim.append({
                'params': non_lora_trainable_params,
                'lr': args.learning_rate,
            })
            trainable_params.extend(non_lora_trainable_params)
            print(f"📊 Non-LoRA trainable parameters: {sum(p.numel() for p in non_lora_trainable_params):,}")
        
        print(f"📊 LoRA trainable parameters: {sum(p.numel() for p in lora_trainable_params):,}")
        print(f"📊 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "vace_lora":
        print("🔧 Setting up VACE + Backbone LoRA training (vace modules trainable + backbone LoRA)...")

        # Ensure we're using WanTransformer3DVace
        if transformer_type != "WanTransformer3DVace":
            raise ValueError(f"vace_lora training mode requires WanTransformer3DVace, but got {transformer_type}")

        # Freeze everything first
        transformer3d.requires_grad_(False)

        # In vace_lora mode, default skip to avoid injecting LoRA into vace_* modules
        lora_skip_name = args.lora_skip_name if args.lora_skip_name is not None else "vace_"
        if args.lora_skip_name is None:
            print("ℹ️  vace_lora mode: lora_skip_name not provided, defaulting to 'vace_'")
        else:
            print(f"ℹ️  vace_lora mode: using user-provided lora_skip_name='{args.lora_skip_name}'")

        # Create LoRA network for backbone
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            skip_name=lora_skip_name,
        )
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)

        # Enable only VACE-specific full-trainable modules
        for name, param in transformer3d.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                param.requires_grad = True
                vace_trainable_params.append(param)
                print(f"   ✓ VACE enabled: {name}")

        if not vace_trainable_params:
            raise ValueError("No VACE parameters found to train! Check if transformer has vace_blocks and vace_patch_embedding.")

        # LoRA optimizer params + VACE full-train params
        trainable_params_optim = network.prepare_optimizer_params(
            args.learning_rate / 2, args.learning_rate, args.learning_rate
        )
        trainable_params_optim.append({
            'params': vace_trainable_params,
            'lr': args.learning_rate,
        })

        lora_trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params = lora_trainable_params + vace_trainable_params

        print(f"📊 LoRA trainable parameters: {sum(p.numel() for p in lora_trainable_params):,}")
        print(f"📊 VACE trainable parameters: {sum(p.numel() for p in vace_trainable_params):,}")
        print(f"📊 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "sft":
        print("🔧 Setting up SFT training...")
        
        transformer3d.train()
        
        # Enable specific modules for training
        if args.trainable_modules:
            for name, param in transformer3d.named_parameters():
                for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
                    if trainable_module_name in name:
                        param.requires_grad = True
                        break
        else:
            # Full fine-tuning
            transformer3d.requires_grad_(True)
        
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
        
        # Setup optimizer params with different learning rates
        trainable_params_optim = [
            {'params': [], 'lr': args.learning_rate},
            {'params': [], 'lr': args.learning_rate / 2},
        ]
        in_already = []
        for name, param in transformer3d.named_parameters():
            if not param.requires_grad or name in in_already:
                continue
            
            high_lr = False
            if args.trainable_modules:
                for module_name in args.trainable_modules:
                    if module_name in name:
                        trainable_params_optim[0]['params'].append(param)
                        in_already.append(name)
                        high_lr = True
                        break
            
            if not high_lr:
                for module_name in args.trainable_modules_low_learning_rate:
                    if module_name in name:
                        trainable_params_optim[1]['params'].append(param)
                        in_already.append(name)
                        break
                else:
                    # Default to high learning rate if no match
                    if not args.trainable_modules:
                        trainable_params_optim[0]['params'].append(param)
        
        print(f"📊 SFT trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "vace":
        print("🔧 Setting up VACE training (backbone frozen, only vace_blocks + vace_patch_embedding trainable)...")

        # Ensure we're using WanTransformer3DVace
        if transformer_type != "WanTransformer3DVace":
            raise ValueError(f"VACE training mode requires WanTransformer3DVace, but got {transformer_type}")

        # Freeze everything first (already done above, but be explicit)
        transformer3d.requires_grad_(False)

        # Enable only VACE-specific modules
        vace_trainable_params = []
        for name, param in transformer3d.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                param.requires_grad = True
                vace_trainable_params.append(param)
                print(f"   ✓ Enabled: {name}")

        if not vace_trainable_params:
            raise ValueError("No VACE parameters found to train! Check if transformer has vace_blocks and vace_patch_embedding.")

        trainable_params = vace_trainable_params
        trainable_params_optim = [
            {'params': vace_trainable_params, 'lr': args.learning_rate},
        ]

        print(f"📊 VACE trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Enable gradient checkpointing (from args or config)
    gradient_checkpointing = args.gradient_checkpointing or training_config.get("custom_settings", {}).get("gradient_checkpointing", False)
    if gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")

    # Enable TF32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ========== Setup Optimizer ==========
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # ========== Setup Learning Rate Scheduler ==========
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    if use_deepspeed_scheduler:
        lr_scheduler = DummyScheduler(
            name=training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=training_config.get("lr_num_cycles", 1),
        )

    # ========== Setup Dataset ==========
    print("🔧 Setting up dataset...")
    
    # Use is_wan2_2 already defined above (for width alignment to 32)
    align_width_to_32 = is_wan2_2
    if align_width_to_32:
        print("📐 WAN 2.2 detected: Aligning width to multiples of 32 for compatibility")
    
    # Setup dataset and dataloader
    # Read load_tensors from training.custom_settings (same as cogvideox pattern)
    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", False)
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    i2v_use_hand_condition = bool(training_config.get("i2v_use_hand_condition", False))
    static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
    static_disparity_i2v_first_frame_only = (
        bool(data_config.get("static_disparity_i2v_first_frame_only", False))
        and conditioning_mode == "i2v"
    )
    if load_tensors:
        print("📦 Using pre-encoded latents (load_tensors=True)")

    dataset_init_kwargs = build_dataset_init_kwargs(
        data_config,
        load_tensors=load_tensors,
        conditioning_mode=conditioning_mode,
        use_preencoded_prompt_embeds=use_preencoded_prompt_embeds,
        align_width_to_32=align_width_to_32,
    )

    if accelerator.is_main_process and data_config.get("temporal_frame_skip_choices"):
        logger.info(
            "Training temporal frame skip config: choices=%s probs=%s",
            data_config.get("temporal_frame_skip_choices"),
            data_config.get("temporal_frame_skip_probs", "uniform"),
        )

    multiview_composition = bool(data_config.get("multiview_composition", False))
    multiview_width_concat = bool(data_config.get("multiview_width_concat", False))
    multiview_prope = bool(data_config.get("multiview_prope", False))
    if sum([multiview_composition, multiview_width_concat, multiview_prope]) > 1:
        raise ValueError(
            "multiview_composition, multiview_width_concat, and multiview_prope are mutually exclusive."
        )
    if multiview_prope:
        prope_config = data_config.get("multiview_prope_config") or {}
        views_spec = prope_config.get("views")
        if not views_spec:
            raise ValueError(
                "multiview_prope=true requires data.multiview_prope_config.views"
            )
        mv_kwargs = dict(dataset_init_kwargs)
        mv_kwargs["views_spec"] = views_spec
        for key in (
            "camera_serials_json",
            "cam2base_json",
            "cam2base_superset_json",
            "intrinsics_json",
            "raw_cache_root",
            "wrist_pose_source",
            "gripper2wrist_json",
            "wrist_proprio_flip",
        ):
            if key in prope_config and prope_config[key] is not None:
                mv_kwargs[key] = prope_config[key]
        if accelerator.is_main_process:
            logger.info(
                "Using MultiViewPropeDataset: views=%s",
                [v.get("suffix") or v.get("suffix_pool") for v in views_spec],
            )
        train_dataset = MultiViewPropeDataset(**mv_kwargs)
    elif multiview_composition:
        raw_layout = data_config.get("multiview_layout")
        if raw_layout is None:
            raise ValueError(
                "multiview_composition=true requires a multiview_layout block in data config."
            )
        layout_spec = _validate_composition_layout(raw_layout)
        declared_height = int(data_config.get("height_buckets", layout_spec["canvas_height"]))
        declared_width = int(data_config.get("width_buckets", layout_spec["canvas_width"]))
        if declared_height != layout_spec["canvas_height"] or declared_width != layout_spec["canvas_width"]:
            raise ValueError(
                "multiview_composition requires height_buckets/width_buckets to match "
                f"canvas dims ({layout_spec['canvas_height']}x{layout_spec['canvas_width']}), "
                f"got {declared_height}x{declared_width}."
            )
        mv_kwargs = dict(dataset_init_kwargs)
        mv_kwargs["layout_spec"] = layout_spec
        wrist_pcd_config = data_config.get("wrist_pcd_config")
        if wrist_pcd_config is not None:
            mv_kwargs["wrist_pcd_config"] = dict(wrist_pcd_config)
        if accelerator.is_main_process:
            logger.info(
                "Using MultiViewCompositeVideoDataset: canvas=%dx%d cells=%d",
                layout_spec["canvas_height"],
                layout_spec["canvas_width"],
                len(layout_spec["cells"]),
            )
        train_dataset = MultiViewCompositeVideoDataset(**mv_kwargs)
    elif multiview_width_concat:
        multiview_views = data_config.get("multiview_views", ["ext1", "ext2"])
        multiview_per_view_width = int(data_config.get("multiview_per_view_width", 832))
        multiview_view_ids: List[int] = []
        for view in multiview_views:
            if isinstance(view, int):
                multiview_view_ids.append(int(view))
                continue
            view_str = str(view).strip().lower()
            if view_str.startswith("ext") and view_str[3:].isdigit():
                multiview_view_ids.append(int(view_str[3:]))
            else:
                raise ValueError(
                    f"multiview_views entries must be of the form 'extN' for now, got: {view}"
                )
        expected_total_width = multiview_per_view_width * len(multiview_view_ids)
        declared_width = int(data_config.get("width_buckets", expected_total_width))
        if declared_width != expected_total_width:
            raise ValueError(
                "multiview_width_concat requires width_buckets to equal "
                f"multiview_per_view_width * len(multiview_views) "
                f"({multiview_per_view_width} * {len(multiview_view_ids)} = {expected_total_width}), "
                f"but width_buckets={declared_width}."
            )
        mv_kwargs = dict(dataset_init_kwargs)
        mv_kwargs["expected_view_ids"] = multiview_view_ids
        mv_kwargs["grouping_mode"] = "ext_suffix"
        mv_kwargs["per_view_width"] = multiview_per_view_width
        if accelerator.is_main_process:
            logger.info(
                "Using MultiViewWidthConcatVideoDataset: views=%s per_view_width=%d total_width=%d",
                multiview_view_ids,
                multiview_per_view_width,
                expected_total_width,
            )
        train_dataset = MultiViewWidthConcatVideoDataset(**mv_kwargs)
    else:
        train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)

    def collate_fn(batch):
        def stack_or_none(key: str):
            if key in batch[0] and batch[0][key] is not None:
                first_tensor = batch[0][key]
                stacked_items = []
                for item in batch:
                    value = item.get(key)
                    if value is None:
                        stacked_items.append(torch.zeros_like(first_tensor))
                    else:
                        stacked_items.append(value)
                return torch.stack(stacked_items)
            return None

        def stack_per_view_list(key: str):
            """For MultiViewPropeDataset, ``key`` carries ``List[Tensor]`` of per-view
            videos with possibly different spatial sizes. Return a list of per-view
            batched tensors (each [B, F, C, H_v, W_v])."""
            if key not in batch[0] or batch[0][key] is None:
                return None
            num_views = len(batch[0][key])
            if any(len(item.get(key) or []) != num_views for item in batch):
                raise RuntimeError(
                    f"All samples must carry the same number of views for key '{key}'"
                )
            return [
                torch.stack([item[key][v_idx] for item in batch], dim=0)
                for v_idx in range(num_views)
            ]

        prope_mode = "video_views" in batch[0]
        if prope_mode:
            return {
                "videos": None,
                "video_views": stack_per_view_list("video_views"),
                "hand_video_views": stack_per_view_list("hand_video_views"),
                "static_video_views": stack_per_view_list("static_video_views"),
                "static_disparity_video_views": stack_per_view_list("static_disparity_video_views"),
                "eef_depth_video_views": stack_per_view_list("eef_depth_video_views"),
                "camera_poses": stack_or_none("camera_poses"),
                "intrinsics": stack_or_none("intrinsics"),
                "view_sizes": stack_or_none("view_sizes"),
                "images": None,
                "fun_inp_i2v_latents": None,
                "clip_image_embeds": None,
                "prompts": (
                    torch.stack([item["prompt"] for item in batch])
                    if isinstance(batch[0]["prompt"], torch.Tensor)
                    else [item["prompt"] for item in batch]
                ),
                "hand_videos": None,
                "static_videos": None,
                "static_disparities": None,
                "hand_disparities": None,
                "state_adaln": None,
                "masks": None,
            }

        return {
            "videos": torch.stack([item["video"] for item in batch]),
            "images": stack_or_none("image"),
            "fun_inp_i2v_latents": stack_or_none("fun_inp_i2v_latents"),
            "clip_image_embeds": stack_or_none("clip_image_embeds"),
            "prompts": (
                torch.stack([item["prompt"] for item in batch])
                if isinstance(batch[0]["prompt"], torch.Tensor)
                else [item["prompt"] for item in batch]
            ),
            "hand_videos": stack_or_none("hand_videos"),
            "static_videos": stack_or_none("static_videos"),
            "static_disparities": stack_or_none("static_disparities"),
            "hand_disparities": stack_or_none("hand_disparities"),
            "state_adaln": stack_or_none("state_adaln"),
            "masks": stack_or_none("masks"),
        }
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch to ensure consistent batch size
        collate_fn=collate_fn,
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    # ========== Prepare with Accelerator ==========
    use_deepspeed_or_fsdp = (fsdp_stage != 0) or (zero_stage != 0)
    
    if args.train_mode in ["lora", "vace_lora"]:
        if use_deepspeed_or_fsdp:
            # Attach network to transformer so DeepSpeed/FSDP can find all params
            transformer3d.network = network
            # FSDP flattens parameters per wrapped unit, so all params inside
            # a unit must share a dtype. Keep the attached LoRA network and
            # frozen backbone in the mixed-precision storage dtype; otherwise
            # 2-GPU FSDP fp32 shards cost about the same VRAM as unsharded bf16.
            transformer3d = transformer3d.to(weight_dtype)
            transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer3d, optimizer, train_dataloader, lr_scheduler
            )
            print(f"📦 Prepared transformer3d with attached LoRA network (DeepSpeed/FSDP mode, train_mode={args.train_mode})")
        else:
            if args.train_mode == "vace_lora":
                # No DeepSpeed/FSDP: prepare both network and transformer
                network, transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    network, transformer3d, optimizer, train_dataloader, lr_scheduler
                )
            else:
                # No DeepSpeed/FSDP: Just prepare network
                network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    network, optimizer, train_dataloader, lr_scheduler
                )
    else:
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )

    # Move models to device (skip transformer3d if already prepared with DeepSpeed/FSDP)
    if not load_tensors:
        # Only keep VAE on device when encoding videos during training
        vae_device_dtype = torch.float32 if use_diffusers_i2v_backend else weight_dtype
        vae.to(accelerator.device, dtype=vae_device_dtype)
    if not (args.train_mode in ["lora", "vace_lora"] and use_deepspeed_or_fsdp):
        transformer3d.to(accelerator.device, dtype=weight_dtype)
    if not load_tensors and text_encoder is not None and not use_preencoded_prompt_embeds:
        # Only keep text_encoder on device when encoding prompts during training
        text_encoder.to(accelerator.device)
    if clip_image_encoder is not None:
        clip_dtype = torch.float32 if use_diffusers_i2v_backend else weight_dtype
        clip_image_encoder.to(accelerator.device, dtype=clip_dtype)

    empty_prompt_embeds = None
    if prompt_mode == PROMPT_MODE_EMPTY or text_dropout_prob > 0.0:
        empty_prompt_embeds = load_or_create_empty_prompt_embeds(
            empty_prompt_embeds_path=empty_prompt_embeds_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_kwargs=text_encoder_kwargs,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
        )
        if (not load_tensors) and text_encoder is not None:
            logger.info("📦 prompt_mode=empty: moving text_encoder to CPU for training.")
            text_encoder = text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(accelerator.device)

    # Free up memory when using pre-encoded latents
    if load_tensors:
        logger.info("📦 load_tensors=True: Deleting VAE and text_encoder to save memory")
        del vae
        vae = None
        del text_encoder
        del tokenizer
        text_encoder = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        logger.info("   VAE and text_encoder deleted. GPU memory freed.")

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ========== Initialize Trackers (WandB) ==========
    if accelerator.is_main_process:
        # Create run name: {date}_{name}_{mode}_{job_id}
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{date_suffix}_{exp_name}_{args.mode}_{slurm_job_id}"
        
        # Prepare tracker config
        tracker_config = {
            "experiment": {
                "name": exp_name,
                "date": exp_date if 'exp_date' in dir() else "unknown",
                "output_dir": args.output_dir,
            },
            "training": {
                "mode": args.train_mode,
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "max_train_steps": args.max_train_steps,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            },
            "model": {
                "base_model_name_or_path": args.pretrained_model_name_or_path,
                "condition_channels": args.condition_channels,
            },
            "system": {
                "num_gpus": accelerator.num_processes,
                "mixed_precision": args.mixed_precision,
            }
        }
        
        # Setup wandb init kwargs
        wandb_init_kwargs = {
            "name": run_name,
        }
        if args.tracker_entity_name:
            wandb_init_kwargs["entity"] = args.tracker_entity_name

        # Prevent frequent startup timeout in restricted/offline environments.
        wandb_init_timeout = float(custom_settings.get("wandb_init_timeout", 180))
        if HAS_WANDB:
            wandb_init_kwargs["settings"] = wandb.Settings(init_timeout=wandb_init_timeout)
        
        # Force offline mode for debug / slurm_test (we don't want their experiments
        # cluttering the cloud project). For ``batch`` mode, defer to the WANDB_MODE env
        # variable that the launcher sets from the yaml ``environment.wandb_mode`` field —
        # this lets aicomputing-style runs (no interactive login) opt into online tracking
        # by exporting WANDB_API_KEY + WANDB_MODE=online in the launcher script.
        if args.mode in ["debug", "slurm_test"]:
            wandb_init_kwargs["mode"] = "offline"
            print(f"🔧 WandB set to offline mode for {args.mode}")
        elif args.mode == "batch":
            env_wandb_mode = os.environ.get("WANDB_MODE")
            if env_wandb_mode:
                wandb_init_kwargs["mode"] = env_wandb_mode
                print(f"🔧 Batch mode: WandB mode={env_wandb_mode} (from env)")
            else:
                wandb_init_kwargs["mode"] = "disabled"
                print(f"🔧 Batch mode: WandB disabled (no WANDB_MODE env set)")
        
        try:
            accelerator.init_trackers(
                project_name=args.tracker_project_name,
                config=tracker_config,
                init_kwargs={"wandb": wandb_init_kwargs}
            )
        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            is_wandb_init_issue = (
                "wandb" in str(args.report_to).lower()
                and (
                    "timeout" in error_text.lower()
                    or "commerror" in error_text.lower()
                    or "timed out waiting for response" in error_text.lower()
                )
            )
            if not is_wandb_init_issue:
                raise

            logger.warning(
                "WandB init failed (%s). Falling back to disabled WandB tracking for this run.",
                error_text,
            )
            fallback_wandb_kwargs = {k: v for k, v in wandb_init_kwargs.items() if k != "settings"}
            fallback_wandb_kwargs["mode"] = "disabled"
            accelerator.init_trackers(
                project_name=args.tracker_project_name,
                config=tracker_config,
                init_kwargs={"wandb": fallback_wandb_kwargs},
            )
            print("🔧 WandB fallback applied: mode=disabled")
        
        print(f"🔗 Experiment initialized: {run_name}")

    # ========== Training Loop ==========
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Resume from checkpoint (supports both config-based and arg-based)
    checkpoint_to_resume = resume_from_checkpoint if 'resume_from_checkpoint' in dir() and resume_from_checkpoint else args.resume_from_checkpoint
    
    if checkpoint_to_resume:
        if checkpoint_to_resume != "latest":
            # Use the provided checkpoint path directly
            checkpoint_path = checkpoint_to_resume
            
            if not os.path.exists(checkpoint_path):
                accelerator.print(f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
                
                # Try to load full accelerator state first, fallback to weight-only loading if it fails
                try:
                    accelerator.load_state(checkpoint_path)
                    accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                except Exception as e:
                    accelerator.print(f"Failed to load full accelerator state: {e}")
                    accelerator.print(f"Falling back to weight-only loading...")
                    
                    if args.train_mode == "lora":
                        # Load LoRA weights
                        lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                        if os.path.exists(lora_path):
                            state_dict = load_file(lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                            print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                        
                        # Load non-LoRA trainable weights (e.g., patch_embedding)
                        non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                        if os.path.exists(non_lora_path):
                            non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                            print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                    elif args.train_mode == "vace":
                        # Load VACE weights (vace_blocks + vace_patch_embedding)
                        vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                        if os.path.exists(vace_path):
                            vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                            print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                    elif args.train_mode == "vace_lora":
                        # Load LoRA + VACE weights
                        lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                        if os.path.exists(lora_path):
                            state_dict = load_file(lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                            print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")

                        vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                        if os.path.exists(vace_path):
                            vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                            print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                    else:
                        # Load transformer weights for SFT
                        transformer_path = os.path.join(checkpoint_path, "transformer")
                        if os.path.exists(transformer_path):
                            if transformer_type == "WanTransformer3DModelWithConcat" and conditioning_mode == "i2v" and use_diffusers_i2v_backend:
                                WanI2VTransformer3DModelWithConcat = _require_wan_i2v_transformer_class()
                                state_dict = WanI2VTransformer3DModelWithConcat.from_pretrained(
                                    transformer_path,
                                    condition_channels=args.condition_channels,
                                ).state_dict()
                            else:
                                state_dict = WanTransformer3DModelWithConcat.from_pretrained(transformer_path).state_dict()
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
                            print(f"Loaded transformer: missing {len(m)}, unexpected {len(u)}")
                
                # Extract global step from checkpoint name
                checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
                if checkpoint_name.startswith("checkpoint-"):
                    global_step = extract_checkpoint_step(checkpoint_name)
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch
                else:
                    accelerator.print(f"Warning: Could not extract global step from checkpoint name '{checkpoint_name}'")
                    initial_global_step = 0
        else:
            # Get the most recent checkpoint from output_dir
            if not os.path.exists(args.output_dir):
                accelerator.print(f"Output directory '{args.output_dir}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                dirs = list_checkpoint_dirs(args.output_dir)
                if len(dirs) == 0:
                    accelerator.print("No checkpoint directories found. Starting a new training run.")
                    checkpoint_to_resume = None
                    initial_global_step = 0
                else:
                    latest_checkpoint = dirs[-1]
                    checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
                    accelerator.print(f"Resuming from latest checkpoint {checkpoint_path}")
                    
                    # Try to load full accelerator state first, fallback to weight-only loading if it fails
                    try:
                        accelerator.load_state(checkpoint_path)
                        accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                    except Exception as e:
                        accelerator.print(f"Failed to load full accelerator state: {e}")
                        accelerator.print(f"Falling back to weight-only loading...")
                        
                        if args.train_mode == "lora":
                            lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                            if os.path.exists(lora_path):
                                state_dict = load_file(lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                                print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                            
                            # Load non-LoRA trainable weights (e.g., patch_embedding)
                            non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                            if os.path.exists(non_lora_path):
                                non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                                print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                        elif args.train_mode == "vace":
                            # Load VACE weights (vace_blocks + vace_patch_embedding)
                            vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                            if os.path.exists(vace_path):
                                vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                                print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                        elif args.train_mode == "vace_lora":
                            lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                            if os.path.exists(lora_path):
                                state_dict = load_file(lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                                print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")

                            vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                            if os.path.exists(vace_path):
                                vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                                print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                        else:
                            # Load transformer weights for SFT
                            transformer_path = os.path.join(checkpoint_path, "transformer")
                            if os.path.exists(transformer_path):
                                if transformer_type == "WanTransformer3DModelWithConcat" and conditioning_mode == "i2v" and use_diffusers_i2v_backend:
                                    WanI2VTransformer3DModelWithConcat = _require_wan_i2v_transformer_class()
                                    state_dict = WanI2VTransformer3DModelWithConcat.from_pretrained(
                                        transformer_path,
                                        condition_channels=args.condition_channels,
                                    ).state_dict()
                                else:
                                    state_dict = WanTransformer3DModelWithConcat.from_pretrained(transformer_path).state_dict()
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
                                print(f"Loaded transformer weights: missing {len(m)}, unexpected {len(u)}")
                    
                    global_step = extract_checkpoint_step(latest_checkpoint)
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch

    # Get step intervals from config (with args as fallback)
    checkpointing_steps = training_config.get("custom_settings", {}).get("checkpointing_steps", args.checkpointing_steps)
    validation_steps = data_config.get("validation_steps", args.validation_steps)
    init_validation_steps = data_config.get("init_validation_steps", 100)
    
    logger.info(f"📊 Step intervals: checkpointing={checkpointing_steps}, validation={validation_steps}, init_validation={init_validation_steps}")
    logger.info(
        "📦 Full-state checkpoint retention: keep_full_state_checkpoints=%s",
        keep_full_state_checkpoints,
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Initialize DiscreteSampling for uniform timestep sampling
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Get patch_size from transformer config (used for seq_len calculation)
    patch_size = accelerator.unwrap_model(transformer3d).config.patch_size

    # VAE mini-batch encoding for memory efficiency
    vae_mini_batch = getattr(args, 'vae_mini_batch', 32)
    
    def _batch_encode_vae(pixel_values):
        """Encode pixel values to latents in mini-batches to save memory."""
        bs = vae_mini_batch
        vae_param_dtype = next(vae.parameters()).dtype
        new_latents = []
        for i in range(0, pixel_values.shape[0], bs):
            pixel_values_bs = pixel_values[i:i + bs].to(dtype=vae_param_dtype)
            latents_bs = vae.encode(pixel_values_bs)[0].sample().to(dtype=weight_dtype)
            new_latents.append(latents_bs)
        return torch.cat(new_latents, dim=0)

    max_validation_videos = int(data_config.get("max_validation_videos", 2) or 0)
    run_initial_validation = bool(custom_settings.get("run_initial_validation", True))

    # Run initial validation at step 0 (before training starts)
    # Run on all GPUs to distribute validation videos (not just main process)
    if (
        run_initial_validation
        and data_config.get("validation_set") is not None
        and max_validation_videos > 0
    ):
        if accelerator.is_main_process:
            logger.info("🎬 Running initial validation at step 0 (before training starts)")
        log_validation(
            vae, text_encoder, tokenizer, clip_image_encoder,
            transformer3d, network, config, args, accelerator,
            weight_dtype, global_step=global_step,
            load_tensors=load_tensors, vae_path=vae_path, vae_kwargs=vae_kwargs,
            text_encoder_path=text_encoder_path, text_encoder_kwargs=text_encoder_kwargs,
            tokenizer_path=tokenizer_path,
        )
    elif (
        not run_initial_validation
        and accelerator.is_main_process
        and data_config.get("validation_set") is not None
        and max_validation_videos > 0
    ):
        logger.info("Skipping initial validation at step 0 because training.custom_settings.run_initial_validation=false")

    requires_hand_videos = transformer_type in [
        "WanTransformer3DModelWithConcat",
        "WanFunControlTransformer3DModelWithConcat",
        "Wan2_2FunControlTransformer3DModelWithConcat",
        "WanTransformer3DVace",
    ]

    # Time-based checkpoint saving (47h45m) for slurm jobs with 48h walltime.
    training_start_time = time.time()
    time_based_checkpoint_saved = False
    TIME_LIMIT_SECONDS = 47 * 3600 + 45 * 60  # 47 hours 45 minutes
    enable_time_limit_checkpoint = (
        args.mode == "slurm"
    )

    def run_post_update_hooks(global_step: int) -> None:
        nonlocal time_based_checkpoint_saved

        elapsed_time = time.time() - training_start_time
        should_save_time_based_checkpoint = (
            enable_time_limit_checkpoint
            and not time_based_checkpoint_saved
            and elapsed_time >= TIME_LIMIT_SECONDS
        )
        if should_save_time_based_checkpoint or global_step % checkpointing_steps == 0:
            if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                if should_save_time_based_checkpoint:
                    time_based_checkpoint_saved = True
                    elapsed_hours = elapsed_time / 3600
                    logger.info(f"⏰ Time-based checkpoint trigger: {elapsed_hours:.2f} hours elapsed (limit: 47.75 hours)")

                # Remove old checkpoint directories before saving a new one.
                if args.checkpoints_total_limit is not None:
                    checkpoints = list_checkpoint_dirs(args.output_dir)
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        if accelerator.is_main_process:
                            for removing_checkpoint in checkpoints[:num_to_remove]:
                                removing_path = os.path.join(args.output_dir, removing_checkpoint)
                                logger.info(f"Removing old checkpoint: {removing_path}")
                                shutil.rmtree(removing_path)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_training_checkpoint(
                    save_path,
                    args.output_dir,
                    accelerator,
                    keep_full_state_checkpoints,
                    lambda path: save_train_mode_weights(
                        path,
                        args,
                        network,
                        weight_dtype,
                        accelerator,
                        transformer3d,
                        trainable_parameter_patterns,
                    ),
                    logger,
                )

        # Validation (runs if validation_set is defined in config)
        # Early phase: every init_validation_steps until first checkpoint; then every validation_steps (CogVideoX style)
        should_run_validation = data_config.get("validation_set") is not None and max_validation_videos > 0 and (
            (global_step % init_validation_steps == 0 and global_step < checkpointing_steps)
            or global_step % validation_steps == 0
        )
        if should_run_validation:
            if accelerator.is_main_process:
                logger.info(f"🎬 Running validation at step {global_step}")
            log_validation(
                vae, text_encoder, tokenizer, clip_image_encoder,
                transformer3d, network, config, args, accelerator,
                weight_dtype, global_step,
                load_tensors=load_tensors, vae_path=vae_path, vae_kwargs=vae_kwargs,
                text_encoder_path=text_encoder_path, text_encoder_kwargs=text_encoder_kwargs,
                tokenizer_path=tokenizer_path,
            )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.train_mode == "vace_lora":
                models_to_accumulate = [network, transformer3d]
            elif args.train_mode == "lora":
                models_to_accumulate = [network]
            else:
                models_to_accumulate = [transformer3d]

            with accelerator.accumulate(models_to_accumulate):
                # ===== PRoPE multi-view branch =====
                # When the dataset yields per-view tensors + camera poses, route training
                # through a compact helper that VAE-encodes each view, builds a per-view
                # DWM/i2v inpaint-condition bundle, and calls the transformer's
                # token-concat multi-view path.
                if batch.get("video_views") is not None:
                    loss = _prope_training_step(
                        batch=batch,
                        transformer3d=transformer3d,
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        accelerator=accelerator,
                        batch_encode_vae=_batch_encode_vae,
                        idx_sampling=idx_sampling,
                        torch_rng=torch_rng,
                        empty_prompt_embeds=empty_prompt_embeds,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        training_config=training_config,
                        args=args,
                        conditioning_mode=conditioning_mode,
                        prompt_mode=prompt_mode,
                        text_dropout_prob=text_dropout_prob,
                        load_tensors=load_tensors,
                        uses_fun_control_y=uses_fun_control_y,
                        is_wan2_2_fun_control=is_wan2_2_fun_control_i2v,
                        i2v_use_hand_condition=i2v_use_hand_condition,
                    )
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if args.train_mode == "lora":
                        grad_sanitize_params = list(network.parameters()) + non_lora_trainable_params
                    elif args.train_mode == "vace_lora":
                        grad_sanitize_params = list(network.parameters()) + vace_trainable_params
                    else:
                        grad_sanitize_params = trainable_params
                    sanitize_nonfinite_gradients(
                        grad_sanitize_params,
                        accelerator=accelerator,
                        label="PRoPE training",
                    )
                    if accelerator.sync_gradients:
                        if args.train_mode == "lora":
                            # Include any non-LoRA trainable params (e.g. patch_embedding,
                            # o_prope) in the clip — they live in their own optimizer
                            # param group and were previously unbounded, which let the
                            # zero-/small-init PRoPE projection blow up its first
                            # gradient (260427 PRoPE divergence root cause per codex
                            # review). Mirrors the vace_lora branch below.
                            clip_params = list(network.parameters()) + non_lora_trainable_params
                            accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                        elif args.train_mode == "vace_lora":
                            clip_params = list(network.parameters()) + vace_trainable_params
                            accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                        else:
                            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0
                        run_post_update_hooks(global_step)
                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    if global_step >= args.max_train_steps:
                        break
                    continue

                # Get batch data
                videos = batch["videos"].to(accelerator.device, dtype=weight_dtype)
                prompts = batch["prompts"]
                images = batch.get("images")
                fun_inp_i2v_latents = batch.get("fun_inp_i2v_latents")
                clip_image_embeds = batch.get("clip_image_embeds")
                hand_videos = batch.get("hand_videos")
                static_videos = batch.get("static_videos")
                static_disparities = batch.get("static_disparities")
                eef_depths = batch.get("hand_disparities")
                state_adaln = batch.get("state_adaln")
                loss_masks = batch.get("masks")

                # ===== Encode videos to latents =====
                videos_bcfhw = None
                if load_tensors:
                    # Pre-encoded latents: already [B, C, F', H', W'] from dataset
                    # encode_with_wan.py already called .sample(), no need for VAE
                    latents = videos  # Already latents
                else:
                    # Raw videos: [B, F, C, H, W] -> encode to latents
                    videos_bcfhw = rearrange(videos, "b f c h w -> b c f h w")
                    with torch.no_grad():
                        latents = _batch_encode_vae(videos_bcfhw)

                loss_mask_latents = None
                if loss_masks is not None:
                    loss_masks = loss_masks.to(accelerator.device, dtype=weight_dtype)
                    loss_masks = rearrange(loss_masks, "b f c h w -> b c f h w")
                    loss_mask_latents = resize_mask(
                        loss_masks,
                        latents,
                        process_first_frame_only=False,
                    )
                    loss_mask_latents = (loss_mask_latents >= 0.5).to(dtype=weight_dtype)

                bsz, latent_channels, latent_frames, latent_height, latent_width = latents.size()
                state_adaln_latents = resize_state_adaln_to_latent_frames(
                    state_adaln,
                    latent_frames,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    temporal_mode=data_config.get("state_adaln_temporal_mode", "sample4"),
                )
                if transformer_type == "WanTransformer3DModelWithStateAdaLN" and state_adaln_latents is None:
                    raise ValueError("WanTransformer3DModelWithStateAdaLN requires batch['state_adaln']")

                # ===== Encode condition videos =====
                dropout_mask = None  # Initialize dropout_mask
                use_hand_condition = requires_hand_videos and (
                    conditioning_mode == "dwm" or i2v_use_hand_condition
                )
                if use_hand_condition and hand_videos is not None:
                    if load_tensors:
                        # Pre-encoded hand latents: already [B, C, F', H', W']
                        hand_latents = hand_videos.to(accelerator.device, dtype=weight_dtype)
                    else:
                        # Raw hand videos: encode to latents
                        hand_videos = hand_videos.to(accelerator.device, dtype=weight_dtype)
                        hand_videos = rearrange(hand_videos, "b f c h w -> b c f h w")
                        with torch.no_grad():
                            hand_latents = _batch_encode_vae(hand_videos)
                else:
                    hand_latents = None

                if static_disparities is not None:
                    if load_tensors:
                        static_disparity_latents = static_disparities.to(
                            accelerator.device, dtype=weight_dtype
                        )
                        if static_disparity_i2v_first_frame_only:
                            first_frame_latents = static_disparity_latents[:, :, 0:1, :, :]
                            if first_frame_latents.shape[-2:] != (latent_height, latent_width):
                                first_frame_latents = F.interpolate(
                                    first_frame_latents.squeeze(2),
                                    size=(latent_height, latent_width),
                                    mode="bilinear",
                                    align_corners=False,
                                ).unsqueeze(2)
                            static_disparity_latents = torch.zeros(
                                bsz,
                                first_frame_latents.shape[1],
                                latent_frames,
                                latent_height,
                                latent_width,
                                device=accelerator.device,
                                dtype=weight_dtype,
                            )
                            static_disparity_latents[:, :, 0:1, :, :] = first_frame_latents[:, :, 0:1, :, :]
                    else:
                        static_disparities = static_disparities.to(
                            accelerator.device, dtype=weight_dtype
                        )
                        static_disparities = rearrange(
                            static_disparities, "b f c h w -> b c f h w"
                        )
                        with torch.no_grad():
                            if static_disparity_i2v_first_frame_only:
                                first_frame_latents = _batch_encode_vae(static_disparities[:, :, 0:1, :, :])
                            else:
                                static_disparity_latents = _batch_encode_vae(static_disparities)
                        if static_disparity_i2v_first_frame_only:
                            if first_frame_latents.shape[-2:] != (latent_height, latent_width):
                                first_frame_latents = F.interpolate(
                                    first_frame_latents.squeeze(2),
                                    size=(latent_height, latent_width),
                                    mode="bilinear",
                                    align_corners=False,
                                ).unsqueeze(2)
                            static_disparity_latents = torch.zeros(
                                bsz,
                                first_frame_latents.shape[1],
                                latent_frames,
                                latent_height,
                                latent_width,
                                device=accelerator.device,
                                dtype=weight_dtype,
                            )
                            static_disparity_latents[:, :, 0:1, :, :] = first_frame_latents[:, :, 0:1, :, :]
                else:
                    static_disparity_latents = None

                if eef_depths is not None:
                    if load_tensors:
                        eef_depth_latents = eef_depths.to(
                            accelerator.device, dtype=weight_dtype
                        )
                    else:
                        eef_depths = eef_depths.to(
                            accelerator.device, dtype=weight_dtype
                        )
                        eef_depths = rearrange(
                            eef_depths, "b f c h w -> b c f h w"
                        )
                        with torch.no_grad():
                            eef_depth_latents = _batch_encode_vae(eef_depths)
                else:
                    eef_depth_latents = None

                control_latents = None
                hand_condition_tensors = []
                if is_fun_control_transformer:
                    if hand_latents is None:
                        raise ValueError(
                            "Wan Fun-Control requires robot mesh/control videos. "
                            "Check data.robot_mesh_video_subdir and training.i2v_use_hand_condition=true."
                        )
                    control_latents = hand_latents
                    if eef_depth_latents is not None:
                        hand_condition_tensors.append(eef_depth_latents)
                else:
                    if hand_latents is not None:
                        hand_condition_tensors.append(hand_latents)
                    if eef_depth_latents is not None:
                        eef_depth_dropout_prob = float(training_config.get("eef_depth_dropout_prob", 0.0) or 0.0)
                        if eef_depth_dropout_prob > 0:
                            eef_mask = torch.rand(eef_depth_latents.shape[0], device=accelerator.device) < eef_depth_dropout_prob
                            if eef_mask.any():
                                eef_depth_latents = eef_depth_latents.clone()
                                eef_depth_latents[eef_mask] = torch.zeros_like(eef_depth_latents[eef_mask])
                                if global_step % 100 == 0:
                                    accelerator.log({"eef_depth_dropout_count": int(eef_mask.sum())}, step=global_step)
                        hand_condition_tensors.append(eef_depth_latents)

                robot_dropout_prob = training_config.get("robot_dropout_prob", 0.0)
                if hand_condition_tensors and robot_dropout_prob > 0:
                    hand_batch_size = hand_condition_tensors[0].shape[0]
                    dropout_mask = (
                        torch.rand(hand_batch_size, device=accelerator.device) < robot_dropout_prob
                    )

                    if dropout_mask.any():
                        for tensor in hand_condition_tensors:
                            tensor[dropout_mask] = torch.zeros_like(tensor[dropout_mask])
                        if global_step % 100 == 0:
                            num_dropped = dropout_mask.sum().item()
                            accelerator.log({"hand_dropout_count": num_dropped}, step=global_step)

                static_latents = None

                # Note: assertion `wan2.2_fun_control_i2v_mesh requires
                # conditioning_mode='i2v'` removed 2026-05-12 by user request to
                # allow single-view Wan2.2 Fun-Control + DWM training. The DWM
                # branch immediately below handles static_videos / mask
                # construction independently of the I2V code path. The
                # multi-view PRoPE path has been running DWM with this same
                # pipeline.type since 2026-05-10 (`exps/260510/wan2.2_5b_mv_*_dwm_*.yaml`).
                if False and is_wan2_2_fun_control_i2v and conditioning_mode != "i2v" and not multiview_prope:
                    raise ValueError("wan2.2_fun_control_i2v_mesh requires training.conditioning_mode='i2v'.")

                if conditioning_mode == "dwm":
                    if static_videos is None:
                        if load_tensors or static_video_source_mode != "copy_first_frame":
                            raise ValueError("conditioning_mode='dwm' requires static_videos in batch")
                        static_videos = videos[:, 0:1, :, :, :].repeat(1, videos.shape[1], 1, 1, 1)

                    if load_tensors:
                        static_latents = static_videos.to(accelerator.device, dtype=weight_dtype)
                    else:
                        static_videos = static_videos.to(accelerator.device, dtype=weight_dtype)
                        static_videos = rearrange(static_videos, "b f c h w -> b c f h w")
                        with torch.no_grad():
                            static_latents = _batch_encode_vae(static_videos)

                    mask_latents = torch.ones(
                        bsz, 4, latent_frames, latent_height, latent_width,
                        device=accelerator.device, dtype=weight_dtype
                    )
                    inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)
                elif conditioning_mode == "i2v":
                    if load_tensors and i2v_condition_latent_type == "fun_inp":
                        if fun_inp_i2v_latents is None:
                            raise ValueError(
                                "i2v_condition_latent_type='fun_inp' requires precomputed "
                                "fun_inp_i2v_latents. Check data.fun_inp_i2v_latents_subdir and re-encode data."
                            )
                        static_latents = fun_inp_i2v_latents.to(accelerator.device, dtype=weight_dtype)
                        if static_latents.ndim != 5:
                            raise ValueError(
                                f"Unexpected fun_inp_i2v_latents shape: {static_latents.shape}. "
                                "Expected [B, C, F, H, W]."
                            )
                        if static_latents.shape[2:] != (latent_frames, latent_height, latent_width):
                            raise ValueError(
                                "fun_inp_i2v_latents shape mismatch with target latents: "
                                f"got {tuple(static_latents.shape[2:])}, "
                                f"expected {(latent_frames, latent_height, latent_width)}."
                            )
                    elif (not load_tensors) and i2v_condition_latent_type == "fun_inp":
                        if videos_bcfhw is None:
                            raise ValueError(
                                "Raw i2v fun_inp conditioning expects decoded videos before VAE encoding."
                            )

                        # Match encode_fun_inp_i2v_video(): keep frame 0, zero later frames,
                        # then encode the full conditioning video with the causal VAE.
                        conditioned_videos = torch.zeros_like(videos_bcfhw)
                        conditioned_videos[:, :, 0:1, :, :] = videos_bcfhw[:, :, 0:1, :, :]
                        with torch.no_grad():
                            static_latents = _batch_encode_vae(conditioned_videos)

                        if static_latents.shape[2:] != (latent_frames, latent_height, latent_width):
                            raise ValueError(
                                "Raw fun_inp i2v latent shape mismatch with target latents: "
                                f"got {tuple(static_latents.shape[2:])}, "
                                f"expected {(latent_frames, latent_height, latent_width)}."
                            )
                    else:
                        if load_tensors:
                            # Prefer precomputed image latents when present, otherwise use first frame of video latents.
                            first_frame_latents = images if images is not None else latents
                            first_frame_latents = first_frame_latents.to(accelerator.device, dtype=weight_dtype)
                            if first_frame_latents.ndim == 4:
                                first_frame_latents = first_frame_latents.unsqueeze(2)
                            elif first_frame_latents.ndim == 5:
                                first_frame_latents = first_frame_latents[:, :, 0:1, :, :]
                            else:
                                raise ValueError(
                                    f"Unexpected latent shape for i2v first-frame conditioning: {first_frame_latents.shape}"
                                )
                        else:
                            if videos_bcfhw is None:
                                raise ValueError(
                                    "Raw i2v conditioning expects decoded videos before VAE encoding."
                                )

                            with torch.no_grad():
                                first_frame_latents = _batch_encode_vae(
                                    videos_bcfhw[:, :, 0:1, :, :]
                                )

                        if first_frame_latents.shape[-2:] != (latent_height, latent_width):
                            first_frame_latents = F.interpolate(
                                first_frame_latents.squeeze(2),
                                size=(latent_height, latent_width),
                                mode="bilinear",
                                align_corners=False,
                            ).unsqueeze(2)

                        static_latents = torch.zeros(
                            bsz, latent_channels, latent_frames, latent_height, latent_width,
                            device=accelerator.device, dtype=weight_dtype
                        )
                        static_latents[:, :, 0:1, :, :] = first_frame_latents[:, :, 0:1, :, :]

                    mask_latents = torch.zeros(
                        bsz, 4, latent_frames, latent_height, latent_width,
                        device=accelerator.device, dtype=weight_dtype
                    )
                    mask_latents[:, :, 0:1, :, :] = 1.0
                    inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)
                else:
                    raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")

                expected_condition_channels = int(getattr(args, "condition_channels", 0) or 0)
                condition_tensors = []
                if static_disparity_latents is not None:
                    condition_tensors.append(static_disparity_latents)
                condition_tensors.extend(hand_condition_tensors)
                condition_latents = torch.cat(condition_tensors, dim=1) if condition_tensors else None
                model_y = inpaint_latents
                if is_wan_fun_control:
                    if control_latents is None:
                        raise ValueError("Wan2.1 Fun-Control path requires control_latents.")
                    if control_latents.shape[2:] != static_latents.shape[2:]:
                        raise ValueError(
                            "Control latent shape mismatch with static latents: "
                            f"control={tuple(control_latents.shape)}, static={tuple(static_latents.shape)}"
                        )
                    model_y = torch.cat([control_latents, static_latents], dim=1)
                elif is_wan2_2_fun_control_i2v:
                    if control_latents is None:
                        raise ValueError("Wan2.2 Fun-Control path requires control_latents.")
                    if control_latents.shape[2:] != inpaint_latents.shape[2:]:
                        raise ValueError(
                            "Control latent shape mismatch with inpaint latents: "
                            f"control={tuple(control_latents.shape)}, inpaint={tuple(inpaint_latents.shape)}"
                        )
                    model_y = torch.cat([control_latents, inpaint_latents], dim=1)

                if expected_condition_channels > 0:
                    if condition_latents is None:
                        raise ValueError(
                            f"transformer.condition_channels={expected_condition_channels} "
                            "but no condition latents were loaded."
                        )
                    actual_condition_channels = int(condition_latents.shape[1])
                    if actual_condition_channels != expected_condition_channels:
                        raise ValueError(
                            "Condition channel mismatch: "
                            f"expected {expected_condition_channels}, got {actual_condition_channels}. "
                            "Check hand/static disparity latent subdir settings."
                        )

                use_empty_prompt = prompt_mode == PROMPT_MODE_EMPTY
                if not use_empty_prompt and text_dropout_prob > 0.0:
                    use_empty_prompt = torch.rand(1).item() < text_dropout_prob
                if use_empty_prompt and global_step % 100 == 0:
                    accelerator.log({"empty_prompt_count": int(bsz)}, step=global_step)

                if use_empty_prompt:
                    if empty_prompt_embeds is None:
                        raise ValueError("prompt_mode=empty but empty_prompt_embeds is not initialized.")

                    empty_prompt_embeds_device = empty_prompt_embeds.to(
                        accelerator.device, dtype=weight_dtype
                    )
                    if isinstance(prompts, list):
                        batch_prompt_count = len(prompts)
                    elif isinstance(prompts, torch.Tensor):
                        batch_prompt_count = int(prompts.shape[0])
                    else:
                        batch_prompt_count = int(bsz)
                    prompt_embeds = [empty_prompt_embeds_device.clone() for _ in range(batch_prompt_count)]
                else:
                    # Encode prompts only when prompt_mode=normal.
                    with torch.no_grad():
                        if load_tensors or use_preencoded_prompt_embeds:
                            # Pre-encoded embeddings from dataset
                            # prompts is a list of tensors [seq_len_i, dim] from collate (variable lengths)
                            # or a stacked tensor [B, seq_len, dim] if same lengths
                            if isinstance(prompts, torch.Tensor):
                                # If stacked tensor, convert to list for WAN transformer
                                prompt_embeds = [p.to(accelerator.device, dtype=weight_dtype) for p in prompts]
                            elif isinstance(prompts, list) and all(isinstance(p, torch.Tensor) for p in prompts):
                                # Already a list of tensors
                                prompt_embeds = [p.to(accelerator.device, dtype=weight_dtype) for p in prompts]
                            else:
                                raise TypeError(
                                    "Expected preencoded prompt tensors when "
                                    "data.use_preencoded_prompt_embeds=true."
                                )
                        elif isinstance(prompts, list):
                            if text_encoder is not None:
                                text_encoder_device = next(text_encoder.parameters()).device
                                if text_encoder_device != accelerator.device:
                                    logger.info(
                                        "📦 Moving text encoder back to GPU for training prompt encoding."
                                    )
                                    text_encoder = text_encoder.to(accelerator.device)
                            prompt_ids = tokenizer(
                                prompts,
                                padding="max_length",
                                max_length=args.tokenizer_max_length,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt"
                            )
                            text_input_ids = prompt_ids.input_ids.to(latents.device)
                            prompt_attention_mask = prompt_ids.attention_mask.to(latents.device)
                            seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                            prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
                            prompt_embeds = [u[:v].to(weight_dtype) for u, v in zip(prompt_embeds, seq_lens)]
                        else:
                            prompt_embeds = prompts.to(dtype=weight_dtype)

                if (
                    text_encoder is not None
                    and not load_tensors
                    and not bool(getattr(args, "train_text_encoder", False))
                    and bool(custom_settings.get("offload_text_encoder_after_prompt_encoding", True))
                ):
                    text_encoder_device = next(text_encoder.parameters()).device
                    if text_encoder_device.type == "cuda":
                        logger.info("📦 Moving text encoder to CPU after training prompt encoding.")
                        text_encoder = text_encoder.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()

                clip_context = None
                if conditioning_mode == "i2v":
                    if clip_image_embeds is not None:
                        clip_context = clip_image_embeds.to(accelerator.device, dtype=weight_dtype)
                    elif not load_tensors:
                        with torch.no_grad():
                            clip_context = encode_clip_context_batch(
                                clip_image_encoder,
                                images.to(accelerator.device, dtype=weight_dtype) if images is not None else None,
                                weight_dtype,
                            )

                # Sample noise
                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                # Sample timesteps
                if not args.uniform_sampling:
                    # Use density-based sampling (weighting scheme based)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Use DiscreteSampling for uniform sampling (supports distributed training)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()

                timesteps = noise_scheduler.timesteps[indices.cpu()].to(device=latents.device)

                # Get sigmas and create noisy latents (Flow Matching)
                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=accelerator.device)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Flow Matching target
                # If hand_latents were dropped out, use static_latents as target instead of full latents
                # This teaches the model to generate static-only content when hand info is missing
                if dropout_mask is not None and dropout_mask.any() and static_latents is not None:
                    # For dropped samples, use static_latents as target
                    # target shape: [B, 16, F, H, W]
                    target = noise - latents.clone()
                    # Replace target for dropped samples with static-based target
                    target[dropout_mask] = noise[dropout_mask] - static_latents[dropout_mask]
                else:
                    target = noise - latents

                # Calculate seq_len for transformer (based on patched dimensions)
                seq_len = math.ceil(
                    (height / patch_size[1]) * (width / patch_size[2]) * (num_frames / patch_size[0])
                )

                # Forward pass
                # x: noisy_latents [B, 16, F, H, W]
                # y: inpaint_latents [B, 17, F, H, W] (mask + static)
                # condition_latents: concatenated extra conditions [B, C_cond, F, H, W]
                #
                # WanTransformer3DModelWithConcat.forward handles conversion to list format
                # and concatenation of condition_latents
                with torch.amp.autocast(device_type='cuda', dtype=weight_dtype):
                    if transformer_type == "WanTransformer3DModel":
                        # Base model: no condition_latents
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                        )
                    elif transformer_type == "WanTransformer3DModelWithConcat":
                        # Concat model: with condition_latents
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                            condition_latents=condition_latents,
                        )
                    elif transformer_type == "WanTransformer3DModelWithStateAdaLN":
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                            state_adaln=state_adaln_latents,
                        )
                    elif _is_fun_control_transformer_type(transformer_type):
                        # Fun-Control keeps the pretrained y layout:
                        # control_video_latents + inpaint info. Wan2.1 uses
                        # 16-channel video latents, Wan2.2 5B uses 48-channel
                        # video latents; config in_dim preserves that base shape.
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=model_y,
                            clip_fea=clip_context,
                            condition_latents=condition_latents,
                        )
                    elif transformer_type == "WanTransformer3DVace":
                        # VACE model: hand condition via vace_context (list format)
                        # Convert hand_latents to list format for VACE
                        if hand_latents is not None:
                            vace_context = [hand_latents[i] for i in range(hand_latents.shape[0])]
                        else:
                            vace_context = None

                        # Get vace_context_scale from config (default 1.0)
                        vace_context_scale = training_config.get("vace_context_scale", 1.0)

                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                            vace_context=vace_context,
                            vace_context_scale=vace_context_scale,
                        )
                    else:
                        raise ValueError(f"Unsupported transformer type: {transformer_type}")

                # Compute loss with weighting
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(
                    noise_pred.float(),
                    target.float(),
                    weighting.float(),
                    spatial_mask=loss_mask_latents.float() if loss_mask_latents is not None else None,
                )
                loss = loss.mean()

                # Gather losses
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.train_mode == "lora":
                        # See comment at the matching block above (PRoPE branch). Clip
                        # both LoRA and non-LoRA trainable params together so o_prope /
                        # patch_embedding gradients are bounded.
                        clip_params = list(network.parameters()) + non_lora_trainable_params
                        accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                    elif args.train_mode == "vace_lora":
                        clip_params = list(network.parameters()) + vace_trainable_params
                        accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                    else:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                run_post_update_hooks(global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_training_checkpoint(
            save_path,
            args.output_dir,
            accelerator,
            keep_full_state_checkpoints,
            lambda path: save_train_mode_weights(
                path,
                args,
                network,
                weight_dtype,
                accelerator,
                transformer3d,
                trainable_parameter_patterns,
            ),
            logger,
        )

    if accelerator.is_main_process:
        logger.info(f"Training completed. Final checkpoint saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
