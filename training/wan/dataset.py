import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from training.wan.robot_state_conditioning import load_robot_state_condition


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]
MULTIVIEW_EXT_PATTERN = re.compile(r"^(?P<base>.+)__ext(?P<view>[0-9]+)$")
MULTIVIEW_SUFFIX_PATTERN = re.compile(
    r"^(?P<base>.+?)__(?P<suffix>[A-Za-z][A-Za-z0-9]*)$"
)
MULTIVIEW_CLIP_SPAN_PATTERN = re.compile(r"^c[0-9]+_(?P<start>[0-9]+)-(?P<end>[0-9]+)$")

_FILL_NAME_TO_UINT8 = {
    "black": 0,
    "white": 255,
}

_COMPOSITION_CONDITION_DEFAULT_FILL = {
    "video": "black",
    "static_video": "black",
    "static_disparity": "black",
    "eef_depth": "black",
    "hand_video": "white",
}


def _parse_multiview_suffix_path(
    path: Path,
    data_root: Path,
    allowed_suffixes: set[str],
) -> Optional[Tuple[str, str]]:
    """Parse `{base}__{view}.mp4` and clip-level `{base}__{view}__{clip}.mp4`.

    The clip-level form must keep the clip id in the group key; otherwise
    different 81-frame clips from the same episode collapse into one group.
    """
    parts = path.stem.split("__")
    base_name: Optional[str] = None
    suffix: Optional[str] = None
    if len(parts) >= 2 and parts[-1] in allowed_suffixes:
        base_name = "__".join(parts[:-1])
        suffix = parts[-1]
    elif len(parts) >= 3 and parts[-2] in allowed_suffixes:
        base_name = "__".join(parts[:-2] + [parts[-1]])
        suffix = parts[-2]

    if not base_name or suffix is None:
        return None
    try:
        rel_parent = path.relative_to(data_root).parent.as_posix()
    except ValueError:
        rel_parent = path.parent.as_posix()
    pair_key = base_name if rel_parent in {"", "."} else f"{rel_parent}/{base_name}"
    return pair_key, suffix


def _strip_multiview_suffix_and_clip(stem: str) -> str:
    parts = stem.split("__")
    if len(parts) >= 3 and MULTIVIEW_CLIP_SPAN_PATTERN.match(parts[-1]):
        return "__".join(parts[:-2])
    if len(parts) >= 2:
        return "__".join(parts[:-1])
    return stem


def _clip_frame_indices_from_stem(
    stem: str,
    *,
    num_frames: int,
    source_len: int,
) -> Optional[np.ndarray]:
    parts = stem.split("__")
    if not parts:
        return None
    match = MULTIVIEW_CLIP_SPAN_PATTERN.match(parts[-1])
    if match is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    if source_len <= 0 or end < start:
        return None
    indices = np.linspace(start, end, int(num_frames)).round().astype(np.int64)
    return np.clip(indices, 0, source_len - 1)


def _depth_frame_to_turbo_rgb(*args, **kwargs) -> np.ndarray:
    """Backwards-compatible wrapper. The canonical implementation lives in
    `data_processing/droid/depth_colormap.py` so the offline disparity-video
    preprocessor and this runtime path agree byte-for-byte. Resolves the import
    by inserting the repo root onto sys.path on demand — accelerate-launched
    training does not always have cwd or repo root in sys.path."""
    try:
        from data_processing.droid.depth_colormap import depth_frame_to_turbo_rgb
    except ImportError:
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from data_processing.droid.depth_colormap import depth_frame_to_turbo_rgb
    return depth_frame_to_turbo_rgb(*args, **kwargs)


def _validate_composition_layout(layout_spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(layout_spec, dict):
        raise ValueError("multiview_layout must be a mapping.")
    canvas_h = int(layout_spec.get("canvas_height", 0))
    canvas_w = int(layout_spec.get("canvas_width", 0))
    if canvas_h <= 0 or canvas_w <= 0:
        raise ValueError(
            f"canvas_height/canvas_width must be positive, got {canvas_h}x{canvas_w}"
        )
    if canvas_h % 16 != 0 or canvas_w % 16 != 0:
        raise ValueError(
            f"canvas dims must be multiples of 16, got {canvas_h}x{canvas_w}"
        )
    cells_raw = layout_spec.get("cells")
    if not cells_raw:
        raise ValueError("multiview_layout.cells must not be empty.")
    normalized_cells: List[Dict[str, Any]] = []
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    for idx, cell in enumerate(cells_raw):
        if not isinstance(cell, dict):
            raise ValueError(f"cell[{idx}] must be a mapping, got {type(cell).__name__}")
        bbox = cell.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError(f"cell[{idx}].bbox must be [y, x, h, w]")
        y, x, h, w = [int(v) for v in bbox]
        if min(y, x, h, w) < 0:
            raise ValueError(f"cell[{idx}].bbox has negative values: {bbox}")
        if y + h > canvas_h or x + w > canvas_w:
            raise ValueError(
                f"cell[{idx}].bbox extends beyond canvas: bbox={bbox} canvas={canvas_h}x{canvas_w}"
            )
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"cell[{idx}].bbox h/w must be multiples of 16: {bbox}")
        coverage[y:y + h, x:x + w] += 1
        has_suffix = "suffix" in cell
        has_pool = "suffix_pool" in cell
        has_fill = "fill" in cell
        src_count = int(has_suffix) + int(has_pool) + int(has_fill)
        if src_count != 1:
            raise ValueError(
                f"cell[{idx}] must have exactly one of suffix/suffix_pool/fill, "
                f"got suffix={has_suffix} suffix_pool={has_pool} fill={has_fill}"
            )
        normalized: Dict[str, Any] = {"bbox": (y, x, h, w)}
        if has_suffix:
            normalized["suffix"] = str(cell["suffix"]).strip()
            if not normalized["suffix"]:
                raise ValueError(f"cell[{idx}].suffix is empty")
        elif has_pool:
            pool = [str(s).strip() for s in cell["suffix_pool"] if str(s).strip()]
            if not pool:
                raise ValueError(f"cell[{idx}].suffix_pool is empty")
            normalized["suffix_pool"] = pool
        else:
            fill = cell.get("fill") or {}
            if not isinstance(fill, dict):
                raise ValueError(f"cell[{idx}].fill must be a mapping")
            normalized_fill: Dict[str, str] = {}
            for key, value in fill.items():
                key_str = str(key)
                value_str = str(value).strip().lower()
                if value_str not in _FILL_NAME_TO_UINT8:
                    raise ValueError(
                        f"cell[{idx}].fill.{key_str} has unsupported color '{value}'. "
                        f"Supported: {sorted(_FILL_NAME_TO_UINT8)}"
                    )
                normalized_fill[key_str] = value_str
            normalized["fill"] = normalized_fill
        normalized_cells.append(normalized)
    if int(coverage.min()) < 1:
        raise ValueError("multiview_layout cells leave gaps on the canvas.")
    if int(coverage.max()) > 1:
        raise ValueError("multiview_layout cells overlap.")
    return {
        "canvas_height": canvas_h,
        "canvas_width": canvas_w,
        "cells": normalized_cells,
    }


def _normalize_exclude_videos_files(exclude_videos_file: Optional[Any]) -> List[str]:
    if not exclude_videos_file:
        return []
    if isinstance(exclude_videos_file, (str, Path)):
        return [str(exclude_videos_file)]
    try:
        return [str(path) for path in exclude_videos_file if path]
    except TypeError:
        return [str(exclude_videos_file)]


def _normalize_temporal_frame_skip_choices(
    temporal_frame_skip_choices: Optional[Any],
) -> List[int]:
    if temporal_frame_skip_choices is None:
        return []
    if isinstance(temporal_frame_skip_choices, (int, np.integer)):
        choices = [int(temporal_frame_skip_choices)]
    else:
        try:
            choices = [int(choice) for choice in temporal_frame_skip_choices]
        except TypeError:
            choices = [int(temporal_frame_skip_choices)]

    normalized_choices: List[int] = []
    seen = set()
    for choice in choices:
        if choice <= 0:
            raise ValueError(f"temporal_frame_skip_choices must be positive, got {choice}")
        if choice in seen:
            continue
        normalized_choices.append(choice)
        seen.add(choice)
    return normalized_choices


def _normalize_temporal_frame_skip_probs(
    temporal_frame_skip_probs: Optional[Any],
    num_choices: int,
) -> Optional[List[float]]:
    if temporal_frame_skip_probs is None:
        return None
    try:
        probs = [float(prob) for prob in temporal_frame_skip_probs]
    except TypeError:
        probs = [float(temporal_frame_skip_probs)]

    if len(probs) != num_choices:
        raise ValueError(
            "temporal_frame_skip_probs must have the same length as temporal_frame_skip_choices: "
            f"{len(probs)} != {num_choices}"
        )
    if any(prob < 0 for prob in probs):
        raise ValueError("temporal_frame_skip_probs must be non-negative")

    total_prob = float(sum(probs))
    if total_prob <= 0:
        raise ValueError("temporal_frame_skip_probs must sum to a positive value")
    return [prob / total_prob for prob in probs]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
        prompt_subdir: str = "prompts",
        prompt_embeds_subdir: str = "prompt_embeds_fun_rewrite_wan",
        use_preencoded_prompt_embeds: bool = False,
        prompt_json_fields: Optional[List[str]] = None,
        static_video_subdir: str = "videos_static",
        robot_mesh_video_subdir: str = "videos_hands",
        robot_mesh_video_latents_subdir: str = "hand_video_latents_wan",
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        image_latents_subdir: str = "image_latents_wan",
        fun_inp_i2v_latents_subdir: str = "fun_inp_i2v_latents_wan",
        clip_image_embeds_subdir: str = "clip_image_embeds",
        exclude_videos_file: Optional[Any] = None,
        eager_load_prompts: bool = True,
        align_width_to_32: bool = False,
        max_prompt_length: int = 512,
        clip_start_frame_stride: int = 1,
        temporal_frame_skip_choices: Optional[Any] = None,
        temporal_frame_skip_probs: Optional[Any] = None,
        action_density_index: Optional[str] = None,
        clip_sampling: str = "uniform",
        action_sampling_alpha: float = 0.5,
        action_sampling_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.align_width_to_32 = align_width_to_32
        self.max_prompt_length = max_prompt_length
        self.clip_start_frame_stride = max(1, int(clip_start_frame_stride))
        self.temporal_frame_skip_choices = _normalize_temporal_frame_skip_choices(
            temporal_frame_skip_choices
        )
        self.temporal_frame_skip_probs = _normalize_temporal_frame_skip_probs(
            temporal_frame_skip_probs,
            len(self.temporal_frame_skip_choices),
        )
        self.clip_sampling = str(clip_sampling)
        self.action_sampling_alpha = float(action_sampling_alpha)
        self.action_sampling_eps = float(action_sampling_eps)
        self.action_density_index_path = action_density_index
        # Lazily loaded; see _ensure_action_density_index()
        self._action_density_index: Optional[Dict[str, Dict[str, Any]]] = None

        self.data_root = Path(data_root)
        # Handle dataset_file as list or single path
        if dataset_file is None:
            self.dataset_file_list: Optional[List[str]] = None
        elif isinstance(dataset_file, (list, tuple)):
            self.dataset_file_list = [str(path) for path in dataset_file]
        else:
            self.dataset_file_list = [str(dataset_file)]
        self.dataset_file = dataset_file
        self.caption_column = caption_column  # not used
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video
        self.prompt_subdir = prompt_subdir
        self.prompt_embeds_subdir = prompt_embeds_subdir
        self.use_preencoded_prompt_embeds = bool(use_preencoded_prompt_embeds)
        if prompt_json_fields:
            self.prompt_json_fields = [str(field) for field in prompt_json_fields]
        elif "qwen3vl" in str(prompt_subdir).lower():
            self.prompt_json_fields = ["general.text"]
        else:
            self.prompt_json_fields = None
        self.static_video_subdir = static_video_subdir
        self.robot_mesh_video_subdir = robot_mesh_video_subdir
        self.robot_mesh_video_latents_subdir = robot_mesh_video_latents_subdir
        self.video_latents_subdir = video_latents_subdir
        self.static_video_latents_subdir = static_video_latents_subdir
        self.image_latents_subdir = image_latents_subdir
        self.fun_inp_i2v_latents_subdir = fun_inp_i2v_latents_subdir
        self.clip_image_embeds_subdir = clip_image_embeds_subdir
        self.exclude_videos_file = exclude_videos_file
        self.eager_load_prompts = bool(eager_load_prompts)
        self.excluded_videos = self._load_excluded_video_entries(exclude_videos_file)
        
        # Ensure buckets are lists
        if not isinstance(self.height_buckets, list):
            self.height_buckets = [self.height_buckets]
        if not isinstance(self.width_buckets, list):
            self.width_buckets = [self.width_buckets]
        if not isinstance(self.frame_buckets, list):
            self.frame_buckets = [self.frame_buckets]
        
        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if self.dataset_file_list is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_datafile()
            # ) = self._load_dataset_from_csv()

        if self.excluded_videos:
            self.prompts, self.video_paths = self._apply_exclude_video_filter(self.prompts, self.video_paths)

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        if self.temporal_frame_skip_choices:
            logger.info(
                "Enabled temporal frame skip sampling: choices=%s probs=%s",
                self.temporal_frame_skip_choices,
                self.temporal_frame_skip_probs or "uniform",
            )

    def _find_nearest_resolution(self, height: int, width: int) -> Tuple[int, int]:
        """Pick the nearest (H, W) resolution bucket, optionally aligning W to a multiple of 32.

        Note: alignment is only meaningful when a subclass actually resizes frames to the chosen bucket.
        """
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        nearest_h, nearest_w = nearest_res[1], nearest_res[2]

        # For WAN 2.2: align width to 32 for compatibility
        # 32 = 8 (VAE spatial_compression) * 4, ensures clean division through the pipeline
        if getattr(self, "align_width_to_32", False):
            # Always round to nearest multiple of 32, regardless of whether it's in buckets
            # This ensures WAN 2.2 compatibility even if the aligned value is not in the bucket list
            nearest_w = round(nearest_w / 32) * 32

        return nearest_h, nearest_w

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_excluded_video_entries(self, exclude_videos_file: Optional[Any]) -> set[str]:
        exclude_paths_raw = _normalize_exclude_videos_files(exclude_videos_file)
        if not exclude_paths_raw:
            return set()

        entries: set[str] = set()
        for exclude_path_raw in exclude_paths_raw:
            exclude_path = Path(exclude_path_raw)
            if not exclude_path.is_absolute():
                exclude_path = self.data_root / exclude_path
            if not exclude_path.exists():
                raise FileNotFoundError(f"Exclude videos file not found: {exclude_path}")

            path_entries = {
                line.strip()
                for line in exclude_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            entries.update(path_entries)
            logger.info("Loaded %d excluded videos from %s", len(path_entries), exclude_path)

        logger.info("Loaded %d unique excluded videos from %d file(s)", len(entries), len(exclude_paths_raw))
        return entries

    def _video_path_to_rel_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.data_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _get_candidate_action_dirs(self, path: Path) -> List[Path]:
        action_dir = path.parent.parent
        candidate_dirs = [action_dir]
        if action_dir.name == "processed2":
            candidate_dirs.append(action_dir.parent)
        return candidate_dirs

    def _resolve_existing_subdir(self, path: Path, subdir: str) -> Path:
        candidate_dirs = [action_dir / subdir for action_dir in self._get_candidate_action_dirs(path)]
        for candidate_dir in candidate_dirs:
            if candidate_dir.exists():
                return candidate_dir
        return candidate_dirs[0]

    def _resolve_existing_file(self, path: Path, subdir: str, filename: str) -> Path:
        candidate_files = [
            action_dir / subdir / filename for action_dir in self._get_candidate_action_dirs(path)
        ]
        if len(candidate_files) == 1:
            return candidate_files[0]
        for candidate_file in candidate_files:
            if candidate_file.exists():
                return candidate_file
        return candidate_files[0]

    def _apply_exclude_video_filter(
        self, prompts: List[str], video_paths: List[Path]
    ) -> Tuple[List[str], List[Path]]:
        filtered_prompts: List[str] = []
        filtered_video_paths: List[Path] = []
        excluded_count = 0

        for prompt, video_path in zip(prompts, video_paths):
            rel_key = self._video_path_to_rel_key(video_path)
            if rel_key in self.excluded_videos:
                excluded_count += 1
                continue
            filtered_prompts.append(prompt)
            filtered_video_paths.append(video_path)

        if excluded_count:
            logger.warning(
                "Excluded %d videos listed in %s from the dataset.",
                excluded_count,
                self.exclude_videos_file,
            )
        if not filtered_video_paths:
            raise ValueError("All dataset videos were excluded. Check exclude_videos_file.")

        return filtered_prompts, filtered_video_paths

    def _build_raw_clip_plan(self, path: Path) -> Dict[str, int]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        return self._build_raw_clip_plan_from_reader(video_reader, path=path)

    # ---- action-density-aware clip-start sampling ---------------------------

    def _ensure_action_density_index(self) -> Optional[Dict[str, Dict[str, Any]]]:
        if self.clip_sampling != "action_weighted" or self.action_density_index_path is None:
            return None
        if self._action_density_index is not None:
            return self._action_density_index
        idx_path = Path(self.action_density_index_path)
        if not idx_path.is_absolute():
            idx_path = self.data_root / idx_path
        if not idx_path.exists():
            logger.warning(
                "action_density_index '%s' not found; falling back to uniform clip sampling",
                idx_path,
            )
            self.clip_sampling = "uniform"
            self._action_density_index = {}
            return self._action_density_index
        index: Dict[str, Dict[str, Any]] = {}
        with idx_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row["episode_id"]
                index[key] = {
                    "starts": row["starts"],
                    "num_transitions": row["num_transitions"],
                    "T": row["T"],
                    "clip_length": row["clip_length"],
                    "stride": row["stride"],
                }
        logger.info(
            "[action-density] loaded %d episode entries from %s (mode=%s alpha=%.2f)",
            len(index), idx_path, self.clip_sampling, self.action_sampling_alpha,
        )
        self._action_density_index = index
        return index

    @staticmethod
    def _episode_key_from_video_path(path: Path) -> Optional[str]:
        """For a render_v2 video path '<root>/<lab>/<hash>/videos/<ts>__extX.mp4'
        return '<lab>/<hash>/<ts>'. None if the structure does not match."""
        try:
            name = path.stem  # e.g. 2023-10-21-19h-02m-09s__ext1
            ts = name.split("__")[0]
            videos_dir = path.parent
            if videos_dir.name != "videos":
                return None
            hash_dir = videos_dir.parent
            lab_dir = hash_dir.parent
            return f"{lab_dir.name}/{hash_dir.name}/{ts}"
        except Exception:
            return None

    def _action_weighted_start(
        self,
        path: Optional[Path],
        max_start: int,
        target_num_frames: int,
        frame_skip: int,
        uniform_candidates: List[int],
    ) -> Optional[int]:
        """Weighted draw of clip start using action-density index.

        Returns None to signal "fall back to uniform". Falls back when:
        - sampling mode is not action_weighted
        - frame_skip != 1 (index was built at frame_skip=1 / clip_length=81)
        - target_num_frames does not match index clip_length
        - episode key not found in index
        - no overlap between index starts and feasible [0, max_start]
        """
        if self.clip_sampling != "action_weighted" or path is None:
            return None
        if frame_skip != 1:
            return None
        index = self._ensure_action_density_index()
        if not index:
            return None
        key = self._episode_key_from_video_path(path)
        if key is None:
            return None
        entry = index.get(key)
        if entry is None:
            return None
        if int(entry["clip_length"]) != int(target_num_frames):
            return None
        starts = entry["starts"]
        nts = entry["num_transitions"]
        # Restrict to feasible starts given current video length-derived max_start.
        feasible = [(s, n) for s, n in zip(starts, nts) if 0 <= s <= max_start]
        if not feasible:
            return None
        eps = self.action_sampling_eps
        scores = np.array(
            [(1.0 if n > 0 else eps) for _, n in feasible], dtype=np.float64
        )
        if self.action_sampling_alpha != 1.0:
            scores = np.power(scores, self.action_sampling_alpha)
        scores /= scores.sum()
        idx = int(np.random.choice(len(feasible), p=scores))
        return int(feasible[idx][0])

    def _sample_temporal_frame_skip(self) -> int:
        if not self.temporal_frame_skip_choices:
            return 1
        if self.temporal_frame_skip_probs is None:
            return int(random.choice(self.temporal_frame_skip_choices))
        return int(random.choices(self.temporal_frame_skip_choices, weights=self.temporal_frame_skip_probs, k=1)[0])

    @staticmethod
    def _num_frames_available_with_skip(video_num_frames: int, start_idx: int, frame_skip: int) -> int:
        if video_num_frames <= 0:
            return 0
        if start_idx >= video_num_frames:
            return 0
        return 1 + max(video_num_frames - 1 - start_idx, 0) // max(1, frame_skip)

    def _resolve_feasible_frame_skip(
        self,
        video_num_frames: int,
        target_num_frames: int,
        sampled_frame_skip: int,
    ) -> int:
        if target_num_frames <= 1:
            return 1

        if video_num_frames >= 1 + (target_num_frames - 1) * sampled_frame_skip:
            return int(sampled_frame_skip)

        fallback_candidates = [
            choice
            for choice in sorted(self.temporal_frame_skip_choices, reverse=True)
            if choice < sampled_frame_skip
        ]
        if 1 not in fallback_candidates:
            fallback_candidates.append(1)

        for candidate in fallback_candidates:
            if video_num_frames >= 1 + (target_num_frames - 1) * candidate:
                return int(candidate)
        return 1

    def _build_raw_clip_plan_from_reader(
        self, video_reader, path: Optional[Path] = None
    ) -> Dict[str, int]:
        video_num_frames = len(video_reader)
        if video_num_frames <= 0:
            raise ValueError("Video has no frames")

        first_frame = video_reader[0]
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        target_height, target_width = self._find_nearest_resolution(height, width)
        target_num_frames = int(self.max_num_frames)
        sampled_frame_skip = self._sample_temporal_frame_skip()
        frame_skip = self._resolve_feasible_frame_skip(
            video_num_frames=video_num_frames,
            target_num_frames=target_num_frames,
            sampled_frame_skip=sampled_frame_skip,
        )
        required_span = 1 + max(target_num_frames - 1, 0) * frame_skip
        if video_num_frames >= required_span:
            max_start = max(video_num_frames - required_span, 0)
        else:
            max_start = max(video_num_frames - target_num_frames, 0)
        if max_start <= 0:
            start_idx = 0
        else:
            start_candidates = list(range(0, max_start + 1, self.clip_start_frame_stride))
            if not start_candidates:
                start_candidates = [0]
            if start_candidates[-1] != max_start:
                start_candidates.append(max_start)
            weighted_start = self._action_weighted_start(
                path=path,
                max_start=max_start,
                target_num_frames=target_num_frames,
                frame_skip=frame_skip,
                uniform_candidates=start_candidates,
            )
            if weighted_start is not None:
                start_idx = weighted_start
            else:
                start_idx = int(random.choice(start_candidates))

        return {
            "start_idx": start_idx,
            "target_num_frames": target_num_frames,
            "target_height": target_height,
            "target_width": target_width,
            "frame_skip": frame_skip,
        }

    def _resolve_raw_clip_plan(self, video_reader, clip_plan: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        if clip_plan is None:
            return self._build_raw_clip_plan_from_reader(video_reader)

        resolved_plan = dict(clip_plan)
        target_num_frames = int(resolved_plan.get("target_num_frames", self.max_num_frames))
        video_num_frames = len(video_reader)
        frame_skip = max(1, int(resolved_plan.get("frame_skip", 1)))
        required_span = 1 + max(target_num_frames - 1, 0) * frame_skip
        if video_num_frames >= required_span:
            max_start = max(video_num_frames - required_span, 0)
        else:
            max_start = max(video_num_frames - target_num_frames, 0)
        resolved_plan["target_num_frames"] = target_num_frames
        resolved_plan["start_idx"] = min(int(resolved_plan.get("start_idx", 0)), max_start)
        resolved_plan["frame_skip"] = frame_skip
        return resolved_plan

    def _load_raw_frames(self, video_reader, clip_plan: Dict[str, int]) -> torch.Tensor:
        video_num_frames = len(video_reader)
        target_num_frames = int(clip_plan["target_num_frames"])
        start_idx = int(clip_plan["start_idx"])
        frame_skip = max(1, int(clip_plan.get("frame_skip", 1)))
        available_frames = self._num_frames_available_with_skip(video_num_frames, start_idx, frame_skip)
        num_selected_frames = min(target_num_frames, available_frames)
        frame_indices = [
            start_idx + frame_skip * frame_idx for frame_idx in range(num_selected_frames)
        ]

        if not frame_indices:
            raise ValueError("No frame indices selected for video clip")

        frames = video_reader.get_batch(frame_indices).float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        if frames.shape[0] < target_num_frames:
            pad_frames = frames[-1:].repeat(target_num_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)

        return frames

    def _get_raw_video_sample(self, index: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
        clip_plan = self._build_raw_clip_plan(self.video_paths[index])
        image, video, _ = self._preprocess_video(self.video_paths[index], clip_plan=clip_plan)
        if self.use_preencoded_prompt_embeds:
            prompt = self._load_prompt_embeds_for_video_path(self.video_paths[index])
        else:
            prompt = (
                self.prompts[index]
                if self.eager_load_prompts
                else self._load_prompt_for_video_path(self.video_paths[index])
            )

        sample = {
            "prompt": prompt if isinstance(prompt, torch.Tensor) else self.id_token + prompt,
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
                "frame_skip": int(clip_plan.get("frame_skip", 1)),
            },
        }
        return sample, clip_plan

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     # Here, index is actually a list of data objects that we need to return.
        #     # The BucketSampler should ideally return indices. But, in the sampler, we'd like
        #     # to have information about num_frames, height and width. Since this is not stored
        #     # as metadata, we need to read the video to get this information. You could read this
        #     # information without loading the full video in memory, but we do it anyway. In order
        #     # to not load the video twice (once to get the metadata, and once to return the loaded video
        #     # based on sampled indices), we cache it in the BucketSampler. When the sampler is
        #     # to yield, we yield the cache data instead of indices. So, this special check ensures
        #     # that data is not loaded a second time. PRs are welcome for improvements.
        #     return index

        # Handle index out of range with mod operation to prevent training interruption
        original_index = index
        if index >= len(self.video_paths):
            logger.warning(f"Index {index} is out of range (dataset size: {len(self.video_paths)}). Using mod operation: {index} % {len(self.video_paths)} = {index % len(self.video_paths)}")
            index = index % len(self.video_paths)

        # try:
        if self.load_tensors:
            first_frame_latents, fun_inp_i2v_latents, video_latents, prompt_embeds, clip_image_embeds = self._load_preprocessed_latents_and_embeds(
                self.video_paths[index]
            )

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": first_frame_latents,
                "fun_inp_i2v_latents": fun_inp_i2v_latents,
                "clip_image_embeds": clip_image_embeds,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            sample, _ = self._get_raw_video_sample(index)
            return sample
        # except Exception as e:
        #     logger.error(f"Failed to load main video data for index {index}: {e}")
        #     # Return a minimal valid data structure to prevent training from crashing
        #     return {
        #         "prompt": "ERROR_LOADING_DATA",
        #         "image": None,
        #         "video": torch.zeros(1, 3, 16, 256, 256),  # Dummy video tensor
        #         "video_metadata": {
        #             "num_frames": 16,
        #             "height": 256,
        #             "width": 256,
        #         },
        #     }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        video_path = self.data_root.joinpath(self.video_column)

        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        # Derive prompt paths from video paths (same as _load_dataset_from_datafile)
        prompt_paths = [path.parent.parent.joinpath("prompts", path.name.replace(".mp4", ".txt")) for path in video_paths]
        # prompts = [path.read_text().strip() for path in prompt_paths]
        prompts = ["" for path in prompt_paths]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    @staticmethod
    def _extract_json_field(data: Dict[str, Any], field: str) -> str:
        value: Any = data
        for key in field.split("."):
            if not isinstance(value, dict):
                return ""
            value = value.get(key, "")
        if isinstance(value, dict) and "text" in value:
            value = value.get("text", "")
        return str(value).strip()

    def _read_prompt_text(self, prompt_path: Path) -> Tuple[str, bool]:
        if not prompt_path.is_file():
            return "", False
        if self.prompt_json_fields:
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to parse structured prompt %s: %s", prompt_path, exc)
                return "", False
            parts = [self._extract_json_field(data, field) for field in self.prompt_json_fields]
            return " ".join(part for part in parts if part), True
        return prompt_path.read_text(encoding="utf-8"), True

    def _load_prompt_texts(self, prompt_paths: List[Path], dataset_path: Path) -> List[str]:
        prompts: List[str] = []
        missing_count = 0

        for prompt_path in prompt_paths:
            prompt, found = self._read_prompt_text(prompt_path)
            prompts.append(prompt)
            if not found:
                missing_count += 1
        if missing_count:
            logger.warning(
                "Prompt files missing for %d samples while loading %s. Using empty prompts.",
                missing_count,
                dataset_path,
            )
        return prompts

    def _load_prompt_for_video_path(self, video_path: Path) -> str:
        prompt_ext = ".json" if self.prompt_json_fields else ".txt"
        prompt_path = self._resolve_existing_file(
            video_path, self.prompt_subdir, video_path.name.replace(".mp4", prompt_ext)
        )
        prompt, _ = self._read_prompt_text(prompt_path)
        return prompt

    def _load_prompt_embeds_for_video_path(self, video_path: Path) -> torch.Tensor:
        embeds_path = self._resolve_existing_file(
            video_path, self.prompt_embeds_subdir, video_path.name.replace(".mp4", ".pt")
        )
        if not embeds_path.is_file():
            raise FileNotFoundError(f"Missing preencoded prompt embeds: {embeds_path}")
        try:
            embeds = torch.load(embeds_path, map_location="cpu", weights_only=True)
        except TypeError:
            embeds = torch.load(embeds_path, map_location="cpu")
        if not isinstance(embeds, torch.Tensor):
            raise TypeError(f"Expected tensor prompt embeds in {embeds_path}, got {type(embeds)}")
        if embeds.shape[0] < self.max_prompt_length:
            padding = torch.zeros(
                self.max_prompt_length - embeds.shape[0],
                embeds.shape[1],
                dtype=embeds.dtype,
            )
            embeds = torch.cat([embeds, padding], dim=0)
        elif embeds.shape[0] > self.max_prompt_length:
            embeds = embeds[: self.max_prompt_length]
        return embeds.contiguous()

    def _load_dataset_from_datafile(self) -> Tuple[List[str], List[Path]]:
        """Load dataset from one or more datafiles (supports list of files)."""
        if not self.dataset_file_list:
            raise ValueError("dataset_file should not be empty when provided.")
        
        all_prompts: List[str] = []
        all_video_paths: List[Path] = []
        
        for dataset_file in self.dataset_file_list:
            dataset_path = self.data_root / dataset_file
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            with open(dataset_path, "r", encoding="utf-8") as f:
                file_video_paths = [
                    self.data_root.joinpath(line.strip()) for line in f.readlines() if len(line.strip()) > 0
                ]
            
            prompt_ext = ".json" if self.prompt_json_fields else ".txt"
            prompt_paths = [
                self._resolve_existing_file(
                    path, self.prompt_subdir, path.name.replace(".mp4", prompt_ext)
                )
                for path in file_video_paths
            ]
            if self.eager_load_prompts:
                file_prompts = self._load_prompt_texts(prompt_paths, dataset_path)
            else:
                file_prompts = ["" for _ in file_video_paths]

            all_video_paths.extend(file_video_paths)
            all_prompts.extend(file_prompts)
        
        return all_prompts, all_video_paths

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single raw video and returns (image, video, prompt_embed=None).

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.
        """
        try:
            video_reader = decord.VideoReader(uri=path.as_posix())
            resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
            frames = self._load_raw_frames(video_reader, resolved_clip_plan)
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None
        except Exception as e:
            logger.error(f"Failed to preprocess video {path}: {e}")
            raise e

    def _load_preprocessed_latents_and_embeds(
        self, path: Path
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        try:
            filename_without_ext = path.name.split(".")[0]
            pt_filename = f"{filename_without_ext}.pt"

            # The current path is something like: /a/b/c/d/videos/00001.mp4
            # We need to reach: /a/b/c/d/video_latents_wan/00001.pt (using configurable subdir)
            video_latents_path = self._resolve_existing_subdir(path, self.video_latents_subdir)
            embeds_path = self._resolve_existing_subdir(path, self.prompt_embeds_subdir)
            image_latents_path = self._resolve_existing_subdir(path, self.image_latents_subdir)
            fun_inp_i2v_latents_path = self._resolve_existing_subdir(path, self.fun_inp_i2v_latents_subdir)
            clip_embeds_path = self._resolve_existing_subdir(path, self.clip_image_embeds_subdir)

            if not video_latents_path.exists() or not embeds_path.exists():
                logger.error(f"Required folders not found for {path}")
                logger.error(f"video_latents_path exists: {video_latents_path.exists()}")
                logger.error(f"embeds_path exists: {embeds_path.exists()}")
                raise ValueError(
                    f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`."
                )

            video_latent_filepath = video_latents_path.joinpath(pt_filename)
            embeds_filepath = embeds_path.joinpath(pt_filename)

            if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
                video_latent_filepath = video_latent_filepath.as_posix()
                embeds_filepath = embeds_filepath.as_posix()
                logger.error(f"Required files not found:")
                logger.error(f"video_latent_filepath exists: {Path(video_latent_filepath).exists()}")
                logger.error(f"embeds_filepath exists: {Path(embeds_filepath).exists()}")
                raise ValueError(
                    f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
                )

            try:
                video_latent_size = Path(video_latent_filepath).stat().st_size
                embeds_size = Path(embeds_filepath).stat().st_size
                if video_latent_size == 0 or embeds_size == 0:
                    raise EOFError(
                        f"Zero-byte tensor file detected: "
                        f"video_latent_size={video_latent_size}, embeds_size={embeds_size}"
                    )

                latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
                embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)
                images = latents[:, :1, :, :].clone() if self.image_to_video else None
                fun_inp_i2v_latents = None
                clip_image_embeds = None

                if self.image_to_video:
                    image_latent_filepath = image_latents_path.joinpath(pt_filename)
                    fun_inp_i2v_latent_filepath = fun_inp_i2v_latents_path.joinpath(pt_filename)
                    clip_embed_filepath = clip_embeds_path.joinpath(pt_filename)

                    if image_latent_filepath.is_file():
                        images = torch.load(image_latent_filepath, map_location="cpu", weights_only=True)
                    if fun_inp_i2v_latent_filepath.is_file():
                        fun_inp_i2v_latents = torch.load(fun_inp_i2v_latent_filepath, map_location="cpu", weights_only=True)
                    if clip_embed_filepath.is_file():
                        clip_image_embeds = torch.load(clip_embed_filepath, map_location="cpu", weights_only=True)

                # Pad prompt embeddings to max_prompt_length for uniform batch sizes
                # embeds shape: [seq_len, dim] -> pad to [max_prompt_length, dim]
                if embeds.shape[0] < self.max_prompt_length:
                    padding = torch.zeros(
                        self.max_prompt_length - embeds.shape[0],
                        embeds.shape[1],
                        dtype=embeds.dtype,
                    )
                    embeds = torch.cat([embeds, padding], dim=0)
                elif embeds.shape[0] > self.max_prompt_length:
                    # Truncate if longer than max_prompt_length
                    embeds = embeds[:self.max_prompt_length]

            except Exception as e:
                logger.error(
                    f"Failed to load torch files for {path}: "
                    f"{type(e).__name__}: {repr(e)}"
                )
                logger.error(f"video_latent_filepath: {video_latent_filepath}")
                logger.error(f"embeds_filepath: {embeds_filepath}")
                raise e

            return images, fun_inp_i2v_latents, latents, embeds, clip_image_embeds
            
        except Exception as e:
            logger.error(
                f"Error in _load_preprocessed_latents_and_embeds for {path}: "
                f"{type(e).__name__}: {repr(e)}"
            )
            raise e


class VideoDatasetWithConditions(VideoDataset):
    """
    Extended VideoDataset that supports additional condition videos:
    - hand_videos: Egocentric hand mesh videos (automatically derived from main video paths)
    - static_videos: Static scene videos (automatically derived from main video paths)
    """

    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
        prompt_subdir: str = "prompts",
        prompt_embeds_subdir: str = "prompt_embeds_ego_fun_rewrite_wan",
        use_preencoded_prompt_embeds: bool = False,
        prompt_json_fields: Optional[List[str]] = None,
        static_video_subdir: str = "videos_static",
        static_video_source_mode: str = "directory",
        robot_mesh_video_subdir: str = "videos_hands",
        robot_mesh_video_latents_subdir: str = "hand_video_latents_wan",
        static_disparity_subdir: Optional[str] = None,
        static_disparity_latents_subdir: Optional[str] = None,
        static_disparity_npz_mode: Optional[str] = None,
        eef_depth_subdir: Optional[str] = None,
        eef_depth_latents_subdir: Optional[str] = None,
        eef_depth_npz_mode: Optional[str] = None,
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        image_latents_subdir: str = "image_latents_wan",
        fun_inp_i2v_latents_subdir: str = "fun_inp_i2v_latents_wan",
        clip_image_embeds_subdir: str = "clip_image_embeds",
        exclude_videos_file: Optional[Any] = None,
        require_static_videos: bool = True,
        validate_condition_paths: bool = True,
        eager_load_prompts: bool = True,
        align_width_to_32: bool = False,
        max_prompt_length: int = 512,
        clip_start_frame_stride: int = 1,
        temporal_frame_skip_choices: Optional[Any] = None,
        temporal_frame_skip_probs: Optional[Any] = None,
        action_density_index: Optional[str] = None,
        clip_sampling: str = "uniform",
        action_sampling_alpha: float = 0.5,
        action_sampling_eps: float = 1e-2,
        state_adaln_source: Optional[str] = None,
        state_adaln_joint_filename: str = "joint_position_isaac_nominal.npy",
        state_adaln_gripper_filename: str = "gripper_position_isaac_nominal.npy",
        state_adaln_action_sidecar_subdir: str = "action_values_clips81",
        state_adaln_union_droid_dim: int = 7,
        state_adaln_union_robocasa_dim: int = 24,
    ) -> None:
        # Initialize parent class with main video column
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
            max_prompt_length=max_prompt_length,
            prompt_subdir=prompt_subdir,
            prompt_embeds_subdir=prompt_embeds_subdir,
            use_preencoded_prompt_embeds=use_preencoded_prompt_embeds,
            prompt_json_fields=prompt_json_fields,
            static_video_subdir=static_video_subdir,
            robot_mesh_video_subdir=robot_mesh_video_subdir,
            robot_mesh_video_latents_subdir=robot_mesh_video_latents_subdir,
            video_latents_subdir=video_latents_subdir,
            static_video_latents_subdir=static_video_latents_subdir,
            image_latents_subdir=image_latents_subdir,
            fun_inp_i2v_latents_subdir=fun_inp_i2v_latents_subdir,
            clip_image_embeds_subdir=clip_image_embeds_subdir,
            exclude_videos_file=exclude_videos_file,
            eager_load_prompts=eager_load_prompts,
            align_width_to_32=align_width_to_32,
            clip_start_frame_stride=clip_start_frame_stride,
            temporal_frame_skip_choices=temporal_frame_skip_choices,
            temporal_frame_skip_probs=temporal_frame_skip_probs,
            action_density_index=action_density_index,
            clip_sampling=clip_sampling,
            action_sampling_alpha=action_sampling_alpha,
            action_sampling_eps=action_sampling_eps,
        )

        # Store the use_gray_hand_videos flag
        self.use_gray_hand_videos = use_gray_hand_videos
        self.static_disparity_subdir = static_disparity_subdir
        self.static_disparity_latents_subdir = static_disparity_latents_subdir
        self.static_disparity_npz_mode = self._normalize_npz_mode(
            static_disparity_npz_mode, "static_disparity_npz_mode"
        )
        self.eef_depth_subdir = eef_depth_subdir
        self.eef_depth_latents_subdir = eef_depth_latents_subdir
        self.eef_depth_npz_mode = self._normalize_npz_mode(
            eef_depth_npz_mode, "eef_depth_npz_mode"
        )
        self.static_video_source_mode = str(static_video_source_mode).strip().lower()
        self.require_static_videos = bool(require_static_videos)
        self.validate_condition_paths = bool(validate_condition_paths)
        self.state_adaln_source = str(state_adaln_source or "").strip().lower()
        if self.state_adaln_source in {"", "none"}:
            self.state_adaln_source = None
        self.state_adaln_joint_filename = str(state_adaln_joint_filename)
        self.state_adaln_gripper_filename = str(state_adaln_gripper_filename)
        self.state_adaln_action_sidecar_subdir = str(state_adaln_action_sidecar_subdir)
        self.state_adaln_union_droid_dim = int(state_adaln_union_droid_dim)
        self.state_adaln_union_robocasa_dim = int(state_adaln_union_robocasa_dim)
        if self.static_video_source_mode not in {
            "directory",
            "copy_first_frame",
            "pcd_from_ext_first_frame",
            "i2v_first_frame_only",
        }:
            raise ValueError(
                f"Unsupported static_video_source_mode: {static_video_source_mode}. "
                "Use one of: directory, copy_first_frame, pcd_from_ext_first_frame, "
                "i2v_first_frame_only."
            )
        # Store the align_width_to_32 flag (for WAN 2.2 compatibility)
        self.align_width_to_32 = align_width_to_32
        
        # Automatically derive hand video and static video paths from main video paths
        if self.use_gray_hand_videos:
            self.robot_mesh_video_paths = self._derive_condition_video_paths("videos_hands_gray")
        else:
            self.robot_mesh_video_paths = (
                self._derive_condition_video_paths(self.robot_mesh_video_subdir)
                if self.robot_mesh_video_subdir
                else None
            )
        self.static_video_paths = self._derive_condition_video_paths(self.static_video_subdir)
        self.static_disparity_paths = (
            self._derive_condition_video_paths(self.static_disparity_subdir)
            if self.static_disparity_subdir
            else None
        )
        if self.static_disparity_paths is not None and self.static_disparity_npz_mode is not None:
            self.static_disparity_paths = [
                p.with_suffix(".npz") for p in self.static_disparity_paths
            ]
        self.eef_depth_paths = (
            self._derive_condition_video_paths(self.eef_depth_subdir)
            if self.eef_depth_subdir
            else None
        )
        if self.eef_depth_paths is not None and self.eef_depth_npz_mode is not None:
            self.eef_depth_paths = [
                p.with_suffix(".npz") for p in self.eef_depth_paths
            ]
        # Validate that all condition videos exist
        if not self.load_tensors and self.validate_condition_paths:
            if self.robot_mesh_video_paths is not None and any(not path.is_file() for path in self.robot_mesh_video_paths):
                missing_hand_videos = [path for path in self.robot_mesh_video_paths if not path.is_file()]
                raise ValueError(
                    f"Some hand video files are missing. First few missing files: {missing_hand_videos[:5]}"
                )
            
            # if any(not path.is_file() for path in self.hand_video_gray_paths):
            #     missing_hand_gray_videos = [path for path in self.hand_video_gray_paths if not path.is_file()]
            #     raise ValueError(
            #         f"Some hand gray video files are missing. First few missing files: {missing_hand_gray_videos[:5]}"
            #     )
            
            if self.require_static_videos and self.static_video_source_mode == "directory":
                missing_static_videos = [path for path in self.static_video_paths if not path.is_file()]
                if missing_static_videos:
                    raise ValueError(
                        f"Some static video files are missing. First few missing files: {missing_static_videos[:5]}"
                    )
            if self.static_disparity_paths is not None and any(
                not path.is_file() for path in self.static_disparity_paths
            ):
                missing_static_disparities = [path for path in self.static_disparity_paths if not path.is_file()]
                raise ValueError(
                    "Some static disparity video files are missing. "
                    f"First few missing files: {missing_static_disparities[:5]}"
                )
            if self.eef_depth_paths is not None and any(
                not path.is_file() for path in self.eef_depth_paths
            ):
                missing_hand_disparities = [path for path in self.eef_depth_paths if not path.is_file()]
                raise ValueError(
                    "Some hand disparity video files are missing. "
                    f"First few missing files: {missing_hand_disparities[:5]}"
                )

    @staticmethod
    def _normalize_npz_mode(mode: Optional[str], field_name: str) -> Optional[str]:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized in {"", "none"}:
            return None
        allowed = {"first_frame_repeat", "per_frame"}
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported {field_name}: {mode!r}. Use one of: {sorted(allowed)} or None."
            )
        return normalized

    def _derive_condition_video_paths(self, condition_folder: str) -> List[Path]:
        """Derive condition video paths from main video paths.
        
        Args:
            condition_folder: Name of the condition folder (e.g., "videos_hands", "videos_static")
            
        Returns:
            List of condition video paths
        """
        condition_paths = []
        
        for video_path in self.video_paths:
            # Example: video_path = /path/to/sequences/videos/00001.mp4
            # We want: /path/to/sequences/videos_hands/00001.mp4
            # or: /path/to/sequences/videos_static/00001.mp4
            
            parent_dir = video_path.parent
            filename = video_path.name
            condition_path = parent_dir.parent / condition_folder / filename
            if self.validate_condition_paths and not condition_path.exists() and parent_dir.parent.name == "processed2":
                fallback_condition_path = parent_dir.parent.parent / condition_folder / filename
                if fallback_condition_path.exists():
                    condition_path = fallback_condition_path
            condition_paths.append(condition_path)
        
        return condition_paths

    def _get_item_once(self, index: int) -> Dict[str, Any]:
        if self.load_tensors:
            # Get main video data from parent class
            main_data = super().__getitem__(index)

            # Load condition videos
            # Load preprocessed latents for condition videos (using configurable subdirs)
            try:
                if self.robot_mesh_video_paths is None:
                    main_data["hand_videos"] = None
                elif self.use_gray_hand_videos:
                    # Gray hand videos use a different subdir pattern
                    hand_latents_subdir = self.robot_mesh_video_latents_subdir.replace("hand_video", "hand_video_gray")
                    hand_video_latents = self._load_condition_video_latents(self.robot_mesh_video_paths[index], hand_latents_subdir)
                    main_data["hand_videos"] = hand_video_latents
                else:
                    hand_video_latents = self._load_condition_video_latents(self.robot_mesh_video_paths[index], self.robot_mesh_video_latents_subdir)
                    main_data["hand_videos"] = hand_video_latents
            except Exception as e:
                logger.warning(f"Failed to load hand video latents for index {index}: {e}")
                main_data["hand_videos"] = None

            if self.require_static_videos:
                try:
                    static_video_latents = self._load_condition_video_latents(self.static_video_paths[index], self.static_video_latents_subdir)
                    main_data["static_videos"] = static_video_latents
                except Exception as e:
                    logger.warning(f"Failed to load static video latents for index {index}: {e}")
                    main_data["static_videos"] = None
            else:
                main_data["static_videos"] = None

            if self.static_disparity_paths is not None and self.static_disparity_latents_subdir is not None:
                try:
                    static_disparity_latents = self._load_condition_video_latents(
                        self.static_disparity_paths[index], self.static_disparity_latents_subdir
                    )
                    main_data["static_disparities"] = static_disparity_latents
                except Exception as e:
                    logger.warning(f"Failed to load static disparity latents for index {index}: {e}")
                    main_data["static_disparities"] = None
            else:
                main_data["static_disparities"] = None

            if self.eef_depth_paths is not None and self.eef_depth_latents_subdir is not None:
                try:
                    eef_depth_latents = self._load_condition_video_latents(
                        self.eef_depth_paths[index], self.eef_depth_latents_subdir
                    )
                    main_data["hand_disparities"] = eef_depth_latents
                except Exception as e:
                    logger.warning(f"Failed to load hand disparity latents for index {index}: {e}")
                    main_data["hand_disparities"] = None
            else:
                main_data["hand_disparities"] = None

            if self.state_adaln_source is not None:
                raw_num_frames = int(
                    main_data.get("video_metadata", {}).get("num_frames", self.max_num_frames)
                )
                main_data["state_adaln"] = load_robot_state_condition(
                    self.video_paths[index],
                    source=self.state_adaln_source,
                    num_frames=raw_num_frames,
                    joint_filename=self.state_adaln_joint_filename,
                    gripper_filename=self.state_adaln_gripper_filename,
                    action_sidecar_subdir=self.state_adaln_action_sidecar_subdir,
                    union_droid_dim=self.state_adaln_union_droid_dim,
                    union_robocasa_dim=self.state_adaln_union_robocasa_dim,
                )
            else:
                main_data["state_adaln"] = None
        else:
            main_data, clip_plan = self._get_raw_video_sample(index)

            # Load raw videos for condition videos
            try:
                if self.robot_mesh_video_paths is None:
                    main_data["hand_videos"] = None
                else:
                    _, hand_video, _ = self._preprocess_video(self.robot_mesh_video_paths[index], clip_plan=clip_plan)
                    main_data["hand_videos"] = hand_video
            except Exception as e:
                logger.warning(f"Failed to load hand video for index {index}: {e}")
                main_data["hand_videos"] = None
            
            if not self.require_static_videos:
                main_data["static_videos"] = None
            elif self.static_video_source_mode == "copy_first_frame":
                main_video = main_data["video"]
                main_data["static_videos"] = main_video[0:1].repeat(main_video.shape[0], 1, 1, 1).clone()
            elif self.static_video_source_mode == "i2v_first_frame_only":
                main_video = main_data["video"]
                static_videos = torch.zeros_like(main_video)
                static_videos[0] = main_video[0]
                main_data["static_videos"] = static_videos
            else:
                try:
                    _, static_video, _ = self._preprocess_video(self.static_video_paths[index], clip_plan=clip_plan)
                    main_data["static_videos"] = static_video
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load static video for index {index}: {self.static_video_paths[index]}"
                    ) from e

            if self.static_disparity_paths is not None:
                try:
                    if self.static_disparity_npz_mode is not None:
                        static_disparity = self._load_depth_npz_as_tensor(
                            self.static_disparity_paths[index],
                            clip_plan=clip_plan,
                            mode=self.static_disparity_npz_mode,
                        )
                    else:
                        _, static_disparity, _ = self._preprocess_video(
                            self.static_disparity_paths[index], clip_plan=clip_plan
                        )
                    main_data["static_disparities"] = static_disparity
                except Exception as e:
                    logger.warning(f"Failed to load static disparity video for index {index}: {e}")
                    main_data["static_disparities"] = None
            else:
                main_data["static_disparities"] = None

            if self.eef_depth_paths is not None:
                try:
                    if self.eef_depth_npz_mode is not None:
                        eef_depth = self._load_depth_npz_as_tensor(
                            self.eef_depth_paths[index],
                            clip_plan=clip_plan,
                            mode=self.eef_depth_npz_mode,
                        )
                    else:
                        _, eef_depth, _ = self._preprocess_video(
                            self.eef_depth_paths[index], clip_plan=clip_plan
                        )
                    main_data["hand_disparities"] = eef_depth
                except Exception as e:
                    logger.warning(f"Failed to load hand disparity video for index {index}: {e}")
                    main_data["hand_disparities"] = None
            else:
                main_data["hand_disparities"] = None

            if self.state_adaln_source is not None:
                main_data["state_adaln"] = load_robot_state_condition(
                    self.video_paths[index],
                    source=self.state_adaln_source,
                    clip_plan=clip_plan,
                    joint_filename=self.state_adaln_joint_filename,
                    gripper_filename=self.state_adaln_gripper_filename,
                    action_sidecar_subdir=self.state_adaln_action_sidecar_subdir,
                    union_droid_dim=self.state_adaln_union_droid_dim,
                    union_robocasa_dim=self.state_adaln_union_robocasa_dim,
                )
            else:
                main_data["state_adaln"] = None

        return main_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     return index

        if index >= len(self.video_paths):
            logger.warning(
                "Index %d is out of range (dataset size: %d). Using mod operation: %d %% %d = %d",
                index,
                len(self.video_paths),
                index,
                len(self.video_paths),
                index % len(self.video_paths),
            )
            index = index % len(self.video_paths)

        return self._get_item_once(index)

    def _load_condition_video_latents(self, path: Path, latent_folder: str) -> Optional[torch.Tensor]:
        """Load preprocessed latents for condition videos.
        
        Returns:
            torch.Tensor if latents are successfully loaded, None if folder or file doesn't exist.
        """
        try:
            filename_without_ext = path.name.split(".")[0]
            pt_filename = f"{filename_without_ext}.pt"
            
            # The current path is something like: /a/b/c/sequences/videos_hands/00001.mp4
            # We need to reach: /a/b/c/hand_video_latents/00001.pt
            # or: /a/b/c/static_video_latents/00001.pt
            
            # Get the action directory (parent of the condition folder)
            # Example: .../SingleHand/Kitchen_xxx/videos_hands/00001.mp4
            # -> .../SingleHand/Kitchen_xxx
            latents_path = self._resolve_existing_subdir(path, latent_folder)
            
            if not latents_path.exists():
                logger.warning(f"Latents folder not found: {latents_path}. Returning None.")
                return None
            
            latent_filepath = latents_path.joinpath(pt_filename)
            
            if not latent_filepath.is_file():
                logger.warning(f"Latent file not found: {latent_filepath}. Returning None.")
                return None
            
            try:
                latents = torch.load(latent_filepath, map_location="cpu", weights_only=True)
                return latents
            except Exception as e:
                logger.error(f"Failed to load latent file {latent_filepath}: {e}")
                raise e
                
        except Exception as e:
            # Only log and re-raise if it's not a FileNotFoundError (which we already handled above)
            if not isinstance(e, FileNotFoundError):
                logger.error(f"Error in _load_condition_video_latents for {path} in {latent_folder}: {e}")
                raise e
            # If it's a FileNotFoundError that we didn't catch above, return None
            logger.warning(f"Latents not found for {path} in {latent_folder}. Returning None.")
            return None


class VideoDatasetWithConditionsAndResizing(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that also supports resizing.
    """

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    def _load_depth_npz_as_tensor(
        self,
        npz_path: Path,
        clip_plan: Optional[Dict[str, int]],
        mode: str,
    ) -> torch.Tensor:
        """Load an inverse-depth NPZ, turbo-colormap each needed frame, resize to
        the main video's target resolution, and apply the same video_transforms so
        the output matches _preprocess_video's convention ([F, 3, H, W] in [-1, 1]).

        mode:
          - "first_frame_repeat": colormap only frame 0 and tile along time axis
            (matches copy_first_frame semantics of the static_video stream).
          - "per_frame": mirror _load_raw_frames's frame-selection logic
            (start_idx + frame_skip * i), pad the tail by repeating the last
            frame if the source is shorter than target_num_frames.
        """
        import cv2  # local import to keep module import time unchanged for non-depth runs

        if clip_plan is None:
            raise ValueError("clip_plan is required for depth npz loading")
        target_num_frames = int(clip_plan["target_num_frames"])
        start_idx = int(clip_plan.get("start_idx", 0))
        frame_skip = max(1, int(clip_plan.get("frame_skip", 1)))
        target_h = int(clip_plan["target_height"])
        target_w = int(clip_plan["target_width"])

        with np.load(str(npz_path)) as payload:
            inv_u16 = np.asarray(payload["inv_depth_u16"])    # (T, H, W) uint16
            valid_all = np.asarray(payload["valid_mask"])     # (T, H, W) uint8
            near_m = float(payload["near_m"])
            far_m = float(payload["far_m"])
            # stride-sampled NPZs (export_stereo_depth with --frame-stride > 1)
            # record the original RGB frame index for every stereo frame so we
            # can map a training clip_start back into the NPZ's sparse layout.
            if "frame_indices" in payload.files:
                frame_indices_src = np.asarray(payload["frame_indices"], dtype=np.int64)
            else:
                frame_indices_src = None
        t_src = int(inv_u16.shape[0])

        def _to_npz_index(rgb_frame_idx: int) -> int:
            """Map an original-video frame index to an NPZ slot. If the NPZ is
            stride-sampled (has frame_indices), snap to the largest entry <=
            rgb_frame_idx so the decoded depth corresponds to the latest
            captured observation at or before the requested RGB frame."""
            if frame_indices_src is None:
                return max(0, min(int(rgb_frame_idx), t_src - 1))
            pos = int(np.searchsorted(frame_indices_src, int(rgb_frame_idx), side="right") - 1)
            if pos < 0:
                pos = 0
            return min(pos, t_src - 1)

        if mode == "first_frame_repeat":
            # Align the static depth with the static_video stream, which uses
            # copy_first_frame on the *clipped* main video (i.e. original frame
            # index == start_idx). Without this, clip_start != 0 would show
            # static RGB at frame clip_start while static depth stayed pinned
            # at episode frame 0 — a silent contradictory signal.
            src_indices = [_to_npz_index(start_idx)]
        elif mode == "per_frame":
            # Mirror _load_raw_frames's per-frame selection, mapped into the
            # NPZ's (possibly sparse) frame layout.
            src_indices = [
                _to_npz_index(start_idx + frame_skip * i)
                for i in range(target_num_frames)
            ]
        else:
            raise ValueError(f"Unsupported depth npz mode: {mode!r}")

        # Colormap + resize each unique source frame
        rendered: List[np.ndarray] = []
        for src_i in src_indices:
            rgb = _depth_frame_to_turbo_rgb(
                inv_u16[src_i], valid_all[src_i], near_m=near_m, far_m=far_m
            )
            rendered.append(
                cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            )

        if mode == "first_frame_repeat":
            frame_stack = np.repeat(rendered[0][None, ...], target_num_frames, axis=0)
        else:
            # per_frame builds exactly target_num_frames entries via
            # _to_npz_index (which clamps out-of-range indices to the last
            # available NPZ slot), so no tail padding is needed.
            frame_stack = np.stack(rendered, axis=0)

        # (T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 255] matching decord output
        frames = torch.from_numpy(frame_stack).permute(0, 3, 1, 2).contiguous().float()
        frames = torch.stack([self.video_transforms(f) for f in frames], dim=0)
        return frames

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class MultiViewVideoDatasetWithConditionsAndResizing(VideoDatasetWithConditionsAndResizing):
    """
    Raw-video multiview dataset that groups files like `{base}__ext1.mp4` and
    `{base}__ext2.mp4` into one sample while leaving the existing single-view
    dataset classes unchanged.
    """

    def __init__(
        self,
        *,
        expected_view_ids: Optional[List[int]] = None,
        grouping_mode: str = "ext_suffix",
        **kwargs,
    ) -> None:
        load_tensors = bool(kwargs.get("load_tensors", False))
        allow_load_tensors = bool(kwargs.pop("_allow_load_tensors", False))
        if load_tensors and not allow_load_tensors:
            raise NotImplementedError(
                "MultiViewVideoDatasetWithConditionsAndResizing currently supports raw videos only "
                "(load_tensors=False)."
            )

        static_video_source_mode = str(kwargs.pop("static_video_source_mode", "directory")).strip().lower()
        static_disparity_subdir = kwargs.pop("static_disparity_subdir", None)
        static_disparity_latents_subdir = kwargs.pop("static_disparity_latents_subdir", None)
        static_disparity_npz_mode = kwargs.pop("static_disparity_npz_mode", None)
        eef_depth_subdir = kwargs.pop("eef_depth_subdir", None)
        eef_depth_latents_subdir = kwargs.pop("eef_depth_latents_subdir", None)
        eef_depth_npz_mode = kwargs.pop("eef_depth_npz_mode", None)
        require_static_videos = bool(kwargs.pop("require_static_videos", True))
        validate_condition_paths = bool(kwargs.pop("validate_condition_paths", True))
        # Composite-only config — drop it here so subclasses that forward
        # **kwargs up through the class hierarchy don't leak it into
        # VideoDataset.__init__ (which doesn't know about pcd rendering).
        kwargs.pop("wrist_pcd_config", None)

        # Intentionally bypass VideoDatasetWithConditions.__init__ so the new MV
        # path does not inherit flat-path condition validation behavior.
        VideoDataset.__init__(self, **kwargs)

        self.static_disparity_subdir = static_disparity_subdir
        self.static_disparity_latents_subdir = static_disparity_latents_subdir
        self.eef_depth_subdir = eef_depth_subdir
        self.eef_depth_latents_subdir = eef_depth_latents_subdir
        self.static_disparity_npz_mode = self._normalize_npz_mode(
            static_disparity_npz_mode, "static_disparity_npz_mode"
        )
        self.eef_depth_npz_mode = self._normalize_npz_mode(
            eef_depth_npz_mode, "eef_depth_npz_mode"
        )
        self.static_video_source_mode = static_video_source_mode
        self.require_static_videos = require_static_videos
        self.validate_condition_paths = validate_condition_paths
        if self.static_video_source_mode not in {
            "directory",
            "copy_first_frame",
            "pcd_from_ext_first_frame",
            "i2v_first_frame_only",
        }:
            raise ValueError(
                f"Unsupported static_video_source_mode: {static_video_source_mode}. "
                "Use one of: directory, copy_first_frame, pcd_from_ext_first_frame, "
                "i2v_first_frame_only."
            )

        self.grouping_mode = str(grouping_mode).strip().lower()
        if self.grouping_mode != "ext_suffix":
            raise ValueError(
                f"Unsupported grouping_mode: {grouping_mode}. "
                "Use one of: ext_suffix."
            )

        self.expected_view_ids = [int(view_id) for view_id in (expected_view_ids or [1, 2])]
        if not self.expected_view_ids:
            raise ValueError("expected_view_ids must not be empty.")

        self.sample_groups = self._build_multiview_groups(self.video_paths, self.prompts)
        if not self.sample_groups:
            raise ValueError(
                "No valid multiview groups were found. Check dataset_file contents and expected_view_ids."
            )

    def __len__(self) -> int:
        return len(self.sample_groups)

    def _parse_multiview_path(self, path: Path) -> Optional[Tuple[str, int]]:
        match = MULTIVIEW_EXT_PATTERN.match(path.stem)
        if match is None:
            return None
        base_name = match.group("base")
        try:
            rel_parent = path.relative_to(self.data_root).parent.as_posix()
        except ValueError:
            rel_parent = path.parent.as_posix()

        pair_key = base_name if rel_parent in {"", "."} else f"{rel_parent}/{base_name}"
        return pair_key, int(match.group("view"))

    def _build_multiview_groups(
        self,
        video_paths: List[Path],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
        invalid_groups: set[str] = set()
        skipped_missing_suffix = 0
        skipped_unexpected_view = 0
        duplicate_groups = 0
        incomplete_groups = 0

        for prompt, video_path in zip(prompts, video_paths):
            parsed = self._parse_multiview_path(video_path)
            if parsed is None:
                skipped_missing_suffix += 1
                continue

            pair_key, view_id = parsed
            if view_id not in self.expected_view_ids:
                skipped_unexpected_view += 1
                continue

            views = grouped.setdefault(pair_key, {})
            if view_id in views:
                invalid_groups.add(pair_key)
                duplicate_groups += 1
                continue

            views[view_id] = {
                "video_path": video_path,
                "prompt": prompt,
            }

        sample_groups: List[Dict[str, Any]] = []
        for pair_key, views in grouped.items():
            if pair_key in invalid_groups:
                continue

            missing_view_ids = [view_id for view_id in self.expected_view_ids if view_id not in views]
            if missing_view_ids:
                incomplete_groups += 1
                continue

            ordered_views = {view_id: views[view_id] for view_id in self.expected_view_ids}
            sample_groups.append(
                {
                    "pair_key": pair_key,
                    "views": ordered_views,
                }
            )

        logger.info(
            "Built %d multiview groups from %d flat samples (missing_suffix=%d, unexpected_view=%d, "
            "duplicate_groups=%d, incomplete_groups=%d)",
            len(sample_groups),
            len(video_paths),
            skipped_missing_suffix,
            skipped_unexpected_view,
            duplicate_groups,
            incomplete_groups,
        )
        return sample_groups

    @staticmethod
    def _derive_condition_path(video_path: Path, condition_folder: Optional[str]) -> Optional[Path]:
        if not condition_folder:
            return None

        parent_dir = video_path.parent
        filename = video_path.name
        condition_path = parent_dir.parent / condition_folder / filename
        if not condition_path.exists() and parent_dir.parent.name == "processed2":
            fallback_condition_path = parent_dir.parent.parent / condition_folder / filename
            if fallback_condition_path.exists():
                condition_path = fallback_condition_path
        return condition_path

    def _select_group_prompt(self, group: Dict[str, Any]) -> str:
        prompts = [group["views"][view_id]["prompt"] for view_id in self.expected_view_ids]
        non_empty_prompts = [prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
        prompt_candidates = non_empty_prompts or prompts
        return random.choice(prompt_candidates) if prompt_candidates else ""

    def _sample_apply_horizontal_flip(self) -> bool:
        if not self.random_flip:
            return False
        return random.random() < float(self.random_flip)

    def _preprocess_video_with_shared_flip(
        self,
        path: Path,
        clip_plan: Dict[str, int],
        *,
        apply_horizontal_flip: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        if apply_horizontal_flip:
            frames = torch.flip(frames, dims=[3])

        frames = frames / 255.0
        frames = (frames - 0.5) / 0.5
        image = frames[:1].clone() if self.image_to_video else None
        return image, frames, None

    def _load_group_target_views(
        self,
        group: Dict[str, Any],
        clip_plan: Dict[str, int],
        *,
        apply_horizontal_flip: bool,
    ) -> Tuple[Dict[int, Optional[torch.Tensor]], Dict[int, torch.Tensor], Dict[int, Dict[str, int]]]:
        images: Dict[int, Optional[torch.Tensor]] = {}
        videos: Dict[int, torch.Tensor] = {}
        metadata: Dict[int, Dict[str, int]] = {}

        for view_id in self.expected_view_ids:
            video_path = group["views"][view_id]["video_path"]
            image, video, _ = self._preprocess_video_with_shared_flip(
                video_path,
                clip_plan,
                apply_horizontal_flip=apply_horizontal_flip,
            )
            images[view_id] = image
            videos[view_id] = video
            metadata[view_id] = {
                "num_frames": int(video.shape[0]),
                "height": int(video.shape[2]),
                "width": int(video.shape[3]),
            }

        return images, videos, metadata

    def _load_group_optional_condition_videos(
        self,
        group: Dict[str, Any],
        clip_plan: Dict[str, int],
        *,
        condition_folder: Optional[str],
        log_name: str,
        apply_horizontal_flip: bool,
    ) -> Optional[Dict[int, Optional[torch.Tensor]]]:
        if not condition_folder:
            return None

        outputs: Dict[int, Optional[torch.Tensor]] = {}
        for view_id in self.expected_view_ids:
            video_path = group["views"][view_id]["video_path"]
            condition_path = self._derive_condition_path(video_path, condition_folder)
            if condition_path is None or not condition_path.is_file():
                logger.warning(
                    "Missing %s for pair=%s view=%s: %s",
                    log_name,
                    group["pair_key"],
                    view_id,
                    condition_path,
                )
                outputs[view_id] = None
                continue

            try:
                _, condition_video, _ = self._preprocess_video_with_shared_flip(
                    condition_path,
                    clip_plan,
                    apply_horizontal_flip=apply_horizontal_flip,
                )
                outputs[view_id] = condition_video
            except Exception as exc:
                logger.warning(
                    "Failed to load %s for pair=%s view=%s: %s",
                    log_name,
                    group["pair_key"],
                    view_id,
                    exc,
                )
                outputs[view_id] = None

        return outputs

    def _load_group_static_videos(
        self,
        group: Dict[str, Any],
        clip_plan: Dict[str, int],
        target_videos: Dict[int, torch.Tensor],
        *,
        apply_horizontal_flip: bool,
    ) -> Optional[Dict[int, Optional[torch.Tensor]]]:
        if not self.require_static_videos:
            return None

        if self.static_video_source_mode == "copy_first_frame":
            return {
                view_id: target_videos[view_id][0:1].repeat(target_videos[view_id].shape[0], 1, 1, 1).clone()
                for view_id in self.expected_view_ids
            }

        outputs: Dict[int, Optional[torch.Tensor]] = {}
        for view_id in self.expected_view_ids:
            video_path = group["views"][view_id]["video_path"]
            static_path = self._derive_condition_path(video_path, self.static_video_subdir)
            if static_path is None or not static_path.is_file():
                raise RuntimeError(
                    f"Missing static video for pair={group['pair_key']} view={view_id}: {static_path}"
                )
            try:
                _, static_video, _ = self._preprocess_video_with_shared_flip(
                    static_path,
                    clip_plan,
                    apply_horizontal_flip=apply_horizontal_flip,
                )
                outputs[view_id] = static_video
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load static video for pair={group['pair_key']} view={view_id}: {static_path}"
                ) from exc
        return outputs

    def _get_group_item_once(self, index: int) -> Dict[str, Any]:
        group = self.sample_groups[index]
        clip_plan_view_id = self.expected_view_ids[0]
        clip_plan = self._build_raw_clip_plan(group["views"][clip_plan_view_id]["video_path"])
        apply_horizontal_flip = self._sample_apply_horizontal_flip()

        multiview_images, multiview_videos, multiview_video_metadata = self._load_group_target_views(
            group,
            clip_plan,
            apply_horizontal_flip=apply_horizontal_flip,
        )

        multiview_hand_videos = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.robot_mesh_video_subdir,
            log_name="hand video",
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_static_videos = self._load_group_static_videos(
            group,
            clip_plan,
            multiview_videos,
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_static_disparities = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.static_disparity_subdir,
            log_name="static disparity",
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_hand_disparities = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.eef_depth_subdir,
            log_name="hand disparity",
            apply_horizontal_flip=apply_horizontal_flip,
        )

        prompt_text = self.id_token + self._select_group_prompt(group)
        return {
            "prompt": prompt_text,
            "pair_key": group["pair_key"],
            "multiview_view_ids": list(self.expected_view_ids),
            "multiview_videos": multiview_videos,
            "multiview_images": multiview_images,
            "multiview_video_metadata": multiview_video_metadata,
            "multiview_hand_videos": multiview_hand_videos,
            "multiview_static_videos": multiview_static_videos,
            "multiview_static_disparities": multiview_static_disparities,
            "multiview_hand_disparities": multiview_hand_disparities,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index >= len(self.sample_groups):
            logger.warning(
                "Index %d is out of range (multiview dataset size: %d). Using mod operation: %d %% %d = %d",
                index,
                len(self.sample_groups),
                index,
                len(self.sample_groups),
                index % len(self.sample_groups),
            )
            index = index % len(self.sample_groups)

        return self._get_group_item_once(index)


class MultiViewWidthConcatVideoDataset(MultiViewVideoDatasetWithConditionsAndResizing):
    """Multi-view dataset that concatenates views along the width dimension.

    Reuses MultiViewVideoDatasetWithConditionsAndResizing for grouping and per-view
    loading, but returns a single-view-shaped sample dict with the views stitched
    horizontally (like LingBot-VA). Downstream code treats the sample as a plain
    single-view wide video, so no model/collate changes are needed.
    """

    def __init__(
        self,
        *,
        per_view_width: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.per_view_width = int(per_view_width)
        if self.per_view_width <= 0:
            raise ValueError(f"per_view_width must be positive, got {per_view_width}")
        self.num_views = len(self.expected_view_ids)
        self.post_concat_width = self.per_view_width * self.num_views

    def _build_per_view_clip_plan(self, group: Dict[str, Any]) -> Dict[str, int]:
        view0_path = group["views"][self.expected_view_ids[0]]["video_path"]
        clip_plan = self._build_raw_clip_plan(view0_path)
        clip_plan["target_width"] = self.per_view_width
        return clip_plan

    def _concat_views_width(
        self, views_dict: Optional[Dict[int, Optional[torch.Tensor]]]
    ) -> Optional[torch.Tensor]:
        if views_dict is None:
            return None
        tensors: List[torch.Tensor] = []
        for view_id in self.expected_view_ids:
            tensor = views_dict.get(view_id)
            if tensor is None:
                return None
            tensors.append(tensor)
        return torch.cat(tensors, dim=-1)

    def _get_group_item_once(self, index: int) -> Dict[str, Any]:
        group = self.sample_groups[index]
        clip_plan = self._build_per_view_clip_plan(group)
        # Horizontal flip is disabled for width-concat multi-view: flipping each view
        # individually reverses the scene-to-view mapping and breaks the layout.
        apply_horizontal_flip = False

        _, multiview_videos, _ = self._load_group_target_views(
            group, clip_plan, apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_hand_videos = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.robot_mesh_video_subdir,
            log_name="hand video",
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_static_videos = self._load_group_static_videos(
            group,
            clip_plan,
            multiview_videos,
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_static_disparities = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.static_disparity_subdir,
            log_name="static disparity",
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_hand_disparities = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.eef_depth_subdir,
            log_name="hand disparity",
            apply_horizontal_flip=apply_horizontal_flip,
        )

        video = self._concat_views_width(multiview_videos)
        if video is None:
            raise RuntimeError(
                f"Failed to load all target views for pair_key={group['pair_key']}"
            )
        hand_videos = self._concat_views_width(multiview_hand_videos)
        static_videos = self._concat_views_width(multiview_static_videos)
        static_disparities = self._concat_views_width(multiview_static_disparities)
        eef_depths = self._concat_views_width(multiview_hand_disparities)

        image = video[:1].clone() if self.image_to_video else None

        prompt_text = self.id_token + self._select_group_prompt(group)
        return {
            "prompt": prompt_text,
            "image": image,
            "video": video,
            "hand_videos": hand_videos,
            "static_videos": static_videos,
            "static_disparities": static_disparities,
            "hand_disparities": eef_depths,
            "video_metadata": {
                "num_frames": int(video.shape[0]),
                "height": int(video.shape[2]),
                "width": int(video.shape[3]),
                "frame_skip": int(clip_plan.get("frame_skip", 1)),
            },
        }


class MultiViewCompositeVideoDataset(MultiViewVideoDatasetWithConditionsAndResizing):
    """Multi-view dataset that composes cells onto a single 2D canvas.

    Each cell is sourced from one suffix (fixed or randomly sampled from a pool)
    or filled with a constant color. Condition videos (robot-mesh, static, etc.)
    are composed in the same way as the target video, with per-condition fill
    colors for padding cells. Returns a single-view-shaped sample dict so the
    trainer treats the canvas as a plain wide video.
    """

    def __init__(
        self,
        *,
        layout_spec: Dict[str, Any],
        wrist_pcd_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._layout = layout_spec
        self._cells: List[Dict[str, Any]] = layout_spec["cells"]
        self._canvas_h = int(layout_spec["canvas_height"])
        self._canvas_w = int(layout_spec["canvas_width"])

        required_suffixes: set[str] = set()
        optional_suffixes: set[str] = set()
        for cell in self._cells:
            if "suffix" in cell:
                required_suffixes.add(cell["suffix"])
            elif "suffix_pool" in cell:
                optional_suffixes.update(cell["suffix_pool"])
        self._required_suffixes = required_suffixes
        self._optional_suffixes = optional_suffixes
        self._all_suffixes = required_suffixes | optional_suffixes
        if not self._all_suffixes:
            raise ValueError(
                "MultiViewCompositeVideoDataset requires at least one cell "
                "with suffix or suffix_pool."
            )

        # Parent expects expected_view_ids but we use suffix-based grouping.
        kwargs_with_placeholder = dict(kwargs)
        kwargs_with_placeholder["expected_view_ids"] = [1]
        super().__init__(**kwargs_with_placeholder)
        # Disable random flip: per-cell flip would scramble the layout.
        self.random_flip = False

        # Wrist-view pcd conditioning (enabled when static_video_source_mode
        # == "pcd_from_ext_first_frame").
        self._wrist_pcd_config: Dict[str, Any] = dict(wrist_pcd_config or {})
        self._pcd_calib_cache = None  # lazy; created per worker on first use

    def _build_multiview_groups(
        self,
        video_paths: List[Path],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        skipped_unknown_suffix = 0
        duplicate_suffix = 0
        for prompt, video_path in zip(prompts, video_paths):
            parsed = self._parse_composite_path(video_path)
            if parsed is None:
                skipped_unknown_suffix += 1
                continue
            pair_key, suffix = parsed
            views = grouped.setdefault(pair_key, {})
            if suffix in views:
                duplicate_suffix += 1
                continue
            views[suffix] = {"video_path": video_path, "prompt": prompt}

        sample_groups: List[Dict[str, Any]] = []
        incomplete_required = 0
        missing_pool = 0
        for pair_key, views in grouped.items():
            missing_required = [s for s in self._required_suffixes if s not in views]
            if missing_required:
                incomplete_required += 1
                continue
            pool_candidates: List[List[str]] = []
            pool_ok = True
            for cell in self._cells:
                if "suffix_pool" not in cell:
                    continue
                present = [s for s in cell["suffix_pool"] if s in views]
                if not present:
                    pool_ok = False
                    break
                pool_candidates.append(present)
            if not pool_ok:
                missing_pool += 1
                continue
            sample_groups.append(
                {
                    "pair_key": pair_key,
                    "views": views,
                    "pool_candidates": pool_candidates,
                }
            )

        logger.info(
            "Built %d composite groups from %d flat samples "
            "(unknown_suffix=%d, duplicate_suffix=%d, "
            "incomplete_required=%d, missing_pool=%d)",
            len(sample_groups),
            len(video_paths),
            skipped_unknown_suffix,
            duplicate_suffix,
            incomplete_required,
            missing_pool,
        )
        return sample_groups

    def _parse_composite_path(self, path: Path) -> Optional[Tuple[str, str]]:
        return _parse_multiview_suffix_path(path, self.data_root, self._all_suffixes)

    def _select_group_prompt(self, group: Dict[str, Any]) -> str:
        prompts = [info["prompt"] for info in group["views"].values()]
        non_empty = [p for p in prompts if isinstance(p, str) and p.strip()]
        candidates = non_empty or prompts
        return random.choice(candidates) if candidates else ""

    def _pick_cell_suffixes(self, group: Dict[str, Any]) -> Dict[int, str]:
        picks: Dict[int, str] = {}
        pool_index = 0
        for cell_idx, cell in enumerate(self._cells):
            if "suffix" in cell:
                picks[cell_idx] = cell["suffix"]
            elif "suffix_pool" in cell:
                present = group["pool_candidates"][pool_index]
                pool_index += 1
                picks[cell_idx] = random.choice(present)
        return picks

    def _fill_tensor(
        self,
        cell: Dict[str, Any],
        condition_key: str,
        num_frames: int,
    ) -> torch.Tensor:
        fill_map = cell.get("fill", {})
        color_name = fill_map.get(
            condition_key,
            _COMPOSITION_CONDITION_DEFAULT_FILL.get(condition_key, "black"),
        )
        uint8_value = _FILL_NAME_TO_UINT8[str(color_name).lower()]
        _, _, h, w = cell["bbox"]
        normalized = (uint8_value / 255.0 - 0.5) / 0.5
        return torch.full(
            (num_frames, 3, int(h), int(w)),
            float(normalized),
            dtype=torch.float32,
        )

    def _load_cell_video(
        self,
        video_path: Path,
        cell_h: int,
        cell_w: int,
        base_clip_plan: Dict[str, int],
    ) -> torch.Tensor:
        cell_plan = dict(base_clip_plan)
        cell_plan["target_height"] = int(cell_h)
        cell_plan["target_width"] = int(cell_w)
        _, frames, _ = self._preprocess_video_with_shared_flip(
            video_path,
            cell_plan,
            apply_horizontal_flip=False,
        )
        return frames

    def _compose_canvas(
        self,
        condition_key: str,
        condition_folder: Optional[str],
        group: Dict[str, Any],
        cell_suffix_picks: Dict[int, str],
        base_clip_plan: Dict[str, int],
        target_num_frames: int,
        *,
        is_target: bool,
    ) -> Optional[torch.Tensor]:
        if not is_target and not condition_folder:
            return None
        # Optional NPZ-runtime mode for disparity conditions (matches single-view
        # `*_disparity_npz_mode`). When set, we read NPZ per cell + (for the
        # wrist cell with PCD static mode) splat ext frame-0 depth into the
        # wrist camera. Falls back to mp4 path when npz_mode is unset.
        npz_mode: Optional[str] = None
        if not is_target:
            if condition_key == "static_disparity":
                npz_mode = self.static_disparity_npz_mode
            elif condition_key == "eef_depth":
                npz_mode = self.eef_depth_npz_mode
        canvas: Optional[torch.Tensor] = None
        for cell_idx, cell in enumerate(self._cells):
            y, x, h, w = cell["bbox"]
            if "fill" in cell:
                tile = self._fill_tensor(cell, condition_key, target_num_frames)
            else:
                suffix = cell_suffix_picks[cell_idx]
                info = group["views"][suffix]
                if is_target:
                    src_path: Optional[Path] = info["video_path"]
                    tile = self._load_cell_video(src_path, int(h), int(w), base_clip_plan)
                elif npz_mode is not None:
                    tile = self._load_cell_disparity_npz(
                        condition_key=condition_key,
                        condition_folder=condition_folder,
                        suffix=suffix,
                        info=info,
                        group=group,
                        cell_hw=(int(h), int(w)),
                        base_clip_plan=base_clip_plan,
                        target_num_frames=target_num_frames,
                        npz_mode=npz_mode,
                    )
                    if tile is None:
                        return None
                else:
                    src_path = self._derive_condition_path(info["video_path"], condition_folder)
                    if src_path is None or not src_path.is_file():
                        logger.warning(
                            "Missing %s for pair=%s suffix=%s: %s",
                            condition_key,
                            group["pair_key"],
                            suffix,
                            src_path,
                        )
                        return None
                    tile = self._load_cell_video(src_path, int(h), int(w), base_clip_plan)
            if canvas is None:
                canvas = torch.zeros(
                    (tile.shape[0], tile.shape[1], self._canvas_h, self._canvas_w),
                    dtype=tile.dtype,
                )
            elif tile.shape[0] != canvas.shape[0]:
                raise RuntimeError(
                    f"Cell frame count mismatch for condition={condition_key}: "
                    f"got {tile.shape[0]} expected {canvas.shape[0]}"
                )
            canvas[:, :, y:y + h, x:x + w] = tile
        return canvas

    def _load_cell_disparity_npz(
        self,
        *,
        condition_key: str,
        condition_folder: Optional[str],
        suffix: str,
        info: Dict[str, Any],
        group: Dict[str, Any],
        cell_hw: Tuple[int, int],
        base_clip_plan: Dict[str, int],
        target_num_frames: int,
        npz_mode: str,
    ) -> Optional[torch.Tensor]:
        """Load a single cell's disparity tile via NPZ runtime decode.

        Routing:
          - condition_key == "static_disparity" + suffix == "wrist" +
            static_video_source_mode == "pcd_from_ext_first_frame":
              splat the SELECTED ext's frame-0 depth into the wrist camera
              at start_idx, then repeat T times.
          - Otherwise: load NPZ at `<clip_dir>/<condition_folder>/<view_stem>.npz`
            and route through `_load_depth_npz_as_tensor`.
        """
        if condition_folder is None:
            return None
        h_cell, w_cell = int(cell_hw[0]), int(cell_hw[1])

        wrist_pcd_static = (
            condition_key == "static_disparity"
            and suffix == "wrist"
            and self.static_video_source_mode == "pcd_from_ext_first_frame"
        )
        if wrist_pcd_static:
            return self._render_wrist_static_disparity_pcd(
                group=group,
                cell_hw=(h_cell, w_cell),
                base_clip_plan=base_clip_plan,
                target_num_frames=target_num_frames,
            )

        # Per-view own NPZ (ext1/ext2/wrist). Path:
        #   <clip_dir>/<condition_folder>/<view_stem>.npz
        view_video_path: Path = info["video_path"]
        npz_path = view_video_path.parent.parent / condition_folder / f"{view_video_path.stem}.npz"
        if not npz_path.is_file():
            logger.warning(
                "Missing %s NPZ for pair=%s suffix=%s: %s",
                condition_key,
                group.get("pair_key"),
                suffix,
                npz_path,
            )
            return None
        cell_plan = dict(base_clip_plan)
        cell_plan["target_height"] = h_cell
        cell_plan["target_width"] = w_cell
        return self._load_depth_npz_as_tensor(npz_path, cell_plan, mode=npz_mode)

    def _render_wrist_static_disparity_pcd(
        self,
        *,
        group: Dict[str, Any],
        cell_hw: Tuple[int, int],
        base_clip_plan: Dict[str, int],
        target_num_frames: int,
    ) -> torch.Tensor:
        """Wrist cell scene depth via PCD: ext frame-0 depth -> base -> splat
        into wrist camera at start_idx, repeat T times. On any failure returns
        an all-black tile of the correct shape (graceful degradation).
        """
        from training.wan.utils import wrist_pcd_rendering as pcd_mod

        H_cell, W_cell = int(cell_hw[0]), int(cell_hw[1])
        black = torch.full(
            (int(target_num_frames), 3, H_cell, W_cell),
            float((0.0 / 255.0 - 0.5) / 0.5),
            dtype=torch.float32,
        )

        wrist_info = group["views"].get("wrist")
        if wrist_info is None:
            return black
        wrist_video_path = Path(wrist_info["video_path"])
        timestamp = wrist_video_path.stem.rsplit("__", 1)[0]
        clip_dir = wrist_video_path.parent.parent
        episode_meta_path = clip_dir / "episodes" / timestamp / "episode_meta.json"
        if not episode_meta_path.exists():
            return black
        if self._pcd_calib_cache is None:
            self._pcd_calib_cache = pcd_mod.PcdCalibrationCache()
        try:
            calibration = self._pcd_calib_cache.get(episode_meta_path)
        except pcd_mod.WristPcdRenderError:
            return black

        # Pick the ext suffix that was actually selected for this sample.
        picked_ext_suffix: Optional[str] = None
        for suffix in group["views"].keys():
            if suffix in {"ext1", "ext2"}:
                picked_ext_suffix = suffix
                break

        cfg = self._wrist_pcd_config
        try:
            tile = pcd_mod.render_wrist_cell_static_disparity(
                calibration=calibration,
                clip_dir=clip_dir,
                timestamp=timestamp,
                start_idx=int(base_clip_plan.get("start_idx", 0)),
                num_frames=int(target_num_frames),
                frame_skip=int(base_clip_plan.get("frame_skip", 1)),
                cell_hw=(H_cell, W_cell),
                preferred_ext_suffix=str(picked_ext_suffix or "ext1"),
                near_m=float(cfg.get("near_m", 0.2)),
                far_m=float(cfg.get("far_m", 3.0)),
                point_radius=float(cfg.get("point_radius", 0.012)),
                points_per_pixel=int(cfg.get("points_per_pixel", 8)),
                unproject_stride=int(cfg.get("unproject_stride", 1)),
                ext_max_depth_m=float(cfg.get("ext_max_depth_m", 1.5)),
                ext_depth_subdir=str(cfg.get("ext_depth_subdir", "stereo_inverse_depth")),
                robot_mesh_depth_subdir=str(
                    cfg.get("robot_mesh_depth_subdir", "robot_mesh_inverse_depth")
                ),
                videos_subdir=str(cfg.get("videos_subdir", "videos")),
            )
        except pcd_mod.WristPcdRenderError as exc:
            logger.warning(
                "wrist static_disparity PCD render failed for pair=%s (%s); using black tile",
                group.get("pair_key"),
                exc,
            )
            return black
        if tile.shape[0] != int(target_num_frames):
            logger.warning(
                "wrist static_disparity PCD frame count mismatch: got %d expected %d; using black",
                tile.shape[0],
                int(target_num_frames),
            )
            return black
        return tile

    def _build_valid_region_mask(self, num_frames: int) -> torch.Tensor:
        """Return a binary mask with 1 for real video cells and 0 for fill cells."""
        mask = torch.zeros(
            (int(num_frames), 1, self._canvas_h, self._canvas_w),
            dtype=torch.float32,
        )
        for cell in self._cells:
            if "fill" in cell:
                continue
            y, x, h, w = cell["bbox"]
            mask[:, :, y:y + h, x:x + w] = 1.0
        return mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Override with bounded retry so a single corrupted video in a group
        does not kill the dataloader worker. Falls back to a random alternate
        index after MAX_RETRIES worker failures (decord/video IO errors)."""
        n = len(self.sample_groups)
        if n == 0:
            raise RuntimeError("sample_groups is empty")
        if index >= n:
            index = index % n
        last_exc: Optional[Exception] = None
        tried = set()
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                return self._get_group_item_once(index)
            except Exception as exc:  # noqa: BLE001 — retry on any video IO error
                last_exc = exc
                tried.add(index)
                group_key = self.sample_groups[index].get("pair_key", index)
                logger.warning(
                    "MV composite __getitem__ attempt %d/%d failed for pair=%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    group_key,
                    exc,
                )
                for _ in range(8):
                    cand = random.randint(0, n - 1)
                    if cand not in tried:
                        index = cand
                        break
                else:
                    index = random.randint(0, n - 1)
        raise RuntimeError(
            f"MultiViewCompositeVideoDataset failed after {max_retries + 1} retries"
        ) from last_exc

    def _compose_static_with_wrist_pcd(
        self,
        *,
        group: Dict[str, Any],
        cell_suffix_picks: Dict[int, str],
        base_clip_plan: Dict[str, int],
        target_canvas: torch.Tensor,
    ) -> torch.Tensor:
        """Build the canvas-shaped static_videos tensor where the wrist cell is
        replaced by a wrist-trajectory render of the first-frame ext pcd (t=1+)
        with t=0 kept as the wrist GT frame at start_idx. All other cells
        remain as copy_first_frame (match existing behavior for ext / fill)."""
        from training.wan.utils import wrist_pcd_rendering as pcd_mod

        # Start with the copy_first_frame baseline for non-wrist cells.
        static_videos = target_canvas[0:1].repeat(
            target_canvas.shape[0], 1, 1, 1
        ).clone()

        # Identify the wrist cell(s). Currently layouts have at most one.
        wrist_cells: List[Tuple[int, Dict[str, Any]]] = []
        for cell_idx, cell in enumerate(self._cells):
            if "fill" in cell:
                continue
            picked = cell_suffix_picks.get(cell_idx)
            if picked == "wrist":
                wrist_cells.append((cell_idx, cell))
        if not wrist_cells:
            return static_videos

        # Pick which ext suffix was chosen for this sample (there should be exactly
        # one non-wrist ext cell in the current MV layout).
        picked_ext_suffix: Optional[str] = None
        for cell_idx, suffix in cell_suffix_picks.items():
            if suffix in {"ext1", "ext2"}:
                picked_ext_suffix = suffix
                break

        # Derive clip_dir + timestamp from the seed wrist path.
        # Expects render_v2 paths (episode_meta with annotation_assets +
        # trajectory_indices + stereo_inverse_depth are v2-only).
        wrist_cell_idx, _wrist_cell = wrist_cells[0]
        wrist_video_path = Path(group["views"]["wrist"]["video_path"])
        timestamp = wrist_video_path.stem.rsplit("__", 1)[0]
        clip_dir = wrist_video_path.parent.parent
        episode_meta_path = clip_dir / "episodes" / timestamp / "episode_meta.json"
        if not episode_meta_path.exists():
            logger.warning(
                "pcd_from_ext_first_frame: missing %s, falling back to copy_first_frame "
                "for pair=%s",
                episode_meta_path,
                group.get("pair_key"),
            )
            return static_videos

        if self._pcd_calib_cache is None:
            self._pcd_calib_cache = pcd_mod.PcdCalibrationCache()

        try:
            calibration = self._pcd_calib_cache.get(episode_meta_path)
        except pcd_mod.WristPcdRenderError as exc:
            logger.warning(
                "pcd_from_ext_first_frame: calibration load failed (%s); falling back",
                exc,
            )
            return static_videos

        cfg = self._wrist_pcd_config
        # Render into each wrist cell. Typically there is one, but loop for safety.
        for cell_idx, cell in wrist_cells:
            y, x, h, w = cell["bbox"]
            try:
                wrist_tile = pcd_mod.render_wrist_cell_static(
                    calibration=calibration,
                    clip_dir=clip_dir,
                    timestamp=timestamp,
                    start_idx=int(base_clip_plan.get("start_idx", 0)),
                    num_frames=int(target_canvas.shape[0]),
                    frame_skip=int(base_clip_plan.get("frame_skip", 1)),
                    cell_hw=(int(h), int(w)),
                    preferred_ext_suffix=str(picked_ext_suffix or "ext1"),
                    point_radius=float(cfg.get("point_radius", 0.01)),
                    points_per_pixel=int(cfg.get("points_per_pixel", 8)),
                    unproject_stride=int(cfg.get("unproject_stride", 1)),
                    ext_max_depth_m=float(cfg.get("ext_max_depth_m", 1.5)),
                    ext_depth_subdir=str(cfg.get("ext_depth_subdir", "stereo_inverse_depth")),
                    robot_mesh_depth_subdir=str(
                        cfg.get("robot_mesh_depth_subdir", "robot_mesh_depth_metric_first")
                    ),
                    videos_subdir=str(cfg.get("videos_subdir", "videos")),
                )
            except pcd_mod.WristPcdRenderError as exc:
                logger.warning(
                    "pcd_from_ext_first_frame: render failed for pair=%s (%s); "
                    "falling back to copy_first_frame for this wrist cell",
                    group.get("pair_key"),
                    exc,
                )
                continue
            # wrist_tile is (T, 3, h, w) float32 in [-1, 1]; stamp into static_videos.
            if wrist_tile.shape[0] != static_videos.shape[0]:
                logger.warning(
                    "pcd wrist tile frame count mismatch: got %d expected %d; skipping",
                    wrist_tile.shape[0],
                    static_videos.shape[0],
                )
                continue
            static_videos[:, :, y:y + h, x:x + w] = wrist_tile.to(static_videos.dtype)
        return static_videos

    def _get_group_item_once(self, index: int) -> Dict[str, Any]:
        group = self.sample_groups[index]
        cell_suffix_picks = self._pick_cell_suffixes(group)
        seed_path: Optional[Path] = None
        for cell_idx, cell in enumerate(self._cells):
            if "fill" in cell:
                continue
            seed_path = group["views"][cell_suffix_picks[cell_idx]]["video_path"]
            break
        if seed_path is None:
            raise RuntimeError(
                "MultiViewCompositeVideoDataset layout must include at least one non-fill cell."
            )
        base_clip_plan = self._build_raw_clip_plan(seed_path)
        target_num_frames = int(base_clip_plan.get("target_num_frames", self.max_num_frames))

        target_canvas = self._compose_canvas(
            condition_key="video",
            condition_folder=None,
            group=group,
            cell_suffix_picks=cell_suffix_picks,
            base_clip_plan=base_clip_plan,
            target_num_frames=target_num_frames,
            is_target=True,
        )
        if target_canvas is None:
            raise RuntimeError(
                f"Failed to compose target canvas for pair_key={group['pair_key']}"
            )

        hand_videos = self._compose_canvas(
            condition_key="hand_video",
            condition_folder=self.robot_mesh_video_subdir,
            group=group,
            cell_suffix_picks=cell_suffix_picks,
            base_clip_plan=base_clip_plan,
            target_num_frames=target_num_frames,
            is_target=False,
        )
        if self.require_static_videos:
            if self.static_video_source_mode == "copy_first_frame":
                static_videos = target_canvas[0:1].repeat(
                    target_canvas.shape[0], 1, 1, 1
                ).clone()
            elif self.static_video_source_mode == "pcd_from_ext_first_frame":
                static_videos = self._compose_static_with_wrist_pcd(
                    group=group,
                    cell_suffix_picks=cell_suffix_picks,
                    base_clip_plan=base_clip_plan,
                    target_canvas=target_canvas,
                )
            else:
                raise NotImplementedError(
                    f"MultiViewCompositeVideoDataset does not support "
                    f"static_video_source_mode={self.static_video_source_mode!r} yet."
                )
        else:
            static_videos = None

        static_disparities = self._compose_canvas(
            condition_key="static_disparity",
            condition_folder=self.static_disparity_subdir,
            group=group,
            cell_suffix_picks=cell_suffix_picks,
            base_clip_plan=base_clip_plan,
            target_num_frames=target_num_frames,
            is_target=False,
        )
        eef_depths = self._compose_canvas(
            condition_key="eef_depth",
            condition_folder=self.eef_depth_subdir,
            group=group,
            cell_suffix_picks=cell_suffix_picks,
            base_clip_plan=base_clip_plan,
            target_num_frames=target_num_frames,
            is_target=False,
        )

        image = target_canvas[:1].clone() if self.image_to_video else None
        prompt_text = self.id_token + self._select_group_prompt(group)
        valid_region_mask = self._build_valid_region_mask(target_canvas.shape[0])

        return {
            "prompt": prompt_text,
            "image": image,
            "video": target_canvas,
            "hand_videos": hand_videos,
            "static_videos": static_videos,
            "static_disparities": static_disparities,
            "hand_disparities": eef_depths,
            "masks": valid_region_mask,
            "video_metadata": {
                "num_frames": int(target_canvas.shape[0]),
                "height": int(target_canvas.shape[2]),
                "width": int(target_canvas.shape[3]),
                "frame_skip": int(base_clip_plan.get("frame_skip", 1)),
            },
        }


class MultiViewPropeDataset(MultiViewVideoDatasetWithConditionsAndResizing):
    """Multi-view dataset that keeps each view as a separate tensor and yields
    per-(view, frame) camera poses and intrinsics for PRoPE-style attention.

    Unlike ``MultiViewCompositeVideoDataset`` which pastes every view into one
    2D canvas, this dataset preserves per-view resolution and returns the raw
    per-view video tensors so the transformer can apply per-camera projection
    blocks on top of standard RoPE.

    Layout ``views_spec`` is a list of dicts, each describing one view:

        - ``suffix``:      fixed suffix, e.g. "wrist"
        - ``suffix_pool``: list of candidate suffixes (random pick per sample)
        - ``size``:        [H, W] output resolution in pixels
        - ``native_size``: [H, W] source resolution for intrinsics scaling;
                           optional — default reads from intrinsics.json entry
    """

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Bounded-retry override mirroring ``MultiViewCompositeVideoDataset``: a
        single corrupt mp4 / decord buffer-source failure should not kill the
        dataloader worker (and through it the entire training run). Falls back
        to a random alternate group up to ``max_retries`` times."""
        n = len(self.sample_groups)
        if n == 0:
            raise RuntimeError("sample_groups is empty")
        if index >= n:
            index = index % n
        last_exc: Optional[Exception] = None
        tried = set()
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                return self._get_group_item_once(index)
            except Exception as exc:  # noqa: BLE001 — retry on any video IO error
                last_exc = exc
                tried.add(index)
                group_key = self.sample_groups[index].get("pair_key", index)
                logger.warning(
                    "PRoPE __getitem__ attempt %d/%d failed for pair=%s: %s",
                    attempt + 1, max_retries + 1, group_key, exc,
                )
                for _ in range(8):
                    cand = random.randint(0, n - 1)
                    if cand not in tried:
                        index = cand
                        break
                else:
                    index = random.randint(0, n - 1)
        raise RuntimeError(
            f"MultiViewPropeDataset failed after {max_retries + 1} retries"
        ) from last_exc

    def __init__(
        self,
        *,
        views_spec: List[Dict[str, Any]],
        camera_serials_json: Optional[str] = None,
        cam2base_json: Optional[str] = None,
        cam2base_superset_json: Optional[str] = None,
        intrinsics_json: Optional[str] = None,
        raw_cache_root: Optional[str] = None,
        wrist_pose_source: str = "trajectory",
        gripper2wrist_json: Optional[str] = None,
        wrist_proprio_flip: str = "none",
        **kwargs,
    ) -> None:
        if not views_spec:
            raise ValueError("views_spec must list at least one view")
        self._views_spec = [self._normalize_view_spec(v) for v in views_spec]
        required, optional = set(), set()
        for view in self._views_spec:
            if view["suffix"] is not None:
                required.add(view["suffix"])
            if view["suffix_pool"] is not None:
                optional.update(view["suffix_pool"])
        self._required_suffixes = required
        self._optional_suffixes = optional
        self._all_suffixes = required | optional
        if not self._all_suffixes:
            raise ValueError("views_spec must define at least one suffix or suffix_pool")

        kwargs_with_placeholder = dict(kwargs)
        kwargs_with_placeholder["expected_view_ids"] = [1]
        kwargs_with_placeholder["_allow_load_tensors"] = True
        super().__init__(**kwargs_with_placeholder)
        self.random_flip = False

        # DROID camera calibration assets (JSON files shared across episodes).
        from pathlib import Path as _P
        repo_root = _P(__file__).resolve().parents[2]
        assets_root = repo_root / "data_processing" / "droid" / "assets"
        self._camera_serials_path = _P(camera_serials_json) if camera_serials_json else assets_root / "camera_serials.json"
        self._cam2base_path = _P(cam2base_json) if cam2base_json else assets_root / "cam2base_extrinsics.json"
        self._cam2base_superset_path = (
            _P(cam2base_superset_json) if cam2base_superset_json else assets_root / "cam2base_extrinsic_superset.json"
        )
        self._intrinsics_path = _P(intrinsics_json) if intrinsics_json else assets_root / "intrinsics.json"
        self._raw_cache_root = _P(raw_cache_root) if raw_cache_root else _P("data/droid/raw_cache_v2")
        self._wrist_pose_source = str(wrist_pose_source)
        self._gripper2wrist_path = (
            _P(gripper2wrist_json)
            if gripper2wrist_json
            else repo_root / "data_processing" / "third_party" / "PointWorld" / "real" / "gripper2wrist_transforms.json"
        )
        self._wrist_proprio_flip = str(wrist_proprio_flip)

        # Large calibration JSON assets (up to ~170 MB total) are only needed when a
        # per-episode ``camera_bundle.npz`` sidecar is absent. Load them lazily so the
        # training loader can work on hosts (e.g. h200) that only carry the sidecars.
        from training.wan.utils.camera_pose import (
            load_json as _load_json,
        )
        self._camera_pose_load_json = _load_json
        self._camera_serials_blob_cached: Any = None
        self._cam2base_blobs_cached: Optional[List[Tuple[str, Dict[str, Any]]]] = None
        self._intrinsics_blob_cached: Any = None

    @staticmethod
    def _normalize_view_spec(view: Dict[str, Any]) -> Dict[str, Any]:
        spec: Dict[str, Any] = {
            "suffix": view.get("suffix"),
            "suffix_pool": view.get("suffix_pool"),
            "size": None,
            "native_size": view.get("native_size"),
        }
        if spec["suffix"] is None and spec["suffix_pool"] is None:
            raise ValueError("each view must set suffix or suffix_pool")
        if spec["suffix"] is not None and spec["suffix_pool"] is not None:
            raise ValueError("suffix and suffix_pool are mutually exclusive in a view")
        size = view.get("size")
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise ValueError("each view must set size=[H, W]")
        spec["size"] = (int(size[0]), int(size[1]))
        if spec["native_size"] is not None:
            nh, nw = spec["native_size"]
            spec["native_size"] = (int(nh), int(nw))
        return spec

    # --- grouping reuses the composite suffix parser ------------------------------

    def _build_multiview_groups(
        self,
        video_paths: List[Path],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        skipped_unknown_suffix = 0
        duplicate_suffix = 0
        for prompt, video_path in zip(prompts, video_paths):
            parsed = self._parse_prope_path(video_path)
            if parsed is None:
                skipped_unknown_suffix += 1
                continue
            pair_key, suffix = parsed
            views = grouped.setdefault(pair_key, {})
            if suffix in views:
                duplicate_suffix += 1
                continue
            views[suffix] = {"video_path": video_path, "prompt": prompt}

        sample_groups: List[Dict[str, Any]] = []
        incomplete_required = 0
        missing_pool = 0
        for pair_key, views in grouped.items():
            missing_required = [s for s in self._required_suffixes if s not in views]
            if missing_required:
                incomplete_required += 1
                continue
            pool_candidates: List[List[str]] = []
            pool_ok = True
            for view in self._views_spec:
                if view["suffix_pool"] is None:
                    continue
                present = [s for s in view["suffix_pool"] if s in views]
                if not present:
                    pool_ok = False
                    break
                pool_candidates.append(present)
            if not pool_ok:
                missing_pool += 1
                continue
            sample_groups.append(
                {
                    "pair_key": pair_key,
                    "views": views,
                    "pool_candidates": pool_candidates,
                }
            )
        logger.info(
            "MultiViewPropeDataset: %d groups from %d flat samples "
            "(unknown_suffix=%d duplicate_suffix=%d incomplete=%d missing_pool=%d)",
            len(sample_groups),
            len(video_paths),
            skipped_unknown_suffix,
            duplicate_suffix,
            incomplete_required,
            missing_pool,
        )
        if self.load_tensors:
            before_filter = len(sample_groups)
            sample_groups = [
                group for group in sample_groups
                if self._group_has_required_preencoded_tensors(group)
            ]
            logger.info(
                "MultiViewPropeDataset preencoded filter: kept %d/%d groups",
                len(sample_groups),
                before_filter,
            )
        return sample_groups

    def _parse_prope_path(self, path: Path) -> Optional[Tuple[str, str]]:
        return _parse_multiview_suffix_path(path, self.data_root, self._all_suffixes)

    def _select_group_prompt(self, group: Dict[str, Any]) -> str:
        prompts = [info["prompt"] for info in group["views"].values()]
        non_empty = [p for p in prompts if isinstance(p, str) and p.strip()]
        candidates = non_empty or prompts
        return random.choice(candidates) if candidates else ""

    def _pick_view_suffixes(self, group: Dict[str, Any]) -> List[str]:
        picks: List[str] = []
        pool_index = 0
        for view in self._views_spec:
            if view["suffix"] is not None:
                picks.append(view["suffix"])
            else:
                present = group["pool_candidates"][pool_index]
                pool_index += 1
                picks.append(random.choice(present))
        return picks

    # --- pose/K lookup -------------------------------------------------------------

    @staticmethod
    def _episode_id_from_video_path(video_path: Path) -> str:
        """Reconstruct `{lab}+{hash}+{timestamp}` from a render_v2 videos/*.mp4 path.

        Expects `<...>/{lab}/{hash}/videos*/{timestamp}__{suffix}[__clip].mp4`.
        """
        parts = video_path.parts
        if len(parts) < 4 or not str(parts[-2]).startswith("videos"):
            raise ValueError(f"Unexpected video path layout for pose lookup: {video_path}")
        lab = parts[-4]
        hash_value = parts[-3]
        timestamp = _strip_multiview_suffix_and_clip(video_path.stem)
        return f"{lab}+{hash_value}+{timestamp}"

    def _lookup_serial(self, episode_id: str, suffix: str) -> Optional[str]:
        entry = self._camera_serials_blob.get(episode_id)
        if not isinstance(entry, dict):
            return None
        key_candidates = {
            "ext1": ("ext1_cam_serial",),
            "ext2": ("ext2_cam_serial",),
            "wrist": ("wrist_cam_serial", "wrist1_cam_serial"),
        }.get(suffix, ())
        for key in key_candidates:
            serial = entry.get(key)
            if serial:
                return str(serial)
        return None

    def _lookup_intrinsics(
        self,
        episode_id: str,
        serial: str,
        target_size: Tuple[int, int],
        native_size: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """Return a 3x3 K scaled to ``target_size=(H, W)``.

        Priority: intrinsics.json (per-episode, per-serial) -> raw_cache_v2 metadata.json fallback.
        """
        from training.wan.utils.camera_pose import (
            read_intrinsics,
            read_intrinsics_size,
            scale_intrinsics,
            load_json as _load_json,
        )
        target_h, target_w = target_size
        # intrinsics.json uses (width, height); scale_intrinsics expects same convention.
        dst_size = (int(target_w), int(target_h))
        entry = self._intrinsics_blob.get(episode_id, {}) if isinstance(self._intrinsics_blob, dict) else {}
        K = None
        src_size_found = None
        if isinstance(entry, dict) and serial in entry:
            K = read_intrinsics(entry[serial])
            src_size_found = read_intrinsics_size(entry[serial])
        if K is None:
            md = self._raw_cache_root / episode_id / "metadata.json"
            if md.exists():
                blob = _load_json(md)
                K = read_intrinsics(blob)
                # Try to find a "{serial}" keyed intrinsics field too
                if K is None and isinstance(blob, dict) and serial in blob:
                    K = read_intrinsics(blob[serial])
                    src_size_found = read_intrinsics_size(blob[serial])
        if K is None:
            raise KeyError(
                f"Missing intrinsics for episode={episode_id} serial={serial}; "
                "checked intrinsics.json and raw_cache_v2/metadata.json"
            )
        src_size = native_size if native_size is not None else src_size_found
        # native_size is stored as (H, W); convert to (W, H) for scale_intrinsics
        if src_size is not None and native_size is not None:
            src_h, src_w = src_size
            src_size = (int(src_w), int(src_h))
        return scale_intrinsics(K.astype(np.float64), src_size, dst_size)

    def _lookup_cam2base(self, episode_id: str, serial: str) -> Optional[np.ndarray]:
        from training.wan.utils.camera_pose import read_matrix
        for _name, blob in self._cam2base_blobs:
            if not isinstance(blob, dict):
                continue
            entry = blob.get(episode_id)
            if not isinstance(entry, dict):
                continue
            candidate = entry.get(serial)
            if candidate is None:
                continue
            mat = read_matrix(candidate)
            if mat is not None:
                return mat
        return None

    @property
    def _camera_serials_blob(self) -> Any:
        if self._camera_serials_blob_cached is None:
            self._camera_serials_blob_cached = self._camera_pose_load_json(
                self._camera_serials_path
            )
        return self._camera_serials_blob_cached

    @property
    def _cam2base_blobs(self) -> List[Tuple[str, Dict[str, Any]]]:
        if self._cam2base_blobs_cached is None:
            blobs: List[Tuple[str, Dict[str, Any]]] = []
            primary_blob = self._camera_pose_load_json(self._cam2base_path)
            blobs.append((self._cam2base_path.name, primary_blob))
            if (
                self._cam2base_superset_path.exists()
                and self._cam2base_superset_path != self._cam2base_path
            ):
                blobs.append(
                    (
                        self._cam2base_superset_path.name,
                        self._camera_pose_load_json(self._cam2base_superset_path),
                    )
                )
            self._cam2base_blobs_cached = blobs
        return self._cam2base_blobs_cached

    @property
    def _intrinsics_blob(self) -> Any:
        if self._intrinsics_blob_cached is None:
            self._intrinsics_blob_cached = self._camera_pose_load_json(
                self._intrinsics_path
            )
        return self._intrinsics_blob_cached

    @staticmethod
    def _camera_bundle_path_for_video(video_path: Path) -> Path:
        """Expected sidecar path: ``.../{lab}/{hash}/episodes/{TS}/camera_bundle.npz``."""
        parts = video_path.parts
        if len(parts) < 4 or not str(parts[-2]).startswith("videos"):
            raise ValueError(f"Unexpected video path layout for camera_bundle: {video_path}")
        timestamp = _strip_multiview_suffix_and_clip(video_path.stem)
        base = Path(*parts[:-2])
        return base / "episodes" / timestamp / "camera_bundle.npz"

    def _load_camera_bundle_for_video(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Load per-episode ``camera_bundle.npz`` sidecar if present.

        Result dict carries all raw arrays plus a ``view_to_idx`` map. Cached via
        ``self._bundle_cache`` so repeated __getitem__ on the same episode reuses
        the npz load.
        """
        if not getattr(self, "_bundle_cache", None):
            self._bundle_cache: Dict[str, Dict[str, Any]] = {}
        try:
            bundle_path = self._camera_bundle_path_for_video(video_path)
        except ValueError:
            return None
        key = str(bundle_path)
        cached = self._bundle_cache.get(key)
        if cached is not None:
            return cached
        if not bundle_path.exists():
            self._bundle_cache[key] = None  # negative cache
            return None
        data = np.load(bundle_path, allow_pickle=True)
        view_names = [str(v) for v in data["view_names"].tolist()]
        bundle: Dict[str, Any] = {
            "path": bundle_path,
            "view_names": view_names,
            "view_to_idx": {v: i for i, v in enumerate(view_names)},
            "serials": [str(s) for s in data["serials"].tolist()],
            "intrinsics": np.asarray(data["intrinsics"], dtype=np.float64),
            "native_sizes": np.asarray(data["native_sizes"], dtype=np.int64),
            "static_c2w": np.asarray(data["static_c2w"], dtype=np.float64),
            "wrist_view_index": int(data["wrist_view_index"]),
        }
        if "wrist_poses" in data.files:
            bundle["wrist_poses"] = np.asarray(data["wrist_poses"], dtype=np.float64)
        self._bundle_cache[key] = bundle
        return bundle

    def _pose_and_K_from_bundle(
        self,
        *,
        bundle: Dict[str, Any],
        suffix: str,
        target_size: Tuple[int, int],
        native_size_override: Optional[Tuple[int, int]],
        num_frames: int,
        video_stem: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        from training.wan.utils.camera_pose import scale_intrinsics

        idx = bundle["view_to_idx"][suffix]
        serial = bundle["serials"][idx]

        # Intrinsics: scale bundle's native K to the per-view training resolution.
        # ``native_sizes`` stored as (W, H) per precompute_camera_bundle.py. Same
        # convention that ``scale_intrinsics`` expects for both src_size and dst_size.
        target_h, target_w = target_size
        dst_size = (int(target_w), int(target_h))
        native = tuple(int(v) for v in bundle["native_sizes"][idx].tolist())
        if native_size_override is not None:
            # config hint: (H, W) → convert to (W, H)
            override_h, override_w = native_size_override
            src_size = (int(override_w), int(override_h))
        elif native[0] > 0 and native[1] > 0:
            src_size = (int(native[0]), int(native[1]))  # already (W, H)
        else:
            src_size = None
        K = scale_intrinsics(bundle["intrinsics"][idx], src_size, dst_size)

        # Poses: wrist view uses per-frame; ext views use static repeated.
        if suffix == "wrist":
            wrist_poses = bundle.get("wrist_poses")
            if wrist_poses is None:
                raise KeyError("camera_bundle has no wrist_poses for wrist view")
            pose_seq = wrist_poses
            clip_indices = _clip_frame_indices_from_stem(
                video_stem,
                num_frames=num_frames,
                source_len=int(pose_seq.shape[0]),
            )
            if clip_indices is not None:
                pose_seq = pose_seq[clip_indices]
        else:
            pose_seq = np.repeat(bundle["static_c2w"][idx][None, ...], num_frames, axis=0)

        # Match the requested frame count.
        if pose_seq.shape[0] == num_frames:
            poses = pose_seq
        elif pose_seq.shape[0] < num_frames:
            last = pose_seq[-1:]
            pad = np.repeat(last, num_frames - pose_seq.shape[0], axis=0)
            poses = np.concatenate([pose_seq, pad], axis=0)
        else:
            idxs = np.linspace(0, pose_seq.shape[0] - 1, num_frames).round().astype(np.int64)
            poses = pose_seq[idxs]
        return poses, K, serial

    def _serial_and_K_from_bundle(
        self,
        *,
        bundle: Dict[str, Any],
        suffix: str,
        target_size: Tuple[int, int],
        native_size_override: Optional[Tuple[int, int]],
    ) -> Tuple[str, np.ndarray]:
        """Read only serial and scaled intrinsics from camera_bundle.npz."""
        from training.wan.utils.camera_pose import scale_intrinsics

        idx = bundle["view_to_idx"][suffix]
        serial = bundle["serials"][idx]
        target_h, target_w = target_size
        dst_size = (int(target_w), int(target_h))
        native = tuple(int(v) for v in bundle["native_sizes"][idx].tolist())
        if native_size_override is not None:
            override_h, override_w = native_size_override
            src_size = (int(override_w), int(override_h))
        elif native[0] > 0 and native[1] > 0:
            src_size = (int(native[0]), int(native[1]))
        else:
            src_size = None
        K = scale_intrinsics(bundle["intrinsics"][idx], src_size, dst_size)
        return serial, K

    def _load_view_poses(
        self,
        suffix: str,
        video_path: Path,
        serial: str,
        num_frames: int,
    ) -> np.ndarray:
        episode_id = self._episode_id_from_video_path(video_path)
        if suffix == "wrist":
            from training.wan.utils.camera_pose import (
                load_raw_wrist_pose_sequence,
            )
            traj = self._raw_cache_root / episode_id / "trajectory.h5"
            md = self._raw_cache_root / episode_id / "metadata.json"
            if not traj.exists():
                raise FileNotFoundError(f"Missing trajectory for wrist pose: {traj}")
            pose_seq = load_raw_wrist_pose_sequence(
                traj,
                serial,
                pose_source=self._wrist_pose_source,
                raw_metadata_path=md if md.exists() else None,
                gripper2wrist_json=self._gripper2wrist_path,
                wrist_proprio_flip=self._wrist_proprio_flip,
            )  # [T_traj, 4, 4]
            clip_indices = _clip_frame_indices_from_stem(
                video_path.stem,
                num_frames=num_frames,
                source_len=int(pose_seq.shape[0]),
            )
            if clip_indices is not None:
                pose_seq = pose_seq[clip_indices]
            # Subsample / clip to `num_frames`.
            if pose_seq.shape[0] == num_frames:
                return pose_seq
            if pose_seq.shape[0] < num_frames:
                last = pose_seq[-1:]
                pad = np.repeat(last, num_frames - pose_seq.shape[0], axis=0)
                return np.concatenate([pose_seq, pad], axis=0)
            idxs = np.linspace(0, pose_seq.shape[0] - 1, num_frames).round().astype(np.int64)
            return pose_seq[idxs]
        # Static ext: repeat static cam2base
        mat = self._lookup_cam2base(episode_id, serial)
        if mat is None:
            raise KeyError(
                f"Missing static cam2base for episode={episode_id} serial={serial} "
                f"(suffix={suffix})"
            )
        return np.repeat(mat[None, ...], num_frames, axis=0)

    # --- main sample loop ---------------------------------------------------------

    @staticmethod
    def _raw_frame_count_from_latents(latents: torch.Tensor) -> int:
        latent_frames = int(latents.shape[1])
        if latent_frames % 2 == 0:
            return latent_frames * 4
        return (latent_frames - 1) * 4 + 1

    def _latent_path_for_subdir(self, path: Path, subdir: str) -> Path:
        return self._resolve_existing_subdir(path, subdir).joinpath(f"{path.stem}.pt")

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _has_required_preencoded_view_tensors(self, path: Path) -> bool:
        if not self._is_nonempty_file(self._latent_path_for_subdir(path, self.video_latents_subdir)):
            return False
        if self.robot_mesh_video_subdir:
            hand_path = self._derive_condition_path(path, self.robot_mesh_video_subdir)
            if hand_path is None:
                return False
            if not self._is_nonempty_file(self._latent_path_for_subdir(hand_path, self.robot_mesh_video_latents_subdir)):
                return False
        return True

    def _group_has_required_preencoded_tensors(self, group: Dict[str, Any]) -> bool:
        prompt_available = False
        pool_index = 0
        for view in self._views_spec:
            if view["suffix"] is not None:
                suffixes = [view["suffix"]]
            else:
                suffixes = list(group["pool_candidates"][pool_index])
                pool_index += 1
            if not suffixes:
                return False
            view_ok = False
            for suffix in suffixes:
                info = group["views"].get(suffix)
                if info is None:
                    continue
                video_path = info["video_path"]
                if self._has_required_preencoded_view_tensors(video_path):
                    view_ok = True
                if self._is_nonempty_file(self._latent_path_for_subdir(video_path, self.prompt_embeds_subdir)):
                    prompt_available = True
            if not view_ok:
                return False
        return prompt_available

    def _load_latents_from_subdir(self, path: Path, subdir: str, *, required: bool) -> Optional[torch.Tensor]:
        tensor_path = self._latent_path_for_subdir(path, subdir)
        if not self._is_nonempty_file(tensor_path):
            if required:
                raise FileNotFoundError(f"Missing tensor file: {tensor_path}")
            return None
        return torch.load(tensor_path, map_location="cpu", weights_only=True)

    def _load_prompt_embeds_if_available(self, path: Path) -> Optional[torch.Tensor]:
        embeds = self._load_latents_from_subdir(path, self.prompt_embeds_subdir, required=False)
        if embeds is None:
            return None
        if embeds.shape[0] < self.max_prompt_length:
            padding = torch.zeros(
                self.max_prompt_length - embeds.shape[0],
                embeds.shape[1],
                dtype=embeds.dtype,
            )
            embeds = torch.cat([embeds, padding], dim=0)
        elif embeds.shape[0] > self.max_prompt_length:
            embeds = embeds[:self.max_prompt_length]
        return embeds

    @staticmethod
    def _first_latent_frame_repeat(latents: torch.Tensor) -> torch.Tensor:
        return latents[:, 0:1, :, :].repeat(1, int(latents.shape[1]), 1, 1).clone()

    def _load_prope_latent_item_once(self, index: int) -> Dict[str, Any]:
        group = self.sample_groups[index]
        view_suffixes = self._pick_view_suffixes(group)

        video_views: List[torch.Tensor] = []
        hand_video_views: List[torch.Tensor] = []
        static_video_views: List[torch.Tensor] = []
        static_disparity_video_views: List[torch.Tensor] = []
        eef_depth_video_views: List[torch.Tensor] = []
        camera_poses_list: List[np.ndarray] = []
        intrinsics_list: List[np.ndarray] = []
        view_sizes: List[Tuple[int, int]] = []
        prompt_embed_candidates: List[torch.Tensor] = []

        for view_idx, (view, suffix) in enumerate(zip(self._views_spec, view_suffixes)):
            target_h, target_w = view["size"]
            info = group["views"][suffix]
            main_path = info["video_path"]

            video_latents = self._load_latents_from_subdir(
                main_path,
                self.video_latents_subdir,
                required=True,
            )
            fun_inp_i2v_latents = self._load_latents_from_subdir(
                main_path,
                self.fun_inp_i2v_latents_subdir,
                required=False,
            )
            prompt_embeds = self._load_prompt_embeds_if_available(main_path)
            video_views.append(video_latents)
            if prompt_embeds is not None:
                prompt_embed_candidates.append(prompt_embeds)
            raw_num_frames = self._raw_frame_count_from_latents(video_latents)

            hand_latents: Optional[torch.Tensor] = None
            if self.robot_mesh_video_subdir:
                hand_path = self._derive_condition_path(main_path, self.robot_mesh_video_subdir)
                if hand_path is not None:
                    hand_latents = self._load_condition_video_latents(
                        hand_path,
                        self.robot_mesh_video_latents_subdir,
                    )
            if hand_latents is None:
                raise FileNotFoundError(
                    f"Missing preencoded hand/control latents for pair={group['pair_key']} "
                    f"suffix={suffix} path={main_path}"
                )
            hand_video_views.append(hand_latents)

            if self.require_static_videos:
                static_latents: Optional[torch.Tensor] = None
                if self.static_video_subdir:
                    static_path = self._derive_condition_path(main_path, self.static_video_subdir)
                    if static_path is not None:
                        static_latents = self._load_condition_video_latents(
                            static_path,
                            self.static_video_latents_subdir,
                        )
                if static_latents is None:
                    static_latents = self._first_latent_frame_repeat(video_latents)
                static_video_views.append(static_latents)
            elif fun_inp_i2v_latents is not None:
                static_video_views.append(fun_inp_i2v_latents)

            if self.static_disparity_subdir and self.static_disparity_latents_subdir:
                sd_path = self._derive_condition_path(main_path, self.static_disparity_subdir)
                sd_latents = (
                    self._load_condition_video_latents(sd_path, self.static_disparity_latents_subdir)
                    if sd_path is not None
                    else None
                )
                if sd_latents is None:
                    sd_latents = torch.zeros_like(video_latents)
                static_disparity_video_views.append(sd_latents)

            if self.eef_depth_subdir and self.eef_depth_latents_subdir:
                hd_path = self._derive_condition_path(main_path, self.eef_depth_subdir)
                hd_latents = (
                    self._load_condition_video_latents(hd_path, self.eef_depth_latents_subdir)
                    if hd_path is not None
                    else None
                )
                if hd_latents is None:
                    hd_latents = torch.zeros_like(video_latents)
                eef_depth_video_views.append(hd_latents)

            view_sizes.append((int(target_h), int(target_w)))

            bundle = self._load_camera_bundle_for_video(main_path)
            use_bundle_pose = not (suffix == "wrist" and self._wrist_pose_source != "trajectory")
            if bundle is not None and suffix in bundle["view_to_idx"] and use_bundle_pose:
                poses, K, serial = self._pose_and_K_from_bundle(
                    bundle=bundle,
                    suffix=suffix,
                    target_size=(int(target_h), int(target_w)),
                    native_size_override=view["native_size"],
                    num_frames=raw_num_frames,
                    video_stem=main_path.stem,
                )
            else:
                episode_id = self._episode_id_from_video_path(main_path)
                if bundle is not None and suffix in bundle["view_to_idx"]:
                    serial, K = self._serial_and_K_from_bundle(
                        bundle=bundle,
                        suffix=suffix,
                        target_size=(int(target_h), int(target_w)),
                        native_size_override=view["native_size"],
                    )
                else:
                    serial = self._lookup_serial(episode_id, suffix)
                    if serial is None:
                        raise KeyError(
                            f"Missing camera serial for episode={episode_id} suffix={suffix}"
                        )
                    K = self._lookup_intrinsics(
                        episode_id=episode_id,
                        serial=serial,
                        target_size=(int(target_h), int(target_w)),
                        native_size=view["native_size"],
                    )
                poses = self._load_view_poses(suffix, main_path, serial, raw_num_frames)
            camera_poses_list.append(poses.astype(np.float32))
            intrinsics_list.append(K.astype(np.float32))

        latent_frames_out = video_views[0].shape[1]
        for v_idx, v in enumerate(video_views[1:], start=1):
            if v.shape[1] != latent_frames_out:
                raise RuntimeError(
                    f"Latent frame count mismatch view[0]={latent_frames_out} vs view[{v_idx}]={v.shape[1]}"
                )

        camera_poses = torch.from_numpy(np.stack(camera_poses_list, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0))
        view_sizes_tensor = torch.tensor(view_sizes, dtype=torch.int32)

        if not prompt_embed_candidates:
            raise RuntimeError(f"No prompt embeddings loaded for pair={group['pair_key']}")
        prompt_embeds = random.choice(prompt_embed_candidates)

        return {
            "prompt": prompt_embeds,
            "pair_key": str(group["pair_key"]),
            "video_views": video_views,
            "hand_video_views": hand_video_views,
            "static_video_views": static_video_views or None,
            "static_disparity_video_views": static_disparity_video_views or None,
            "eef_depth_video_views": eef_depth_video_views or None,
            "camera_poses": camera_poses,
            "intrinsics": intrinsics,
            "view_sizes": view_sizes_tensor,
            "video_metadata": {
                "num_frames": int(self._raw_frame_count_from_latents(video_views[0])),
                "num_views": int(len(video_views)),
                "frame_skip": 1,
            },
        }

    def _get_group_item_once(self, index: int) -> Dict[str, Any]:
        if self.load_tensors:
            return self._load_prope_latent_item_once(index)

        group = self.sample_groups[index]
        view_suffixes = self._pick_view_suffixes(group)

        seed_view_idx = 0
        seed_suffix = view_suffixes[seed_view_idx]
        seed_path = group["views"][seed_suffix]["video_path"]
        base_clip_plan = self._build_raw_clip_plan(seed_path)
        target_num_frames = int(base_clip_plan.get("target_num_frames", self.max_num_frames))

        video_views: List[torch.Tensor] = []
        hand_video_views: List[torch.Tensor] = []
        static_video_views: List[torch.Tensor] = []
        static_disparity_video_views: List[torch.Tensor] = []
        eef_depth_video_views: List[torch.Tensor] = []
        camera_poses_list: List[np.ndarray] = []
        intrinsics_list: List[np.ndarray] = []
        view_sizes: List[Tuple[int, int]] = []

        for view_idx, (view, suffix) in enumerate(zip(self._views_spec, view_suffixes)):
            target_h, target_w = view["size"]
            info = group["views"][suffix]
            main_path = info["video_path"]

            cell_plan = dict(base_clip_plan)
            cell_plan["target_height"] = int(target_h)
            cell_plan["target_width"] = int(target_w)
            _, frames, _ = self._preprocess_video_with_shared_flip(
                main_path,
                cell_plan,
                apply_horizontal_flip=False,
            )
            video_views.append(frames)

            hand_frames: Optional[torch.Tensor] = None
            if self.robot_mesh_video_subdir:
                hand_path = self._derive_condition_path(main_path, self.robot_mesh_video_subdir)
                if hand_path is not None and hand_path.is_file():
                    _, hand_frames, _ = self._preprocess_video_with_shared_flip(
                        hand_path,
                        cell_plan,
                        apply_horizontal_flip=False,
                    )
                else:
                    logger.warning(
                        "Missing hand_video for pair=%s suffix=%s: %s",
                        group["pair_key"],
                        suffix,
                        hand_path,
                    )
            if hand_frames is None:
                hand_frames = torch.full_like(
                    frames, fill_value=(255.0 / 255.0 - 0.5) / 0.5  # white pad, matches composite fill
                )
            hand_video_views.append(hand_frames)

            # Per-view static_video (DWM scene anchor). Only emitted when the
            # outer training mode actually consumes it (DWM mode sets
            # `require_static_videos=True`; I2V mode sets it to False and the
            # training step rebuilds [frame0, 0...] from the main video).
            #
            # Per-view source routing for DWM:
            #   - If static_video_subdir is set AND
            #     `<static_video_subdir>/<main_stem>.mp4` exists: load full video.
            #     For DROID this populates only the wrist view (pre-rendered
            #     PCD splat along the wrist trajectory).
            #   - Otherwise: copy_first_frame fallback (frame 0 replicated 81×).
            #     This is the right anchor for stationary ext cameras and any
            #     view without a dedicated pre-render.
            if self.require_static_videos:
                static_path = None
                if self.static_video_subdir:
                    static_path = self._derive_condition_path(main_path, self.static_video_subdir)
                if static_path is not None and static_path.is_file():
                    _, static_frames, _ = self._preprocess_video_with_shared_flip(
                        static_path,
                        cell_plan,
                        apply_horizontal_flip=False,
                    )
                else:
                    static_frames = frames[0:1].repeat(frames.shape[0], 1, 1, 1).clone()
                static_video_views.append(static_frames)

            # Disparity streams are OPTIONAL conditioning. If the subdir is not
            # configured we don't emit anything for that view (the per-view list
            # ends up shorter than `video_views`, signaling "no disparity") —
            # this keeps the model's condition_channels=0 contract intact for
            # mesh-only training. If the subdir IS configured but the file is
            # missing on disk, we black-pad so the per-clip batch shapes line up.
            if self.static_disparity_subdir:
                sd_path = self._derive_condition_path(main_path, self.static_disparity_subdir)
                if sd_path is not None and sd_path.is_file():
                    _, sd_frames, _ = self._preprocess_video_with_shared_flip(
                        sd_path,
                        cell_plan,
                        apply_horizontal_flip=False,
                    )
                else:
                    logger.warning(
                        "Missing static_disparity for pair=%s suffix=%s: %s",
                        group["pair_key"],
                        suffix,
                        sd_path,
                    )
                    sd_frames = torch.full_like(frames, fill_value=-1.0)
                static_disparity_video_views.append(sd_frames)

            if self.eef_depth_subdir:
                hd_path = self._derive_condition_path(main_path, self.eef_depth_subdir)
                if hd_path is not None and hd_path.is_file():
                    _, hd_frames, _ = self._preprocess_video_with_shared_flip(
                        hd_path,
                        cell_plan,
                        apply_horizontal_flip=False,
                    )
                else:
                    logger.warning(
                        "Missing eef_depth for pair=%s suffix=%s: %s",
                        group["pair_key"],
                        suffix,
                        hd_path,
                    )
                    hd_frames = torch.full_like(frames, fill_value=-1.0)
                eef_depth_video_views.append(hd_frames)

            view_sizes.append((int(target_h), int(target_w)))

            bundle = self._load_camera_bundle_for_video(main_path)
            use_bundle_pose = not (suffix == "wrist" and self._wrist_pose_source != "trajectory")
            if bundle is not None and suffix in bundle["view_to_idx"] and use_bundle_pose:
                poses, K, serial = self._pose_and_K_from_bundle(
                    bundle=bundle,
                    suffix=suffix,
                    target_size=(int(target_h), int(target_w)),
                    native_size_override=view["native_size"],
                    num_frames=int(frames.shape[0]),
                    video_stem=main_path.stem,
                )
            else:
                episode_id = self._episode_id_from_video_path(main_path)
                if bundle is not None and suffix in bundle["view_to_idx"]:
                    serial, K = self._serial_and_K_from_bundle(
                        bundle=bundle,
                        suffix=suffix,
                        target_size=(int(target_h), int(target_w)),
                        native_size_override=view["native_size"],
                    )
                else:
                    serial = self._lookup_serial(episode_id, suffix)
                    if serial is None:
                        raise KeyError(
                            f"Missing camera serial for episode={episode_id} suffix={suffix}"
                        )
                    K = self._lookup_intrinsics(
                        episode_id=episode_id,
                        serial=serial,
                        target_size=(int(target_h), int(target_w)),
                        native_size=view["native_size"],
                    )
                poses = self._load_view_poses(suffix, main_path, serial, int(frames.shape[0]))
            camera_poses_list.append(poses.astype(np.float32))
            intrinsics_list.append(K.astype(np.float32))

        num_frames_out = video_views[0].shape[0]
        for v_idx, v in enumerate(video_views[1:], start=1):
            if v.shape[0] != num_frames_out:
                raise RuntimeError(
                    f"Frame count mismatch view[0]={num_frames_out} vs view[{v_idx}]={v.shape[0]}"
                )

        camera_poses = torch.from_numpy(np.stack(camera_poses_list, axis=0))  # [V, F, 4, 4]
        intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0))        # [V, 3, 3]
        view_sizes_tensor = torch.tensor(view_sizes, dtype=torch.int32)          # [V, 2] (H, W)

        prompt_text = self.id_token + self._select_group_prompt(group)
        return {
            "prompt": prompt_text,
            "pair_key": str(group["pair_key"]),
            "video_views": video_views,
            "hand_video_views": hand_video_views,
            # static_video_views populated only in DWM mode
            # (`require_static_videos=True`). Wrist view: pre-rendered PCD
            # splat from `static_video_subdir`. Ext views: frame-0 replicate
            # built at dataset time so the training step gets a uniform
            # per-view list. None in I2V mode (training step rebuilds from
            # main video first frame).
            "static_video_views": static_video_views or None,
            # Disparity streams are only present when the corresponding subdir
            # is configured on this dataset instance; otherwise they're None so
            # the training step's condition_channels=0 contract stays clean.
            "static_disparity_video_views": static_disparity_video_views or None,
            "eef_depth_video_views": eef_depth_video_views or None,
            "camera_poses": camera_poses,
            "intrinsics": intrinsics,
            "view_sizes": view_sizes_tensor,
            "video_metadata": {
                "num_frames": int(num_frames_out),
                "num_views": int(len(video_views)),
                "frame_skip": int(base_clip_plan.get("frame_skip", 1)),
            },
        }


class VideoDatasetWithConditionsAndResizeAndRectangleCrop(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that supports resizing and rectangle crop.
    """
    
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode
    
    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithHumanMotions(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that also supports human_motions data.
    
    This dataset loads:
    - video_latents/*.pt: Main video latents
    - prompt_embeds/*.pt: Text prompt embeddings
    - first-frame image latent derived from video_latents (for image-to-video)
    - human_motions/*.pt: SMPL pose parameters for AdaLN conditioning
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
    ) -> None:
        # Initialize parent class with main video column
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
            use_gray_hand_videos=use_gray_hand_videos,
        )
        
        # Automatically derive human_motions paths from main video paths
        self.human_motions_paths = self._derive_human_motions_paths()
        
        # Validate that all human_motions files exist
        if not self.load_tensors:
            if any(not path.is_file() for path in self.human_motions_paths):
                missing_human_motions = [path for path in self.human_motions_paths if not path.is_file()]
                raise ValueError(
                    f"Some human_motions files are missing. First few missing files: {missing_human_motions[:5]}"
                )

    def _derive_human_motions_paths(self) -> List[Path]:
        """Derive human_motions paths from main video paths.
        
        Returns:
            List of human_motions file paths
        """
        human_motions_paths = []
        
        for video_path in self.video_paths:
            # Example: video_path = /path/to/sequences/videos/00001.mp4
            # We want: /path/to/sequences/human_motions/00001.pt
            
            # Get the parent directory (sequences)
            parent_dir = video_path.parent
            # Get the filename without extension (00001)
            filename_without_ext = video_path.stem
            # Construct the human_motions path
            human_motions_path = parent_dir.parent / "human_motions" / f"{filename_without_ext}.pt"
            human_motions_paths.append(human_motions_path)
        
        return human_motions_paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     return index

        # Get main video data from parent class
        main_data = super().__getitem__(index)
        
        # Load human_motions data
        if self.load_tensors:
            # Load preprocessed human_motions latents
            try:
                human_motions = self._load_human_motions(self.human_motions_paths[index])
                main_data["human_motions"] = human_motions
            except Exception as e:
                logger.warning(f"Failed to load human_motions for index {index}: {e}")
                main_data["human_motions"] = None
        else:
            # For raw video mode, we don't load human_motions as it's SMPL data
            main_data["human_motions"] = None

        return main_data

    def _load_human_motions(self, path: Path) -> torch.Tensor:
        """Load preprocessed human_motions data and extract body_pose."""
        if not path.exists():
            raise FileNotFoundError(f"Human motions file not found: {path}")
        
        # Load SMPL pose parameters dictionary
        human_motions_dict = torch.load(path, map_location="cpu", weights_only=True)
        
        # Extract body_pose from the dictionary
        if isinstance(human_motions_dict, dict) and "body_pose" in human_motions_dict:
            body_pose = human_motions_dict["body_pose"]
            logger.debug(f"Loaded body_pose with shape: {body_pose.shape}")
            return body_pose
        else:
            available_keys = list(human_motions_dict.keys()) if isinstance(human_motions_dict, dict) else 'Not a dict'
            raise KeyError(f"body_pose not found in human_motions data: {path}. Available keys: {available_keys}")


class VideoDatasetWithHumanMotionsAndResizing(VideoDatasetWithHumanMotions):
    """
    Extended VideoDatasetWithHumanMotions that also supports resizing.
    """
    
    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithHumanMotionsAndResizeAndRectangleCrop(VideoDatasetWithHumanMotions):
    """
    Extended VideoDatasetWithHumanMotions that supports resizing and rectangle crop.
    """
    
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []
