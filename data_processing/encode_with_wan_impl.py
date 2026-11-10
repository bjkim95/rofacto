"""
Encode videos and prompts using WAN VAE and WAN T5 Text Encoder.

Supports:
- Dataset types: trumans (nested scene/action structure), taste_rob (hand-class/scene structure),
  droid (lab/hash structure), robocasa_gr1 (render_v2 view/task/demo structure)
- Modalities: videos, static_videos, hand_videos, static_disparity, hand_disparity,
  static_disparity_image_latents, prompts, clip_image_embeds
- Multi-GPU processing via rank/world_size sharding
- SLURM array job support
- Optional dataset-file filtering by canonical videos/*.mp4 entries
- WAN versions: 2.1 and 2.2

Output naming convention:
- WAN 2.1: video_latents_wan/, static_video_latents_wan/, etc.
- WAN 2.2: video_latents_wan_2_2/, static_video_latents_wan_2_2/, etc.
- Optional I2V image conditioning: image_latents_wan*/ stores only the
  first-frame latent; training pads the temporal tail with zeros.
- Optional Fun-InP I2V conditioning: fun_inp_i2v_latents_wan*/ stores
  VAE-encoded masked videos where only frame 0 is visible.
- Optional CLIP image conditioning: clip_image_embeds_wan*/ stores first-frame
  WAN CLIP hidden states for I2V training.
"""

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import decord
import torch
import torch.nn.functional as F
from decord import VideoReader
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from dataset_layout_utils import iter_taste_rob_sample_dirs, iter_trumans_action_dirs
except ImportError:
    from data_processing.dataset_layout_utils import iter_taste_rob_sample_dirs, iter_trumans_action_dirs


decord.bridge.set_bridge("torch")


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# WAN base-model weights (VAE + T5) — external checkpoint (~modelscope cache).
# Override with WAN_MODEL_PATH_2_1 / WAN_MODEL_PATH_2_2 env, or --model_path.
DEFAULT_MODEL_PATH_2_1 = os.environ.get(
    "WAN_MODEL_PATH_2_1",
    str(Path(__file__).resolve().parents[1] / "models" / "wan-base"))
DEFAULT_MODEL_PATH_2_2 = os.environ.get(
    "WAN_MODEL_PATH_2_2",
    str(Path(__file__).resolve().parents[1] / "models" / "wan22-base"))

DEFAULT_TEXT_ENCODER_KWARGS = {
    "vocab": 256384,
    "dim": 4096,
    "dim_attn": 4096,
    "dim_ffn": 10240,
    "num_heads": 64,
    "num_layers": 24,
    "num_buckets": 32,
    "shared_pos": False,
    "dropout": 0.0,
}

VIDEO_MODALITIES = {
    "videos",
    "static_videos",
    "hand_videos",
    "static_disparity",
    "hand_disparity",
}

CLIP_IMAGE_MODALITY = "clip_image_embeds"
STATIC_DISPARITY_IMAGE_LATENTS_MODALITY = "static_disparity_image_latents"
VAE_REQUIRED_MODALITIES = VIDEO_MODALITIES | {STATIC_DISPARITY_IMAGE_LATENTS_MODALITY}


def iter_droid_sample_dirs(data_root: Path):
    for lab_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for hash_dir in sorted(path for path in lab_dir.iterdir() if path.is_dir()):
            yield lab_dir, hash_dir


ROBOCASA_GR1_SAMPLE_MARKERS = (
    "videos_clips81",
    "videos_clips121",
    "videos_static_clips81",
    "videos_static_clips121",
    "robot_mesh_videos_nominal_clips81",
    "robot_mesh_videos_nominal_clips121",
    "robot_mesh_videos_nominal_flat_clips81",
    "robot_mesh_videos_nominal_flat_clips121",
    "robot_eef_nominal_depth_videos_clips81",
    "robot_eef_nominal_depth_videos_clips121",
    "stereo_depth_videos_clips81",
    "stereo_depth_videos_clips121",
    "videos_clips81_prompts_qwen3vl",
    "videos_clips121_prompts_qwen3vl",
)


def is_robocasa_gr1_sample_dir(path: Path) -> bool:
    return any((path / marker).is_dir() for marker in ROBOCASA_GR1_SAMPLE_MARKERS)


def iter_robocasa_gr1_sample_dirs(data_root: Path):
    """Yield RoboCasa GR1 render_v2 episode directories.

    Supported roots:
    - data/robocasa/render_v2
    - data/robocasa/render_v2/<view_root>
    - data/robocasa/render_v2/<view_root>/<task>
    - data/robocasa/render_v2/<view_root>/<task>/<demo>
    """
    if is_robocasa_gr1_sample_dir(data_root):
        yield data_root.parent.name, data_root.name, data_root
        return

    for first_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if is_robocasa_gr1_sample_dir(first_dir):
            yield data_root.name, first_dir.name, first_dir
            continue
        for second_dir in sorted(path for path in first_dir.iterdir() if path.is_dir()):
            if is_robocasa_gr1_sample_dir(second_dir):
                yield first_dir.name, second_dir.name, second_dir
                continue
            for third_dir in sorted(path for path in second_dir.iterdir() if path.is_dir()):
                if is_robocasa_gr1_sample_dir(third_dir):
                    yield f"{first_dir.name}/{second_dir.name}", third_dir.name, third_dir


def normalize_prompt_subdir(prompt_subdir: str) -> str:
    return prompt_subdir[len("prompts_") :] if prompt_subdir.startswith("prompts_") else prompt_subdir


def load_wan_vae(model_path: str, device: torch.device, dtype: torch.dtype, wan_version: str = "2.1"):
    """Load WAN VAE from pretrained path."""
    from training.wan.models import AutoencoderKLWan, AutoencoderKLWan3_8

    if wan_version == "2.2":
        vae_kwargs = {
            "vae_type": "AutoencoderKLWan3_8",
            "vae_subpath": "Wan2.2_VAE.pth",
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 16,
        }
    else:
        vae_kwargs = {
            "vae_type": "AutoencoderKLWan",
            "vae_subpath": "Wan2.1_VAE.pth",
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 8,
        }

    chosen_autoencoder = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[vae_kwargs.get("vae_type", "AutoencoderKLWan")]
    vae_path = os.path.join(model_path, vae_kwargs.get("vae_subpath", "vae"))
    vae = chosen_autoencoder.from_pretrained(
        vae_path,
        additional_kwargs=vae_kwargs,
    ).eval()
    vae = vae.to(device=device, dtype=dtype)
    return vae


def load_wan_text_encoder(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load WAN T5 Text Encoder and tokenizer from pretrained path."""
    from training.wan.models.wan_text_encoder import WanT5EncoderModel

    text_encoder_path = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(model_path, "google/umt5-xxl")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_encoder_path,
        additional_kwargs=DEFAULT_TEXT_ENCODER_KWARGS,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    text_encoder = text_encoder.to(device=device)
    text_encoder.eval()

    return tokenizer, text_encoder


def load_wan_clip_image_encoder(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    image_encoder_subpath: str = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
):
    """Load the self-impl WAN CLIP image encoder used by Wan2.1 I2V."""
    from training.wan.models import CLIPModel

    image_encoder_path = os.path.join(model_path, image_encoder_subpath)
    clip_image_encoder = CLIPModel.from_pretrained(image_encoder_path).eval()
    clip_image_encoder.requires_grad_(False)
    clip_image_encoder = clip_image_encoder.to(device=device, dtype=dtype)
    return clip_image_encoder


@torch.no_grad()
def encode_video(
    vae,
    video_path: str,
    device: torch.device,
    align_width_to_32: bool = False,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Encode a video file to latents using WAN VAE."""
    vr = VideoReader(video_path)
    video_length = len(vr)
    video = vr.get_batch(list(range(video_length)))
    video = video.float() / 255.0
    video = video * 2.0 - 1.0

    video = resize_video_if_needed(
        video,
        align_width_to_32=align_width_to_32,
        target_height=target_height,
        target_width=target_width,
    )

    video = rearrange(video, "f h w c -> c f h w").unsqueeze(0)
    video = video.to(device=device, dtype=vae.model.decoder.conv1.weight.dtype)

    latents = vae.encode(video).latent_dist.sample()
    return latents.squeeze(0).cpu()


def resize_video_if_needed(
    video: torch.Tensor,
    align_width_to_32: bool = False,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Resize decoded video frames before VAE encoding."""
    frames, height, width, channels = video.shape
    if (target_height is None) != (target_width is None):
        raise ValueError("target_height and target_width must be set together")
    if target_height is not None and target_width is not None:
        out_height = int(target_height)
        out_width = int(target_width)
    elif align_width_to_32:
        out_height = height
        aligned_width = round(width / 32) * 32
        out_width = aligned_width
    else:
        return video

    if out_height == height and out_width == width:
        return video

    video = video.permute(0, 3, 1, 2)
    video = F.interpolate(
        video,
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
    )
    return video.permute(0, 2, 3, 1)


@torch.no_grad()
def encode_first_frame_image_latent(
    vae,
    video_path: str,
    device: torch.device,
    align_width_to_32: bool = False,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Encode only frame 0 for WAN I2V image conditioning."""
    vr = VideoReader(video_path)
    if len(vr) == 0:
        raise ValueError(f"empty video: {video_path}")

    video = vr.get_batch([0]).float() / 255.0
    video = video * 2.0 - 1.0

    video = resize_video_if_needed(
        video,
        align_width_to_32=align_width_to_32,
        target_height=target_height,
        target_width=target_width,
    )

    video = rearrange(video, "f h w c -> c f h w").unsqueeze(0)
    video = video.to(device=device, dtype=vae.model.decoder.conv1.weight.dtype)

    latents = vae.encode(video).latent_dist.sample()
    return latents.squeeze(0).cpu()


@torch.no_grad()
def encode_first_frame_clip_image_embeds(
    clip_image_encoder,
    video_path: str,
    device: torch.device,
    dtype: torch.dtype,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Encode frame 0 into WAN CLIP hidden states for I2V conditioning."""
    vr = VideoReader(video_path)
    if len(vr) == 0:
        raise ValueError(f"empty video: {video_path}")

    video = vr.get_batch([0]).float() / 255.0
    video = video * 2.0 - 1.0

    video = resize_video_if_needed(
        video,
        target_height=target_height,
        target_width=target_width,
    )

    image = rearrange(video, "f h w c -> f c h w")[0]
    image = image.to(device=device, dtype=dtype)
    embeds = clip_image_encoder([image[:, None, :, :]])
    return embeds.squeeze(0).cpu()


@torch.no_grad()
def encode_fun_inp_i2v_video(
    vae,
    video_path: str,
    device: torch.device,
    align_width_to_32: bool = False,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Encode Fun-InP I2V conditioning video where only frame 0 is visible."""
    vr = VideoReader(video_path)
    video_length = len(vr)

    video = vr.get_batch(list(range(video_length))).float() / 255.0
    video = video * 2.0 - 1.0

    video = resize_video_if_needed(
        video,
        align_width_to_32=align_width_to_32,
        target_height=target_height,
        target_width=target_width,
    )

    conditioned_video = torch.zeros_like(video)
    conditioned_video[0] = video[0]
    conditioned_video = rearrange(conditioned_video, "f h w c -> c f h w").unsqueeze(0)
    conditioned_video = conditioned_video.to(device=device, dtype=vae.model.decoder.conv1.weight.dtype)

    latents = vae.encode(conditioned_video).latent_dist.sample()
    return latents.squeeze(0).cpu()


@torch.no_grad()
def encode_first_frame_repeat_video(
    vae,
    video_path: str,
    device: torch.device,
    align_width_to_32: bool = False,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Encode a synthetic static video made by repeating frame 0."""
    vr = VideoReader(video_path)
    video_length = len(vr)
    if video_length == 0:
        raise ValueError(f"empty video: {video_path}")

    video = vr.get_batch([0]).float() / 255.0
    video = video.repeat(video_length, 1, 1, 1)
    video = video * 2.0 - 1.0

    video = resize_video_if_needed(
        video,
        align_width_to_32=align_width_to_32,
        target_height=target_height,
        target_width=target_width,
    )

    video = rearrange(video, "f h w c -> c f h w").unsqueeze(0)
    video = video.to(device=device, dtype=vae.model.decoder.conv1.weight.dtype)

    latents = vae.encode(video).latent_dist.sample()
    return latents.squeeze(0).cpu()


def load_video_for_vae(
    video_path: str,
    encode_mode: str,
    align_width_to_32: bool = False,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> torch.Tensor:
    """Load and normalize a video as [C, F, H, W] on CPU for batched VAE encoding."""
    vr = VideoReader(video_path)
    video_length = len(vr)
    if video_length == 0:
        raise ValueError(f"empty video: {video_path}")

    if encode_mode == "first_frame_repeat":
        video = vr.get_batch([0]).float() / 255.0
        video = video.repeat(video_length, 1, 1, 1)
    else:
        video = vr.get_batch(list(range(video_length))).float() / 255.0
    video = video * 2.0 - 1.0

    video = resize_video_if_needed(
        video,
        align_width_to_32=align_width_to_32,
        target_height=target_height,
        target_width=target_width,
    )
    return rearrange(video, "f h w c -> c f h w").contiguous()


@torch.no_grad()
def encode_vae_tensor_batch(
    vae,
    videos: Sequence[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """Encode a same-shaped batch of [C, F, H, W] CPU tensors and return CPU latents."""
    if not videos:
        return []
    dtype = vae.model.decoder.conv1.weight.dtype
    batch = torch.stack(list(videos), dim=0).to(device=device, dtype=dtype)
    latents = vae.encode(batch).latent_dist.sample()
    return [latent.detach().cpu() for latent in latents]


@torch.no_grad()
def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    device: torch.device,
    max_length: int = 512,
    pad_to_max_length: bool = True,
) -> torch.Tensor:
    """Encode a text prompt using WAN T5 Text Encoder."""
    inputs = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

    if not pad_to_max_length:
        actual_length = attention_mask.sum().item()
        embeds = embeds[:, :actual_length, :]

    return embeds.squeeze(0).cpu()


def build_modality_mapping(
    prompt_subdir: str,
    wan_version: str,
    videos_dir_name: str = "videos",
    videos_output_dir_stem: str = "video_latents",
    image_latents_output_dir_stem: str = "image_latents",
    fun_inp_i2v_output_dir_stem: str = "fun_inp_i2v_latents",
    clip_image_embeds_output_dir_stem: str = "clip_image_embeds",
    static_videos_dir_name: str = "videos_static",
    static_videos_output_dir_stem: str = "static_video_latents",
    hand_videos_dir_name: str = "videos_hands",
    hand_videos_output_dir_stem: str = "hand_video_latents",
    static_disparity_dir_name: str = "disparity_static_sqrt",
    hand_disparity_dir_name: str = "disparity_hands_sqrt",
    static_disparity_output_dir_stem: str = "static_disparity_sqrt_latents",
    static_disparity_image_latents_output_dir_stem: str = "static_disparity_image_latents",
    hand_disparity_output_dir_stem: str = "hand_disparity_sqrt_latents",
    prompt_dir_name: Optional[str] = None,
    prompt_output_dir_stem: Optional[str] = None,
) -> Dict[str, Tuple[str, str]]:
    suffix = "_wan" if wan_version == "2.1" else "_wan_2_2"
    normalized_prompt_subdir = normalize_prompt_subdir(prompt_subdir)
    prompt_src_dir = prompt_dir_name or f"prompts_{normalized_prompt_subdir}"
    prompt_out_stem = prompt_output_dir_stem or f"prompt_embeds_{normalized_prompt_subdir}"
    return {
        "videos": (videos_dir_name, f"{videos_output_dir_stem}{suffix}"),
        "static_videos": (static_videos_dir_name, f"{static_videos_output_dir_stem}{suffix}"),
        "hand_videos": (hand_videos_dir_name, f"{hand_videos_output_dir_stem}{suffix}"),
        "static_disparity": (static_disparity_dir_name, f"{static_disparity_output_dir_stem}{suffix}"),
        STATIC_DISPARITY_IMAGE_LATENTS_MODALITY: (
            static_disparity_dir_name,
            f"{static_disparity_image_latents_output_dir_stem}{suffix}",
        ),
        "hand_disparity": (hand_disparity_dir_name, f"{hand_disparity_output_dir_stem}{suffix}"),
        "prompts": (prompt_src_dir, f"{prompt_out_stem}{suffix}"),
        "image_latents": ("", f"{image_latents_output_dir_stem}{suffix}"),
        "fun_inp_i2v_latents": ("", f"{fun_inp_i2v_output_dir_stem}{suffix}"),
        CLIP_IMAGE_MODALITY: (videos_dir_name, f"{clip_image_embeds_output_dir_stem}{suffix}"),
    }


def safe_read_first_line(txt_path: Path) -> Optional[str]:
    """Read first line of a text file safely."""
    if not txt_path.exists():
        return None
    try:
        with txt_path.open("r", encoding="utf-8") as file:
            line = file.readline()
        return line.strip()
    except Exception:
        return None


def extract_json_field(data: Any, field_path: str) -> Any:
    current = data
    for key in field_path.split("."):
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, list):
            current = current[int(key)]
        else:
            raise KeyError(field_path)
    return current


def read_prompt_file(prompt_path: Path, prompt_json_fields: Optional[Sequence[str]]) -> Optional[str]:
    if not prompt_path.exists():
        return None
    if prompt_json_fields:
        try:
            with prompt_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            parts = []
            for field in prompt_json_fields:
                value = extract_json_field(data, field)
                if value is not None:
                    parts.append(str(value).strip())
            prompt = " ".join(part for part in parts if part)
            return prompt or None
        except Exception:
            return None
    return safe_read_first_line(prompt_path)


def iter_prompt_files(src_dir: Path, prompt_json_fields: Optional[Sequence[str]]) -> Iterable[Path]:
    pattern = "*.json" if prompt_json_fields else "*.txt"
    return sorted(src_dir.glob(pattern))


def atomic_torch_save(tensor: torch.Tensor, out_path: Path) -> None:
    """Save to a temp file first so interrupted writes do not corrupt final outputs."""
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            torch.save(tensor, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
        dir_fd = os.open(str(out_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        if out_path.stat().st_size == 0:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(f"zero-byte output after save: {out_path}")
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def has_valid_output(out_path: Path) -> bool:
    """Treat zero-byte outputs as broken and remove them so they can be regenerated."""
    if not out_path.exists():
        return False
    try:
        if out_path.stat().st_size == 0:
            print(f"[WARN] removing zero-byte output before re-encode: {out_path}")
            out_path.unlink()
            return False
    except OSError as exc:
        print(f"[WARN] failed to inspect existing output {out_path}: {exc}")
        return False
    return True


def dataset_entry_to_clip_key(entry: str) -> Optional[Tuple[str, ...]]:
    parts = Path(entry).parts
    video_indices = [
        index
        for index, part in enumerate(parts)
        if part == "videos" or part.startswith("videos_")
    ]
    if not video_indices:
        return None
    video_index = video_indices[-1]
    if video_index + 1 >= len(parts):
        return None
    stem = Path(parts[video_index + 1]).stem
    return tuple(parts[:video_index] + (stem,))


def job_clip_key(sample_root: Path, stem: str, data_root: Path) -> Tuple[str, ...]:
    try:
        rel_parts = sample_root.relative_to(data_root).parts
    except ValueError:
        rel_parts = sample_root.parts
    return tuple(rel_parts + (stem,))


def load_selected_clip_keys(dataset_files: Sequence[str]) -> Dict[str, List[Tuple[str, ...]]]:
    selected: DefaultDict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for dataset_file in dataset_files:
        file_path = Path(dataset_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                clip_key = dataset_entry_to_clip_key(line)
                if clip_key is not None:
                    selected[clip_key[-1]].append(clip_key)
    return dict(selected)


def clip_key_selected(
    clip_key: Tuple[str, ...],
    selected_clip_keys: Optional[Dict[str, List[Tuple[str, ...]]]],
) -> bool:
    if not selected_clip_keys:
        return True
    for selected_key in selected_clip_keys.get(clip_key[-1], []):
        if len(selected_key) >= len(clip_key) and tuple(selected_key[-len(clip_key) :]) == clip_key:
            return True
    return False


def build_jobs(
    data_root: Path,
    dataset_type: str,
    modalities: List[str],
    prompt_subdir: str = "ego_fun_rewrite",
    wan_version: str = "2.1",
    videos_dir_name: str = "videos",
    videos_output_dir_stem: str = "video_latents",
    image_latents_output_dir_stem: str = "image_latents",
    fun_inp_i2v_output_dir_stem: str = "fun_inp_i2v_latents",
    clip_image_embeds_output_dir_stem: str = "clip_image_embeds",
    encode_i2v_image_latents: bool = False,
    encode_fun_inp_i2v_latents: bool = False,
    static_videos_dir_name: str = "videos_static",
    static_videos_output_dir_stem: str = "static_video_latents",
    static_videos_fallback_dir_name: Optional[str] = None,
    hand_videos_dir_name: str = "videos_hands",
    hand_videos_output_dir_stem: str = "hand_video_latents",
    static_disparity_dir_name: str = "disparity_static_sqrt",
    hand_disparity_dir_name: str = "disparity_hands_sqrt",
    static_disparity_output_dir_stem: str = "static_disparity_latents_sqrt",
    static_disparity_image_latents_output_dir_stem: str = "static_disparity_image_latents",
    hand_disparity_output_dir_stem: str = "hand_disparity_latents_sqrt",
    prompt_dir_name: Optional[str] = None,
    prompt_output_dir_stem: Optional[str] = None,
    prompt_json_fields: Optional[Sequence[str]] = None,
    selected_clip_keys: Optional[Dict[str, List[Tuple[str, ...]]]] = None,
) -> Dict[str, List[Dict]]:
    """Build jobs for each modality."""
    jobs = {modality: [] for modality in modalities}
    modality_mapping = build_modality_mapping(
        prompt_subdir=prompt_subdir,
        wan_version=wan_version,
        videos_dir_name=videos_dir_name,
        videos_output_dir_stem=videos_output_dir_stem,
        image_latents_output_dir_stem=image_latents_output_dir_stem,
        fun_inp_i2v_output_dir_stem=fun_inp_i2v_output_dir_stem,
        clip_image_embeds_output_dir_stem=clip_image_embeds_output_dir_stem,
        static_videos_dir_name=static_videos_dir_name,
        static_videos_output_dir_stem=static_videos_output_dir_stem,
        hand_videos_dir_name=hand_videos_dir_name,
        hand_videos_output_dir_stem=hand_videos_output_dir_stem,
        static_disparity_dir_name=static_disparity_dir_name,
        hand_disparity_dir_name=hand_disparity_dir_name,
        static_disparity_output_dir_stem=static_disparity_output_dir_stem,
        static_disparity_image_latents_output_dir_stem=static_disparity_image_latents_output_dir_stem,
        hand_disparity_output_dir_stem=hand_disparity_output_dir_stem,
        prompt_dir_name=prompt_dir_name,
        prompt_output_dir_stem=prompt_output_dir_stem,
    )

    if dataset_type == "trumans":
        iterator = ((scene_dir.name, action_dir.name, action_dir) for scene_dir, action_dir in iter_trumans_action_dirs(data_root))
    elif dataset_type == "taste_rob":
        iterator = ((hand_dir.name, sample_dir.name, sample_dir) for hand_dir, sample_dir in iter_taste_rob_sample_dirs(data_root))
    elif dataset_type == "droid":
        iterator = ((lab_dir.name, hash_dir.name, hash_dir) for lab_dir, hash_dir in iter_droid_sample_dirs(data_root))
    elif dataset_type == "robocasa_gr1":
        iterator = (
            (scene_name, episode_name, episode_dir)
            for scene_name, episode_name, episode_dir in iter_robocasa_gr1_sample_dirs(data_root)
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    for scene_name, action_name, sample_root in iterator:
        for modality in modalities:
            if modality not in modality_mapping:
                continue

            src_dir_name, out_dir_name = modality_mapping[modality]
            src_dir = sample_root / src_dir_name
            out_dir = sample_root / out_dir_name

            if modality == "static_videos" and static_videos_fallback_dir_name:
                actual_files = {path.stem: path for path in sorted(src_dir.glob("*.mp4"))} if src_dir.exists() else {}
                fallback_dir = sample_root / static_videos_fallback_dir_name
                fallback_files = {
                    path.stem: path for path in sorted(fallback_dir.glob("*.mp4"))
                } if fallback_dir.exists() else {}
                stems = sorted(set(actual_files) | set(fallback_files))
                source_files = []
                for stem in stems:
                    if stem in actual_files:
                        source_files.append((actual_files[stem], "video"))
                    else:
                        source_files.append((fallback_files[stem], "first_frame_repeat"))
            else:
                if not src_dir.exists():
                    continue
                if modality == "prompts":
                    source_files = [(path, "prompt") for path in iter_prompt_files(src_dir, prompt_json_fields)]
                else:
                    source_files = [(path, "video") for path in sorted(src_dir.glob("*.mp4"))]

            if not source_files:
                continue

            for src_file, encode_mode in source_files:
                stem = src_file.stem
                clip_key = job_clip_key(sample_root, stem, data_root)
                if not clip_key_selected(clip_key, selected_clip_keys):
                    continue

                out_file = out_dir / f"{stem}.pt"
                job = {
                    "scene": scene_name,
                    "action": action_name,
                    "src": str(src_file),
                    "out": str(out_file),
                    "sample_root": str(sample_root),
                    "stem": stem,
                    "encode_mode": encode_mode,
                }
                if modality == "videos" and (encode_i2v_image_latents or encode_fun_inp_i2v_latents):
                    image_out_dir = sample_root / modality_mapping["image_latents"][1]
                    fun_inp_out_dir = sample_root / modality_mapping["fun_inp_i2v_latents"][1]
                    if encode_i2v_image_latents:
                        job["image_out"] = str(image_out_dir / f"{stem}.pt")
                    if encode_fun_inp_i2v_latents:
                        job["fun_inp_out"] = str(fun_inp_out_dir / f"{stem}.pt")
                jobs[modality].append(job)

    return jobs


def shard_jobs(jobs: List[Dict], rank: int, world_size: int) -> List[Dict]:
    return jobs[rank::world_size]


def shard_jobs_by_key(jobs: List[Dict], rank: int, world_size: int) -> List[Dict]:
    sharded = []
    for job in jobs:
        key = job.get("out") or job.get("src") or job.get("stem", "")
        digest = hashlib.sha1(str(key).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % world_size
        if bucket == rank:
            sharded.append(job)
    return sharded


def job_outputs_complete(job: Dict) -> bool:
    output_paths = [Path(job["out"])]
    if "image_out" in job:
        output_paths.append(Path(job["image_out"]))
    if "fun_inp_out" in job:
        output_paths.append(Path(job["fun_inp_out"]))
    return all(has_valid_output(path) for path in output_paths)


def process_vae_batch_items(
    batch_items: List[Tuple[Dict, torch.Tensor]],
    vae,
    device: torch.device,
    errors: List[Dict],
) -> int:
    """Encode decoded VAE jobs, splitting recursively on batch OOM/failure."""
    if not batch_items:
        return 0

    try:
        latents = encode_vae_tensor_batch(vae, [tensor for _job, tensor in batch_items], device)
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        if len(batch_items) > 1:
            mid = len(batch_items) // 2
            return process_vae_batch_items(batch_items[:mid], vae, device, errors) + process_vae_batch_items(
                batch_items[mid:], vae, device, errors
            )
        job, _tensor = batch_items[0]
        errors.append({"src": str(job.get("src", "?")), "reason": str(exc)})
        return 0

    processed = 0
    for (job, _tensor), latent in zip(batch_items, latents):
        out_path = Path(job["out"])
        try:
            atomic_torch_save(latent, out_path)
            processed += 1
        except Exception as exc:
            errors.append({"src": str(job.get("src", "?")), "reason": str(exc)})
    return processed


def flush_vae_decode_batch(
    pending_jobs: List[Dict],
    vae,
    device: torch.device,
    args,
    errors: List[Dict],
) -> int:
    """Decode pending jobs on CPU, group by shape, then batch VAE encode."""
    if not pending_jobs:
        return 0

    align_width_to_32 = args.wan_version == "2.2"
    grouped: DefaultDict[Tuple[int, ...], List[Tuple[Dict, torch.Tensor]]] = defaultdict(list)
    for job in pending_jobs:
        try:
            tensor = load_video_for_vae(
                str(job["src"]),
                str(job.get("encode_mode", "video")),
                align_width_to_32=align_width_to_32,
                target_height=args.target_height,
                target_width=args.target_width,
            )
            grouped[tuple(tensor.shape)].append((job, tensor))
        except Exception as exc:
            errors.append({"src": str(job.get("src", "?")), "reason": str(exc)})

    processed = 0
    for items in grouped.values():
        processed += process_vae_batch_items(items, vae, device, errors)
    return processed


def process_vae_modality_batched(
    modality: str,
    jobs: List[Dict],
    vae,
    device: torch.device,
    args,
    rank: int,
    world_size: int,
):
    """Process VAE video modalities with N clips per VAE forward pass."""
    if args.skip_existing and args.shard_pending_after_skip_existing:
        total_jobs = len(jobs)
        jobs = [job for job in jobs if not job_outputs_complete(job)]
        print(
            f"[rank {rank}/{world_size}] {modality}: "
            f"pending_before_shard={len(jobs)}, complete_before_shard={total_jobs - len(jobs)}"
        )

    if args.shard_jobs_by_key:
        my_jobs = shard_jobs_by_key(jobs, rank, world_size)
    else:
        my_jobs = shard_jobs(jobs, rank, world_size)

    if len(my_jobs) == 0:
        print(f"[rank {rank}/{world_size}] No {modality} jobs assigned")
        return

    processed = 0
    skipped = 0
    errors: List[Dict] = []
    pending: List[Dict] = []
    batch_size = max(1, int(args.vae_batch_size))

    progress = tqdm(my_jobs, desc=f"[rank {rank}/{world_size}] {modality}", dynamic_ncols=True)
    for job in progress:
        out_path = Path(job["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and has_valid_output(out_path):
            skipped += 1
            continue

        pending.append(job)
        if len(pending) >= batch_size:
            processed += flush_vae_decode_batch(pending, vae, device, args, errors)
            pending.clear()
            progress.set_postfix(done=processed, skipped=skipped, errors=len(errors), batch=batch_size)

    if pending:
        processed += flush_vae_decode_batch(pending, vae, device, args, errors)
        progress.set_postfix(done=processed, skipped=skipped, errors=len(errors), batch=batch_size)

    print(f"[rank {rank}/{world_size}] {modality}: processed={processed}, skipped={skipped}, errors={len(errors)}")

    if len(errors) > 0:
        if args.log_dir:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            error_log = log_dir / f"errors_{modality}_rank{rank}.jsonl"
            with error_log.open("w", encoding="utf-8") as handle:
                for err in errors:
                    handle.write(json.dumps(err) + "\n")
            print(f"[rank {rank}] Error log: {error_log}")
        print(f"[rank {rank}] First {min(5, len(errors))} error(s):")
        for err in errors[:5]:
            print(f"  src={err.get('src', '?')} reason={err.get('reason', '?')}")


def process_modality(
    modality: str,
    jobs: List[Dict],
    vae,
    tokenizer,
    text_encoder,
    clip_image_encoder,
    device: torch.device,
    args,
    rank: int,
    world_size: int,
):
    """Process all jobs for a given modality."""
    can_batch_vae = (
        modality in VIDEO_MODALITIES
        and int(getattr(args, "vae_batch_size", 1)) > 1
        and not any("image_out" in job or "fun_inp_out" in job for job in jobs)
    )
    if can_batch_vae:
        process_vae_modality_batched(
            modality=modality,
            jobs=jobs,
            vae=vae,
            device=device,
            args=args,
            rank=rank,
            world_size=world_size,
        )
        return

    if args.skip_existing and args.shard_pending_after_skip_existing:
        total_jobs = len(jobs)
        jobs = [job for job in jobs if not job_outputs_complete(job)]
        print(
            f"[rank {rank}/{world_size}] {modality}: "
            f"pending_before_shard={len(jobs)}, complete_before_shard={total_jobs - len(jobs)}"
        )

    if args.shard_jobs_by_key:
        my_jobs = shard_jobs_by_key(jobs, rank, world_size)
    else:
        my_jobs = shard_jobs(jobs, rank, world_size)

    if len(my_jobs) == 0:
        print(f"[rank {rank}/{world_size}] No {modality} jobs assigned")
        return

    processed = 0
    skipped = 0
    errors = []

    progress = tqdm(my_jobs, desc=f"[rank {rank}/{world_size}] {modality}", dynamic_ncols=True)

    for job in progress:
        src_path = Path(job["src"])
        out_path = Path(job["out"])
        image_out_path = Path(job["image_out"]) if "image_out" in job else None
        fun_inp_out_path = Path(job["fun_inp_out"]) if "fun_inp_out" in job else None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if image_out_path is not None:
            image_out_path.parent.mkdir(parents=True, exist_ok=True)
        if fun_inp_out_path is not None:
            fun_inp_out_path.parent.mkdir(parents=True, exist_ok=True)

        out_exists = has_valid_output(out_path)
        image_exists = image_out_path is None or has_valid_output(image_out_path)
        fun_inp_exists = fun_inp_out_path is None or has_valid_output(fun_inp_out_path)

        if args.skip_existing and out_exists and image_exists and fun_inp_exists:
            skipped += 1
            continue

        try:
            if modality == "prompts":
                prompt = read_prompt_file(src_path, args.prompt_json_fields)
                if prompt is None:
                    errors.append({"src": str(src_path), "reason": "cannot_read_prompt"})
                    continue

                embeds = encode_prompt(
                    tokenizer,
                    text_encoder,
                    prompt,
                    device,
                    max_length=args.max_sequence_length,
                    pad_to_max_length=args.pad_to_max_length,
                )
                atomic_torch_save(embeds, out_path)
            elif modality == CLIP_IMAGE_MODALITY:
                if clip_image_encoder is None:
                    raise RuntimeError("clip_image_encoder is required for clip_image_embeds modality")
                clip_embeds = encode_first_frame_clip_image_embeds(
                    clip_image_encoder,
                    str(src_path),
                    device,
                    dtype=DTYPE_MAPPING[args.dtype],
                    target_height=args.target_height,
                    target_width=args.target_width,
                )
                atomic_torch_save(clip_embeds, out_path)
            elif modality == STATIC_DISPARITY_IMAGE_LATENTS_MODALITY:
                align_width_to_32 = args.wan_version == "2.2"
                image_latents = encode_first_frame_image_latent(
                    vae,
                    str(src_path),
                    device,
                    align_width_to_32=align_width_to_32,
                    target_height=args.target_height,
                    target_width=args.target_width,
                )
                atomic_torch_save(image_latents, out_path)
            else:
                align_width_to_32 = args.wan_version == "2.2"
                if not (args.skip_existing and out_exists):
                    if job.get("encode_mode") == "first_frame_repeat":
                        latents = encode_first_frame_repeat_video(
                            vae,
                            str(src_path),
                            device,
                            align_width_to_32=align_width_to_32,
                            target_height=args.target_height,
                            target_width=args.target_width,
                        )
                    else:
                        latents = encode_video(
                            vae,
                            str(src_path),
                            device,
                            align_width_to_32=align_width_to_32,
                            target_height=args.target_height,
                            target_width=args.target_width,
                        )
                    atomic_torch_save(latents, out_path)
                if modality == "videos" and image_out_path is not None and not (
                    args.skip_existing and image_exists
                ):
                    image_latents = encode_first_frame_image_latent(
                        vae,
                        str(src_path),
                        device,
                        align_width_to_32=align_width_to_32,
                        target_height=args.target_height,
                        target_width=args.target_width,
                    )
                    atomic_torch_save(image_latents, image_out_path)
                if modality == "videos" and fun_inp_out_path is not None and not (
                    args.skip_existing and fun_inp_exists
                ):
                    fun_inp_i2v_latents = encode_fun_inp_i2v_video(
                        vae,
                        str(src_path),
                        device,
                        align_width_to_32=align_width_to_32,
                        target_height=args.target_height,
                        target_width=args.target_width,
                    )
                    atomic_torch_save(fun_inp_i2v_latents, fun_inp_out_path)

            processed += 1

        except Exception as exc:
            errors.append({"src": str(src_path), "reason": str(exc)})

        if processed % 10 == 0:
            progress.set_postfix(done=processed, skipped=skipped, errors=len(errors))

    print(f"[rank {rank}/{world_size}] {modality}: processed={processed}, skipped={skipped}, errors={len(errors)}")

    if len(errors) > 0:
        if args.log_dir:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            error_log = log_dir / f"errors_{modality}_rank{rank}.jsonl"
            with error_log.open("w", encoding="utf-8") as handle:
                for err in errors:
                    handle.write(json.dumps(err) + "\n")
            print(f"[rank {rank}] Error log: {error_log}")
        print(f"[rank {rank}] First {min(5, len(errors))} error(s):")
        for err in errors[:5]:
            print(f"  src={err.get('src', '?')} reason={err.get('reason', '?')}")


def resolve_data_root(args) -> Path:
    if args.root_dir is not None:
        return Path(args.root_dir)
    if args.dataset_subpath:
        return Path(args.data_root) / args.dataset_subpath
    if args.dataset_type == "trumans":
        return Path(args.data_root) / "trumans" / "ego_render_fov90"
    if args.dataset_type == "taste_rob":
        return Path(args.data_root) / "taste_rob"
    if args.dataset_type == "droid":
        return Path(args.data_root) / "droid" / "render_v1"
    if args.dataset_type == "robocasa_gr1":
        return Path(args.data_root) / "robocasa" / "render_v2"
    return Path(args.data_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode videos/prompts with WAN VAE/T5")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Canonical dataset root (overrides --data_root/--dataset_subpath resolution)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DATA_ROOT", "data"),
        help="Base directory for datasets",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["trumans", "taste_rob", "droid", "robocasa_gr1"],
        help="Dataset type",
    )
    parser.add_argument(
        "--dataset_subpath",
        type=str,
        default=None,
        help="Subpath within data_root (e.g. trumans/ego_render_fov90)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
        choices=[
            "videos",
            "static_videos",
            "hand_videos",
            "static_disparity",
            "hand_disparity",
            STATIC_DISPARITY_IMAGE_LATENTS_MODALITY,
            "prompts",
            CLIP_IMAGE_MODALITY,
        ],
        help="Modalities to process",
    )
    parser.add_argument(
        "--prompt_subdir",
        type=str,
        default="ego_fun_rewrite",
        help="Prompt subdirectory suffix (prompts_{subdir} -> prompt_embeds_{subdir}_wan)",
    )
    parser.add_argument(
        "--prompt_dir_name",
        type=str,
        default=None,
        help="Exact prompt source subdirectory name. Overrides prompts_{prompt_subdir} when set.",
    )
    parser.add_argument(
        "--prompt_output_dir_stem",
        type=str,
        default=None,
        help="Prompt embedding output directory stem before WAN version suffix.",
    )
    parser.add_argument(
        "--prompt_json_fields",
        type=str,
        nargs="*",
        default=None,
        help="Dotted JSON fields to concatenate when prompt files are JSON, e.g. general.text.",
    )
    parser.add_argument(
        "--videos_dir_name",
        type=str,
        default="videos",
        help="Source subdirectory name for the primary video modality.",
    )
    parser.add_argument(
        "--videos_output_dir_stem",
        type=str,
        default="video_latents",
        help="Output directory stem for primary video latents before WAN version suffix.",
    )
    parser.add_argument(
        "--image_latents_output_dir_stem",
        type=str,
        default="image_latents",
        help="Output directory stem for I2V first-frame image latents before WAN version suffix.",
    )
    parser.add_argument(
        "--fun_inp_i2v_output_dir_stem",
        type=str,
        default="fun_inp_i2v_latents",
        help="Output directory stem for Fun-InP I2V latents before WAN version suffix.",
    )
    parser.add_argument(
        "--encode_i2v_image_latents",
        action="store_true",
        help="Also emit first-frame-only I2V image latents for the primary video modality.",
    )
    parser.add_argument(
        "--encode_fun_inp_i2v_latents",
        action="store_true",
        help="Also emit Fun-InP I2V latents for the primary video modality.",
    )
    parser.add_argument(
        "--clip_image_embeds_output_dir_stem",
        type=str,
        default="clip_image_embeds",
        help="Output directory stem for first-frame CLIP image embeddings before WAN version suffix.",
    )
    parser.add_argument(
        "--image_encoder_subpath",
        type=str,
        default="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="WAN CLIP image encoder checkpoint path relative to --model_path.",
    )
    parser.add_argument(
        "--static_videos_dir_name",
        type=str,
        default="videos_static",
        help="Source subdirectory name for static-video modality.",
    )
    parser.add_argument(
        "--static_videos_output_dir_stem",
        type=str,
        default="static_video_latents",
        help="Output directory stem for static-video latents before WAN version suffix.",
    )
    parser.add_argument(
        "--static_videos_fallback_dir_name",
        type=str,
        default=None,
        help=(
            "Optional fallback source subdirectory for static videos. Missing static clips are "
            "encoded by repeating frame 0 from the fallback video."
        ),
    )
    parser.add_argument(
        "--hand_videos_dir_name",
        type=str,
        default="videos_hands",
        help="Source subdirectory name for hand-video modality",
    )
    parser.add_argument(
        "--hand_videos_output_dir_stem",
        type=str,
        default="hand_video_latents",
        help="Output directory stem for hand-video latents before WAN version suffix",
    )
    parser.add_argument(
        "--static_disparity_dir_name",
        type=str,
        default="disparity_static_sqrt",
        help="Source subdirectory name for static disparity videos",
    )
    parser.add_argument(
        "--hand_disparity_dir_name",
        type=str,
        default="disparity_hands_sqrt",
        help="Source subdirectory name for hand disparity videos",
    )
    parser.add_argument(
        "--static_disparity_output_dir_stem",
        type=str,
        default="static_disparity_sqrt_latents",
        help="Output directory stem for static disparity latents before WAN version suffix",
    )
    parser.add_argument(
        "--static_disparity_image_latents_output_dir_stem",
        type=str,
        default="static_disparity_image_latents",
        help="Output directory stem for static disparity first-frame VAE latents before WAN version suffix",
    )
    parser.add_argument(
        "--hand_disparity_output_dir_stem",
        type=str,
        default="hand_disparity_sqrt_latents",
        help="Output directory stem for hand disparity latents before WAN version suffix",
    )
    parser.add_argument(
        "--dataset_files",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset split files containing canonical videos/*.mp4 paths to restrict encoding",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to WAN model directory")
    parser.add_argument("--wan_version", type=str, default="2.1", choices=["2.1", "2.2"], help="WAN version")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Model dtype")
    parser.add_argument("--target_height", type=int, default=None, help="Resize video frames to this height before VAE encoding")
    parser.add_argument("--target_width", type=int, default=None, help="Resize video frames to this width before VAE encoding")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum text length")
    parser.add_argument("--pad_to_max_length", action="store_true", help="Pad prompt embeddings to max length")
    parser.add_argument("--rank", type=int, default=None, help="Process rank")
    parser.add_argument("--world_size", type=int, default=None, help="Total number of processes")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing outputs")
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=1,
        help="Number of same-shaped clips to encode in one VAE forward pass for video modalities.",
    )
    parser.add_argument(
        "--shard_pending_after_skip_existing",
        action="store_true",
        help=(
            "When --skip_existing is set, filter complete jobs before rank sharding so reruns "
            "rebalance only missing outputs across the array."
        ),
    )
    parser.add_argument(
        "--shard_jobs_by_key",
        action="store_true",
        help=(
            "Assign jobs to ranks by a stable output-path hash instead of list index. Useful for "
            "reruns where missing outputs are concentrated in a few original index shards."
        ),
    )
    parser.add_argument("--log_dir", type=str, default="logs/encode_wan", help="Directory for log files")
    parser.add_argument("--debug", action="store_true", help="Debug mode: limit samples")
    parser.add_argument("--dry_run", action="store_true", help="Build jobs and print counts without loading models or encoding")
    parser.add_argument("--local_multi_gpu", action="store_true", help="Reserved compatibility flag")
    parser.add_argument("--debug_limit", type=int, default=10, help="Number of samples in debug mode")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.prompt_subdir = normalize_prompt_subdir(args.prompt_subdir)

    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH_2_2 if args.wan_version == "2.2" else DEFAULT_MODEL_PATH_2_1
    if not args.model_path.endswith("/"):
        args.model_path += "/"

    data_root = resolve_data_root(args)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Dataset type: {args.dataset_type}")
    print(f"[INFO] WAN version: {args.wan_version}")
    print(f"[INFO] Modalities: {args.modalities}")
    print(f"[INFO] Videos dir: {args.videos_dir_name}")
    if CLIP_IMAGE_MODALITY in args.modalities:
        print(f"[INFO] CLIP image embeds output stem: {args.clip_image_embeds_output_dir_stem}")
        print(f"[INFO] CLIP image encoder subpath: {args.image_encoder_subpath}")
    if (args.target_height is None) != (args.target_width is None):
        raise ValueError("--target_height and --target_width must be set together")
    if args.target_height is not None:
        print(f"[INFO] Target video size: {args.target_height}x{args.target_width}")
    if args.vae_batch_size > 1:
        print(f"[INFO] VAE batch size: {args.vae_batch_size}")
    if "prompts" in args.modalities:
        print(f"[INFO] Prompt dir: {args.prompt_dir_name or f'prompts_{args.prompt_subdir}'}")
    if "prompts" in args.modalities and args.prompt_json_fields:
        print(f"[INFO] Prompt JSON fields: {args.prompt_json_fields}")
    if "static_videos" in args.modalities and args.static_videos_fallback_dir_name:
        print(f"[INFO] Static-video fallback dir: {args.static_videos_fallback_dir_name}")

    if args.rank is not None and args.world_size is not None:
        rank = args.rank
        world_size = args.world_size
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        world_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank = 0
        world_size = 1

    print(f"[INFO] Rank: {rank}/{world_size}")

    if torch.cuda.is_available():
        if world_size > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"[INFO] Device: {device}")

    selected_clip_keys = None
    if args.dataset_files:
        selected_clip_keys = load_selected_clip_keys(args.dataset_files)
        print(
            f"[INFO] Dataset-file filter enabled: {len(selected_clip_keys)} clip entries from {len(args.dataset_files)} file(s)"
        )

    all_jobs = build_jobs(
        data_root=data_root,
        dataset_type=args.dataset_type,
        modalities=args.modalities,
        prompt_subdir=args.prompt_subdir,
        wan_version=args.wan_version,
        videos_dir_name=args.videos_dir_name,
        videos_output_dir_stem=args.videos_output_dir_stem,
        image_latents_output_dir_stem=args.image_latents_output_dir_stem,
        fun_inp_i2v_output_dir_stem=args.fun_inp_i2v_output_dir_stem,
        clip_image_embeds_output_dir_stem=args.clip_image_embeds_output_dir_stem,
        encode_i2v_image_latents=args.encode_i2v_image_latents,
        encode_fun_inp_i2v_latents=args.encode_fun_inp_i2v_latents,
        static_videos_dir_name=args.static_videos_dir_name,
        static_videos_output_dir_stem=args.static_videos_output_dir_stem,
        static_videos_fallback_dir_name=args.static_videos_fallback_dir_name,
        hand_videos_dir_name=args.hand_videos_dir_name,
        hand_videos_output_dir_stem=args.hand_videos_output_dir_stem,
        static_disparity_dir_name=args.static_disparity_dir_name,
        hand_disparity_dir_name=args.hand_disparity_dir_name,
        static_disparity_output_dir_stem=args.static_disparity_output_dir_stem,
        static_disparity_image_latents_output_dir_stem=args.static_disparity_image_latents_output_dir_stem,
        hand_disparity_output_dir_stem=args.hand_disparity_output_dir_stem,
        prompt_dir_name=args.prompt_dir_name,
        prompt_output_dir_stem=args.prompt_output_dir_stem,
        prompt_json_fields=args.prompt_json_fields,
        selected_clip_keys=selected_clip_keys,
    )

    for modality, jobs in all_jobs.items():
        print(f"[INFO] {modality}: {len(jobs)} total jobs")

    if args.debug:
        for modality in all_jobs:
            all_jobs[modality] = all_jobs[modality][: args.debug_limit]
        print(f"[DEBUG] Limited to {args.debug_limit} samples per modality")

    if args.dry_run:
        print("[DRY-RUN] Skipping model loading and encoding.")
        return

    dtype = DTYPE_MAPPING[args.dtype]
    vae = None
    tokenizer = None
    text_encoder = None
    clip_image_encoder = None

    if VAE_REQUIRED_MODALITIES & set(args.modalities):
        print(f"[INFO] Loading WAN {args.wan_version} VAE...")
        vae = load_wan_vae(args.model_path, device, dtype, args.wan_version)
        print(f"[INFO] WAN {args.wan_version} VAE loaded")

    if "prompts" in args.modalities:
        print("[INFO] Loading WAN T5 Text Encoder...")
        tokenizer, text_encoder = load_wan_text_encoder(args.model_path, device, dtype)
        print("[INFO] WAN T5 Text Encoder loaded")

    if CLIP_IMAGE_MODALITY in args.modalities:
        print("[INFO] Loading WAN CLIP Image Encoder...")
        clip_image_encoder = load_wan_clip_image_encoder(
            args.model_path,
            device,
            dtype,
            image_encoder_subpath=args.image_encoder_subpath,
        )
        print("[INFO] WAN CLIP Image Encoder loaded")

    for modality in args.modalities:
        jobs = all_jobs.get(modality, [])
        if len(jobs) == 0:
            print(f"[WARN] No jobs found for {modality}")
            continue

        print(f"\n[INFO] Processing {modality}...")
        process_modality(
            modality=modality,
            jobs=jobs,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            clip_image_encoder=clip_image_encoder,
            device=device,
            args=args,
            rank=rank,
            world_size=world_size,
        )

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
