"""MiniMax-Remover with SAM3 text-based masking for all DexYCB cameras.

Phase 1: SAM3 (text: "person") → per-camera masks saved as .npy
Phase 2: MiniMax-Remover inpainting using SAM3 masks

Output: explore_out/minimax_sam3_inpaint_{SESSION}_{cam}.mp4

    conda activate world_model
    python data_processing/dexycb/precompute_minimax_sam3_dexycb.py \
        --subject 20200709-subject-01 --session 20200709_141754
"""
import sys, os, getpass, shutil, subprocess, tempfile, argparse
SAM3_DIR    = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "third_party/sam3")
MINIMAX_DIR = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "third_party/MiniMax-Remover")
sys.path.insert(0, MINIMAX_DIR)
sys.path.insert(0, SAM3_DIR)

import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from transformer_minimax_remover import Transformer3DModel
from pipeline_minimax_remover import Minimax_Remover_Pipeline

# ── Args ─────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description="SAM3 person-masking + MiniMax-Remover inpaint for DexYCB cameras.")
_parser.add_argument("--subject", default=os.environ.get("SUBJECT", "20200709-subject-01"))
_parser.add_argument("--session", default=os.environ.get("SESSION", "20200709_141754"))
_parser.add_argument("--cam", nargs="*", default=None,
                     help="Limit to these camera serials (default: all in session).")
_parser.add_argument("--output-root", type=Path,
                     default=Path(os.environ.get(
                         "DEXYCB_OUTPUT_ROOT",
                         Path(__file__).resolve().parent / "outputs" / "dexycb" / "inpaint")),
                     help="Local output dir for masks + inpaint mp4s (never SRC by default).")
_args = _parser.parse_args()

# ── Config ──────────────────────────────────────────────────────────────────
# DexYCB raw share (read-only input). Override with DEX_YCB_DIR.
DEX_YCB_DIR = Path(os.environ.get("DEX_YCB_DIR", Path(__file__).resolve().parents[2] / "data" / "dexycb"))
SUBJECT     = _args.subject
SESSION     = _args.session
OUT_DIR     = _args.output_root
OUT_DIR.mkdir(parents=True, exist_ok=True)
# MiniMax-Remover MODEL WEIGHTS (vae/transformer/scheduler) — large external checkpoint
# (~2.6 GB). Code lives vendored under third_party/MiniMax-Remover; weights are NOT
# vendored. MINIMAX_MODEL_ROOT must point at a dir holding vae/ transformer/ scheduler/
# WITH their diffusion_pytorch_model.safetensors. Default = models/minimax-remover under this repository.
MINIMAX_DIR_P = Path(os.environ.get(
    "MINIMAX_MODEL_ROOT",
    Path(__file__).resolve().parents[2] / "models" / "minimax-remover"))
if not (MINIMAX_DIR_P / "transformer").is_dir():
    raise FileNotFoundError(
        f"MiniMax-Remover weights not found at {MINIMAX_DIR_P}. "
        "Set MINIMAX_MODEL_ROOT to a checkout holding vae/ transformer/ scheduler/ weights.")
# SAM3 checkpoint (external weight). Override with SAM3_CKPT env.
SAM3_CKPT   = Path(os.environ.get(
    "SAM3_CKPT",
    Path(__file__).resolve().parents[2] / "models" / "sam3" / "sam3.pt"))
if not SAM3_CKPT.exists():
    raise FileNotFoundError(
        f"SAM3 checkpoint not found: {SAM3_CKPT}. Download facebook/sam3 and set SAM3_CKPT.")
SAM3_TEXT   = "person"

VIDEO_LENGTH = 81
IMG_H, IMG_W = 480, 832
DEVICE = torch.device("cuda:0")

# ── Session meta ─────────────────────────────────────────────────────────────
sess_dir = DEX_YCB_DIR / SUBJECT / SESSION
with open(sess_dir / "meta.yml") as f:
    meta = yaml.safe_load(f)
serials = meta["serials"]
if _args.cam:
    serials = [c for c in serials if c in set(_args.cam)]
    if not serials:
        raise SystemExit(f"--cam {_args.cam} not in session serials {meta['serials']}")
print(f"Session: {SESSION}   cameras: {serials}")

pending = [c for c in serials
           if not (OUT_DIR / f"minimax_sam3_inpaint_{SESSION}_{c}.mp4").exists()]
if not pending:
    print("All cameras already processed.")
    sys.exit(0)
print(f"Processing {len(pending)}/{len(serials)} cameras: {pending}")


def resize_pad_width(img, target_h, target_w, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    img_r = cv2.resize(img, (new_w, target_h), interpolation=interp)
    if new_w < target_w:
        pad = target_w - new_w
        img_r = cv2.copyMakeBorder(img_r, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_REFLECT)
    elif new_w > target_w:
        start = (new_w - target_w) // 2
        img_r = img_r[:, start:start + target_w]
    return img_r


# ── Phase 1: SAM3 masks ───────────────────────────────────────────────────────
print("\n=== Phase 1: SAM3 masking ===")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{getpass.getuser()}"
os.environ["USE_PERFLIB"] = "1"

from sam3 import build_sam3_predictor

print("Building SAM3 predictor...")
sam3_model = build_sam3_predictor(
    version="sam3",
    checkpoint_path=str(SAM3_CKPT),
    compile=False,
    async_loading_frames=False,
)
print("SAM3 loaded.")

mask_paths = {}
for cam in pending:
    mask_path = OUT_DIR / f"sam3_masks_{SESSION}_{cam}.npy"
    mask_paths[cam] = mask_path
    if mask_path.exists():
        print(f"\n── {cam} ── (masks already cached)")
        continue

    print(f"\n── {cam} ──")
    cam_dir   = sess_dir / cam
    col_files = sorted(cam_dir.glob("color_*.jpg"))
    if not col_files:
        print(f"  no color files, skip")
        continue

    frames_bgr = [cv2.imread(str(cf)) for cf in col_files]
    while len(frames_bgr) < VIDEO_LENGTH:
        frames_bgr.append(frames_bgr[-1].copy())
    frames_bgr = frames_bgr[:VIDEO_LENGTH]
    frames_r = [resize_pad_width(f, IMG_H, IMG_W) for f in frames_bgr]

    # Write frames to temp dir
    tmpdir = tempfile.mkdtemp(prefix="sam3_dex_")
    for i, bgr in enumerate(frames_r):
        cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), bgr)

    resp = sam3_model.handle_request({"type": "start_session", "resource_path": tmpdir})
    session_id = resp["session_id"]

    sam3_model.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": SAM3_TEXT,
    })

    T, H, W = VIDEO_LENGTH, IMG_H, IMG_W
    masks_arr = np.zeros((T, H, W), dtype=np.uint8)

    for response in sam3_model.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        fi = response.get("frame_index")
        if fi is None or fi >= T:
            continue
        outputs = response.get("outputs", {})
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")
        if binary_masks is None:
            continue
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.cpu().numpy()
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.cpu().numpy()
        combined = np.zeros((H, W), dtype=bool)
        for i in range(len(obj_ids)):
            m = binary_masks[i]
            if m.ndim == 3:
                m = m[0]
            combined |= m.astype(bool)
        masks_arr[fi] = combined.astype(np.uint8) * 255

    torch.cuda.synchronize()
    sam3_model.handle_request({"type": "close_session", "session_id": session_id})
    shutil.rmtree(tmpdir)

    coverage = (masks_arr > 0).mean()
    print(f"  SAM3 coverage: {coverage:.4f}")

    # Fallback: if SAM3 found nothing, use seg==255
    if coverage < 1e-4:
        print(f"  WARNING: no SAM3 detections, falling back to seg==255")
        lbl_files = sorted(cam_dir.glob("labels_*.npz"))
        for i, lf in enumerate(lbl_files[:VIDEO_LENGTH]):
            seg = np.load(str(lf))["seg"]
            m = (seg == 255).astype(np.uint8) * 255
            masks_arr[i] = resize_pad_width(m, IMG_H, IMG_W, cv2.INTER_NEAREST)

    np.save(str(mask_path), masks_arr)
    print(f"  → {mask_path.name}")

# Unload SAM3 before loading MiniMax
del sam3_model
torch.cuda.empty_cache()
print("\nSAM3 unloaded.")

# ── Phase 2: MiniMax inpainting ───────────────────────────────────────────────
print("\n=== Phase 2: MiniMax inpainting ===")
print("Loading MiniMax-Remover model...")
vae         = AutoencoderKLWan.from_pretrained(str(MINIMAX_DIR_P / "vae"),         torch_dtype=torch.float16)
transformer = Transformer3DModel.from_pretrained(str(MINIMAX_DIR_P / "transformer"), torch_dtype=torch.float16)
scheduler   = UniPCMultistepScheduler.from_pretrained(str(MINIMAX_DIR_P / "scheduler"))
pipe = Minimax_Remover_Pipeline(vae=vae, transformer=transformer, scheduler=scheduler).to(DEVICE)
print("MiniMax loaded.")

for cam in pending:
    out_path = OUT_DIR / f"minimax_sam3_inpaint_{SESSION}_{cam}.mp4"
    print(f"\n── {cam} ──")

    mask_path = mask_paths[cam]
    if not mask_path.exists():
        print(f"  no masks cached, skip")
        continue

    cam_dir   = sess_dir / cam
    col_files = sorted(cam_dir.glob("color_*.jpg"))
    if not col_files:
        print(f"  no color files, skip")
        continue

    frames_bgr = [cv2.imread(str(cf)) for cf in col_files]
    while len(frames_bgr) < VIDEO_LENGTH:
        frames_bgr.append(frames_bgr[-1].copy())
    frames_bgr = frames_bgr[:VIDEO_LENGTH]
    frames_r = [resize_pad_width(f, IMG_H, IMG_W) for f in frames_bgr]

    masks_arr = np.load(str(mask_path))   # (T, H, W) uint8

    imgs_np  = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_r], 0).astype(np.float32)
    images   = torch.from_numpy(imgs_np / 127.5 - 1.0)
    masks_np = masks_arr.astype(np.float32)[..., np.newaxis]
    masks_np[masks_np > 20] = 255
    masks_np[masks_np < 255] = 0
    masks    = torch.from_numpy(masks_np / 255.0)

    print(f"  images: {images.shape}  mask coverage: {masks.mean():.4f}")

    out = pipe(
        images=images, masks=masks,
        num_frames=VIDEO_LENGTH, height=IMG_H, width=IMG_W,
        num_inference_steps=12,
        generator=torch.Generator(device=DEVICE).manual_seed(42),
        iterations=6,
    )
    raw_frames = (np.clip(out.frames[0], 0, 1) * 255).astype(np.uint8)

    h, w = raw_frames.shape[1:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", "15",
         "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frm in raw_frames:
        proc.stdin.write(frm.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"  → {out_path.name}")

print("\nAll done.")
