"""Assemble final DexYCB conditioning videos (5-channel, starting from FIRST_FRAME).

Pre-requisites:
  1. render_dexycb.py  → explore_out/renders/{SESSION}_{cam}/{robot_mesh,eef_depth,scene_depth,robot_mask.npy}
  2. precompute_minimax_sam3_dexycb.py → explore_out/minimax_sam3_inpaint_{SESSION}_{cam}.mp4

Usage (world_model env):
    python data_processing/dexycb/assemble_dexycb_conditioning.py

Env overrides:
    FIRST_FRAME=8         first frame to use (default 10)
    DEPTH_MODE=moge       depth source: "realdep" (default) or "moge"
    CAMS=840412060917     comma-separated subset of cameras (default: all)
    OUT_SUBDIR=custom     override output subdirectory name
    RENDER_TAG=custom     override render dir prefix (default: SESSION)
                          e.g. RENDER_TAG=20200709_141754_arm → renders/{RENDER_TAG}_{cam}/

For each camera, writes to explore_out/renders/{SESSION}_{cam}/{OUT_SUBDIR}/:
    rgb.mp4                 – inpainted RGB, FIRST_FRAME–71
    scene_static.mp4        – inpainted FIRST_FRAME repeated
    scene_depth.mp4         – depth of FIRST_FRAME repeated (real or MoGe)
    robot_mesh.mp4          – robot mesh, FIRST_FRAME–71
    eef_depth.mp4           – EEF depth, FIRST_FRAME–71
    scene_static_robot0.mp4 – scene_static + robot frame-FIRST_FRAME (exact mask)
    scene_depth_robot0.mp4  – scene_depth + robot depth (exact mask)
"""
import argparse
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import subprocess
import yaml
import numpy as np
import cv2
from pathlib import Path

# ── Config (env-overridable) ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument(
    "--output-root",
    type=Path,
    default=Path(os.environ.get("DEXYCB_OUTPUT_ROOT", REPO_ROOT / "outputs" / "dexycb" / "conditioning")),
)
_cli_args = _cli.parse_args()
DEX_YCB_DIR = Path(os.environ.get("DEX_YCB_DIR", REPO_ROOT / "data" / "dexycb"))
SUBJECT     = "20200709-subject-01"
SESSION     = "20200709_141754"
EXPLORE_DIR = _cli_args.output_root.expanduser()
RENDERS_DIR = EXPLORE_DIR / "renders"

FIRST_FRAME    = int(os.environ.get("FIRST_FRAME", "10"))
RENDER_TAG     = os.environ.get("RENDER_TAG", SESSION)  # prefix for renders/{RENDER_TAG}_{cam}/
N_TOTAL        = 72
N_OUT          = N_TOTAL - FIRST_FRAME
FPS            = 15
IMG_H, IMG_W   = 480, 640
INPAINT_W      = 832
CROP_START     = (INPAINT_W - IMG_W) // 2
INPAINT_PREFIX = "minimax_sam3_inpaint"
DEPTH_MODE     = os.environ.get("DEPTH_MODE", "realdep")   # "realdep" | "moge"
_cams_env      = os.environ.get("CAMS", "")
CAMS_FILTER    = [c.strip() for c in _cams_env.split(",") if c.strip()]

_frame_tag  = f"_f{FIRST_FRAME}" if FIRST_FRAME != 10 else ""
_depth_tag  = "moge" if DEPTH_MODE == "moge" else "realdep"
OUT_SUBDIR  = os.environ.get(
    "OUT_SUBDIR",
    f"5ch_{INPAINT_PREFIX.replace('minimax_', '')}_{_depth_tag}{_frame_tag}"
)

# ── MoGe (loaded only when needed) ─────────────────────────────────────────
if DEPTH_MODE == "moge":
    # Optional alternative depth mode. MoGe lives in an external checkout; the
    # default DexYCB e2e path uses real sensor depth and never enters this branch.
    _MOGE = Path(os.environ.get(
        "MOGE_ROOT",
        REPO_ROOT / "data_processing" / "third_party" / "MoGe"))
    if not _MOGE.is_dir():
        raise FileNotFoundError(
            f"DEPTH_MODE=moge requires a MoGe checkout; set MOGE_ROOT (got {_MOGE})")
    sys.path.insert(0, str(_MOGE))
    import torch
    DEVICE = torch.device("cuda:0")
    print("Loading MoGe model...")
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version('v1')
    moge = MoGeModel.from_pretrained('Ruicheng/moge-vitl').to(DEVICE).eval()
    print("MoGe loaded.")

# ── Load session meta ───────────────────────────────────────────────────────
sess_dir = DEX_YCB_DIR / SUBJECT / SESSION
with open(sess_dir / "meta.yml") as f:
    meta = yaml.safe_load(f)
serials = meta["serials"]
if CAMS_FILTER:
    serials = [c for c in serials if c in CAMS_FILTER]
print(f"Session: {SESSION}   FIRST_FRAME={FIRST_FRAME}   DEPTH_MODE={DEPTH_MODE}   OUT_SUBDIR={OUT_SUBDIR}")
print(f"cameras: {serials}")


# ── Helpers ─────────────────────────────────────────────────────────────────
def save_mp4(frames, path):
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(FPS),
         "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"    → {path.parent.name}/{path.name}")


def read_video_frames(path, start, count):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def load_intrinsics(cam):
    raw = (DEX_YCB_DIR / "calibration" / "intrinsics" / f"{cam}_640x480.yml"
           ).read_text().replace("!!python/tuple", "")
    c = yaml.safe_load(raw)["color"]
    return c["fx"], c["fy"], c["ppx"], c["ppy"]


def depth_to_turbo_bgr(depth_m, mask, near_m, far_m):
    clipped = np.clip(depth_m, near_m, far_m)
    u8  = ((clipped - near_m) / max(far_m - near_m, 1e-6) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~mask] = 0
    return bgr


# ── Process each camera ─────────────────────────────────────────────────────
for cam in serials:
    print(f"\n── {cam} ──")
    render_dir   = RENDERS_DIR / f"{RENDER_TAG}_{cam}"
    out_dir      = render_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    inpaint_path = EXPLORE_DIR / f"{INPAINT_PREFIX}_{SESSION}_{cam}.mp4"

    if not inpaint_path.exists():
        print(f"  no inpainted video, skip")
        continue
    if not (render_dir / "robot_mesh.mp4").exists():
        print(f"  no robot_mesh.mp4, skip")
        continue

    # 1. Inpainted frames FIRST_FRAME–71, crop 832→640
    raw_inp = read_video_frames(inpaint_path, FIRST_FRAME, N_OUT)
    if len(raw_inp) < N_OUT:
        print(f"  only {len(raw_inp)} inpainted frames (need {N_OUT}), skip")
        continue
    inpaint_frames = [f[:, CROP_START:CROP_START + IMG_W] for f in raw_inp]

    # 2. Static scene = inpainted FIRST_FRAME
    scene_bgr = inpaint_frames[0].copy()

    # 3. Scene depth
    if DEPTH_MODE == "moge":
        fx, fy, cx, cy = load_intrinsics(cam)
        fov_x_deg = float(2 * np.degrees(np.arctan(IMG_W / (2 * fx))))
        rgb_np = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t  = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            moge_out = moge.infer(img_t, fov_x=fov_x_deg)
        depth_m = moge_out['depth'].squeeze().cpu().numpy()
        dmask   = moge_out['mask'].squeeze().cpu().numpy() > 0.5
        vd      = depth_m[dmask]
        near_m  = float(np.percentile(vd, 2))  if vd.size else 0.1
        far_m   = float(np.percentile(vd, 98)) if vd.size else 5.0
        print(f"  MoGe depth [{near_m:.3f}, {far_m:.3f}] m")
        depth_bgr = depth_to_turbo_bgr(depth_m, dmask, near_m, far_m)
    else:
        dep_frames = read_video_frames(render_dir / "scene_depth.mp4", 0, 1)
        if not dep_frames:
            print(f"  no scene_depth.mp4 in render_dir, skip")
            continue
        depth_bgr = dep_frames[0]
        print(f"  real sensor depth")

    # 4. Robot mesh + EEF depth trimmed to FIRST_FRAME–71
    mesh_frames = read_video_frames(render_dir / "robot_mesh.mp4", FIRST_FRAME, N_OUT)
    eef_frames  = read_video_frames(render_dir / "eef_depth.mp4",  FIRST_FRAME, N_OUT)
    blank = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    while len(mesh_frames) < N_OUT: mesh_frames.append(blank)
    while len(eef_frames)  < N_OUT: eef_frames.append(blank)

    # 5. Write 5-channel conditioning set
    save_mp4(inpaint_frames,        out_dir / "rgb.mp4")
    save_mp4([scene_bgr]  * N_OUT,  out_dir / "scene_static.mp4")
    save_mp4([depth_bgr]  * N_OUT,  out_dir / "scene_depth.mp4")
    save_mp4(mesh_frames,           out_dir / "robot_mesh.mp4")
    save_mp4(eef_frames,            out_dir / "eef_depth.mp4")

    # 6. Robot-FIRST_FRAME composited variants (exact renderer depth-buffer mask)
    mask_npy_path = render_dir / "robot_mask.npy"
    rdf_path      = render_dir / "robot_depth_full.mp4"
    if mask_npy_path.exists() and rdf_path.exists():
        robot_mask = np.load(str(mask_npy_path))[FIRST_FRAME].astype(bool)
        rdf0_list  = read_video_frames(rdf_path, FIRST_FRAME, 1)
        rdf0       = rdf0_list[0] if rdf0_list else np.zeros((IMG_H, IMG_W, 3), np.uint8)

        static_robot0 = scene_bgr.copy()
        static_robot0[robot_mask] = mesh_frames[0][robot_mask]

        depth_robot0 = depth_bgr.copy()
        depth_robot0[robot_mask] = rdf0[robot_mask]

        save_mp4([static_robot0] * N_OUT, out_dir / "scene_static_robot0.mp4")
        save_mp4([depth_robot0]  * N_OUT, out_dir / "scene_depth_robot0.mp4")
    else:
        print(f"  no robot_mask.npy or robot_depth_full.mp4, skip robot0 composites")

print("\nAll done.")
