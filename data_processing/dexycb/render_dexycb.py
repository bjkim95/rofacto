"""Render DexYCB session as 4 conditioning videos for all cameras (world_model env).

Pre-requisites (run in order):
  1. conda activate spider  && python data_processing/dexycb/precompute_retarget.py
  2. conda activate pyroki  && python data_processing/dexycb/precompute_ik_dexycb.py
  3. conda activate world_model && python data_processing/dexycb/render_dexycb.py

Outputs (one sub-folder per camera in explore_out/renders/):
  explore_out/renders/{SESSION}_{cam}/
      rgb.mp4                — original RGB from DexYCB
      robot_mesh.mp4         — Panda arm + Inspire hand, gray background
      robot_depth_full.mp4   — full arm+hand depth (turbo), same scale as scene
      scene_depth.mp4        — first-frame scene depth (turbo), repeated
      eef_depth.mp4          — Inspire hand depth only, same near/far scale
"""
import argparse
import sys, os, re
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import subprocess, yaml
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2
import pyrender
import yourdfpy


# ── Config (env-overridable) ───────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument(
    "--output-root",
    type=Path,
    default=Path(os.environ.get("DEXYCB_OUTPUT_ROOT", REPO_ROOT / "outputs" / "dexycb" / "render")),
)
_cli_args = _cli.parse_args()
DEX_YCB_DIR  = Path(os.environ.get("DEX_YCB_DIR", REPO_ROOT / "data" / "dexycb"))
SUBJECT      = os.environ.get("SUBJECT", "20200709-subject-01")
SESSION      = os.environ.get("SESSION", "20200709_141754")
SRC_CAM      = "840412060917"   # camera used for IK / retargeting
HAND_ONLY    = os.environ.get("HAND_ONLY", "0") == "1"  # skip arm, hand only
HAND_BLACK   = os.environ.get("HAND_BLACK", "0") == "1"  # render hand mesh in black
OUT_TAG      = os.environ.get("OUT_TAG", "")  # if set, use {OUT_TAG}_{cam} as output dir name
BG_VIDEO     = os.environ.get("BG_VIDEO", "")  # if set, composite robot onto these frames → rgb_overlay.mp4
EXPLORE_DIR  = _cli_args.output_root.expanduser()
PANDA_URDF   = Path(__file__).resolve().parent / "assets" / "panda_description" / "urdf" / "panda.urdf"
INSPIRE_URDF = (Path(__file__).resolve().parents[1]
                / "third_party/dex-urdf/robots/hands/inspire_hand/inspire_hand_right_glb.urdf")
IMG_H, IMG_W = 480, 640
FPS          = 15

_M = np.diag([1., -1., -1.])   # DexYCB cam ↔ viser world


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_extrinsics(ext_tag):
    path = DEX_YCB_DIR / "calibration" / f"extrinsics_{ext_tag}" / "extrinsics.yml"
    raw  = path.read_text()
    pat  = re.compile(r"  '(\d+)': !!python/tuple\n((?:  - [^\n]+\n)+)")
    out  = {}
    for m in pat.finditer(raw):
        vals = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', m.group(2))]
        if len(vals) == 12:
            RT = np.array(vals).reshape(3, 4)
            out[m.group(1)] = {"R": RT[:, :3], "t": RT[:, 3]}
    return out   # p_world = R @ p_cam + t


def load_intrinsics(cam):
    raw = (DEX_YCB_DIR / "calibration" / "intrinsics" / f"{cam}_640x480.yml"
           ).read_text().replace("!!python/tuple", "")
    c = yaml.safe_load(raw)["color"]
    return c["fx"], c["fy"], c["ppx"], c["ppy"]


def wrist_frame_cam(j3):
    w, m, i, p = j3[0], j3[9], j3[5], j3[17]
    if not (np.linalg.norm(m - w) > 1e-4 and np.linalg.norm(p - i) > 1e-4):
        return None
    fd  = m - w;  fd  /= np.linalg.norm(fd)
    lat = p - i;  lat -= np.dot(lat, fd) * fd;  lat /= np.linalg.norm(lat)
    pn  = np.cross(-fd, lat)
    return np.column_stack([pn, lat, fd])


def save_mp4(frames, path, fps=FPS):
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
         "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"    → {path.parent.name}/{path.name}")


def depth_to_turbo_bgr(depth_m, valid, near_m, far_m):
    clipped = np.clip(depth_m, near_m, far_m)
    denom   = max(far_m - near_m, 1e-6)
    u8      = ((clipped - near_m) / denom * 255).clip(0, 255).astype(np.uint8)
    bgr     = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


# ── Session meta ───────────────────────────────────────────────────────────────
sess_dir = DEX_YCB_DIR / SUBJECT / SESSION
with open(sess_dir / "meta.yml") as f:
    meta = yaml.safe_load(f)
serials   = meta["serials"]
_cams_env = os.environ.get("CAMS", "")
if _cams_env:
    serials = [c for c in serials if c in _cams_env.split(",")]
ext_tag   = meta["extrinsics"]
extrinsics = load_extrinsics(ext_tag)
print(f"Session: {SESSION}   cameras: {serials}")

# ── Precomputed inputs (source camera frame) ───────────────────────────────────
# Retarget: try session-specific path first, fall back to legacy
_rq_path = EXPLORE_DIR / f"{SESSION}_{SRC_CAM}_retarget_rq.npz"
if not _rq_path.exists():
    _rq_path = EXPLORE_DIR / "retarget_rq.npz"
rq_npz   = np.load(str(_rq_path), allow_pickle=True)
j3_all   = rq_npz["j3"]       # (N, 21, 3) in SRC_CAM frame
rq_all   = rq_npz["rq"]       # (N, 18)
rq_valid = rq_npz["valid"]
rq_names = list(rq_npz["joint_names"])
N        = len(j3_all)
print(f"Retarget: {_rq_path.name}  ({N} frames, {rq_valid.sum()} valid)")

# IK + panda base: optional (skipped in HAND_ONLY mode or when files missing)
_ik_path = EXPLORE_DIR / f"{SESSION}_{SRC_CAM}_panda_ik_joints.npz"
if not _ik_path.exists() and SESSION == "20200709_141754":
    _ik_path = EXPLORE_DIR / "panda_ik_joints.npz"  # legacy fallback for first session only
if not HAND_ONLY and _ik_path.exists():
    ik_npz    = np.load(str(_ik_path), allow_pickle=True)
    ik_joints = ik_npz["joints"]
    ik_valid  = ik_npz["valid"]
    ik_names  = list(ik_npz["joint_names"])
    saved      = yaml.safe_load((EXPLORE_DIR / "panda_base.yml").read_text())
    bp_vis     = np.array(saved["position"])
    bwxyz      = np.array(saved["wxyz"])
    mnt        = saved.get("mount_rot", {})
    R_base_vis = Rotation.from_quat([bwxyz[1], bwxyz[2], bwxyz[3], bwxyz[0]]).as_matrix()
    R_mount    = Rotation.from_euler(
        'xyz', [mnt.get("rx", 0), mnt.get("ry", 180), mnt.get("rz", 0)],
        degrees=True).as_matrix()
    bp_src     = _M @ bp_vis
    R_base_src = _M @ R_base_vis
else:
    ik_valid  = np.zeros(N, dtype=bool)
    ik_joints = np.zeros((N, 8), dtype=np.float32)
    ik_names  = []
    bp_src = R_base_src = None
    if HAND_ONLY:
        print("HAND_ONLY mode: arm rendering disabled")

# R_mount above is for IK EEF target (panda_hand ↔ wrist frame).
# For inspire hand MESH rendering: URDF +Z = finger forward (preserved), but lateral axis
# is flipped relative to wrist frame → Rz(180°) = diag([-1,-1,1]).
R_mount_hand = Rotation.from_euler('z', 180, degrees=True).as_matrix()

# Source camera extrinsics
ext_src  = extrinsics[SRC_CAM]
R_src, t_src = ext_src["R"], ext_src["t"]

# ── Load URDFs once ────────────────────────────────────────────────────────────
print("Loading URDFs …")
if not HAND_ONLY and bp_src is not None:
    panda_urdf = yourdfpy.URDF.load(str(PANDA_URDF))
else:
    panda_urdf = None
inspire_urdf = yourdfpy.URDF.load(str(INSPIRE_URDF), mesh_dir=str(INSPIRE_URDF.parent))


def _pr_mesh_flat(tm, color_rgba):
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color_rgba, metallicFactor=0.1,
        roughnessFactor=0.7, alphaMode="OPAQUE")
    return pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)

print("Building pyrender meshes …")
panda_pr   = ({n: pyrender.Mesh.from_trimesh(tm, smooth=True)
               for n, tm in panda_urdf.scene.geometry.items()
               if "hand" not in n and "finger" not in n}
              if panda_urdf else {})
inspire_pr = {n: (_pr_mesh_flat(tm, [0.02, 0.02, 0.02, 1.0]) if HAND_BLACK
                  else pyrender.Mesh.from_trimesh(tm, smooth=True))
              for n, tm in inspire_urdf.scene.geometry.items()}
print(f"  panda: {len(panda_pr)}   inspire: {len(inspire_pr)}")


# ── Render one camera ──────────────────────────────────────────────────────────
def render_camera(cam):
    print(f"\n── {cam} ──")
    _dir_tag = OUT_TAG if OUT_TAG else SESSION
    out_dir = EXPLORE_DIR / "renders" / f"{_dir_tag}_{cam}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intrinsics
    fx, fy, cx, cy = load_intrinsics(cam)

    # Relative extrinsic: SRC_CAM → world → this cam
    ext_tgt  = extrinsics.get(cam)
    if ext_tgt is None:
        print(f"  no extrinsic, skip")
        return
    R_tgt, t_tgt = ext_tgt["R"], ext_tgt["t"]
    R_rel = R_tgt.T @ R_src                      # rotation: src → tgt cam
    t_rel = R_tgt.T @ (t_src - t_tgt)            # translation

    # Panda base in this camera's frame (None in HAND_ONLY mode)
    if bp_src is not None:
        bp_cam     = R_rel @ bp_src + t_rel
        R_base_cam = R_rel @ R_base_src
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R_base_cam
        T_base_cam[:3, 3]  = bp_cam
    else:
        T_base_cam = None

    # Pyrender camera: DexYCB cam (+Y down, +Z fwd) → OpenGL (+Y up, -Z fwd)
    T_cam_pr  = np.diag([1., -1., -1., 1.])
    _camera   = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                           znear=0.01, zfar=5.0)
    _dirlight = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=2.5)
    renderer  = pyrender.OffscreenRenderer(IMG_W, IMG_H)

    def build_scene(gray_bg):
        bg = [0.5, 0.5, 0.5, 1.0] if gray_bg else [0., 0., 0., 1.0]
        sc = pyrender.Scene(bg_color=bg, ambient_light=[0.3, 0.3, 0.3])
        sc.add(_camera,   pose=T_cam_pr)
        sc.add(_dirlight, pose=T_cam_pr)
        return sc

    # Scene depth from first frame of THIS camera
    dep_files = sorted((sess_dir / cam).glob("aligned_depth_to_color_*.png"))
    col_files = sorted((sess_dir / cam).glob("color_*.jpg"))
    if not dep_files:
        print(f"  no depth files, skip")
        renderer.delete(); return

    dep0    = cv2.imread(str(dep_files[0]), cv2.IMREAD_ANYDEPTH)
    depth0  = dep0.astype(np.float32) / 1000.0
    valid0  = dep0 > 0
    vd      = depth0[valid0]
    near_m  = float(np.percentile(vd, 2))
    far_m   = float(np.percentile(vd, 98))
    print(f"  depth [{near_m:.3f}, {far_m:.3f}] m")
    scene_depth_bgr = depth_to_turbo_bgr(depth0, valid0, near_m, far_m)

    rgb_frames              = []
    robot_frames            = []
    robot_depth_full_frames = []
    robot_mask_frames       = []   # exact renderer binary masks (H, W) bool
    eef_frames              = []

    for fi in range(N):
        # Original RGB
        bgr = cv2.imread(str(col_files[fi])) if fi < len(col_files) else \
              np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        rgb_frames.append(bgr)

        # Transform j3 from SRC_CAM to this camera
        j3_tgt = (R_rel @ j3_all[fi].T).T + t_rel   # (21, 3)

        # Panda arm
        T_eef_cam = None
        if ik_valid[fi]:
            jdict = {n: float(ik_joints[fi][k]) for k, n in enumerate(ik_names)}
            panda_urdf.update_cfg(jdict)
            T_eef_cam = T_base_cam @ panda_urdf.scene.graph.get('panda_hand')[0]

        # Inspire hand root in this camera
        R_wrist = wrist_frame_cam(j3_tgt)
        hand_valid = R_wrist is not None
        if hand_valid:
            if T_eef_cam is not None:
                # Snap inspire root to actual panda_hand FK pose → eliminates IK residual gap
                T_hand_root = T_eef_cam.copy()
            else:
                R_hand_cam = R_wrist @ R_mount_hand
                T_hand_root = np.eye(4)
                T_hand_root[:3, :3] = R_hand_cam
                T_hand_root[:3, 3]  = j3_tgt[0]
            if rq_valid[fi]:
                rq_dict = {n: float(v) for n, v in zip(rq_names, rq_all[fi])}
                cfg_arr = np.array([rq_dict.get(n, 0.0)
                                    for n in inspire_urdf.actuated_joint_names])
                inspire_urdf.update_cfg(cfg_arr)

        # Robot mesh
        sc = build_scene(gray_bg=True)
        if ik_valid[fi]:
            for name, pm in panda_pr.items():
                sc.add(pm, pose=T_base_cam @ panda_urdf.scene.graph.get(name)[0])
        if hand_valid:
            for name, pm in inspire_pr.items():
                sc.add(pm, pose=T_hand_root @ inspire_urdf.scene.graph.get(name)[0])
        color, depth_robot = renderer.render(sc)
        robot_frames.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        robot_mask_frames.append(depth_robot > 0)
        robot_depth_full_frames.append(
            depth_to_turbo_bgr(depth_robot, depth_robot > 0, near_m, far_m))

        # EEF depth
        sc_eef = build_scene(gray_bg=False)
        if hand_valid:
            for name, pm in inspire_pr.items():
                sc_eef.add(pm, pose=T_hand_root @ inspire_urdf.scene.graph.get(name)[0])
        _, depth_eef = renderer.render(sc_eef)
        valid_eef = depth_eef > 0
        eef_frames.append(depth_to_turbo_bgr(depth_eef, valid_eef, near_m, far_m))

        if (fi + 1) % 20 == 0 or fi == N - 1:
            print(f"  {fi+1}/{N}")

    renderer.delete()

    # ── Composite: robot mask onto BG_VIDEO frames ──────────────────────────
    if BG_VIDEO:
        cap = cv2.VideoCapture(BG_VIDEO)
        bg_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bg_frames.append(frame)
        cap.release()
        composite_frames = []
        for fi in range(N):
            bg = bg_frames[fi] if fi < len(bg_frames) else np.zeros((IMG_H, IMG_W, 3), np.uint8)
            if bg.shape[:2] != (IMG_H, IMG_W):
                bg = cv2.resize(bg, (IMG_W, IMG_H))
            mask = robot_mask_frames[fi][:, :, np.newaxis]
            comp = np.where(mask, robot_frames[fi], bg).astype(np.uint8)
            composite_frames.append(comp)
        save_mp4(composite_frames, out_dir / "rgb_overlay.mp4")

    save_mp4(rgb_frames,                    out_dir / "rgb.mp4")
    save_mp4(robot_frames,                  out_dir / "robot_mesh.mp4")
    save_mp4(robot_depth_full_frames,       out_dir / "robot_depth_full.mp4")
    save_mp4([scene_depth_bgr] * N,         out_dir / "scene_depth.mp4")
    save_mp4(eef_frames,                    out_dir / "eef_depth.mp4")
    np.save(str(out_dir / "robot_mask.npy"),
            np.stack(robot_mask_frames).astype(np.uint8))   # (N, H, W) 0/1
    print(f"    → {out_dir.name}/robot_mask.npy")


# ── Run all cameras ────────────────────────────────────────────────────────────
for cam in serials:
    render_camera(cam)

print("\nAll done.")
