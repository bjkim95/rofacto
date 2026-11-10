"""Interactive Panda+Robotiq base placement using viser + pyroki IK.

Shows DexYCB scene as a point cloud; drag the Panda base handle to position
the arm. IK runs in real-time to reach the hand wrist target (panda_link8).
All 8 DexYCB cameras are shown as frustums with live RGB textures.
Inspire hand is shown directly at the human wrist pose for alignment tuning.

Usage (pyroki env):
    python data_processing/dexycb/interactive_panda_base.py
    Then open http://localhost:7979

Press "Save base position" to write to explore_out/panda_base.yml.
NOTE: panda_base.yml must be authored against the Robotiq URDF / panda_link8 EEF.
      The old Inspire/panda_hand base is invalid for this setup.
"""
import argparse
from pathlib import Path
import sys, os
_REPO_ROOT = Path(__file__).resolve().parents[2]
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument(
    "--output-root",
    type=Path,
    default=Path(os.environ.get("DEXYCB_OUTPUT_ROOT", _REPO_ROOT / "outputs" / "dexycb" / "panda_base")),
)
_cli_args = _cli.parse_args()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/pyroki/examples"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/dex-ycb-toolkit/manopth"))

import time, yaml, tempfile
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2, trimesh
import viser
import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import jax.numpy as jnp
import yourdfpy

from dexycb_layout import load_extrinsics, extrinsic_to_pyrender_pose  # noqa: F401

_ROBOTIQ_URDF = str(_REPO_ROOT / "data_processing" / "droid_urdf_overlay" / "assets" / "panda_arm_robotiq_2f85_fts300.urdf")
_PANDA_VIS_URDF = str(_REPO_ROOT / "data_processing" / "dexycb" / "assets" / "panda_description" / "urdf" / "panda.urdf")
_MANO_ROOT = os.environ.get("MANO_ROOT", str(_REPO_ROOT / "data" / "dexycb" / "calibration" / "mano"))
_MANO_BETAS_YML = os.environ.get("MANO_BETAS_YML", str(_REPO_ROOT / "data" / "dexycb" / "calibration" / "mano_20200709_140042_subject-01_right" / "mano.yml"))

DEX_YCB_DIR = Path(os.environ.get("DEX_YCB_DIR", Path(__file__).resolve().parents[2] / "data" / "dexycb"))
SUBJECT     = "20200709-subject-01"
SESSION     = "20200709_141754"
CAM         = "840412060917"
OUT_DIR     = _cli_args.output_root.expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

_YCB_NAMES = [
    "002_master_chef_can","003_cracker_box","004_sugar_box","005_tomato_soup_can",
    "006_mustard_bottle","007_tuna_fish_can","008_pudding_box","009_gelatin_box",
    "010_potted_meat_can","011_banana","019_pitcher_base","021_bleach_cleanser",
    "024_bowl","025_mug","035_power_drill","036_wood_block","037_scissors",
    "040_large_marker","051_large_clamp","052_extra_large_clamp",
]

import torch
# chumpy 0.70 expects numpy<2 aliases removed in numpy 2.x; restore them before import.
for _np_alias, _builtin in [("bool", bool), ("int", int), ("float", float),
                            ("complex", complex), ("object", object),
                            ("str", str), ("unicode", str)]:
    if not hasattr(np, _np_alias):
        setattr(np, _np_alias, _builtin)
from manopth.manolayer import ManoLayer as ManopthLayer

def _build_mano():
    mano = ManopthLayer(flat_hand_mean=False, ncomps=45, side='right',
                        mano_root=_MANO_ROOT, use_pca=True)
    with open(_MANO_BETAS_YML) as f:
        betas_np = np.array(yaml.safe_load(f)['betas'], dtype=np.float32)
    betas = torch.from_numpy(betas_np).unsqueeze(0)
    faces = mano.th_faces.numpy()          # (1538, 3)
    return mano, betas, faces

_mano_layer, _mano_betas, _mano_faces = _build_mano()

HAND_CONNECTIONS = np.array([
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
], dtype=int)

# ── Session meta ──────────────────────────────────────────────────────────────
sess_dir = DEX_YCB_DIR / SUBJECT / SESSION
with open(sess_dir / "meta.yml") as f:
    meta = yaml.safe_load(f)
grasp_idx = meta["ycb_grasp_ind"]
grasp_obj = _YCB_NAMES[meta["ycb_ids"][grasp_idx] - 1]
print(f"Grasp object: {grasp_obj}")

# ── Extrinsics for all cameras (cam-to-world) ─────────────────────────────────
_ext_tag  = str(meta["extrinsics"])
_extrinsics = load_extrinsics(_ext_tag)  # {serial: {"R": (3,3), "t": (3,)}}
_serials  = [str(s) for s in meta["serials"]]  # all 8 camera serials

# ── Intrinsics ────────────────────────────────────────────────────────────────
intr_path = DEX_YCB_DIR / "calibration" / "intrinsics" / f"{CAM}_640x480.yml"
raw = intr_path.read_text().replace("!!python/tuple", "")
_c  = yaml.safe_load(raw)["color"]
FX, FY, CX, CY = _c["fx"], _c["fy"], _c["ppx"], _c["ppy"]

cam_dir   = sess_dir / CAM
col_files = sorted(cam_dir.glob("color_*.jpg"))
dep_files = sorted(cam_dir.glob("aligned_depth_to_color_*.png"))
lbl_files = sorted(cam_dir.glob("labels_*.npz"))
N = len(col_files)

# ── Precomputed MANO pose_m ───────────────────────────────────────────────────
_all_pose_m = np.stack([np.load(str(lf))["pose_m"][0] for lf in lbl_files])  # (N,51)

def _mano_verts_vis(fi: int) -> np.ndarray:
    """Return MANO hand vertices in viser world frame for frame fi."""
    pm = _all_pose_m[fi]
    p  = torch.from_numpy(pm[:48]).float().unsqueeze(0)
    t  = torch.from_numpy(pm[48:]).float().unsqueeze(0)
    with torch.no_grad():
        v, _ = _mano_layer(p, _mano_betas, t)
    verts = v[0].numpy() / 1000.0   # mm → m, camera frame
    return cam2vis(verts)

# ── Precomputed retargeting ───────────────────────────────────────────────────
_rq_npz        = np.load(str(OUT_DIR / "retarget_rq.npz"), allow_pickle=True)
frame_rq_all   = _rq_npz["rq"]
frame_rq_valid = _rq_npz["valid"]
frame_j3_all   = _rq_npz["j3"]
rq_joint_names = list(_rq_npz["joint_names"])
print(f"Loaded retargeting: {frame_rq_valid.sum()}/{N} valid frames")

# ── Object mesh ───────────────────────────────────────────────────────────────
obj_mesh_path = DEX_YCB_DIR / "models" / grasp_obj / "textured.obj"
print(f"Loading object mesh: {obj_mesh_path.name} …")
obj_mesh = trimesh.load(str(obj_mesh_path), process=False, force='mesh')

# ── Helpers ───────────────────────────────────────────────────────────────────
_M = np.diag([1., -1., -1.])  # camera frame → viser world frame

def load_frame(fi):
    col = cv2.imread(str(col_files[fi]))
    dep = cv2.imread(str(dep_files[fi]), cv2.IMREAD_ANYDEPTH)
    return col, dep, frame_j3_all[fi]

def load_obj_pose_vis(fi):
    """Object [R|t] from labels_*.npz → viser world (pos, wxyz)."""
    d  = np.load(str(lbl_files[fi]))
    RT = d["pose_y"][grasp_idx].astype(float)
    t_vis = _M @ RT[:, 3]
    R_vis = _M @ RT[:, :3] @ _M.T
    q = Rotation.from_matrix(R_vis).as_quat()
    return t_vis, np.array([q[3], q[0], q[1], q[2]])

def depth_to_pc(dep, col, stride=4, max_z=3.0):
    h, w = dep.shape
    u = np.arange(0, w, stride); v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)
    z  = dep[::stride, ::stride].astype(np.float32) / 1000.0
    ok = (z > 0.1) & (z < max_z)
    X  = ((uu - CX) / FX * z)[ok]
    Y  = ((vv - CY) / FY * z)[ok]
    Z  = z[ok]
    pts    = np.stack([X, Y, Z], -1)
    col_rgb = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)[::stride, ::stride]
    colors  = col_rgb.reshape(-1, 3)[ok.ravel()] / 255.0
    return pts, colors

def cam2vis(pts):
    return pts * np.array([1., -1., -1.])

def wrist_wxyz_vis(j3):
    """Wrist frame in viser: X=palm_normal, Y=lateral, Z=finger_direction."""
    w, m, i, p = j3[0], j3[9], j3[5], j3[17]
    if not (np.linalg.norm(m-w) > 1e-4 and np.linalg.norm(p-i) > 1e-4):
        return np.array([1., 0., 0., 0.])
    fd  = m - w;   fd  /= np.linalg.norm(fd)
    lat = p - i;   lat -= np.dot(lat, fd)*fd;  lat /= np.linalg.norm(lat)
    pn  = np.cross(-fd, lat)
    R_cam = np.column_stack([pn, lat, fd])
    R_vis = _M @ R_cam @ _M.T
    q = Rotation.from_matrix(R_vis).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def _load_cam_img(serial: str, fi: int):
    """Load color frame for one camera serial as RGB ndarray, or None."""
    p = DEX_YCB_DIR / SUBJECT / SESSION / serial / f"color_{fi:06d}.jpg"
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading panda+Robotiq URDF (IK) …")
    urdf      = yourdfpy.URDF.load(_ROBOTIQ_URDF)
    robot     = pk.Robot.from_urdf(urdf)
    target_link = "panda_link8"  # Robotiq URDF has no panda_hand
    ik_joint_names = list(urdf.actuated_joint_names)

    print("Loading panda URDF (visualization) …")
    panda_vis_urdf = yourdfpy.URDF.load(_PANDA_VIS_URDF)

    saved_path = OUT_DIR / "panda_base.yml"
    if saved_path.exists():
        saved = yaml.safe_load(saved_path.read_text())
        init_base_pos  = tuple(saved["position"])
        init_base_wxyz = tuple(saved["wxyz"])
        init_rx = float(saved.get("mount_rot", {}).get("rx", 0))
        init_ry = float(saved.get("mount_rot", {}).get("ry", 180))
        init_rz = float(saved.get("mount_rot", {}).get("rz", 0))
        print(f"Loaded {saved_path}")
    else:
        _, _, j3_init  = load_frame(35)
        wv0            = cam2vis(j3_init[0:1])[0]
        init_base_pos  = tuple(wv0 + np.array([0.0, -0.5, 0.6]))
        init_base_wxyz = (1, 0, 0, 0)
        init_rx, init_ry, init_rz = 0, 180, 0

    fi = 35
    col, dep, j3 = load_frame(fi)
    pts, colors  = depth_to_pc(dep, col)
    col0, dep0, _ = load_frame(0)
    pts0, colors0  = depth_to_pc(dep0, col0)
    wrist_vis    = cam2vis(j3[0:1])[0]
    wrist_q      = wrist_wxyz_vis(j3)
    obj_pos, obj_wxyz = load_obj_pose_vis(fi)

    print("Starting viser server at http://localhost:7979 …")
    server = viser.ViserServer(port=7979)
    server.scene.set_up_direction("+y")

    # Scene elements
    pc_handle = server.scene.add_point_cloud("/scene", cam2vis(pts0),
                                              colors=colors0, point_size=0.003)

    def _set_scene_pc(pts_cam, cols01):
        """Re-add /scene point cloud (handles varying point counts cleanly).
        Also hides points below a height threshold along the robot base +Z axis."""
        nonlocal pc_handle
        pw   = cam2vis(pts_cam).astype(np.float32)
        cols = cols01.astype(np.float32)
        cut  = float(base_cut_sl.value)
        if cut > -0.999:  # -1.0 means "keep all"
            bwxyz = np.array(base_handle.wxyz)
            R_base = Rotation.from_quat([bwxyz[1], bwxyz[2], bwxyz[3], bwxyz[0]]).as_matrix()
            z_base = (pw - np.array(base_handle.position)) @ R_base[:, 2]
            keep = z_base >= cut
            pw, cols = pw[keep], cols[keep]
        pc_handle = server.scene.add_point_cloud(
            "/scene", pw, colors=cols, point_size=0.003, visible=scene_cb.value)
    wrist_fh = server.scene.add_frame("/wrist_target",
        position=tuple(wrist_vis), wxyz=tuple(wrist_q),
        axes_length=0.06, axes_radius=0.004)
    obj_fh = server.scene.add_frame("/obj_frame",
        position=tuple(obj_pos), wxyz=tuple(obj_wxyz), show_axes=False)
    server.scene.add_mesh_trimesh("/obj_frame/mesh", obj_mesh)

    # Human hand skeleton
    _dummy_j3 = cam2vis(frame_j3_all[fi])
    _skel_pts = np.stack([_dummy_j3[HAND_CONNECTIONS[:, 0]],
                          _dummy_j3[HAND_CONNECTIONS[:, 1]]], axis=1).astype(np.float32)
    _init_mano_verts = _mano_verts_vis(fi)
    _MANO_COLOR_DEFAULT = (235, 160, 120)
    _mano_mesh_fh = server.scene.add_mesh_simple(
        "/human_hand/mesh", _init_mano_verts.astype(np.float32), _mano_faces,
        color=_MANO_COLOR_DEFAULT, side="double", visible=False)

    human_pts_h = server.scene.add_point_cloud(
        "/human_hand/pts", _dummy_j3.astype(np.float32),
        colors=np.array([[1.0, 0.7, 0.3]] * 21, dtype=np.float32), point_size=0.008)
    human_skel_h = server.scene.add_line_segments(
        "/human_hand/skel", _skel_pts,
        colors=np.array([1.0, 0.7, 0.3]), line_width=2.5, visible=True)

    # Panda arm
    base_handle = server.scene.add_transform_controls(
        "/panda_base_ctrl", scale=0.15,
        position=init_base_pos, wxyz=init_base_wxyz)
    panda_fh = server.scene.add_frame("/panda_base",
        position=init_base_pos, wxyz=init_base_wxyz, show_axes=False)
    urdf_vis = ViserUrdf(server, panda_vis_urdf, root_node_name="/panda_base")

    # Inspire hand
    hand_urdf_path = Path("data_processing/third_party/dex-urdf/robots/hands/inspire_hand/inspire_hand_right.urdf")
    hand_urdf = yourdfpy.URDF.load(str(hand_urdf_path), mesh_dir=str(hand_urdf_path.parent))
    hand_fh = server.scene.add_frame("/hand_ee",
        position=init_base_pos, wxyz=init_base_wxyz, show_axes=False)
    _ROBOT_HAND_RGB = (0.82, 0.82, 0.86)   # light metallic gray, matches panda arm
    hand_vis       = ViserUrdf(server, hand_urdf, root_node_name="/hand_ee",
                               mesh_color_override=(*_ROBOT_HAND_RGB, 1.0))
    hand_vis_black = ViserUrdf(server, hand_urdf, root_node_name="/hand_ee_black",
                               mesh_color_override=(0.02, 0.02, 0.02, 1.0))
    hand_fh_black  = server.scene.add_frame("/hand_ee_black",
        position=init_base_pos, wxyz=init_base_wxyz, show_axes=False)
    hand_vis_black.show_visual = False  # initially hidden

    # GUI
    frame_slider = server.gui.add_slider("Frame", 0, N-1, 1, fi)
    save_btn     = server.gui.add_button("Save base position")
    ik_err_num   = server.gui.add_number("IK pos err (mm)", 0.0, disabled=True)
    scene_cb     = server.gui.add_checkbox("Show scene",  True)
    arm_cb       = server.gui.add_checkbox("Show arm",    True)
    hand_cb      = server.gui.add_checkbox("Show hand",   True)
    obj_cb       = server.gui.add_checkbox("Show object", True)
    axes_cb      = server.gui.add_checkbox("Show wrist axes", True)
    lock_depth_cb    = server.gui.add_checkbox("Lock depth to frame 0", True)
    max_depth_sl     = server.gui.add_slider("Max depth (m)", 0.1, 5.0, 0.05, 2.5)
    base_cut_sl      = server.gui.add_slider("Base height cut (m)", -1.0, 1.0, 0.01, -1.0)
    hand_black_cb    = server.gui.add_checkbox("Black hand", False)
    mano_color_rgb   = server.gui.add_rgb("MANO color", _MANO_COLOR_DEFAULT)
    frustum_scale_sl = server.gui.add_slider("Frustum scale", 0.01, 1.0, 0.01, 0.15)
    mode_dd = server.gui.add_dropdown(
        "Record mode",
        ("human_hand", "robot_hand", "arm_and_hand"),
        initial_value="arm_and_hand")
    save_view_btn  = server.gui.add_button("Save view")
    rec_start_sl   = server.gui.add_slider("Rec start", 0, N-1, 1, 0)
    rec_end_sl     = server.gui.add_slider("Rec end",   0, N-1, 1, N-1)
    record_btn     = server.gui.add_button("Record")
    trail_frames_text = server.gui.add_text("Trail frames", "35,45,55,65")
    capture_trail_btn = server.gui.add_button("Capture trail")
    render_scale_sl   = server.gui.add_slider("Render scale", 1, 6, 1, 3)

    def _toggle_hand_color(_=None):
        use_black = hand_black_cb.value
        hand_vis.show_visual       = not use_black
        hand_vis_black.show_visual = use_black

    hand_black_cb.on_update(_toggle_hand_color)

    def _apply_mode(_=None):
        mode = mode_dd.value
        is_human = mode == "human_hand"
        is_robot = mode in ("robot_hand", "arm_and_hand")
        urdf_vis.show_visual       = mode == "arm_and_hand"
        hand_vis.show_visual       = is_robot and not hand_black_cb.value
        hand_vis_black.show_visual = is_robot and hand_black_cb.value
        human_pts_h.visible        = is_human
        human_skel_h.visible       = is_human
        # MANO mesh visibility via a dummy frame trick (add_mesh_trimesh has no handle)
        _mano_mesh_fh.visible      = is_human
    def _set_mano_mesh(verts):
        """Re-add MANO mesh (vertex assignment alone does not sync to client)."""
        nonlocal _mano_mesh_fh
        _mano_mesh_fh = server.scene.add_mesh_simple(
            "/human_hand/mesh", verts.astype(np.float32), _mano_faces,
            color=tuple(mano_color_rgb.value), side="double",
            visible=(mode_dd.value == "human_hand"))

    mano_color_rgb.on_update(
        lambda _: _set_mano_mesh(_mano_verts_vis(int(frame_slider.value))))

    mode_dd.on_update(_apply_mode)
    _apply_mode()  # set initial visibility

    @save_view_btn.on_click
    def _save_view(_):
        clients = server.get_clients()
        if not clients:
            print("No client connected"); return
        cam = list(clients.values())[0].camera
        view = {"wxyz": list(map(float, cam.wxyz)),
                "position": list(map(float, cam.position)),
                "fov": float(cam.fov)}
        (OUT_DIR / "viser_view.yml").write_text(yaml.dump(view))
        print(f"View saved: {view}")

    @record_btn.on_click
    def _record(_):
        clients = server.get_clients()
        if not clients:
            print("No client connected"); return
        client = list(clients.values())[0]
        view_file = OUT_DIR / "viser_view.yml"
        render_kw: dict = {}
        if view_file.exists():
            v = yaml.safe_load(view_file.read_text())
            render_kw = dict(wxyz=tuple(v["wxyz"]), position=tuple(v["position"]),
                             fov=float(v["fov"]))
        import subprocess, shutil
        start = int(rec_start_sl.value)
        end   = int(rec_end_sl.value)
        mode  = mode_dd.value
        out   = OUT_DIR / f"record_{mode}_{start}_{end}.mp4"
        # Save PNGs in-process (no fork), then assemble with ffmpeg via subprocess.run
        # (posix_spawn on Py3.10, so it avoids the fork()+JAX-thread deadlock).
        tmp = OUT_DIR / f".rec_tmp_{mode}_{start}_{end}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        print(f"Recording {mode} frames {start}-{end} → {out}")
        sc = int(render_scale_sl.value)
        rh, rw = 480 * sc, 640 * sc
        for k, fi in enumerate(range(start, end + 1)):
            frame_slider.value = fi
            time.sleep(0.12)
            img = client.get_render(height=rh, width=rw, **render_kw)
            cv2.imwrite(str(tmp / f"frame_{k:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "15", "-i", str(tmp / "frame_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp)
        print(f"Saved ({end - start + 1} frames): {out}")
    play_cb      = server.gui.add_checkbox("Play", False)
    fps_sl    = server.gui.add_slider("Playback FPS", 1, 30, 1, 15)
    rx_sl = server.gui.add_slider("Mount rot X (deg)", -180, 180, 1, init_rx)
    ry_sl = server.gui.add_slider("Mount rot Y (deg)", -180, 180, 1, init_ry)
    rz_sl = server.gui.add_slider("Mount rot Z (deg)", -180, 180, 1, init_rz)

    # Camera frustums — all 8 DexYCB cameras
    _fov_y = float(2 * np.arctan(240.0 / FY))
    _ext_src = _extrinsics.get(CAM, {})
    frustum_handles: dict = {}
    frustum_cbs: dict = {}
    if _ext_src:
        R_src, t_src = _ext_src["R"], _ext_src["t"]
        for idx, serial in enumerate(_serials):
            ext_i = _extrinsics.get(serial)
            if ext_i is None:
                continue
            R_i, t_i = ext_i["R"], ext_i["t"]
            pos_vis = _M @ (R_src.T @ (t_i - t_src))
            R_vis   = _M @ R_src.T @ R_i
            q       = Rotation.from_matrix(R_vis).as_quat()
            wxyz_vis = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            enabled = idx < 4
            fh = server.scene.add_camera_frustum(
                f"/frustum_{serial}", fov=_fov_y, aspect=640.0 / 480.0,
                scale=0.15, position=tuple(pos_vis.tolist()),
                wxyz=wxyz_vis, image=_load_cam_img(serial, fi), visible=enabled)
            cb = server.gui.add_checkbox(f"Cam {serial[-4:]}", enabled)
            frustum_handles[serial] = fh
            frustum_cbs[serial] = cb

            def _make_toggle(fh_=fh, cb_=cb):
                @cb_.on_update
                def _(_): fh_.visible = cb_.value
            _make_toggle()

    @scene_cb.on_update
    def _(_): pc_handle.visible = scene_cb.value

    @arm_cb.on_update
    def _(_): panda_fh.visible = arm_cb.value

    @hand_cb.on_update
    def _(_): hand_fh.visible = hand_cb.value

    @obj_cb.on_update
    def _(_): obj_fh.visible = obj_cb.value

    @axes_cb.on_update
    def _(_): wrist_fh.visible = axes_cb.value

    @save_btn.on_click
    def _save(_):
        out = {
            "position":  list(map(float, base_handle.position)),
            "wxyz":      list(map(float, base_handle.wxyz)),
            "cam":       CAM,
            "mount_rot": {"rx": float(rx_sl.value),
                          "ry": float(ry_sl.value),
                          "rz": float(rz_sl.value)},
        }
        saved_path.write_text(yaml.dump(out))
        print(f"Saved → {saved_path}\n{out}")

    print("Ready.")
    nonlocal_state = {"fi": fi, "wrist_vis": wrist_vis, "wrist_q": wrist_q,
                      "last_max_z": 2.5, "capturing": False,
                      "last_cut": -1.0, "last_bp": np.array(base_handle.position),
                      "last_bwxyz": np.array(base_handle.wxyz)}
    last_play_t = time.time()

    # ── Motion-trail figure capture ───────────────────────────────────────────
    def _compute_frame_pose(fi_):
        j3 = frame_j3_all[fi_]
        wrist_vis_ = cam2vis(j3[0:1])[0]
        wq = wrist_wxyz_vis(j3)
        bp = np.array(base_handle.position)
        bwxyz = np.array(base_handle.wxyz)
        R_base = Rotation.from_quat([bwxyz[1], bwxyz[2], bwxyz[3], bwxyz[0]]).as_matrix()
        R_mount = Rotation.from_euler('xyz',
                      [rx_sl.value, ry_sl.value, rz_sl.value], degrees=True).as_matrix()
        target_pos_base = R_base.T @ (wrist_vis_ - bp)
        R_wrist = Rotation.from_quat([wq[1], wq[2], wq[3], wq[0]]).as_matrix()
        R_hand = R_wrist @ R_mount
        q_base = Rotation.from_matrix(R_base.T @ R_hand).as_quat()
        target_wxyz_base = np.array([q_base[3], q_base[0], q_base[1], q_base[2]])
        solution = pks.solve_ik(robot=robot, target_link_name=target_link,
                                 target_wxyz=target_wxyz_base,
                                 target_position=target_pos_base)
        sol_dict = {n: float(solution[i]) for i, n in enumerate(ik_joint_names)}
        arm_cfg = np.array([sol_dict.get(jn, 0.0) for jn in panda_vis_urdf.actuated_joint_names])
        fk = robot.forward_kinematics(jnp.array(solution))
        li = robot.links.names.index(target_link)
        ee_pos_vis = bp + R_base @ np.array(fk[li, 4:])
        qh = Rotation.from_matrix(R_hand).as_quat()
        hand_wxyz = np.array([qh[3], qh[0], qh[1], qh[2]])
        if frame_rq_valid[fi_]:
            rq_dict = {n: float(v) for n, v in zip(rq_joint_names, frame_rq_all[fi_])}
            hand_cfg = np.array([rq_dict.get(n, 0.0) for n in hand_urdf.actuated_joint_names])
        else:
            hand_cfg = np.zeros(len(hand_urdf.actuated_joint_names))
        return dict(bp=bp, bwxyz=bwxyz, arm_cfg=arm_cfg,
                    ee_pos_vis=ee_pos_vis, hand_wxyz=hand_wxyz, hand_cfg=hand_cfg)

    @capture_trail_btn.on_click
    def _capture_trail(_):
        clients = server.get_clients()
        if not clients:
            print("No client connected"); return
        client = list(clients.values())[0]
        view_file = OUT_DIR / "viser_view.yml"
        render_kw: dict = {}
        if view_file.exists():
            v = yaml.safe_load(view_file.read_text())
            render_kw = dict(wxyz=tuple(v["wxyz"]), position=tuple(v["position"]),
                             fov=float(v["fov"]))
        try:
            frames = [int(x.strip()) for x in trail_frames_text.value.split(",") if x.strip()]
        except ValueError:
            print("Bad trail frames"); return
        if not frames:
            return
        ops = np.linspace(0.3, 1.0, len(frames))
        mode = mode_dd.value
        nonlocal_state["capturing"] = True
        time.sleep(0.06)
        # hide live geometry
        urdf_vis.show_visual = False
        hand_vis.show_visual = False
        hand_vis_black.show_visual = False
        _mano_mesh_fh.visible = False
        human_pts_h.visible = False
        human_skel_h.visible = False
        ghosts = []
        for fi_, op in zip(frames, ops):
            op = float(op)
            if mode == "human_hand":
                ghosts.append(server.scene.add_mesh_simple(
                    f"/trail/mano_{fi_}", _mano_verts_vis(fi_).astype(np.float32),
                    _mano_faces, color=tuple(mano_color_rgb.value),
                    opacity=op, side="double"))
            else:
                P = _compute_frame_pose(fi_)
                hcol = (0.05, 0.05, 0.05, op) if hand_black_cb.value else (*_ROBOT_HAND_RGB, op)
                fr_h = server.scene.add_frame(f"/trail/hand_{fi_}",
                    position=tuple(P["ee_pos_vis"]), wxyz=tuple(P["hand_wxyz"]),
                    show_axes=False)
                ViserUrdf(server, hand_urdf, root_node_name=f"/trail/hand_{fi_}",
                          mesh_color_override=hcol).update_cfg(P["hand_cfg"])
                ghosts.append(fr_h)
                if mode == "arm_and_hand":
                    fr_a = server.scene.add_frame(f"/trail/arm_{fi_}",
                        position=tuple(P["bp"]), wxyz=tuple(P["bwxyz"]), show_axes=False)
                    # opaque frame keeps original URDF colors; faded ghosts must be flat
                    # (ViserUrdf transparency forces a single color override).
                    arm_kw = {} if op >= 0.99 else dict(mesh_color_override=(0.85, 0.85, 0.85, op))
                    ViserUrdf(server, panda_vis_urdf, root_node_name=f"/trail/arm_{fi_}",
                              **arm_kw).update_cfg(P["arm_cfg"])
                    ghosts.append(fr_a)
        time.sleep(0.5)
        sc = int(render_scale_sl.value)
        img = client.get_render(height=480 * sc, width=640 * sc, **render_kw)
        out = OUT_DIR / f"trail_{mode}_{'_'.join(map(str, frames))}.png"
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for g in ghosts:
            g.remove()
        nonlocal_state["capturing"] = False
        _apply_mode()  # restore live geometry visibility
        print(f"Saved trail figure: {out}")

    while True:
        if nonlocal_state["capturing"]:
            time.sleep(0.05); continue
        # Auto-advance frame when playing
        if play_cb.value:
            now = time.time()
            if now - last_play_t >= 1.0 / fps_sl.value:
                next_fi = (int(frame_slider.value) + 1) % N
                frame_slider.value = next_fi
                last_play_t = now

        cur_fi = int(frame_slider.value)
        if cur_fi != nonlocal_state["fi"]:
            nonlocal_state["fi"] = cur_fi
            col, dep, j3 = load_frame(cur_fi)
            wv = cam2vis(j3[0:1])[0]
            wq = wrist_wxyz_vis(j3)
            nonlocal_state["wrist_vis"] = wv
            nonlocal_state["wrist_q"]   = wq
            if not lock_depth_cb.value:
                pts_new, col_new = depth_to_pc(dep, col, max_z=float(max_depth_sl.value))
                _set_scene_pc(pts_new, col_new)
            wrist_fh.position = tuple(wv)
            wrist_fh.wxyz     = tuple(wq)
            # Update human hand skeleton
            j3_vis = cam2vis(j3)
            human_pts_h.centers = j3_vis.astype(np.float32)
            skel_pts = np.stack([j3_vis[HAND_CONNECTIONS[:, 0]],
                                 j3_vis[HAND_CONNECTIONS[:, 1]]], axis=1).astype(np.float32)
            human_skel_h.positions = skel_pts
            _set_mano_mesh(_mano_verts_vis(cur_fi))
            op, owxyz = load_obj_pose_vis(cur_fi)
            obj_fh.position = tuple(op)
            obj_fh.wxyz     = tuple(owxyz)
            for serial, fh in frustum_handles.items():
                if fh.visible:
                    img = _load_cam_img(serial, cur_fi)
                    if img is not None:
                        fh.image = img

        # Reload point cloud if max-depth, base-cut, or base pose changed
        cur_max_z = float(max_depth_sl.value)
        cur_cut   = float(base_cut_sl.value)
        bp_now    = np.array(base_handle.position)
        bwxyz_now = np.array(base_handle.wxyz)
        base_moved = (np.abs(bp_now - nonlocal_state["last_bp"]).max() > 1e-4 or
                      np.abs(bwxyz_now - nonlocal_state["last_bwxyz"]).max() > 1e-4)
        if (abs(cur_max_z - nonlocal_state["last_max_z"]) > 0.04 or
                abs(cur_cut - nonlocal_state["last_cut"]) > 0.005 or base_moved):
            nonlocal_state["last_max_z"] = cur_max_z
            nonlocal_state["last_cut"]   = cur_cut
            nonlocal_state["last_bp"]    = bp_now
            nonlocal_state["last_bwxyz"] = bwxyz_now
            src_dep = dep0 if lock_depth_cb.value else cv2.imread(
                str(dep_files[cur_fi]), cv2.IMREAD_ANYDEPTH)
            src_col = col0 if lock_depth_cb.value else cv2.imread(str(col_files[cur_fi]))
            pts_r, col_r = depth_to_pc(src_dep, src_col, max_z=cur_max_z)
            _set_scene_pc(pts_r, col_r)

        # Update frustum scale for all handles
        new_scale = float(frustum_scale_sl.value)
        for fh in frustum_handles.values():
            fh.scale = new_scale

        wrist_vis = nonlocal_state["wrist_vis"]
        wrist_q   = nonlocal_state["wrist_q"]

        # Panda arm IK
        bp    = np.array(base_handle.position)
        bwxyz = np.array(base_handle.wxyz)
        R_base = Rotation.from_quat([bwxyz[1], bwxyz[2], bwxyz[3], bwxyz[0]]).as_matrix()
        R_mount = Rotation.from_euler('xyz',
                      [rx_sl.value, ry_sl.value, rz_sl.value], degrees=True).as_matrix()

        target_pos_base = R_base.T @ (wrist_vis - bp)
        R_wrist = Rotation.from_quat([wrist_q[1], wrist_q[2], wrist_q[3], wrist_q[0]]).as_matrix()
        R_hand = R_wrist @ R_mount
        q_base = Rotation.from_matrix(R_base.T @ R_hand).as_quat()
        target_wxyz_base = np.array([q_base[3], q_base[0], q_base[1], q_base[2]])

        solution = pks.solve_ik(robot=robot, target_link_name=target_link,
                                 target_wxyz=target_wxyz_base,
                                 target_position=target_pos_base)
        panda_fh.position = tuple(bp)
        panda_fh.wxyz     = tuple(bwxyz)
        sol_dict = {n: float(solution[i]) for i, n in enumerate(ik_joint_names)}
        vis_cfg  = np.array([sol_dict.get(jn, 0.0) for jn in panda_vis_urdf.actuated_joint_names])
        urdf_vis.update_cfg(vis_cfg)

        fk = robot.forward_kinematics(jnp.array(solution))
        link_idx    = robot.links.names.index(target_link)
        ee_pos_base = np.array(fk[link_idx, 4:])
        ik_err_num.value = round(float(np.linalg.norm(ee_pos_base - target_pos_base)) * 1000, 1)

        # Inspire hand: position at IK EEF, orientation from human wrist (R_wrist @ R_mount)
        try:
            ee_pos_vis = bp + R_base @ ee_pos_base
            q_hand = Rotation.from_matrix(R_hand).as_quat()
            hand_fh.position = tuple(ee_pos_vis)
            hand_fh.wxyz     = np.array([q_hand[3], q_hand[0], q_hand[1], q_hand[2]])
            if frame_rq_valid[cur_fi]:
                rq_dict = {n: float(v) for n, v in zip(rq_joint_names, frame_rq_all[cur_fi])}
                cfg_arr = np.array([rq_dict.get(n, 0.0) for n in hand_urdf.actuated_joint_names])
                hand_vis.update_cfg(cfg_arr)
                hand_vis_black.update_cfg(cfg_arr)
            hand_fh_black.position = tuple(ee_pos_vis)
            hand_fh_black.wxyz     = np.array([q_hand[3], q_hand[0], q_hand[1], q_hand[2]])
        except (np.linalg.LinAlgError, ValueError):
            pass

        time.sleep(0.02)

if __name__ == "__main__":
    main()
