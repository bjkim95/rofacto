"""DexYCB session indexer, output-path formatter, and coordinate bridge.

Single source of truth for:
- Enumerating subjects / sessions / cameras from the DexYCB dataset
- Resolving object names from meta.yml
- Parsing flat-12 cam-to-world extrinsics
- Computing the six locked output stream paths under outputs/dexycb_retarget/
- Converting DexYCB cam-to-world extrinsics to pyrender camera poses

Usage (any env):
    python data_processing/dexycb/dexycb_layout.py
    → prints each object with total session×cam counts
"""

import os
import re
import sys
import yaml
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEX_YCB_DIR = Path(os.environ.get("DEX_YCB_DIR", Path(__file__).resolve().parents[2] / "data" / "dexycb"))
OUTPUT_ROOT = Path(os.environ.get("DEXYCB_OUTPUT_ROOT", _REPO_ROOT / "outputs" / "dexycb_retarget"))

# droid_urdf_overlay on sys.path for camera_pose_opengl_from_base_to_camera
sys.path.insert(0, str(_REPO_ROOT / "data_processing" / "droid_urdf_overlay"))

# ── YCB object names (1-indexed, entry 0 = id 1) ─────────────────────────────
_YCB_NAMES = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
    "011_banana", "019_pitcher_base", "021_bleach_cleanser",
    "024_bowl", "025_mug", "035_power_drill", "036_wood_block",
    "037_scissors", "040_large_marker", "051_large_clamp",
    "052_extra_large_clamp",
]


# ── Extrinsics ────────────────────────────────────────────────────────────────

def load_extrinsics(ext_tag: str) -> dict[str, dict]:
    """Parse flat-12 cam-to-world extrinsics for one calibration tag.

    Returns: {serial: {"R": (3,3), "t": (3,)}}
    Convention: p_world = R @ p_cam + t  (cam-to-world).
    Matches render_dexycb.load_extrinsics:59.
    """
    path = DEX_YCB_DIR / "calibration" / f"extrinsics_{ext_tag}" / "extrinsics.yml"
    raw = path.read_text()
    # The YAML uses !!python/tuple; strip the tag so safe_load can handle it
    raw_clean = raw.replace("!!python/tuple", "")
    data = yaml.safe_load(raw_clean)
    ext_dict = data.get("extrinsics", {})
    out = {}
    for serial, vals in ext_dict.items():
        serial = str(serial)
        if not isinstance(vals, (list, tuple)) or len(vals) != 12:
            continue
        RT = np.array(vals, dtype=float).reshape(3, 4)
        out[serial] = {"R": RT[:, :3], "t": RT[:, 3]}
    return out  # p_world = R @ p_cam + t


# ── Session meta ──────────────────────────────────────────────────────────────

def load_session_meta(subject_dir: Path, session: str) -> dict:
    """Load meta.yml for one session. Returns dict with parsed fields."""
    meta_path = subject_dir / session / "meta.yml"
    raw = meta_path.read_text().replace("!!python/tuple", "")
    meta = yaml.safe_load(raw)
    grasp_idx = meta["ycb_grasp_ind"]
    object_name = _YCB_NAMES[meta["ycb_ids"][grasp_idx] - 1]
    return {
        "object_name": object_name,
        "serials": [str(s) for s in meta["serials"]],
        "ext_tag": str(meta["extrinsics"]),
        "num_frames": meta.get("num_frames", 72),
        "mano_sides": meta.get("mano_sides", []),
    }


# ── Subject / session enumeration ─────────────────────────────────────────────

def _subject_id(subject_dir_name: str) -> str:
    """'20200709-subject-01' → 's01'"""
    m = re.search(r"subject-(\d+)$", subject_dir_name)
    return f"s{m.group(1).zfill(2)}" if m else subject_dir_name


def enumerate_sessions(dex_ycb_dir: Path = DEX_YCB_DIR):
    """Yield one dict per (subject, session, cam) triple.

    Keys: subject_id, subject_dir, session, cam_serial, object_name,
          ext_tag, num_frames, extrinsics_tag.
    """
    for subj_dir in sorted(dex_ycb_dir.glob("*-subject-*")):
        if not subj_dir.is_dir():
            continue
        subject_id = _subject_id(subj_dir.name)
        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            meta_path = sess_dir / "meta.yml"
            if not meta_path.exists():
                continue
            try:
                meta = load_session_meta(subj_dir, sess_dir.name)
            except Exception:
                continue
            for serial in meta["serials"]:
                yield {
                    "subject_id": subject_id,
                    "subject_dir": subj_dir,
                    "session": sess_dir.name,
                    "cam_serial": serial,
                    "object_name": meta["object_name"],
                    "ext_tag": meta["ext_tag"],
                    "num_frames": meta["num_frames"],
                }


# ── Output paths ──────────────────────────────────────────────────────────────

def _key(subject_id: str, session: str, cam_serial: str) -> str:
    """Canonical filename stem: s{N}_{datetime}_{cam_serial}"""
    return f"{subject_id}_{session}_{cam_serial}"


def output_paths(
    object_name: str,
    subject_id: str,
    session: str,
    cam_serial: str,
    root: Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    """Return the six locked output stream paths for one session×cam.

    Streams:
        videos                    raw RGB mp4
        robot_mesh_videos_nominal Franka+Robotiq wrist+gripper mesh render
        robot_eef_nominal_depth_videos  EEF depth conditioning
        scene_depth_realdep_videos RealSense scene depth (first frame repeated)
        inpaint_videos            hand-removed + inpainted RGB
        hand_retarget             IK joint angles npz
    """
    key = _key(subject_id, session, cam_serial)
    base = root / object_name
    return {
        "videos":                        base / "videos"                        / f"{key}.mp4",
        "robot_mesh_videos_nominal":     base / "robot_mesh_videos_nominal"     / f"{key}.mp4",
        "robot_eef_nominal_depth_videos":base / "robot_eef_nominal_depth_videos"/ f"{key}.mp4",
        "scene_depth_realdep_videos":    base / "scene_depth_realdep_videos"    / f"{key}.mp4",
        "inpaint_videos":                base / "inpaint_videos"                / f"{key}.mp4",
        "hand_retarget":                 base / "hand_retarget"                 / f"{key}.npz",
    }


# ── Coordinate bridge ─────────────────────────────────────────────────────────

def extrinsic_to_pyrender_pose(R_cam2world: np.ndarray, t_world: np.ndarray) -> np.ndarray:
    """Convert a DexYCB cam-to-world extrinsic to a pyrender camera pose (4×4).

    Input convention (cam-to-world):
        p_world = R_cam2world @ p_cam + t_world
        This matches render_dexycb.load_extrinsics:59 and the flat-12 DexYCB format.

    Bridge logic:
        1. Build the 4×4 cam-to-world homogeneous matrix.
        2. Invert to world→cam (OpenCV convention).
        3. Delegate the OpenGL flip to camera_pose_opengl_from_base_to_camera
           from droid_urdf_overlay, which computes inv(base_to_camera) @ cv_to_gl.

    _M = diag([1,-1,-1]) is the viser-display-only flip used in
    interactive_panda_base.py (Phase 2). It is NOT applied here and must
    not leak into Phase 1 (IK) or Phase 3 (render) coordinate math.

    Returns:
        4×4 camera pose matrix in OpenGL convention, suitable for
        build_pyrender_context / render_pyrender_frame.
    """
    from droid_urdf_overlay import camera_pose_opengl_from_base_to_camera

    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_cam2world
    T_c2w[:3, 3] = t_world
    # Invert: cam-to-world → world-to-cam (what camera_pose_opengl expects)
    T_w2c = np.linalg.inv(T_c2w)
    return camera_pose_opengl_from_base_to_camera(T_w2c)


# ── Main: enumerate and summarise ─────────────────────────────────────────────

def _summarise(dex_ycb_dir: Path = DEX_YCB_DIR):
    from collections import defaultdict
    obj_counts: dict[str, int] = defaultdict(int)
    for rec in enumerate_sessions(dex_ycb_dir):
        obj_counts[rec["object_name"]] += 1
    print(f"{'Object':<35} {'session×cam':>12}")
    print("-" * 49)
    for obj, cnt in sorted(obj_counts.items()):
        print(f"{obj:<35} {cnt:>12}")
    print("-" * 49)
    print(f"{'TOTAL':<35} {sum(obj_counts.values()):>12}")


def _self_test():
    """Spot-check output_paths and extrinsic_to_pyrender_pose."""
    paths = output_paths("002_master_chef_can", "s01", "20200709_141754", "840412060917")
    assert len(paths) == 6, f"expected 6 paths, got {len(paths)}"
    key = "s01_20200709_141754_840412060917"
    assert paths["videos"].name == f"{key}.mp4"
    assert paths["hand_retarget"].name == f"{key}.npz"
    assert "002_master_chef_can" in str(paths["videos"])
    assert "robot_mesh_videos_nominal" in str(paths["robot_mesh_videos_nominal"])
    print("output_paths: OK")

    # Spot-check bridge with identity extrinsic (cam at world origin looking +Z)
    R_id = np.eye(3)
    t_id = np.zeros(3)
    pose = extrinsic_to_pyrender_pose(R_id, t_id)
    assert pose.shape == (4, 4), f"expected 4x4, got {pose.shape}"
    print(f"extrinsic_to_pyrender_pose (identity): OK  shape={pose.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        _self_test()
    else:
        _summarise()
