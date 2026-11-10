"""Precompute dex-retargeting for all DexYCB frames.

Run in spider env:
    python data_processing/dexycb/precompute_retarget.py

Env overrides:
    SUBJECT=20200813-subject-02
    SESSION=20200813_145341
    CAM=840412060917

Output: explore_out/{SESSION}_{CAM}_retarget_rq.npz
(Legacy default session saves to explore_out/retarget_rq.npz for backward compat.)
"""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import numpy as np
from pathlib import Path
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEX_YCB_DIR = Path(os.environ.get("DEX_YCB_DIR", REPO_ROOT / "data" / "dexycb"))
SUBJECT      = os.environ.get("SUBJECT", "20200709-subject-01")
SESSION      = os.environ.get("SESSION", "20200709_141754")
CAM          = os.environ.get("CAM",     "840412060917")
OUT_DIR = Path(os.environ.get("DEXYCB_OUTPUT_ROOT", REPO_ROOT / "outputs" / "dexycb" / "retarget"))
DEX_URDF_DIR = REPO_ROOT / "data_processing" / "third_party" / "dex-urdf" / "robots" / "hands"
RETARGET_CFG = REPO_ROOT / "data_processing" / "third_party" / "dex-retargeting" / "src" / "dex_retargeting" / "configs" / "offline" / "inspire_hand_right.yml"

_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--output-root", type=Path, default=OUT_DIR)
_cli_args = _cli.parse_args()
OUT_DIR = _cli_args.output_root.expanduser()

RetargetingConfig.set_default_urdf_dir(str(DEX_URDF_DIR))
retargeter = RetargetingConfig.load_from_file(str(RETARGET_CFG)).build()

sess_dir  = DEX_YCB_DIR / SUBJECT / SESSION / CAM
lbl_files = sorted(sess_dir.glob("labels_*.npz"))
N = len(lbl_files)
print(f"Subject={SUBJECT}  Session={SESSION}  CAM={CAM}  frames={N}")
print(f"Joint names: {retargeter.joint_names}")

rq_all  = np.zeros((N, len(retargeter.joint_names)), dtype=np.float32)
valid   = np.zeros(N, dtype=bool)
j3_all  = np.zeros((N, 21, 3), dtype=np.float32)

for i, lf in enumerate(lbl_files):
    j3 = np.load(str(lf))["joint_3d"][0]
    j3_all[i] = j3
    if (j3[:, 2] > 0.01).any():
        rq_all[i] = retargeter.retarget(j3[[4, 8, 12, 16, 20]])
        valid[i] = True

# Session-specific output; also write legacy path for default session
out = OUT_DIR / f"{SESSION}_{CAM}_retarget_rq.npz"
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(str(out), rq=rq_all, valid=valid, joint_names=np.array(retargeter.joint_names), j3=j3_all)
print(f"Saved {valid.sum()}/{N} valid frames → {out}")
if SESSION == "20200709_141754" and CAM == "840412060917":
    legacy = OUT_DIR / "retarget_rq.npz"
    np.savez(str(legacy), rq=rq_all, valid=valid, joint_names=np.array(retargeter.joint_names), j3=j3_all)
    print(f"  (also → {legacy} for backward compat)")
