"""Sequence-level (trajectory) Panda arm IK for DexYCB inspire-hand rendering.

Unlike precompute_panda_ik_dexycb.py (independent per-frame IK -> jitter), this solves
the WHOLE trajectory jointly with pose-tracking + smoothness + acceleration costs via
jaxls, giving temporally coherent joints. Run in pyroki env.

Usage:
    conda activate pyroki
    python data_processing/dexycb/precompute_panda_ik_traj_dexycb.py \\
        --session 20200709_141754 --cam 840412060917 [--overwrite]

Reads:  explore_out/panda_base.yml  (CURRENT base)
        explore_out/{session}_{cam}_retarget_rq.npz
Writes: explore_out/{session}_{cam}_panda_ik_joints.npz   (same path render_dexycb reads)
"""
import sys, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/pyroki/examples"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_PANDA_URDF  = str(_REPO_ROOT / "data_processing" / "dexycb" / "assets" / "panda_description" / "urdf" / "panda.urdf")
_EXPLORE_OUT = Path(
    os.environ.get("DEXYCB_OUTPUT_ROOT", _REPO_ROOT / "outputs" / "dexycb" / "panda_ik_traj")
)
_M = np.diag([1., -1., -1.])  # cam <-> viser world


def wrist_frame_cam(j3):
    w, m, i, p = j3[0], j3[9], j3[5], j3[17]
    if not (np.linalg.norm(m - w) > 1e-4 and np.linalg.norm(p - i) > 1e-4):
        return None
    fd  = m - w;  fd  /= np.linalg.norm(fd)
    lat = p - i;  lat -= np.dot(lat, fd) * fd;  lat /= np.linalg.norm(lat)
    pn  = np.cross(-fd, lat)
    return np.column_stack([pn, lat, fd])


def solve_traj_ik(robot, target_link_name, target_wxyz, target_position,
                  pos_weight=50.0, ori_weight=10.0,
                  smooth_weight=1.0, accel_weight=0.5, dt=1.0 / 16.0):
    """Joint trajectory IK over all frames: per-frame pose + smoothness + accel."""
    N = target_position.shape[0]
    tli = robot.links.names.index(target_link_name)
    traj_vars = robot.joint_var_cls(jnp.arange(N))
    targets = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.array(target_wxyz)), jnp.array(target_position))

    robot_b = jax.tree.map(lambda x: x[None], robot)  # batch dim for vectorized costs

    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot_b, traj_vars, targets,
            jnp.full((N,), tli),
            pos_weight=jnp.full((N,), pos_weight),
            ori_weight=jnp.full((N,), ori_weight),
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, N)),
            robot.joint_var_cls(jnp.arange(0, N - 1)),
            jnp.array([smooth_weight])[None],
        ),
        pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, N - 2)),
            robot.joint_var_cls(jnp.arange(4, N)),
            robot.joint_var_cls(jnp.arange(3, N - 1)),
            robot.joint_var_cls(jnp.arange(1, N - 3)),
            robot.joint_var_cls(jnp.arange(0, N - 4)),
            dt,
            jnp.array([accel_weight])[None],
        ),
        pk.costs.limit_constraint(robot_b, traj_vars),
        pk.costs.limit_velocity_constraint(
            robot_b,
            robot.joint_var_cls(jnp.arange(1, N)),
            robot.joint_var_cls(jnp.arange(0, N - 1)),
            dt,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars])
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky",
               trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0))
    )
    return np.asarray(sol[traj_vars])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20200709_141754")
    ap.add_argument("--cam",     default="840412060917")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-root", type=Path, default=_EXPLORE_OUT)
    ap.add_argument("--smooth_weight", type=float, default=1.0)
    ap.add_argument("--accel_weight", type=float, default=0.5)
    args = ap.parse_args()

    explore_out = args.output_root.expanduser()
    out_path = explore_out / f"{args.session}_{args.cam}_panda_ik_joints.npz"
    if out_path.exists() and not args.overwrite:
        print(f"Already exists: {out_path}  (use --overwrite)")
        return

    saved   = yaml.safe_load((explore_out / "panda_base.yml").read_text())
    bp_vis  = np.array(saved["position"])
    bwxyz   = np.array(saved["wxyz"])
    mnt     = saved.get("mount_rot", {})
    R_base  = Rotation.from_quat([bwxyz[1], bwxyz[2], bwxyz[3], bwxyz[0]]).as_matrix()
    R_mount = Rotation.from_euler('xyz',
                [mnt.get("rx", 0), mnt.get("ry", 180), mnt.get("rz", 0)],
                degrees=True).as_matrix()

    rq_path = explore_out / f"{args.session}_{args.cam}_retarget_rq.npz"
    if not rq_path.exists():
        rq_path = explore_out / "retarget_rq.npz"
    j3_all  = np.load(str(rq_path), allow_pickle=True)["j3"]
    N       = len(j3_all)

    # per-frame wrist pose targets (base frame); forward/back-fill invalid frames
    tgt_pos  = np.zeros((N, 3), np.float32)
    tgt_wxyz = np.zeros((N, 4), np.float32)
    valid    = np.zeros(N, bool)
    last = None
    for fi in range(N):
        Rw = wrist_frame_cam(j3_all[fi])
        if Rw is None:
            continue
        wrist_vis  = _M @ j3_all[fi][0]
        R_hand_vis = (_M @ Rw @ _M.T) @ R_mount
        p = R_base.T @ (wrist_vis - bp_vis)
        q = Rotation.from_matrix(R_base.T @ R_hand_vis).as_quat()
        tgt_pos[fi]  = p
        tgt_wxyz[fi] = [q[3], q[0], q[1], q[2]]
        valid[fi] = True
        last = fi
    # fill invalid frames by nearest valid (hold) so trajectory stays smooth
    if not valid.any():
        print("No valid wrist frames"); return
    idx = np.where(valid)[0]
    for fi in range(N):
        if not valid[fi]:
            j = idx[np.argmin(np.abs(idx - fi))]
            tgt_pos[fi], tgt_wxyz[fi] = tgt_pos[j], tgt_wxyz[j]

    print(f"Loading panda URDF ({N} frames, {valid.sum()} valid) trajectory IK …")
    urdf  = yourdfpy.URDF.load(_PANDA_URDF)
    robot = pk.Robot.from_urdf(urdf)

    joints = solve_traj_ik(robot, "panda_hand", tgt_wxyz, tgt_pos,
                           smooth_weight=args.smooth_weight,
                           accel_weight=args.accel_weight).astype(np.float32)

    # report pose tracking residual + jitter (max abs joint step)
    step = np.abs(np.diff(joints[valid], axis=0)).max() if valid.sum() > 1 else 0.0
    print(f"  max joint step (valid frames): {step:.4f} rad")

    np.savez(str(out_path), joints=joints, valid=valid,
             joint_names=np.array(list(urdf.actuated_joint_names)))
    print(f"Saved {valid.sum()}/{N} valid -> {out_path}")


if __name__ == "__main__":
    main()
