from __future__ import annotations

import numpy as np

def update_env_after_qpos(env) -> None:
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()

def qpos_limits_for_indices(env, qpos_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    low = np.full(len(qpos_indices), -np.inf, dtype=np.float64)
    high = np.full(len(qpos_indices), np.inf, dtype=np.float64)
    qpos_to_offset = {int(qpos_idx): offset for offset, qpos_idx in enumerate(qpos_indices)}
    for joint_id in range(int(raw_model.njnt)):
        if not bool(raw_model.jnt_limited[joint_id]):
            continue
        qpos_adr = int(raw_model.jnt_qposadr[joint_id])
        if qpos_adr not in qpos_to_offset:
            continue
        offset = qpos_to_offset[qpos_adr]
        low[offset] = float(raw_model.jnt_range[joint_id][0])
        high[offset] = float(raw_model.jnt_range[joint_id][1])
    return low, high
