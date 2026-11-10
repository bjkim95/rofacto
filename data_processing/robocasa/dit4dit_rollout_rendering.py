"""DiT4DiT rendering helpers extracted from rerender_robocasa_hdf5.py.

The deferred HDF5 entrypoint is intentionally not migrated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

DROID_STYLE_DEPTH_NEAR_M = 0.2
DROID_STYLE_DEPTH_FAR_M = 3.0


def sanitize_valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    return np.isfinite(depth_m) & (depth_m > 0.0)

def encode_inverse_depth_u16(
    depth_m: np.ndarray,
    *,
    near_m: float,
    far_m: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if near_m <= 0.0 or far_m <= 0.0 or far_m <= near_m:
        raise ValueError(f"Invalid near/far values: near={near_m}, far={far_m}")
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if valid_mask is None:
        valid_mask = sanitize_valid_depth_mask(depth_m)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & sanitize_valid_depth_mask(depth_m)

    encoded = np.zeros(depth_m.shape, dtype=np.uint16)
    if not np.any(valid_mask):
        return encoded, valid_mask.astype(np.uint8)

    clipped = np.clip(depth_m[valid_mask], float(near_m), float(far_m))
    inv_depth = 1.0 / clipped
    inv_near = 1.0 / float(near_m)
    inv_far = 1.0 / float(far_m)
    normalized = (inv_depth - inv_far) / (inv_near - inv_far)
    quantized = np.clip(np.round(normalized * 65534.0) + 1.0, 1.0, 65535.0).astype(np.uint16)
    encoded[valid_mask] = quantized
    return encoded, valid_mask.astype(np.uint8)

def depth_frame_to_turbo_rgb(
    inv_u16: np.ndarray,
    valid: np.ndarray,
    *,
    near_m: float,
    far_m: float,
    invalid_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Render one inverse-depth u16 frame as an (H, W, 3) uint8 RGB image
    using cv2's turbo colormap.

    invalid_color:
      - None (default): invalid pixels are clipped to far_m so they map to
        the colormap's "far" end (dark red in turbo). Suits stereo depth
        where most pixels are valid and the few invalid ones blending into
        "far" is fine.
      - (R, G, B) tuple: invalid pixels are overwritten with this constant
        color. Use (0, 0, 0) to mark "no signal" for robot mesh depth where
        the foreground covers a small fraction of the frame and the
        all-far-red background otherwise dominates.
    """
    import cv2

    inv_u16_f = inv_u16.astype(np.float32)
    valid_bool = (np.asarray(valid).astype(bool)) & (inv_u16_f > 0)

    inv_near = 1.0 / max(float(near_m), 1e-8)
    inv_far = 1.0 / max(float(far_m), 1e-8)
    normalized = np.clip((inv_u16_f - 1.0) / 65534.0, 0.0, 1.0)
    inv_depth = normalized * (inv_near - inv_far) + inv_far
    depth_m = np.full(inv_u16_f.shape, far_m, dtype=np.float32)
    np.divide(1.0, np.maximum(inv_depth, 1e-8), out=depth_m, where=valid_bool)

    clipped = np.clip(depth_m, near_m, far_m)
    denom = max(float(far_m) - float(near_m), 1e-8)
    depth_u8 = ((clipped - float(near_m)) / denom * 255.0).clip(0.0, 255.0).astype(np.uint8)

    bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if invalid_color is not None:
        rgb[~valid_bool] = np.asarray(invalid_color, dtype=np.uint8)
    return rgb

def hide_ik_indicator_sites(model_xml: str | bytes) -> str:
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    raw_xml = ET.fromstring(model_xml)
    changed = False
    for site in raw_xml.findall(".//site"):
        name = site.get("name", "")
        if "pinch_spheres" in name:
            site.set("rgba", "0 0 0 0")
            changed = True
    if not changed:
        return model_xml
    return ET.tostring(raw_xml, encoding="unicode")

def import_runtime_modules(args: argparse.Namespace):
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    try:
        import mujoco  # type: ignore
        import robocasa  # noqa: F401  # type: ignore
        import robosuite  # type: ignore
        from robosuite.utils import camera_utils as robosuite_camera_utils  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mujoco / robocasa / robosuite. Activate the RoboCasa render environment first."
        ) from exc
    return mujoco, robosuite, robosuite_camera_utils

def reset_to(env, state: dict[str, Any], robosuite_module) -> None:
    should_reset = False
    if "model" in state:
        ep_meta_raw = state.get("ep_meta", None)
        if isinstance(ep_meta_raw, bytes):
            ep_meta_raw = ep_meta_raw.decode("utf-8")
        ep_meta = json.loads(ep_meta_raw) if ep_meta_raw else {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

        robosuite_version_minor = int(str(robosuite_module.__version__).split(".")[1])
        if robosuite_version_minor <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml  # type: ignore

            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])

        try:
            env.reset()
        except Exception as exc:
            print(f"[reset_to] pre-reset failed, retrying direct XML restore: {exc}")
        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_reset = True

    if should_reset:
        if hasattr(env, "update_sites"):
            env.update_sites()
        if hasattr(env, "update_state"):
            env.update_state()

def get_camera_id(sim_model, camera_name: str, mujoco_module) -> int:
    if hasattr(sim_model, "camera_name2id"):
        try:
            return int(sim_model.camera_name2id(camera_name))
        except Exception:
            pass

    raw_models = [sim_model]
    for attr_name in ("_model", "model"):
        raw_model = getattr(sim_model, attr_name, None)
        if raw_model is not None and raw_model is not sim_model:
            raw_models.append(raw_model)

    for raw_model in raw_models:
        try:
            camera_id = mujoco_module.mj_name2id(raw_model, mujoco_module.mjtObj.mjOBJ_CAMERA, camera_name)
        except TypeError:
            continue
        if camera_id >= 0:
            return int(camera_id)

    raise KeyError(f"Camera '{camera_name}' was not found in the instantiated MuJoCo model.")

def _camera_xmat_to_quat(camera_xmat: np.ndarray, mujoco_module) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco_module.mju_mat2Quat(quat, np.asarray(camera_xmat, dtype=np.float64))
    return quat

@contextmanager
def override_camera_pose(
    env,
    *,
    camera_name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    mujoco_module,
):
    sim_model = env.sim.model
    camera_id = get_camera_id(sim_model, camera_name, mujoco_module)

    original_body_id = int(sim_model.cam_bodyid[camera_id])
    original_position = np.array(sim_model.cam_pos[camera_id], copy=True)
    original_quaternion = np.array(sim_model.cam_quat[camera_id], copy=True)
    original_mode = int(sim_model.cam_mode[camera_id]) if hasattr(sim_model, "cam_mode") else None

    sim_model.cam_bodyid[camera_id] = 0
    sim_model.cam_pos[camera_id] = position
    sim_model.cam_quat[camera_id] = quaternion
    if original_mode is not None:
        sim_model.cam_mode[camera_id] = int(mujoco_module.mjtCamLight.mjCAMLIGHT_FIXED)
    env.sim.forward()

    try:
        yield
    finally:
        sim_model.cam_bodyid[camera_id] = original_body_id
        sim_model.cam_pos[camera_id] = original_position
        sim_model.cam_quat[camera_id] = original_quaternion
        if original_mode is not None:
            sim_model.cam_mode[camera_id] = original_mode
        env.sim.forward()

def get_robot_name_prefixes(env) -> set[str]:
    prefixes: set[str] = set()
    for robot in getattr(env, "robots", []):
        robot_model = getattr(robot, "robot_model", None)
        for candidate in (
            getattr(robot_model, "naming_prefix", None),
            getattr(getattr(robot_model, "base", None), "naming_prefix", None),
        ):
            if candidate:
                prefixes.add(candidate)
        for gripper in getattr(robot, "gripper", {}).values():
            prefix = getattr(gripper, "naming_prefix", None)
            if prefix:
                prefixes.add(prefix)
    return prefixes

def _iter_named_geom_candidates(robot) -> list[str]:
    candidates: list[str] = []
    robot_model = getattr(robot, "robot_model", None)
    components = [robot_model, getattr(robot_model, "base", None)]
    components.extend(getattr(robot, "gripper", {}).values())
    for component in components:
        if component is None:
            continue
        for attr_name in ("visual_geoms", "contact_geoms"):
            values = getattr(component, attr_name, None)
            if values:
                candidates.extend(values)
    return candidates

def collect_robot_geom_ids(env) -> np.ndarray:
    geom_ids: set[int] = set()
    prefixes = get_robot_name_prefixes(env)
    sim_model = env.sim.model

    for geom_id in range(sim_model.ngeom):
        geom_name = sim_model.geom(geom_id).name
        if geom_name and any(geom_name.startswith(prefix) for prefix in prefixes):
            geom_ids.add(geom_id)

    for robot in getattr(env, "robots", []):
        for geom_name in _iter_named_geom_candidates(robot):
            try:
                geom_ids.add(sim_model.geom_name2id(geom_name))
            except Exception:
                continue

    if not geom_ids:
        raise RuntimeError("Could not identify robot geoms for robot-only rendering.")
    return np.array(sorted(geom_ids), dtype=np.int32)

def collect_eef_geom_ids(env) -> np.ndarray:
    geom_ids: set[int] = set()
    sim_model = env.sim.model

    for robot in getattr(env, "robots", []):
        for gripper in getattr(robot, "gripper", {}).values():
            prefix = getattr(gripper, "naming_prefix", None)
            if prefix:
                for geom_id in range(sim_model.ngeom):
                    geom_name = sim_model.geom(geom_id).name
                    if geom_name and geom_name.startswith(prefix):
                        geom_ids.add(geom_id)
            for attr_name in ("visual_geoms", "contact_geoms"):
                for geom_name in getattr(gripper, attr_name, []) or []:
                    try:
                        geom_ids.add(sim_model.geom_name2id(geom_name))
                    except Exception:
                        continue

    if not geom_ids:
        for geom_id in range(sim_model.ngeom):
            geom_name = sim_model.geom(geom_id).name or ""
            lowered = geom_name.lower()
            if "gripper" in lowered or "hand" in lowered or "finger" in lowered or "thumb" in lowered:
                geom_ids.add(geom_id)

    if not geom_ids:
        raise RuntimeError("Could not identify end-effector geoms for eef-only rendering.")
    return np.array(sorted(geom_ids), dtype=np.int32)

def select_hidden_geom_group(env, robot_geom_ids: np.ndarray) -> int:
    offscreen_context = getattr(env.sim, "_render_context_offscreen", None)
    if offscreen_context is None:
        raise RuntimeError("Offscreen MuJoCo render context is not initialized.")

    num_groups = len(offscreen_context.vopt.geomgroup)
    robot_groups = set(int(value) for value in np.asarray(env.sim.model.geom_group)[robot_geom_ids])
    for group_id in reversed(range(num_groups)):
        if group_id not in robot_groups:
            return group_id
    raise RuntimeError("Could not find a temporary geom group for robot-only renders.")

@contextmanager
def isolate_robot_scene(env, robot_geom_ids: np.ndarray):
    offscreen_context = getattr(env.sim, "_render_context_offscreen", None)
    if offscreen_context is None:
        raise RuntimeError("Offscreen MuJoCo render context is not initialized.")

    hidden_group = select_hidden_geom_group(env, robot_geom_ids)
    sim_model = env.sim.model
    non_robot_mask = np.ones(sim_model.ngeom, dtype=bool)
    non_robot_mask[robot_geom_ids] = False

    original_geom_groups = np.array(sim_model.geom_group, copy=True)
    original_visible_groups = np.array(offscreen_context.vopt.geomgroup, copy=True)
    original_site_groups = np.array(offscreen_context.vopt.sitegroup, copy=True)

    sim_model.geom_group[non_robot_mask] = hidden_group
    offscreen_context.vopt.geomgroup[:] = original_visible_groups
    offscreen_context.vopt.geomgroup[hidden_group] = 0
    offscreen_context.vopt.sitegroup[:] = 0
    try:
        yield
    finally:
        sim_model.geom_group[:] = original_geom_groups
        offscreen_context.vopt.geomgroup[:] = original_visible_groups
        offscreen_context.vopt.sitegroup[:] = original_site_groups

def disable_geom_collisions(env, geom_ids: np.ndarray) -> None:
    sim_model = env.sim.model
    sim_model.geom_contype[geom_ids] = 0
    sim_model.geom_conaffinity[geom_ids] = 0

@contextmanager
def hide_robot_scene(env, robot_geom_ids: np.ndarray):
    offscreen_context = getattr(env.sim, "_render_context_offscreen", None)
    if offscreen_context is None:
        raise RuntimeError("Offscreen MuJoCo render context is not initialized.")

    hidden_group = select_hidden_geom_group(env, robot_geom_ids)
    sim_model = env.sim.model

    original_geom_groups = np.array(sim_model.geom_group, copy=True)
    original_visible_groups = np.array(offscreen_context.vopt.geomgroup, copy=True)

    sim_model.geom_group[robot_geom_ids] = hidden_group
    offscreen_context.vopt.geomgroup[:] = original_visible_groups
    offscreen_context.vopt.geomgroup[hidden_group] = 0
    try:
        yield
    finally:
        sim_model.geom_group[:] = original_geom_groups
        offscreen_context.vopt.geomgroup[:] = original_visible_groups

def render_segmentation_geom_ids(
    env,
    *,
    height: int,
    width: int,
    camera_name: str,
    expected_geom_ids: np.ndarray | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        segmentation = env.sim.render(
            height=height,
            width=width,
            camera_name=camera_name,
            segmentation=True,
        )
    except TypeError as exc:
        raise RuntimeError("This MuJoCo render backend does not support segmentation=True.") from exc
    if isinstance(segmentation, tuple):
        segmentation = segmentation[0]
    segmentation = np.asarray(segmentation)[::-1]
    if segmentation.ndim == 3 and segmentation.shape[-1] >= 1:
        if expected_geom_ids is None:
            geom_ids = segmentation[..., 0]
        else:
            expected = np.asarray(expected_geom_ids, dtype=np.int64)
            best_channel = 0
            best_overlap = -1
            for channel in range(segmentation.shape[-1]):
                channel_ids = np.asarray(segmentation[..., channel], dtype=np.int64)
                overlap = int(np.isin(channel_ids, expected).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_channel = channel
            geom_ids = segmentation[..., best_channel]
    elif segmentation.ndim == 2:
        geom_ids = segmentation
    else:
        raise RuntimeError(f"Unexpected segmentation render shape: {segmentation.shape}")
    geom_ids = np.asarray(geom_ids, dtype=np.int64)
    if not np.any((geom_ids >= 0) & (geom_ids < int(env.sim.model.ngeom))):
        if allow_empty:
            return np.full((height, width), -1, dtype=np.int64)
        sample_values = np.unique(geom_ids)[:20].tolist()
        raise RuntimeError(
            "Segmentation render did not expose geom ids in any selected channel. "
            f"shape={segmentation.shape} sample_values={sample_values}"
        )
    invalid = (geom_ids < 0) | (geom_ids >= int(env.sim.model.ngeom))
    if invalid.any():
        geom_ids = geom_ids.copy()
        geom_ids[invalid] = -1
    return geom_ids

def flat_geom_color_frame(
    *,
    geom_id_image: np.ndarray,
    mask: np.ndarray,
    background_rgb: np.ndarray,
    robot_rgb: np.ndarray,
) -> np.ndarray:
    frame = np.broadcast_to(background_rgb, (*geom_id_image.shape, 3)).copy()
    if not np.any(mask):
        return frame
    frame[mask] = robot_rgb.reshape(1, 3)
    return frame

def render_robot_only_outputs(
    env,
    *,
    camera_name: str,
    width: int,
    height: int,
    robot_geom_ids: np.ndarray,
    background_rgb: np.ndarray,
    rgb_style: str,
    flat_robot_rgb: np.ndarray,
    near_m: float | None = None,
    far_m: float | None = None,
    robosuite_camera_utils_module=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    with isolate_robot_scene(env, robot_geom_ids):
        geom_id_image = None
        if rgb_style in {"flat", "shaded"}:
            geom_id_image = render_segmentation_geom_ids(
                env,
                height=height,
                width=width,
                camera_name=camera_name,
                expected_geom_ids=robot_geom_ids,
                allow_empty=True,
            )
            robot_mask = np.isin(geom_id_image, robot_geom_ids)

        if rgb_style == "flat":
            if not np.any(robot_mask):
                # Ego cameras often do not see the selected robot subset on every frame,
                # especially for EEF-only depth. Treat that as a valid empty condition.
                isolated_rgb = np.broadcast_to(background_rgb, (*geom_id_image.shape, 3)).copy()
            else:
                isolated_rgb = flat_geom_color_frame(
                    geom_id_image=geom_id_image,
                    mask=robot_mask,
                    background_rgb=background_rgb,
                    robot_rgb=flat_robot_rgb,
                )
        elif rgb_style == "shaded":
            isolated_rgb = env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise ValueError(f"Unsupported robot-only RGB style: {rgb_style}")
        depth_buffer = None
        if near_m is not None and far_m is not None:
            _, depth_buffer = env.sim.render(
                height=height,
                width=width,
                camera_name=camera_name,
                depth=True,
            )

    robot_only = np.broadcast_to(background_rgb, isolated_rgb.shape).copy()
    robot_only[robot_mask] = isolated_rgb[robot_mask]

    robot_depth_frame = None
    if depth_buffer is not None:
        depth_m = robosuite_camera_utils_module.get_real_depth_map(env.sim, depth_buffer[::-1])
        inv_depth_u16, valid_u8 = encode_inverse_depth_u16(
            depth_m,
            near_m=near_m,
            far_m=far_m,
            valid_mask=robot_mask,
        )
        robot_depth_frame = depth_frame_to_turbo_rgb(
            inv_depth_u16,
            valid_u8,
            near_m=near_m,
            far_m=far_m,
            invalid_color=(0, 0, 0),
        )
    return robot_only, robot_depth_frame

def render_depth_colormap_frame(
    env,
    *,
    camera_name: str,
    width: int,
    height: int,
    near_m: float,
    far_m: float,
    invalid_color: tuple[int, int, int] | None = None,
    robosuite_camera_utils_module,
) -> np.ndarray:
    _, depth_buffer = env.sim.render(
        height=height,
        width=width,
        camera_name=camera_name,
        depth=True,
    )
    depth_buffer = depth_buffer[::-1]
    depth_m = robosuite_camera_utils_module.get_real_depth_map(env.sim, depth_buffer)
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    inv_depth_u16, valid_u8 = encode_inverse_depth_u16(
        depth_m,
        near_m=near_m,
        far_m=far_m,
        valid_mask=valid_mask,
    )
    return depth_frame_to_turbo_rgb(
        inv_depth_u16,
        valid_u8,
        near_m=near_m,
        far_m=far_m,
        invalid_color=invalid_color,
    )
