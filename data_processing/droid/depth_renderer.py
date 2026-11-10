"""Robot mesh depth sequence renderer for the DROID render_v2 pipeline.

This is the depth-only counterpart of `droid_urdf_overlay.render_video`. The
existing rendering primitives (build_pyrender_context, update_pyrender_context,
compose_joint_values, frame indexing helpers, etc.) live inside the legacy
`droid_urdf_overlay/droid_urdf_overlay.py` module; this file imports them but
keeps the orchestration localised to `data_processing/droid/` so the depth
entry point (`render_robot_mesh_depth.py`) does not depend on a function
that has not been implemented inside the legacy module.

Wrist support follows the same pattern as the mesh-RGB pipeline: pass a
per-frame `base_to_camera_sequence` ([T, 4, 4]) instead of a static
`base_to_camera` and the renderer will update the camera pose every frame.

Public API:
    render_depth_sequence(...) -> dict
        Keys: depth_m, valid_mask, output_fps, output_frame_indices,
              output_frame_size, source_frame_size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# The legacy primitives live in data_processing/droid_urdf_overlay/. We
# extend sys.path the same way the existing entry scripts do.
REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_OVERLAY_DIR = REPO_ROOT / "data_processing" / "droid_urdf_overlay"
if str(LEGACY_OVERLAY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_OVERLAY_DIR))

from droid_urdf_overlay import (  # type: ignore  # noqa: E402
    build_output_frame_indices,
    build_pyrender_context,
    compose_joint_values,
    get_gripper_joint_spec,
    get_video_meta,
    resolve_render_frame_size,
    scale_intrinsics_to_frame,
    update_pyrender_context,
)


def _render_pyrender_depth(context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Render the current pyrender scene and return (depth_m, valid_mask).

    Mirrors the metric depth that
    ``droid_urdf_overlay.render_pyrender_frame`` already computes
    internally; we just keep the depth buffer instead of the RGB.
    """
    _color, depth = context["renderer"].render(context["scene"], flags=context["render_flags"])
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > 0.0
    return depth, valid


def render_depth_sequence(
    *,
    video_path: Path,
    robot: Any,
    joints: np.ndarray,
    gripper: np.ndarray | None,
    base_to_camera: np.ndarray,
    base_to_camera_sequence: np.ndarray | None,
    intrinsics: np.ndarray,
    calibration_image_size: tuple[int, int],
    fps_override: float | None,
    max_faces: int,
    frame_limit: int | None,
    target_short_side: int,
    robot_only_render_batch_size: int = 1,  # accepted for parity, unused
    pyrender_style: str = "flat",
    progress_interval: int = 50,
) -> dict[str, Any]:
    """Render per-frame metric depth of the robot mesh aligned to the source RGB.

    Parameters mirror ``droid_urdf_overlay.render_video`` so the call sites
    in ``render_robot_mesh_depth.py`` can switch with no signature change.
    Wrist views must pass ``base_to_camera_sequence`` (per-frame [T,4,4]);
    static views pass ``base_to_camera`` ([4,4]) and ``base_to_camera_sequence=None``.

    Returns a dict with:
        depth_m: (T, H, W) float32 metric depth (0 where invalid).
        valid_mask: (T, H, W) bool.
        output_fps: float.
        output_frame_indices: (T,) int32 -- indices into the SOURCE video.
        output_frame_size: (H, W) int32.
        source_frame_size: (H, W) int32.
    """
    if max_faces and max_faces > 0:
        # We render with pyrender, which uses the URDF meshes directly. The
        # max_faces decimation path only exists for the pytorch3d backend
        # (kept as a parameter for API parity with render_video).
        raise ValueError(
            "render_depth_sequence currently only supports the pyrender backend; "
            "pass max_faces=0 (or rely on a pre-decimated URDF)."
        )

    cap, source_width, source_height, source_fps, frame_count = get_video_meta(video_path)
    try:
        width, height = resolve_render_frame_size(
            frame_size=(source_width, source_height),
            target_short_side=target_short_side,
        )
        render_intrinsics = scale_intrinsics_to_frame(
            intrinsics=intrinsics,
            calibration_image_size=calibration_image_size,
            frame_size=(width, height),
        )

        total_frames = min(
            frame_count if frame_count > 0 else joints.shape[0],
            int(joints.shape[0]),
        )
        if gripper is not None:
            total_frames = min(total_frames, int(gripper.shape[0]))
        if base_to_camera_sequence is not None:
            total_frames = min(total_frames, int(base_to_camera_sequence.shape[0]))
        if frame_limit is not None and frame_limit > 0:
            total_frames = min(total_frames, int(frame_limit))
        if total_frames <= 0:
            raise RuntimeError(
                f"render_depth_sequence: no frames to render (video={video_path})"
            )

        output_fps = float(fps_override or source_fps or 30.0)
        output_frame_indices = build_output_frame_indices(
            total_frames=total_frames,
            source_fps=float(source_fps or 0.0),
            output_fps=output_fps,
        )
        output_frame_total = int(output_frame_indices.shape[0])

        gripper_spec = get_gripper_joint_spec(robot)

        # Pyrender context. For the static-view case the camera pose is set
        # once; for wrist we update it every frame inside the loop.
        initial_base_to_camera = (
            base_to_camera_sequence[0] if base_to_camera_sequence is not None else base_to_camera
        )
        context = build_pyrender_context(
            robot=robot,
            frame_size=(width, height),
            base_to_camera=initial_base_to_camera,
            intrinsics=render_intrinsics,
            style=pyrender_style,
        )

        depth_buf = np.zeros((output_frame_total, height, width), dtype=np.float32)
        valid_buf = np.zeros((output_frame_total, height, width), dtype=bool)

        progress_start = time.time()
        produced = 0
        # We iterate over the source video frame count and only act on the
        # exported frame indices; this matches render_video's traversal so
        # the output indices line up with the existing RGB / mesh outputs.
        index_set = set(int(i) for i in output_frame_indices.tolist())
        next_slot = 0
        for frame_idx in range(total_frames):
            ok, _frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx not in index_set:
                continue

            arm_joint = joints[frame_idx]
            gripper_signal = None if gripper is None else float(gripper[frame_idx])
            joint_values = compose_joint_values(
                robot=robot,
                arm_joint_values=arm_joint,
                gripper_signal=gripper_signal,
                gripper_spec=gripper_spec,
            )
            cam_pose = (
                base_to_camera_sequence[frame_idx]
                if base_to_camera_sequence is not None
                else base_to_camera
            )
            update_pyrender_context(
                context=context,
                robot=robot,
                joint_values=joint_values,
                base_to_camera=cam_pose,
            )
            depth, valid = _render_pyrender_depth(context)
            depth_buf[next_slot] = depth
            valid_buf[next_slot] = valid
            next_slot += 1
            produced += 1

            if progress_interval > 0 and (
                produced == 1
                or produced == output_frame_total
                or produced % progress_interval == 0
            ):
                elapsed = time.time() - progress_start
                rate = produced / max(elapsed, 1e-6)
                print(
                    f"[depth] {video_path.name}: {produced}/{output_frame_total} "
                    f"frames @ {rate:.2f} fps",
                    flush=True,
                )

        if next_slot != output_frame_total:
            # Trim if the source ended early.
            depth_buf = depth_buf[:next_slot]
            valid_buf = valid_buf[:next_slot]
            output_frame_indices = output_frame_indices[:next_slot]

    finally:
        cap.release()

    return {
        "depth_m": depth_buf,
        "valid_mask": valid_buf,
        "output_fps": float(output_fps),
        "output_frame_indices": np.asarray(output_frame_indices, dtype=np.int32),
        "output_frame_size": np.asarray((height, width), dtype=np.int32),
        "source_frame_size": np.asarray((source_height, source_width), dtype=np.int32),
    }


__all__ = ["render_depth_sequence"]
