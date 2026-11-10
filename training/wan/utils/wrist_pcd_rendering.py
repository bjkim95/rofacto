# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
"""Runtime wrist-view point-cloud rendering for MV DWM training.

Replaces the wrist cell's "copy first frame" static conditioning with a video
rendered by unprojecting the SELECTED ext view's first-frame RGB+depth into a
colored point cloud (in the robot base frame) and projecting it along the wrist
camera trajectory.

Semantics (per user spec, 2026-04-23):
    * frame 0  = wrist GT RGB at training start_idx (no pcd; this matches what
      the model will actually see as its initial observation).
    * frame t>0 = pcd render from selected ext's first-frame depth, projected
      through the wrist camera pose at frame (start_idx + t * frame_skip).
    * If neither ext view has a usable calibration, frame 0 still uses wrist GT
      and frames 1+ fall back to black.

The module is self-contained and device-agnostic for the public API.
Heavy rasterization runs on the GPU via pytorch3d; if CUDA is unavailable the
caller gets a RuntimeError (fallback to pure copy_first_frame is the
responsibility of the caller).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch

__all__ = [
    "PcdCalibrationCache",
    "render_wrist_cell_static",
    "render_wrist_cell_static_disparity",
    "render_wrist_cell_dynamic_disparity",
    "WristPcdRenderError",
]


# ---------------- errors ----------------


class WristPcdRenderError(RuntimeError):
    """Raised when pcd rendering cannot proceed — caller should fall back."""


# ---------------- small helpers ----------------


def _decode_inv_depth_u16(
    inv_u16: np.ndarray,
    *,
    near_m: float,
    far_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of `data_processing/droid/common.py::encode_inverse_depth_u16`."""
    inv_near = 1.0 / float(near_m)
    inv_far = 1.0 / float(far_m)
    valid = inv_u16 >= 1
    depth = np.zeros(inv_u16.shape, dtype=np.float32)
    if np.any(valid):
        normalized = (inv_u16[valid].astype(np.float32) - 1.0) / 65534.0
        inv_depth = inv_far + normalized * (inv_near - inv_far)
        depth[valid] = 1.0 / inv_depth
    return depth, valid


def _pose_vector_to_matrix(vec: np.ndarray) -> np.ndarray:
    """[tx,ty,tz,rx,ry,rz] (euler XYZ, R=Rz@Ry@Rx) → 4x4. Matches
    data_processing/droid/common.py::_pose_vector_to_matrix."""
    tx, ty, tz, rx, ry, rz = (float(v) for v in vec)
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Rz @ Ry @ Rx
    M[:3, 3] = [tx, ty, tz]
    return M


# ---------------- calibration cache ----------------


@dataclass
class _EpisodeCalibration:
    """Everything we need for one episode to render from ext→wrist."""

    episode_id: str
    views: Dict[str, Dict[str, Any]]  # keyed by view_name ("ext1"|"ext2"|"wrist")
    trajectory_h5_path: Path
    wrist_cam2base: np.ndarray  # (T_traj, 4, 4) — full trajectory, pre-loaded
    # Per-ext view cam2base (4,4), only present if calibration exists.
    ext_cam2base: Dict[str, np.ndarray]
    trajectory_indices: np.ndarray  # (N_export,) maps exported frame idx → trajectory idx


class PcdCalibrationCache:
    """Thread-safe lazy cache keyed by episode_meta.json path.

    The dataset calls `get(episode_meta_path)` inside __getitem__; the first hit
    loads and caches; subsequent hits reuse in-memory data. Intended to be
    attached as `dataset._pcd_calib_cache` so each worker has its own copy.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, _EpisodeCalibration] = {}
        self._lock = threading.Lock()
        # Cross-episode blobs — loaded lazily from the first episode_meta we see.
        self._cam2base_blobs: Optional[List[dict]] = None
        self._intrinsics_blob: Optional[dict] = None

    def _load_shared_blobs(self, annotation_assets: Dict[str, str]) -> None:
        if self._cam2base_blobs is None:
            blobs: List[dict] = []
            for key in ("cam2base_json", "cam2base_superset_json"):
                p = annotation_assets.get(key)
                if p is None:
                    continue
                path = Path(p)
                if path.exists():
                    with open(path) as f:
                        blobs.append(json.load(f))
            self._cam2base_blobs = blobs
        if self._intrinsics_blob is None:
            p = annotation_assets.get("intrinsics_json")
            if p is not None and Path(p).exists():
                with open(p) as f:
                    self._intrinsics_blob = json.load(f)
            else:
                self._intrinsics_blob = {}

    def _lookup_ext_cam2base(self, episode_id: str, serial: str) -> Optional[np.ndarray]:
        assert self._cam2base_blobs is not None
        for blob in self._cam2base_blobs:
            entry = blob.get(episode_id)
            if not isinstance(entry, dict):
                continue
            vec = entry.get(serial)
            if vec is None:
                continue
            try:
                arr = np.asarray(vec, dtype=np.float64)
                if arr.size == 6:
                    return _pose_vector_to_matrix(arr)
            except (TypeError, ValueError):
                continue
        return None

    def get(self, episode_meta_path: Path) -> _EpisodeCalibration:
        key = str(episode_meta_path)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        # Load outside the lock to avoid blocking other workers on IO.
        with open(episode_meta_path) as f:
            meta = json.load(f)
        annotation_assets = meta.get("annotation_assets") or {}
        self._load_shared_blobs(annotation_assets)
        episode_id = str(meta["episode_id"])

        views: Dict[str, Dict[str, Any]] = {
            v["view_name"]: dict(v) for v in meta.get("views_saved", [])
        }
        traj_rel = meta.get("localized_raw_trajectory_path")
        if traj_rel is None:
            raise WristPcdRenderError(
                f"episode_meta missing localized_raw_trajectory_path: {episode_meta_path}"
            )
        repo_root = Path(__file__).resolve().parents[3]
        traj_h5_path = (repo_root / traj_rel).resolve() if not Path(traj_rel).is_absolute() else Path(traj_rel)
        if not traj_h5_path.exists():
            raise WristPcdRenderError(
                f"trajectory.h5 not found for episode {episode_id}: {traj_h5_path}"
            )
        wrist_info = views.get("wrist")
        if wrist_info is None:
            raise WristPcdRenderError(f"episode {episode_id} has no wrist view")
        wrist_serial = str(wrist_info["serial"])
        with h5py.File(traj_h5_path, "r") as h:
            key_name = f"{wrist_serial}_left"
            group = h.get("observation/camera_extrinsics")
            if group is None or key_name not in group:
                raise WristPcdRenderError(
                    f"missing observation/camera_extrinsics/{key_name} in {traj_h5_path}"
                )
            pose_vectors = np.asarray(group[key_name], dtype=np.float64)
        if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6:
            raise WristPcdRenderError(
                f"unexpected wrist pose shape {pose_vectors.shape} in {traj_h5_path}"
            )
        wrist_c2b = np.stack(
            [_pose_vector_to_matrix(pose_vectors[i]) for i in range(pose_vectors.shape[0])],
            axis=0,
        )
        trajectory_indices = np.asarray(meta.get("trajectory_indices") or [], dtype=np.int64)

        ext_c2b: Dict[str, np.ndarray] = {}
        for view_name in ("ext1", "ext2"):
            if view_name not in views:
                continue
            serial = str(views[view_name]["serial"])
            M = self._lookup_ext_cam2base(episode_id, serial)
            if M is not None:
                ext_c2b[view_name] = M

        calib = _EpisodeCalibration(
            episode_id=episode_id,
            views=views,
            trajectory_h5_path=traj_h5_path,
            wrist_cam2base=wrist_c2b,
            ext_cam2base=ext_c2b,
            trajectory_indices=trajectory_indices,
        )
        with self._lock:
            self._cache[key] = calib
        return calib


# ---------------- per-frame inputs ----------------


def _resolve_ext_depth_frame(
    inv_depth_u16_all: np.ndarray,
    frame_indices: np.ndarray,
    target_export_idx: int,
) -> int:
    """Pick the slice in inv_depth_u16_all whose recorded frame_index is
    closest to target_export_idx. Returns the slice index (0..T-1)."""
    if frame_indices.size == 0:
        raise WristPcdRenderError("ext depth NPZ has empty frame_indices")
    diffs = np.abs(frame_indices.astype(np.int64) - int(target_export_idx))
    return int(np.argmin(diffs))


def _read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise WristPcdRenderError(f"cannot open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise WristPcdRenderError(f"cannot read frame {frame_idx} of {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _scale_intrinsics(K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _unproject_to_base(
    *,
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    cam2base: np.ndarray,
    valid_mask: np.ndarray,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth_m.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = depth_m[::stride, ::stride]
    v = valid_mask[::stride, ::stride] & (d > 0)
    xs = xs[v].astype(np.float32)
    ys = ys[v].astype(np.float32)
    d = d[v].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d
    Z = d
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)
    pts_base = (cam2base @ pts_cam.T).T[:, :3]
    rgb_f = rgb[::stride, ::stride][v].astype(np.float32) / 255.0
    return pts_base.astype(np.float32), rgb_f.astype(np.float32)


# ---------------- public render function ----------------


def _black_wrist_static(
    *,
    num_frames: int,
    cell_h: int,
    cell_w: int,
    wrist_gt_frame0: Optional[np.ndarray],
) -> torch.Tensor:
    """Fallback: frame 0 = wrist GT (if given), frames 1+ = black. (T, 3, H, W) in [-1, 1]."""
    out = torch.full(
        (num_frames, 3, int(cell_h), int(cell_w)),
        float((0.0 / 255.0 - 0.5) / 0.5),
        dtype=torch.float32,
    )
    if wrist_gt_frame0 is not None:
        resized = cv2.resize(wrist_gt_frame0, (int(cell_w), int(cell_h)), interpolation=cv2.INTER_AREA)
        frame0_t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        out[0] = (frame0_t - 0.5) / 0.5
    return out


def _render_from_pcd(
    *,
    points_base: np.ndarray,  # (N, 3) float32
    colors: np.ndarray,        # (N, 3) float32 in [0, 1]
    wrist_base2cam_seq: np.ndarray,  # (T, 4, 4)
    K_target: np.ndarray,  # (3, 3) at (cell_h, cell_w)
    cell_hw: Tuple[int, int],
    point_radius: float,
    points_per_pixel: int,
    device: torch.device,
    chunk: int = 16,
) -> np.ndarray:
    """Batched pytorch3d point splatting. Returns (T, H, W, 3) uint8 RGB."""
    from pytorch3d.renderer import (
        AlphaCompositor,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
    )
    from pytorch3d.structures import Pointclouds
    from pytorch3d.utils import cameras_from_opencv_projection

    H, W = cell_hw
    T = wrist_base2cam_seq.shape[0]
    pts_t = torch.from_numpy(points_base).to(device)
    col_t = torch.from_numpy(colors).to(device)
    R_cv = torch.from_numpy(wrist_base2cam_seq[:, :3, :3]).float()
    t_cv = torch.from_numpy(wrist_base2cam_seq[:, :3, 3]).float()
    K_batch = torch.from_numpy(
        np.broadcast_to(K_target.astype(np.float32), (T, 3, 3)).copy()
    ).float()
    image_size = torch.tensor([[H, W]], dtype=torch.int64).expand(T, -1)
    raster = PointsRasterizationSettings(
        image_size=(H, W),
        radius=float(point_radius),
        points_per_pixel=int(points_per_pixel),
    )
    out = np.empty((T, H, W, 3), dtype=np.uint8)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        cams = cameras_from_opencv_projection(
            R=R_cv[s:e].to(device),
            tvec=t_cv[s:e].to(device),
            camera_matrix=K_batch[s:e].to(device),
            image_size=image_size[s:e].to(device),
        )
        rasterizer = PointsRasterizer(cameras=cams, raster_settings=raster)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0)),
        )
        pc = Pointclouds(points=[pts_t] * (e - s), features=[col_t] * (e - s))
        img = renderer(pc).clamp(0.0, 1.0)
        out[s:e] = (img * 255).to(torch.uint8).cpu().numpy()
    return out


def render_wrist_cell_static(
    *,
    calibration: _EpisodeCalibration,
    clip_dir: Path,  # directory containing videos/, stereo_inverse_depth/, etc.
    timestamp: str,
    start_idx: int,
    num_frames: int,
    frame_skip: int,
    cell_hw: Tuple[int, int],  # (H_cell, W_cell)
    preferred_ext_suffix: str,  # "ext1" | "ext2"
    ext_depth_subdir: str = "stereo_inverse_depth",
    robot_mesh_depth_subdir: str = "robot_mesh_inverse_depth",
    videos_subdir: str = "videos",
    point_radius: float = 0.01,
    points_per_pixel: int = 8,
    unproject_stride: int = 1,
    ext_max_depth_m: float = 1.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Produce the wrist cell's static video as a (T, 3, H, W) float32 tensor
    normalized to [-1, 1].

    Falls back to {frame0 = wrist GT, frames 1+ = black} if no usable ext
    calibration is available for this episode.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise WristPcdRenderError("CUDA not available; cannot render wrist pcd")

    H_cell, W_cell = int(cell_hw[0]), int(cell_hw[1])
    wrist_info = calibration.views.get("wrist")
    if wrist_info is None:
        raise WristPcdRenderError(f"no wrist view for {calibration.episode_id}")

    # Always read wrist GT frame at start_idx for frame 0 / fallback.
    wrist_video_path = clip_dir / videos_subdir / f"{timestamp}__wrist.mp4"
    wrist_gt_frame0 = _read_video_frame(wrist_video_path, int(start_idx))

    # Pick the ext view to source the pcd from: preferred first, else the other.
    candidate_suffixes: List[str] = []
    if preferred_ext_suffix in calibration.ext_cam2base:
        candidate_suffixes.append(preferred_ext_suffix)
    for alt in ("ext1", "ext2"):
        if alt == preferred_ext_suffix:
            continue
        if alt in calibration.ext_cam2base:
            candidate_suffixes.append(alt)

    if not candidate_suffixes:
        return _black_wrist_static(
            num_frames=num_frames,
            cell_h=H_cell,
            cell_w=W_cell,
            wrist_gt_frame0=wrist_gt_frame0,
        )

    # Per-frame wrist base→cam for the training clip.
    # trajectory_indices maps exported frame idx → raw trajectory.h5 idx.
    traj_idx = calibration.trajectory_indices
    frame_skip = max(1, int(frame_skip))
    target_export_indices = np.asarray(
        [int(start_idx) + i * frame_skip for i in range(num_frames)], dtype=np.int64
    )
    if traj_idx.size == 0 or target_export_indices.max() >= traj_idx.size:
        return _black_wrist_static(
            num_frames=num_frames, cell_h=H_cell, cell_w=W_cell, wrist_gt_frame0=wrist_gt_frame0
        )
    wrist_c2b = calibration.wrist_cam2base[traj_idx[target_export_indices]]
    wrist_b2c = np.linalg.inv(wrist_c2b)

    # Wrist intrinsics at target cell resolution.
    wrist_depth_npz_path = clip_dir / ext_depth_subdir / f"{timestamp}__wrist.npz"
    if not wrist_depth_npz_path.exists():
        # we only need the wrist intrinsics; fall back to loading from the RGB video shape
        K_wrist_src = None
        src_hw_wrist: Optional[Tuple[int, int]] = None
    else:
        z_w = np.load(wrist_depth_npz_path)
        K_wrist_src = z_w["intrinsics"].astype(np.float64)
        src_hw_wrist = (int(z_w["source_size"][1]), int(z_w["source_size"][0]))
    if K_wrist_src is None or src_hw_wrist is None:
        # final fallback: black video
        return _black_wrist_static(
            num_frames=num_frames, cell_h=H_cell, cell_w=W_cell, wrist_gt_frame0=wrist_gt_frame0
        )
    K_target = _scale_intrinsics(K_wrist_src, src_hw_wrist, (H_cell, W_cell))

    # Try each candidate ext suffix in order.
    last_exc: Optional[Exception] = None
    for ext_suffix in candidate_suffixes:
        try:
            pts_base, cols = _unproject_ext_first_frame(
                clip_dir=clip_dir,
                timestamp=timestamp,
                ext_suffix=ext_suffix,
                cam2base=calibration.ext_cam2base[ext_suffix],
                start_idx_export=int(start_idx),
                ext_depth_subdir=ext_depth_subdir,
                robot_mesh_depth_subdir=robot_mesh_depth_subdir,
                videos_subdir=videos_subdir,
                unproject_stride=unproject_stride,
                max_depth_m=ext_max_depth_m,
            )
            rendered_u8 = _render_from_pcd(
                points_base=pts_base,
                colors=cols,
                wrist_base2cam_seq=wrist_b2c,
                K_target=K_target,
                cell_hw=(H_cell, W_cell),
                point_radius=point_radius,
                points_per_pixel=points_per_pixel,
                device=device,
            )
            return _pack_output(
                rendered_u8=rendered_u8,
                wrist_gt_frame0=wrist_gt_frame0,
                cell_hw=(H_cell, W_cell),
            )
        except WristPcdRenderError as exc:
            last_exc = exc
            continue
    # All candidate exts failed → fallback.
    return _black_wrist_static(
        num_frames=num_frames,
        cell_h=H_cell,
        cell_w=W_cell,
        wrist_gt_frame0=wrist_gt_frame0,
    )


def _unproject_ext_first_frame(
    *,
    clip_dir: Path,
    timestamp: str,
    ext_suffix: str,
    cam2base: np.ndarray,
    start_idx_export: int,
    ext_depth_subdir: str,
    robot_mesh_depth_subdir: str,
    videos_subdir: str,
    unproject_stride: int,
    max_depth_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ext depth/RGB at the frame closest to start_idx_export, mask out
    robot pixels, and return (points_base[N,3], colors[N,3] in [0,1]).

    Raises WristPcdRenderError if any required file is missing."""
    depth_npz_path = clip_dir / ext_depth_subdir / f"{timestamp}__{ext_suffix}.npz"
    if not depth_npz_path.exists():
        raise WristPcdRenderError(f"missing ext depth npz: {depth_npz_path}")
    z = np.load(depth_npz_path)
    inv_u16_all = z["inv_depth_u16"]
    frame_indices = np.asarray(z["frame_indices"], dtype=np.int64)
    depth_slice_idx = _resolve_ext_depth_frame(inv_u16_all, frame_indices, start_idx_export)
    depth_export_idx = int(frame_indices[depth_slice_idx])

    inv_u16 = inv_u16_all[depth_slice_idx]
    valid_0 = z["valid_mask"][depth_slice_idx].astype(bool)
    near_m = float(z["near_m"])
    far_m = float(z["far_m"])
    depth_m, valid_d = _decode_inv_depth_u16(inv_u16, near_m=near_m, far_m=far_m)
    K_depth = z["intrinsics"].astype(np.float64)

    valid_mask = valid_0 & valid_d
    valid_mask &= depth_m <= float(max_depth_m)
    valid_mask &= depth_m > 0.0

    # Robot mesh mask: exclude pixels where the robot is in the ext view at
    # the exported frame we sampled for depth. `robot_mesh_inverse_depth` NPZ
    # stores a full (T, H_src, W_src) mask aligned with the exported RGB
    # video — index directly by the exported frame index.
    robot_npz_path = clip_dir / robot_mesh_depth_subdir / f"{timestamp}__{ext_suffix}.npz"
    if robot_npz_path.exists():
        try:
            zr = np.load(robot_npz_path, mmap_mode="r")
            robot_valid_all = zr["valid_mask"]  # (T, H_src, W_src) uint8 (1=robot present)
            if robot_valid_all.ndim == 3 and depth_export_idx < robot_valid_all.shape[0]:
                mask_src = np.asarray(robot_valid_all[depth_export_idx]).astype(bool)
                # downsample to match our depth/RGB resolution if needed
                if mask_src.shape != valid_mask.shape:
                    mask_src = cv2.resize(
                        mask_src.astype(np.uint8),
                        (valid_mask.shape[1], valid_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                valid_mask &= ~mask_src
        except Exception:
            # best-effort: if robot mask is unreadable, keep going without it
            pass

    # Read ext RGB at the matching export frame.
    ext_video_path = clip_dir / videos_subdir / f"{timestamp}__{ext_suffix}.mp4"
    rgb = _read_video_frame(ext_video_path, depth_export_idx)
    if rgb.shape[:2] != depth_m.shape:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)

    pts, cols = _unproject_to_base(
        depth_m=depth_m,
        rgb=rgb,
        K=K_depth,
        cam2base=cam2base,
        valid_mask=valid_mask,
        stride=int(unproject_stride),
    )
    if pts.shape[0] == 0:
        raise WristPcdRenderError(
            f"no valid ext points after masking for {ext_suffix} @ frame {depth_export_idx}"
        )
    return pts, cols


def _splat_pcd_depth_zbuf(
    *,
    points_base: np.ndarray,  # (N, 3) float32
    wrist_base2cam: np.ndarray,  # (4, 4) — single wrist pose
    K_target: np.ndarray,  # (3, 3) at (cell_h, cell_w)
    cell_hw: Tuple[int, int],
    point_radius: float,
    points_per_pixel: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render the PCD into the wrist camera at a single pose using PointsRasterizer
    and return per-pixel metric depth (from fragments.zbuf) plus a valid mask.

    Returns:
      depth_m: (H, W) float32 — z-depth (m) of the closest splat per pixel.
      valid:   (H, W) uint8  — 1 where any splat hit, 0 otherwise.
    """
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRasterizer,
    )
    from pytorch3d.structures import Pointclouds
    from pytorch3d.utils import cameras_from_opencv_projection

    H, W = int(cell_hw[0]), int(cell_hw[1])
    pts_t = torch.from_numpy(points_base).to(device)
    R_cv = torch.from_numpy(wrist_base2cam[None, :3, :3]).float().to(device)
    t_cv = torch.from_numpy(wrist_base2cam[None, :3, 3]).float().to(device)
    K_batch = torch.from_numpy(
        K_target.astype(np.float32)[None, ...].copy()
    ).to(device)
    image_size = torch.tensor([[H, W]], dtype=torch.int64, device=device)
    raster = PointsRasterizationSettings(
        image_size=(H, W),
        radius=float(point_radius),
        points_per_pixel=int(points_per_pixel),
    )
    cams = cameras_from_opencv_projection(
        R=R_cv, tvec=t_cv, camera_matrix=K_batch, image_size=image_size
    )
    rasterizer = PointsRasterizer(cameras=cams, raster_settings=raster)
    pc = Pointclouds(points=[pts_t])
    fragments = rasterizer(pc)  # zbuf: (1, H, W, K), idx: (1, H, W, K)
    zbuf = fragments.zbuf[0, :, :, 0].detach().cpu().numpy()  # (H, W)
    # PyTorch3D writes -1 where no splat reached the pixel.
    valid = (zbuf > 0.0).astype(np.uint8)
    depth_m = np.where(valid > 0, zbuf, 0.0).astype(np.float32)
    return depth_m, valid


def render_wrist_cell_static_disparity(
    *,
    calibration: _EpisodeCalibration,
    clip_dir: Path,
    timestamp: str,
    start_idx: int,
    num_frames: int,
    frame_skip: int,
    cell_hw: Tuple[int, int],
    preferred_ext_suffix: str,
    near_m: float,
    far_m: float,
    ext_depth_subdir: str = "stereo_inverse_depth",
    robot_mesh_depth_subdir: str = "robot_mesh_inverse_depth",
    videos_subdir: str = "videos",
    point_radius: float = 0.012,
    points_per_pixel: int = 8,
    unproject_stride: int = 1,
    ext_max_depth_m: float = 1.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Wrist-cell scene depth (static_disparity) via PCD: take the SELECTED ext's
    first-frame depth at start_idx, unproject -> base, splat into the wrist
    camera at the start_idx pose, read fragments.zbuf, turbo-colormap, and
    repeat T = num_frames times (matches static_disparity_npz_mode='first_frame_repeat').

    Returns (T, 3, H_cell, W_cell) float32 in [-1, 1]. On any failure falls
    back to all-black tile.
    """
    from data_processing.droid.depth_colormap import depth_frame_to_turbo_rgb

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise WristPcdRenderError("CUDA not available; cannot render wrist pcd depth")

    H_cell, W_cell = int(cell_hw[0]), int(cell_hw[1])

    def _black_tile() -> torch.Tensor:
        out = torch.full(
            (int(num_frames), 3, H_cell, W_cell),
            float((0.0 / 255.0 - 0.5) / 0.5),
            dtype=torch.float32,
        )
        return out

    # Pick ext suffix: preferred first, else any other with calibration.
    candidate_suffixes: List[str] = []
    if preferred_ext_suffix in calibration.ext_cam2base:
        candidate_suffixes.append(preferred_ext_suffix)
    for alt in ("ext1", "ext2"):
        if alt == preferred_ext_suffix:
            continue
        if alt in calibration.ext_cam2base:
            candidate_suffixes.append(alt)
    if not candidate_suffixes:
        return _black_tile()

    # Wrist intrinsics from wrist NPZ (same shape used by RGB renderer).
    wrist_depth_npz_path = clip_dir / ext_depth_subdir / f"{timestamp}__wrist.npz"
    if not wrist_depth_npz_path.exists():
        return _black_tile()
    z_w = np.load(wrist_depth_npz_path)
    if "intrinsics" not in z_w.files or "source_size" not in z_w.files:
        return _black_tile()
    K_wrist_src = z_w["intrinsics"].astype(np.float64)
    src_hw_wrist = (int(z_w["source_size"][1]), int(z_w["source_size"][0]))
    K_target = _scale_intrinsics(K_wrist_src, src_hw_wrist, (H_cell, W_cell))

    # Wrist cam2base at start_idx.
    traj_idx = calibration.trajectory_indices
    if traj_idx.size == 0 or int(start_idx) >= traj_idx.size:
        return _black_tile()
    wrist_c2b = calibration.wrist_cam2base[traj_idx[int(start_idx)]]
    wrist_b2c = np.linalg.inv(wrist_c2b)

    last_exc: Optional[Exception] = None
    for ext_suffix in candidate_suffixes:
        try:
            pts_base, _cols = _unproject_ext_first_frame(
                clip_dir=clip_dir,
                timestamp=timestamp,
                ext_suffix=ext_suffix,
                cam2base=calibration.ext_cam2base[ext_suffix],
                start_idx_export=int(start_idx),
                ext_depth_subdir=ext_depth_subdir,
                robot_mesh_depth_subdir=robot_mesh_depth_subdir,
                videos_subdir=videos_subdir,
                unproject_stride=int(unproject_stride),
                max_depth_m=float(ext_max_depth_m),
            )
            depth_m, valid_mask = _splat_pcd_depth_zbuf(
                points_base=pts_base,
                wrist_base2cam=wrist_b2c,
                K_target=K_target,
                cell_hw=(H_cell, W_cell),
                point_radius=float(point_radius),
                points_per_pixel=int(points_per_pixel),
                device=device,
            )
            # Turbo-colormap by re-encoding metric depth -> inv_u16 (matches the
            # NPZ pipeline's encoding so the model sees the same color mapping).
            inv_near = 1.0 / float(near_m)
            inv_far = 1.0 / float(far_m)
            inv_d = np.where(depth_m > 0, 1.0 / np.maximum(depth_m, 1e-6), inv_far)
            normalized = (inv_d - inv_far) / (inv_near - inv_far)
            normalized = np.clip(normalized, 0.0, 1.0)
            inv_u16 = np.where(
                valid_mask > 0,
                (normalized * 65534.0 + 1.0).astype(np.uint16),
                np.uint16(0),
            )
            rgb = depth_frame_to_turbo_rgb(
                inv_u16,
                valid_mask,
                near_m=float(near_m),
                far_m=float(far_m),
            )
            # First-frame repeat semantics.
            stack = np.repeat(rgb[None, ...], int(num_frames), axis=0)  # (T, H, W, 3)
            tensor = torch.from_numpy(stack).float().permute(0, 3, 1, 2) / 255.0
            return (tensor - 0.5) / 0.5
        except WristPcdRenderError as exc:
            last_exc = exc
            continue
    return _black_tile()


def _pack_output(
    *,
    rendered_u8: np.ndarray,  # (T, H, W, 3) uint8
    wrist_gt_frame0: np.ndarray,  # (H_src, W_src, 3) uint8
    cell_hw: Tuple[int, int],
) -> torch.Tensor:
    """Overwrite frame 0 with wrist GT (resized) and convert to normalized
    (T, 3, H, W) float32 in [-1, 1]."""
    H_cell, W_cell = int(cell_hw[0]), int(cell_hw[1])
    resized_gt = cv2.resize(
        wrist_gt_frame0, (W_cell, H_cell), interpolation=cv2.INTER_AREA
    )
    rendered_u8 = rendered_u8.copy()
    rendered_u8[0] = resized_gt
    tensor = torch.from_numpy(rendered_u8).float().permute(0, 3, 1, 2) / 255.0
    return (tensor - 0.5) / 0.5


def _splat_pcd_depth_zbuf_batched(
    *,
    points_base: np.ndarray,  # (N, 3) float32
    wrist_base2cam_seq: np.ndarray,  # (T, 4, 4)
    K_target: np.ndarray,  # (3, 3) at (cell_h, cell_w)
    cell_hw: Tuple[int, int],
    point_radius: float,
    points_per_pixel: int,
    device: torch.device,
    chunk: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched per-frame PCD depth splatting (the depth analogue of
    `_render_from_pcd`). Returns:
      depth_m: (T, H, W) float32 — z-depth (m) of the closest splat per pixel
      valid:   (T, H, W) uint8  — 1 where any splat hit, 0 otherwise
    """
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRasterizer,
    )
    from pytorch3d.structures import Pointclouds
    from pytorch3d.utils import cameras_from_opencv_projection

    H, W = int(cell_hw[0]), int(cell_hw[1])
    T = int(wrist_base2cam_seq.shape[0])
    pts_t = torch.from_numpy(points_base).to(device)
    R_cv = torch.from_numpy(wrist_base2cam_seq[:, :3, :3]).float()
    t_cv = torch.from_numpy(wrist_base2cam_seq[:, :3, 3]).float()
    K_batch = torch.from_numpy(
        np.broadcast_to(K_target.astype(np.float32), (T, 3, 3)).copy()
    ).float()
    image_size = torch.tensor([[H, W]], dtype=torch.int64).expand(T, -1)
    raster = PointsRasterizationSettings(
        image_size=(H, W),
        radius=float(point_radius),
        points_per_pixel=int(points_per_pixel),
    )

    depth_out = np.empty((T, H, W), dtype=np.float32)
    valid_out = np.empty((T, H, W), dtype=np.uint8)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        cams = cameras_from_opencv_projection(
            R=R_cv[s:e].to(device),
            tvec=t_cv[s:e].to(device),
            camera_matrix=K_batch[s:e].to(device),
            image_size=image_size[s:e].to(device),
        )
        rasterizer = PointsRasterizer(cameras=cams, raster_settings=raster)
        pc = Pointclouds(points=[pts_t] * (e - s))
        fragments = rasterizer(pc)  # zbuf: (B, H, W, K)
        zbuf = fragments.zbuf[:, :, :, 0].detach().cpu().numpy()  # (B, H, W)
        valid_chunk = (zbuf > 0.0).astype(np.uint8)
        depth_chunk = np.where(valid_chunk > 0, zbuf, 0.0).astype(np.float32)
        depth_out[s:e] = depth_chunk
        valid_out[s:e] = valid_chunk
    return depth_out, valid_out


def render_wrist_cell_dynamic_disparity(
    *,
    calibration: _EpisodeCalibration,
    clip_dir: Path,
    timestamp: str,
    start_idx: int,
    num_frames: int,
    frame_skip: int,
    cell_hw: Tuple[int, int],
    preferred_ext_suffix: str,
    near_m: float,
    far_m: float,
    ext_depth_subdir: str = "stereo_inverse_depth",
    robot_mesh_depth_subdir: str = "robot_mesh_inverse_depth",
    videos_subdir: str = "videos",
    point_radius: float = 0.012,
    points_per_pixel: int = 8,
    unproject_stride: int = 1,
    ext_max_depth_m: float = 1.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Wrist scene-depth observed through the wrist trajectory.

    Same colored point cloud as `render_wrist_cell_static` (ext first-frame
    RGB+depth -> base) but the output channel is *per-frame* depth (zbuf from
    pytorch3d), turbo-colormapped, rendered at every wrist pose in the clip.
    Companion to the RGB renderer: same scene, dynamic observer, dynamic
    depth — not first-frame-repeat.

    Returns (T, 3, H_cell, W_cell) float32 in [-1, 1]. On any failure falls
    back to an all-black tile (matches `render_wrist_cell_static_disparity`).
    """
    from data_processing.droid.depth_colormap import depth_frame_to_turbo_rgb

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise WristPcdRenderError("CUDA not available; cannot render wrist pcd depth")

    H_cell, W_cell = int(cell_hw[0]), int(cell_hw[1])

    def _black_tile() -> torch.Tensor:
        return torch.full(
            (int(num_frames), 3, H_cell, W_cell),
            float((0.0 / 255.0 - 0.5) / 0.5),
            dtype=torch.float32,
        )

    # Candidate ext suffixes: preferred first, else any other with calibration.
    candidate_suffixes: List[str] = []
    if preferred_ext_suffix in calibration.ext_cam2base:
        candidate_suffixes.append(preferred_ext_suffix)
    for alt in ("ext1", "ext2"):
        if alt == preferred_ext_suffix:
            continue
        if alt in calibration.ext_cam2base:
            candidate_suffixes.append(alt)
    if not candidate_suffixes:
        return _black_tile()

    # Wrist intrinsics — same source as RGB renderer.
    wrist_depth_npz_path = clip_dir / ext_depth_subdir / f"{timestamp}__wrist.npz"
    if not wrist_depth_npz_path.exists():
        return _black_tile()
    z_w = np.load(wrist_depth_npz_path)
    if "intrinsics" not in z_w.files or "source_size" not in z_w.files:
        return _black_tile()
    K_wrist_src = z_w["intrinsics"].astype(np.float64)
    src_hw_wrist = (int(z_w["source_size"][1]), int(z_w["source_size"][0]))
    K_target = _scale_intrinsics(K_wrist_src, src_hw_wrist, (H_cell, W_cell))

    # Per-frame wrist base->cam for the training clip (same trajectory math as
    # the RGB renderer).
    traj_idx = calibration.trajectory_indices
    frame_skip = max(1, int(frame_skip))
    target_export_indices = np.asarray(
        [int(start_idx) + i * frame_skip for i in range(int(num_frames))],
        dtype=np.int64,
    )
    if traj_idx.size == 0 or target_export_indices.max() >= traj_idx.size:
        return _black_tile()
    wrist_c2b = calibration.wrist_cam2base[traj_idx[target_export_indices]]
    wrist_b2c = np.linalg.inv(wrist_c2b)

    last_exc: Optional[Exception] = None
    for ext_suffix in candidate_suffixes:
        try:
            pts_base, _cols = _unproject_ext_first_frame(
                clip_dir=clip_dir,
                timestamp=timestamp,
                ext_suffix=ext_suffix,
                cam2base=calibration.ext_cam2base[ext_suffix],
                start_idx_export=int(start_idx),
                ext_depth_subdir=ext_depth_subdir,
                robot_mesh_depth_subdir=robot_mesh_depth_subdir,
                videos_subdir=videos_subdir,
                unproject_stride=int(unproject_stride),
                max_depth_m=float(ext_max_depth_m),
            )
            depth_m_seq, valid_seq = _splat_pcd_depth_zbuf_batched(
                points_base=pts_base,
                wrist_base2cam_seq=wrist_b2c,
                K_target=K_target,
                cell_hw=(H_cell, W_cell),
                point_radius=float(point_radius),
                points_per_pixel=int(points_per_pixel),
                device=device,
            )
            # Turbo-encode each frame independently (matches the per-frame
            # encoding used by bake_robot_depth_mp4.py).
            inv_near = 1.0 / float(near_m)
            inv_far = 1.0 / float(far_m)
            T = int(num_frames)
            stack = np.empty((T, H_cell, W_cell, 3), dtype=np.uint8)
            for t in range(T):
                depth_m = depth_m_seq[t]
                valid_mask = valid_seq[t]
                inv_d = np.where(depth_m > 0, 1.0 / np.maximum(depth_m, 1e-6), inv_far)
                normalized = (inv_d - inv_far) / (inv_near - inv_far)
                normalized = np.clip(normalized, 0.0, 1.0)
                inv_u16 = np.where(
                    valid_mask > 0,
                    (normalized * 65534.0 + 1.0).astype(np.uint16),
                    np.uint16(0),
                )
                stack[t] = depth_frame_to_turbo_rgb(
                    inv_u16,
                    valid_mask,
                    near_m=float(near_m),
                    far_m=float(far_m),
                    invalid_color=(0, 0, 0),
                )
            tensor = torch.from_numpy(stack).float().permute(0, 3, 1, 2) / 255.0
            return (tensor - 0.5) / 0.5
        except WristPcdRenderError as exc:
            last_exc = exc
            continue
    return _black_tile()
