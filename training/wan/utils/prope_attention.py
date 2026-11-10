# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
# MIT License — adapted from HY-WorldPlay's hyvideo/prope/camera_rope.py,
# which in turn derives from the PRoPE paper and reference repo:
#   "Cameras as Relative Positional Encoding" (arXiv:2507.10496, NeurIPS 2025)
#   Reference: third_party/prope/prope/torch.py
#
# This module keeps the original projection-only experiment available, but the
# default path used by new runs should be the official-style split variant:
# normalized intrinsics, projective transform on half the head dim, and spatial
# RoPE on the remaining x/y quarters.
"""PRoPE (Projective Positional Encoding) for multi-view self-attention."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


__all__ = [
    "prope_qkv",
    "prope_dot_product_attention",
]


def prope_qkv(
    q: torch.Tensor,  # [batch, num_heads, seqlen, head_dim]
    k: torch.Tensor,  # [batch, num_heads, seqlen, head_dim]
    v: torch.Tensor,  # [batch, num_heads, seqlen, head_dim]
    *,
    viewmats: torch.Tensor,  # [batch, cameras, 4, 4] world->camera (or c2w, consistent across call)
    Ks: Optional[torch.Tensor],  # [batch, cameras, 3, 3]
    view_token_offsets: Optional[torch.Tensor] = None,  # [cameras+1] cumulative token counts
    view_grid_sizes: Optional[torch.Tensor] = None,  # [views, 3] as (F, H_tok, W_tok)
    image_sizes: Optional[torch.Tensor] = None,  # [batch, views|cameras, 2] as (H_px, W_px)
    variant: str = "projection_only",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Apply PRoPE-style per-camera projection to Q/K/V.

    Returns (q', k', v', apply_fn_o). The caller is expected to run
    scaled_dot_product_attention on (q', k', v') and then call apply_fn_o on the
    output to undo the Q-side transform.

    If ``view_token_offsets`` is ``None`` (default) the helper assumes
    ``seqlen == cameras * (seqlen // cameras)`` with a uniform per-camera block
    size, matching the stock reference implementation. When views have different
    token counts (heterogeneous resolutions), pass an explicit
    ``view_token_offsets`` tensor with length ``cameras + 1``; per-camera blocks
    are then sliced using those offsets.
    """
    del kwargs
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        view_token_offsets=view_token_offsets,
        view_grid_sizes=view_grid_sizes,
        image_sizes=image_sizes,
        variant=variant,
    )

    query = apply_fn_q(q)
    key = apply_fn_kv(k)
    value = apply_fn_kv(v)
    return query, key, value, apply_fn_o


def prope_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
    view_token_offsets: Optional[torch.Tensor] = None,
    view_grid_sizes: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    variant: str = "projection_only",
    **attention_kwargs,
) -> torch.Tensor:
    """Drop-in wrapper that runs scaled_dot_product_attention with PRoPE applied."""
    q_p, k_p, v_p, apply_fn_o = prope_qkv(
        q,
        k,
        v,
        viewmats=viewmats,
        Ks=Ks,
        view_token_offsets=view_token_offsets,
        view_grid_sizes=view_grid_sizes,
        image_sizes=image_sizes,
        variant=variant,
    )
    out = F.scaled_dot_product_attention(q_p, k_p, v_p, **attention_kwargs)
    return apply_fn_o(out)


def _prepare_apply_fns_all_dim(
    head_dim: int,
    viewmats: torch.Tensor,  # [batch, cameras, 4, 4]
    Ks: Optional[torch.Tensor],  # [batch, cameras, 3, 3]
    view_token_offsets: Optional[torch.Tensor] = None,
    view_grid_sizes: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    variant: str = "projection_only",
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Compose block-diagonal projection transforms for Q / K / V / O."""
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for PRoPE; got {head_dim}")

    variant = str(variant).strip().lower()
    if variant in {"official", "official_split", "normalized_split"}:
        return _prepare_apply_fns_official_split(
            head_dim=head_dim,
            viewmats=viewmats,
            Ks=Ks,
            view_token_offsets=view_token_offsets,
            view_grid_sizes=view_grid_sizes,
            image_sizes=image_sizes,
        )
    if variant not in {"projection_only", "raw_projection_only", "legacy"}:
        raise ValueError(f"Unsupported PRoPE variant: {variant}")

    # Legacy projection-only path: raw pixel-unit intrinsics and no spatial RoPE split.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)

        # P = image<-world = lift(K) @ viewmats
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)
    else:
        # GTA / pose-only formula (no intrinsics).
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    cameras = viewmats.shape[1]
    batch = viewmats.shape[0]
    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    if view_token_offsets is None:
        apply_q = partial(_apply_tiled_projmat_uniform, matrix=P_T)
        apply_kv = partial(_apply_tiled_projmat_uniform, matrix=P_inv)
        apply_o = partial(_apply_tiled_projmat_uniform, matrix=P)
    else:
        if view_token_offsets.shape[0] != cameras + 1:
            raise ValueError(
                f"view_token_offsets length must be cameras+1={cameras + 1}, "
                f"got {tuple(view_token_offsets.shape)}"
            )
        apply_q = partial(_apply_tiled_projmat_hetero, matrix=P_T, offsets=view_token_offsets)
        apply_kv = partial(_apply_tiled_projmat_hetero, matrix=P_inv, offsets=view_token_offsets)
        apply_o = partial(_apply_tiled_projmat_hetero, matrix=P, offsets=view_token_offsets)

    return apply_q, apply_kv, apply_o


def _prepare_apply_fns_official_split(
    *,
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
    view_token_offsets: Optional[torch.Tensor],
    view_grid_sizes: Optional[torch.Tensor],
    image_sizes: Optional[torch.Tensor],
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Official-style PRoPE: normalized K, projective half, spatial RoPE quarters."""
    if view_token_offsets is None:
        raise ValueError("official_split PRoPE requires explicit view_token_offsets.")
    if view_grid_sizes is None:
        raise ValueError("official_split PRoPE requires view_grid_sizes.")
    if head_dim % 8 != 0:
        raise ValueError(
            f"official_split PRoPE requires head_dim divisible by 8; got {head_dim}"
        )

    view_grid_sizes = view_grid_sizes.to(device=viewmats.device, dtype=torch.int64)
    cameras = int(viewmats.shape[1])
    batch = int(viewmats.shape[0])

    if Ks is not None:
        if image_sizes is None:
            image_sizes = _derive_image_sizes_from_grid(view_grid_sizes, batch, viewmats.device)
        camera_image_sizes = _expand_image_sizes_to_cameras(
            image_sizes=image_sizes,
            view_grid_sizes=view_grid_sizes,
            cameras=cameras,
            batch=batch,
            device=viewmats.device,
            dtype=Ks.dtype,
        )
        Ks_norm = _normalize_intrinsics(Ks, camera_image_sizes)
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)
    else:
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    if int(view_token_offsets[-1].item()) <= 0:
        raise ValueError("view_token_offsets must contain positive token counts.")

    transforms_q = [
        (partial(_apply_tiled_projmat_hetero, matrix=P_T, offsets=view_token_offsets), head_dim // 2),
        (partial(_rope_apply_hetero_spatial, view_grid_sizes=view_grid_sizes, offsets=view_token_offsets, axis="x"), head_dim // 4),
        (partial(_rope_apply_hetero_spatial, view_grid_sizes=view_grid_sizes, offsets=view_token_offsets, axis="y"), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat_hetero, matrix=P_inv, offsets=view_token_offsets), head_dim // 2),
        (partial(_rope_apply_hetero_spatial, view_grid_sizes=view_grid_sizes, offsets=view_token_offsets, axis="x"), head_dim // 4),
        (partial(_rope_apply_hetero_spatial, view_grid_sizes=view_grid_sizes, offsets=view_token_offsets, axis="y"), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat_hetero, matrix=P, offsets=view_token_offsets), head_dim // 2),
        (partial(_rope_apply_hetero_spatial, view_grid_sizes=view_grid_sizes, offsets=view_token_offsets, axis="x", inverse=True), head_dim // 4),
        (partial(_rope_apply_hetero_spatial, view_grid_sizes=view_grid_sizes, offsets=view_token_offsets, axis="y", inverse=True), head_dim // 4),
    ]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )


def _apply_tiled_projmat_uniform(
    feats: torch.Tensor,  # [batch, num_heads, seqlen, feat_dim]
    matrix: torch.Tensor,  # [batch, cameras, D, D]
) -> torch.Tensor:
    """Uniform per-camera block size: seqlen must be divisible by cameras."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    if seqlen % cameras != 0:
        raise ValueError(
            "Uniform PRoPE path requires seqlen % cameras == 0. "
            "Pass explicit view_token_offsets for heterogeneous views."
        )
    if feat_dim % D != 0:
        raise ValueError(f"feat_dim={feat_dim} must be divisible by D={D} for PRoPE tiling.")
    # Caller is expected to wrap the PRoPE branch in autocast(enabled=False) and pass
    # fp32 viewmats/Ks (see FullSelfAttentionWithPrope.forward), so the einsum already
    # runs in fp32 in practice. The .to(feats.dtype) is a one-liner safety net for
    # callers that pass mixed-dtype inputs — torch.einsum requires both operands to
    # share dtype.
    matrix = matrix.to(feats.dtype)
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape(batch, num_heads, cameras, seqlen // cameras, feat_dim // D, D),
    ).reshape(feats.shape)


def _apply_tiled_projmat_hetero(
    feats: torch.Tensor,  # [batch, num_heads, seqlen, feat_dim]
    *,
    matrix: torch.Tensor,  # [batch, cameras, D, D]
    offsets: torch.Tensor,  # [cameras+1] cumulative token counts
) -> torch.Tensor:
    """Heterogeneous per-camera block size: each camera's slice is ``offsets[i] .. offsets[i+1]``."""
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    if offsets.shape[0] != cameras + 1:
        raise ValueError("offsets length must equal cameras + 1")
    if int(offsets[-1].item()) != seqlen:
        raise ValueError(
            f"offsets[-1]={int(offsets[-1].item())} must equal seqlen={seqlen}"
        )
    if feat_dim % D != 0:
        raise ValueError(f"feat_dim={feat_dim} must be divisible by D={D} for PRoPE tiling.")

    # Same dtype-alignment rationale as ``_apply_tiled_projmat_uniform`` above.
    matrix = matrix.to(feats.dtype)
    chunks: List[torch.Tensor] = []
    for cam in range(cameras):
        start = int(offsets[cam].item())
        end = int(offsets[cam + 1].item())
        block = feats[:, :, start:end, :]  # [B, H, Tc, feat_dim]
        block_reshaped = block.reshape(batch, num_heads, end - start, feat_dim // D, D)
        transformed = torch.einsum(
            "bij,bnpkj->bnpki",
            matrix[:, cam],
            block_reshaped,
        ).reshape(batch, num_heads, end - start, feat_dim)
        chunks.append(transformed)
    return torch.cat(chunks, dim=2)


def _apply_block_diagonal(
    feats: torch.Tensor,
    *,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply independent transforms to contiguous feature chunks."""
    funcs, block_sizes = zip(*func_size_pairs)
    if feats.shape[-1] != sum(block_sizes):
        raise ValueError(
            f"Feature dim {feats.shape[-1]} does not match PRoPE block sizes {block_sizes}"
        )
    chunks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat([fn(chunk) for fn, chunk in zip(funcs, chunks)], dim=-1)
    if out.shape != feats.shape:
        raise RuntimeError(f"PRoPE block-diagonal output shape {out.shape} != input {feats.shape}")
    return out


def _rope_apply_hetero_spatial(
    feats: torch.Tensor,  # [batch, heads, seqlen, feat_dim]
    *,
    view_grid_sizes: torch.Tensor,  # [views, 3] as (F, H_tok, W_tok)
    offsets: torch.Tensor,
    axis: str,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply x/y RoPE per camera block for heterogeneous per-view token grids."""
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")
    batch, num_heads, seqlen, feat_dim = feats.shape
    del batch, num_heads
    if feat_dim % 2 != 0:
        raise ValueError(f"RoPE feature dim must be even; got {feat_dim}")
    if int(offsets[-1].item()) != seqlen:
        raise ValueError(f"offsets[-1]={int(offsets[-1].item())} must equal seqlen={seqlen}")

    chunks: List[torch.Tensor] = []
    cam_idx = 0
    for f_tok, h_tok, w_tok in view_grid_sizes.to(torch.int64).tolist():
        for _ in range(int(f_tok)):
            start = int(offsets[cam_idx].item())
            end = int(offsets[cam_idx + 1].item())
            h_i, w_i = int(h_tok), int(w_tok)
            expected = h_i * w_i
            if end - start != expected:
                raise ValueError(
                    f"camera {cam_idx} token count {end - start} != grid {h_i}*{w_i}"
                )
            block = feats[:, :, start:end, :]
            if axis == "x":
                positions = torch.tile(torch.arange(w_i, device=feats.device), (h_i,))
            else:
                positions = torch.repeat_interleave(torch.arange(h_i, device=feats.device), w_i)
            coeffs = _rope_precompute_coeffs(
                positions,
                freq_base=100.0,
                freq_scale=1.0,
                feat_dim=feat_dim,
            )
            chunks.append(_rope_apply_coeffs(block, coeffs=coeffs, inverse=inverse))
            cam_idx += 1
    if cam_idx + 1 != offsets.shape[0]:
        raise ValueError(
            f"view_grid_sizes imply {cam_idx} cameras, offsets imply {offsets.shape[0] - 1}"
        )
    return torch.cat(chunks, dim=2)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    *,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if positions.dim() != 1:
        raise ValueError("positions must be a 1D tensor")
    if feat_dim % 2 != 0:
        raise ValueError(f"RoPE feat_dim must be even; got {feat_dim}")
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device, dtype=torch.float32) / num_freqs)
    )
    angles = positions.to(torch.float32)[None, None, :, None] * freqs[None, None, None, :]
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    *,
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    cos, sin = coeffs
    cos = cos.to(device=feats.device, dtype=feats.dtype)
    sin = sin.to(device=feats.device, dtype=feats.dtype)
    if cos.shape[2] != feats.shape[2]:
        if feats.shape[2] % cos.shape[2] != 0:
            raise ValueError("RoPE coeff sequence length must divide feature sequence length")
        repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, repeats, 1)
        sin = sin.repeat(1, 1, repeats, 1)
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    if inverse:
        return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)
    return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)


def _derive_image_sizes_from_grid(
    view_grid_sizes: torch.Tensor,
    batch: int,
    device: torch.device,
    token_pixel_size: int = 16,
) -> torch.Tensor:
    """Fallback image sizes from token grid; Wan VAE stride 8 * patch size 2 = 16 px."""
    hw = view_grid_sizes[:, 1:3].to(device=device, dtype=torch.float32) * float(token_pixel_size)
    return hw.view(1, hw.shape[0], 2).expand(batch, -1, -1).contiguous()


def _expand_image_sizes_to_cameras(
    *,
    image_sizes: torch.Tensor,
    view_grid_sizes: torch.Tensor,
    cameras: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_sizes = image_sizes.to(device=device, dtype=dtype)
    if image_sizes.dim() == 2:
        image_sizes = image_sizes.view(1, image_sizes.shape[0], 2).expand(batch, -1, -1)
    if image_sizes.shape[0] != batch or image_sizes.shape[-1] != 2:
        raise ValueError(
            f"image_sizes must be [B, views|cameras, 2] or [views|cameras, 2], got {tuple(image_sizes.shape)}"
        )
    if image_sizes.shape[1] == cameras:
        return image_sizes
    if image_sizes.shape[1] != view_grid_sizes.shape[0]:
        raise ValueError(
            f"image_sizes second dim must be views={view_grid_sizes.shape[0]} or cameras={cameras}, "
            f"got {image_sizes.shape[1]}"
        )
    per_camera: List[torch.Tensor] = []
    for view_idx, (f_tok, _h_tok, _w_tok) in enumerate(view_grid_sizes.to(torch.int64).tolist()):
        size_v = image_sizes[:, view_idx:view_idx + 1, :]
        per_camera.append(size_v.expand(-1, int(f_tok), -1))
    out = torch.cat(per_camera, dim=1)
    if out.shape[1] != cameras:
        raise ValueError(f"expanded image_sizes has {out.shape[1]} cameras, expected {cameras}")
    return out


def _normalize_intrinsics(Ks: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    """Normalize pixel-space K into official PRoPE image-coordinate convention."""
    if Ks.shape[:-2] != image_sizes.shape[:-1]:
        raise ValueError(f"Ks shape {tuple(Ks.shape)} and image_sizes {tuple(image_sizes.shape)} mismatch")
    height = image_sizes[..., 0].clamp_min(1.0)
    width = image_sizes[..., 1].clamp_min(1.0)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = Ks[..., 0, 0] / width
    out[..., 1, 1] = Ks[..., 1, 1] / height
    out[..., 0, 2] = Ks[..., 0, 2] / width - 0.5
    out[..., 1, 2] = Ks[..., 1, 2] / height - 0.5
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift a 3x3 intrinsics matrix into a 4x4 projection block."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out.to(dtype=Ks.dtype)


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics (no skew)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)
