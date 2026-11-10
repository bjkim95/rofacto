# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
from typing import Dict, Optional

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.utils import is_torch_version

from ..utils import cfg_skip
from .wan_transformer3d import (
    WanAttentionBlock,
    WanSelfAttention,
    rope_apply,
    sinusoidal_embedding_1d,
)
from .wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat

try:
    import flash_attn

    FLASH_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn = None
    FLASH_ATTN_AVAILABLE = False


def _require_flash_attn() -> None:
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError(
            "flash_attn is required for this fused attention path."
        )


def _flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_flash_attn()

    out, softmax_lse, _ = flash_attn.flash_attn_varlen_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        return_attn_probs=True,
    )
    return out, softmax_lse


def _compute_view_token_offsets(view_grid_sizes: torch.Tensor) -> torch.Tensor:
    """Return cumulative token offsets of shape ``[B, V+1]``."""
    token_counts = view_grid_sizes.prod(dim=-1)  # [B, V]
    zeros = torch.zeros(token_counts.shape[0], 1, dtype=token_counts.dtype, device=token_counts.device)
    return torch.cat([zeros, token_counts.cumsum(dim=-1)], dim=-1)


def _compute_view_x_offsets(view_grid_sizes: torch.Tensor) -> torch.Tensor:
    """Return cumulative per-view RoPE x offsets of shape ``[B, V]``."""
    widths = view_grid_sizes[..., 2].to(torch.long)
    zeros = torch.zeros(widths.shape[0], 1, dtype=widths.dtype, device=widths.device)
    return torch.cat([zeros, widths.cumsum(dim=-1)[:, :-1]], dim=-1)


def _rope_apply_with_x_offsets(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan 3D RoPE while shifting the width-axis phase per sample.

    This keeps per-view latent tensors separate, but gives later views the same
    width-axis RoPE phase they would have had in a physical width-concat canvas.
    """
    num_heads, complex_dim = x.size(2), x.size(3) // 2
    freqs_t, freqs_h, freqs_w = freqs.split(
        [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
        dim=1,
    )

    output = []
    for sample_idx, ((frames, height, width), x_offset) in enumerate(
        zip(grid_sizes.tolist(), x_offsets.to(torch.long).tolist())
    ):
        frames, height, width, x_offset = int(frames), int(height), int(width), int(x_offset)
        seq_len = frames * height * width
        if x_offset < 0 or x_offset + width > freqs_w.shape[0]:
            raise ValueError(
                f"RoPE x-offset range [{x_offset}, {x_offset + width}) exceeds "
                f"available width freqs ({freqs_w.shape[0]})."
            )

        x_i = torch.view_as_complex(
            x[sample_idx, :seq_len].to(torch.float32).reshape(seq_len, num_heads, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs_t[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                freqs_h[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freqs_w[x_offset:x_offset + width].view(1, 1, width, -1).expand(
                    frames, height, width, -1
                ),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[sample_idx, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def _run_multiview_standard_attention(
    module: WanSelfAttention,
    *,
    x: torch.Tensor,
    view_grid_sizes: torch.Tensor,
    total_seq_lens: torch.Tensor,
    freqs: torch.Tensor,
    dtype: torch.dtype,
    rope_x_offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Full self-attention over view-concatenated tokens with per-view 3D RoPE."""
    use_flash = (
        FLASH_ATTN_AVAILABLE
        and x.is_cuda
        and dtype in (torch.float16, torch.bfloat16)
    )
    batch_size, seq_max, _ = x.shape
    num_views = int(view_grid_sizes.shape[1])
    if not torch.equal(total_seq_lens[:1].expand_as(total_seq_lens), total_seq_lens):
        raise ValueError("multiview full attention requires identical total_seq_len across batch.")
    total_seq = int(total_seq_lens[0].item())
    if seq_max < total_seq:
        raise ValueError(f"x seq_len {seq_max} < required total {total_seq}")

    x_valid = x[:, :total_seq].to(dtype)
    q = module.norm_q(module.q(x_valid))
    k = module.norm_k(module.k(x_valid))
    v = module.v(x_valid)

    token_offsets = _compute_view_token_offsets(view_grid_sizes)
    q_rope = q.clone()
    k_rope = k.clone()
    if rope_x_offsets is not None:
        if tuple(rope_x_offsets.shape) != tuple(view_grid_sizes.shape[:2]):
            raise ValueError(
                f"rope_x_offsets must be [B,V], got {tuple(rope_x_offsets.shape)}"
            )
        rope_x_offsets = rope_x_offsets.to(device=view_grid_sizes.device, dtype=torch.long)

    for view_idx in range(num_views):
        view_grid = view_grid_sizes[:, view_idx].to(torch.int64)  # [B, 3]
        per_view_tokens = int(view_grid.prod(dim=-1)[0].item())
        for batch_idx in range(batch_size):
            if int(view_grid[batch_idx].prod().item()) != per_view_tokens:
                raise ValueError("multiview full attention requires identical per-view grid across batch.")

        start = int(token_offsets[0, view_idx].item())
        end = int(token_offsets[0, view_idx + 1].item())
        q_slice = q[:, start:end].view(batch_size, per_view_tokens, module.num_heads, module.head_dim)
        k_slice = k[:, start:end].view(batch_size, per_view_tokens, module.num_heads, module.head_dim)
        if rope_x_offsets is None:
            q_roped = rope_apply(q_slice, view_grid, freqs).view(batch_size, per_view_tokens, module.dim)
            k_roped = rope_apply(k_slice, view_grid, freqs).view(batch_size, per_view_tokens, module.dim)
        else:
            view_x_offsets = rope_x_offsets[:, view_idx]
            q_roped = _rope_apply_with_x_offsets(q_slice, view_grid, freqs, view_x_offsets).view(
                batch_size, per_view_tokens, module.dim
            )
            k_roped = _rope_apply_with_x_offsets(k_slice, view_grid, freqs, view_x_offsets).view(
                batch_size, per_view_tokens, module.dim
            )
        q_rope[:, start:end] = q_roped.to(q_rope.dtype)
        k_rope[:, start:end] = k_roped.to(k_rope.dtype)

    q_std = q_rope.view(batch_size, total_seq, module.num_heads, module.head_dim)
    k_std = k_rope.view(batch_size, total_seq, module.num_heads, module.head_dim)
    v_std = v.view(batch_size, total_seq, module.num_heads, module.head_dim)

    if use_flash:
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * total_seq,
            total_seq,
            device=q_std.device,
            dtype=torch.int32,
        )
        std_out, _ = _flash_attn_with_lse(
            q_std.reshape(batch_size * total_seq, module.num_heads, module.head_dim),
            k_std.reshape(batch_size * total_seq, module.num_heads, module.head_dim),
            v_std.reshape(batch_size * total_seq, module.num_heads, module.head_dim),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=total_seq,
            max_seqlen_k=total_seq,
        )
        std_out = std_out.view(batch_size, total_seq, module.dim)
    else:
        q_sdpa = q_std.permute(0, 2, 1, 3).contiguous()
        k_sdpa = k_std.permute(0, 2, 1, 3).contiguous()
        v_sdpa = v_std.permute(0, 2, 1, 3).contiguous()
        std_out = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        std_out = std_out.permute(0, 2, 1, 3).contiguous().view(batch_size, total_seq, module.dim)

    return std_out, q, k, v, batch_size, seq_max, total_seq


def _pad_multiview_attention_output(
    x: torch.Tensor,
    *,
    seq_max: int,
    total_seq: int,
    dim: int,
) -> torch.Tensor:
    if x.shape[1] == seq_max:
        return x
    padded = x.new_zeros(x.shape[0], seq_max, dim)
    padded[:, :total_seq] = x
    return padded


class FullSelfAttentionWithRopeOffset(WanSelfAttention):
    """Full multi-view self-attention with width-concat-style RoPE x offsets."""

    def forward(
        self,
        x: torch.Tensor,
        view_grid_sizes: torch.Tensor,
        total_seq_lens: torch.Tensor,
        freqs: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        std_out, _, _, _, _, seq_max, total_seq = _run_multiview_standard_attention(
            self,
            x=x,
            view_grid_sizes=view_grid_sizes,
            total_seq_lens=total_seq_lens,
            freqs=freqs,
            dtype=dtype,
            rope_x_offsets=_compute_view_x_offsets(view_grid_sizes),
        )
        combined = self.o(std_out)
        return _pad_multiview_attention_output(
            combined,
            seq_max=seq_max,
            total_seq=total_seq,
            dim=self.dim,
        )


class FullSelfAttentionWithPrope(WanSelfAttention):
    """Full (N-view, heterogeneous grid) self-attention with a parallel PRoPE branch.

    Inherits q/k/v/o/norm_q/norm_k from :class:`WanSelfAttention` so that loading
    state dicts from the base block is cheap (we don't allocate a duplicate set
    of Linears). The only new parameter is ``o_prope`` which is zero-initialised.

    Layout expectations:
      * ``x`` has shape ``[B, total_seq, dim]`` where ``total_seq`` is the sum of
        per-view token counts across all views. Token order is view-major, then
        standard (F, H, W) order within each view.
      * ``view_grid_sizes`` has shape ``[B, V, 3]`` and gives ``(F, H_tok, W_tok)``
        per view. Different views may have different grid sizes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        prope_attention_variant: str = "projection_only",
    ):
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.o_prope = nn.Linear(dim, dim)
        self.prope_attention_variant = str(prope_attention_variant).strip().lower()
        # Zero-init so initial PRoPE contribution is exactly 0; preserves base behaviour.
        nn.init.zeros_(self.o_prope.weight)
        if self.o_prope.bias is not None:
            nn.init.zeros_(self.o_prope.bias)

    @staticmethod
    def _compute_offsets(view_grid_sizes: torch.Tensor) -> torch.Tensor:
        """Return cumulative token counts of shape ``[B, V+1]``."""
        token_counts = view_grid_sizes.prod(dim=-1)  # [B, V]
        zeros = torch.zeros(token_counts.shape[0], 1, dtype=token_counts.dtype, device=token_counts.device)
        cumulative = torch.cat([zeros, token_counts.cumsum(dim=-1)], dim=-1)
        return cumulative  # [B, V+1]

    def forward(
        self,
        x: torch.Tensor,               # [B, T_max, dim] (possibly padded)
        view_grid_sizes: torch.Tensor,  # [B, V, 3]
        total_seq_lens: torch.Tensor,   # [B]
        freqs: torch.Tensor,
        viewmats: torch.Tensor,         # [B, V*F, 4, 4]
        Ks: torch.Tensor,               # [B, V*F, 3, 3]
        image_sizes: Optional[torch.Tensor],  # [B, V, 2] or [B, V*F, 2] as (H, W)
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # flash_attn is CUDA-only + fp16/bf16-only. Fall back to SDPA on CPU or fp32 smoke.
        use_flash = (
            FLASH_ATTN_AVAILABLE
            and x.is_cuda
            and dtype in (torch.float16, torch.bfloat16)
        )
        batch_size, seq_max, _ = x.shape
        num_views = int(view_grid_sizes.shape[1])
        # Sanity: identical per-batch totals.
        if not torch.equal(total_seq_lens[:1].expand_as(total_seq_lens), total_seq_lens):
            raise ValueError("FullSelfAttentionWithPrope requires identical total_seq_len across batch.")
        total_seq = int(total_seq_lens[0].item())
        if seq_max < total_seq:
            raise ValueError(f"x seq_len {seq_max} < required total {total_seq}")

        x_valid = x[:, :total_seq].to(dtype)
        q = self.norm_q(self.q(x_valid))
        k = self.norm_k(self.k(x_valid))
        v = self.v(x_valid)

        # --- standard branch with per-view RoPE applied on the fly ---------------
        offsets = self._compute_offsets(view_grid_sizes)  # [B, V+1]
        # Per-sample, per-view grids must match across batch for a single rope_apply call,
        # but individual views can differ from each other. We call rope_apply per view with
        # grid_sizes [B, 3] carrying that view's (F, H, W). Apply to the view's slice of q/k.
        q_rope = q.clone()
        k_rope = k.clone()
        for v_idx in range(num_views):
            view_grid = view_grid_sizes[:, v_idx].to(torch.int64)  # [B, 3]
            per_view_tokens = int(view_grid.prod(dim=-1)[0].item())
            # Assume identical per-view grid across batch (same spec per config).
            for b in range(batch_size):
                if int(view_grid[b].prod().item()) != per_view_tokens:
                    raise ValueError(
                        "FullSelfAttentionWithPrope requires identical per-view grid across batch."
                    )
            start = int(offsets[0, v_idx].item())
            end = int(offsets[0, v_idx + 1].item())
            q_slice = q[:, start:end].view(
                batch_size, per_view_tokens, self.num_heads, self.head_dim
            )
            k_slice = k[:, start:end].view(
                batch_size, per_view_tokens, self.num_heads, self.head_dim
            )
            q_roped = rope_apply(q_slice, view_grid, freqs).view(batch_size, per_view_tokens, self.dim)
            k_roped = rope_apply(k_slice, view_grid, freqs).view(batch_size, per_view_tokens, self.dim)
            q_rope[:, start:end] = q_roped.to(q_rope.dtype)
            k_rope[:, start:end] = k_roped.to(k_rope.dtype)

        # Pack for flash_attn_varlen
        q_std = q_rope.view(batch_size, total_seq, self.num_heads, self.head_dim)
        k_std = k_rope.view(batch_size, total_seq, self.num_heads, self.head_dim)
        v_std = v.view(batch_size, total_seq, self.num_heads, self.head_dim)

        if use_flash:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * total_seq,
                total_seq,
                device=q_std.device,
                dtype=torch.int32,
            )
            q_flat = q_std.reshape(batch_size * total_seq, self.num_heads, self.head_dim)
            k_flat = k_std.reshape(batch_size * total_seq, self.num_heads, self.head_dim)
            v_flat = v_std.reshape(batch_size * total_seq, self.num_heads, self.head_dim)
            std_out, _ = _flash_attn_with_lse(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=total_seq,
                max_seqlen_k=total_seq,
            )
            std_out = std_out.view(batch_size, total_seq, self.dim)
        else:
            # SDPA expects [B, H, S, D]
            q_sdpa = q_std.permute(0, 2, 1, 3).contiguous()
            k_sdpa = k_std.permute(0, 2, 1, 3).contiguous()
            v_sdpa = v_std.permute(0, 2, 1, 3).contiguous()
            std_out = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            std_out = std_out.permute(0, 2, 1, 3).contiguous().view(batch_size, total_seq, self.dim)

        # --- PRoPE branch (parallel) --------------------------------------------
        # Keep the PRoPE branch in fp32. The legacy projection-only variant used raw
        # pixel-unit fx/fy and could multiply Q/K/V by O(10^2-10^3); the official_split
        # variant normalizes intrinsics and splits head_dim into projective + x/y RoPE
        # chunks, but still benefits from fp32 cancellation around P/P_inv.
        from ..utils.prope_attention import prope_qkv

        q_bhsd = q.view(batch_size, total_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().to(torch.float32)
        k_bhsd = k.view(batch_size, total_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().to(torch.float32)
        v_bhsd = v.view(batch_size, total_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().to(torch.float32)

        # Camera count per batch: V * F (= sum per-view frame counts).
        # For now assume cameras are flat across (view, frame). Token offsets per camera =
        # H_tok_i * W_tok_i (unrolled F times per view). We build a flat offsets tensor
        # that maps each camera to its contiguous token block.
        cam_offsets = self._build_per_camera_offsets(view_grid_sizes[0])  # [cameras+1]

        with torch.amp.autocast(device_type="cuda", enabled=False):
            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                viewmats=viewmats.to(torch.float32),
                Ks=Ks.to(torch.float32) if Ks is not None else None,
                view_token_offsets=cam_offsets,
                view_grid_sizes=view_grid_sizes[0],
                image_sizes=image_sizes.to(torch.float32) if image_sizes is not None else None,
                variant=getattr(self, "prope_attention_variant", "projection_only"),
            )

            # Run SDPA for PRoPE branch in fp32.
            prope_out = torch.nn.functional.scaled_dot_product_attention(q_p, k_p, v_p)
            prope_out = apply_fn_o(prope_out)
            prope_out = prope_out.permute(0, 2, 1, 3).contiguous().view(batch_size, total_seq, self.dim)

            # ``self.o_prope`` is a bf16 module under the outer autocast; explicitly
            # invoke it in fp32 here so the zero-init weight's first gradient is also
            # computed in fp32. We cast the result back to ``std_out`` dtype below.
            o_prope_w = self.o_prope.weight.to(torch.float32)
            o_prope_b = self.o_prope.bias.to(torch.float32) if self.o_prope.bias is not None else None
            prope_proj_f32 = torch.nn.functional.linear(prope_out, o_prope_w, o_prope_b)

        # NaN bypass: occasionally a sample's PRoPE projection produces non-finite
        # values (sample-specific P/P_inv conditioning issue at high-sigma timesteps).
        # If that happens, drop the PRoPE contribution for this forward — the std
        # attention path is finite and the loss-side guard will skip the optimizer
        # update for this step. Without this, NaN propagates into self.o(std_out) +
        # NaN and poisons the entire block output even though std_out itself is fine.
        prope_contrib = prope_proj_f32.to(std_out.dtype)
        if not torch.isfinite(prope_contrib).all():
            prope_contrib = torch.zeros_like(prope_contrib)
        combined = self.o(std_out) + prope_contrib
        if combined.shape[1] == seq_max:
            return combined
        padded = combined.new_zeros(batch_size, seq_max, self.dim)
        padded[:, :total_seq] = combined
        return padded

    @staticmethod
    def _build_per_camera_offsets(view_grid: torch.Tensor) -> torch.Tensor:
        """Given one sample's view_grid_sizes ``[V, 3]`` (F, H, W per view) build
        cumulative offsets ``[cameras+1]`` where ``cameras = sum_v F_v`` and the
        i-th camera covers that view's ``H_v * W_v`` tokens at frame ``(i mod F_v)``.
        Assumes the seq is packed ``[view0_frame0_HxW, view0_frame1_HxW, ..., view1_frame0_HxW, ...]``.
        """
        view_grid = view_grid.to(torch.int64).tolist()
        offsets = [0]
        for (f_v, h_v, w_v) in view_grid:
            per_frame_tokens = int(h_v) * int(w_v)
            for _ in range(int(f_v)):
                offsets.append(offsets[-1] + per_frame_tokens)
        return torch.tensor(offsets, dtype=torch.int64)


class MultiViewWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        cross_view_mode: str = "full_prope",
        prope_attention_variant: str = "projection_only",
    ):
        super().__init__(
            cross_attn_type=cross_attn_type,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.cross_view_mode = str(cross_view_mode).strip().lower()
        self.prope_attention_variant = str(prope_attention_variant).strip().lower()
        if self.cross_view_mode not in ("full_prope", "full_rope_offset"):
            raise ValueError(f"Unsupported cross_view_mode: {cross_view_mode}")

        # Promote the existing WanSelfAttention created by the parent __init__ into
        # the multiview self-attention in-place: this avoids allocating a duplicate
        # set of q/k/v/o/norm_q/norm_k Linears.
        if self.cross_view_mode == "full_prope":
            self.self_attn.__class__ = FullSelfAttentionWithPrope
            self.self_attn.prope_attention_variant = self.prope_attention_variant
            self.self_attn.o_prope = nn.Linear(dim, dim)
            nn.init.zeros_(self.self_attn.o_prope.weight)
            if self.self_attn.o_prope.bias is not None:
                nn.init.zeros_(self.self_attn.o_prope.bias)
        else:
            self.self_attn.__class__ = FullSelfAttentionWithRopeOffset

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.float32,
        t=0,
        multiview_batch_size: Optional[int] = None,
        num_views: Optional[int] = None,
        view_grid_sizes: Optional[torch.Tensor] = None,
        view_seq_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
    ):
        del grid_sizes, multiview_batch_size, num_views

        del view_seq_lens

        if self.cross_view_mode not in ("full_prope", "full_rope_offset"):
            raise ValueError(f"Unsupported cross_view_mode: {self.cross_view_mode}")
        if view_grid_sizes is None or total_seq_lens is None:
            raise ValueError("multiview block requires view_grid_sizes and total_seq_lens")
        if self.cross_view_mode == "full_prope" and (viewmats is None or Ks is None):
            raise ValueError("full_prope requires viewmats and Ks.")

        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        if self.cross_view_mode == "full_prope":
            y = self.self_attn(
                temp_x,
                view_grid_sizes=view_grid_sizes,
                total_seq_lens=total_seq_lens,
                freqs=freqs,
                viewmats=viewmats,
                Ks=Ks,
                image_sizes=image_sizes,
                dtype=dtype,
            )
        else:
            y = self.self_attn(
                temp_x,
                view_grid_sizes=view_grid_sizes,
                total_seq_lens=total_seq_lens,
                freqs=freqs,
                dtype=dtype,
            )
        x = x + y * e[2]

        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)

        temp_x = self.norm2(x) * (1 + e[4]) + e[3]
        temp_x = temp_x.to(dtype)
        y = self.ffn(temp_x)
        x = x + y * e[5]
        return x


class MultiViewWanTransformer3DModelWithConcat(WanTransformer3DModelWithConcat):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        fps: int = 16,
        condition_channels: int = 0,
        is_wan2_2: bool = False,
        multiview_mode: str = "full_prope",
        cross_view_num_heads: Optional[int] = None,
        prope_attention_variant: str = "projection_only",
        uses_fun_control_y: bool = False,
    ):
        del cross_view_num_heads

        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type,
            fps=fps,
            condition_channels=condition_channels,
            is_wan2_2=is_wan2_2,
        )

        self.multiview_mode = str(multiview_mode).strip().lower()
        self.prope_attention_variant = str(prope_attention_variant).strip().lower()
        self.uses_fun_control_y = bool(uses_fun_control_y)
        if self.multiview_mode not in ("full_prope", "full_rope_offset"):
            raise ValueError(f"Unsupported multiview_mode: {multiview_mode}")

        self._replace_multiview_blocks()
        self.register_to_config(
            multiview_mode=self.multiview_mode,
            prope_attention_variant=self.prope_attention_variant,
            uses_fun_control_y=self.uses_fun_control_y,
        )

    def _replace_multiview_blocks(self) -> None:
        """Promote each already-loaded ``WanAttentionBlock`` into a ``MultiViewWanAttentionBlock``
        **in-place**. Avoids reconstructing fresh Linear/Conv/Norm modules for 40 blocks,
        which at 14B scale was taking >10 min and tripping the accelerate elastic timeout.

        ``full_prope`` additionally gets one small-init ``o_prope`` Linear per
        self-attention. ``full_rope_offset`` keeps the pretrained attention weights
        unchanged and only changes how RoPE phases are assigned across views.
        """
        total = len(self.blocks)
        print(
            f"🔁 Promoting {total} blocks to MultiViewWanAttentionBlock "
            f"(mode={self.multiview_mode}, prope_variant={self.prope_attention_variant}) in-place ...",
            flush=True,
        )
        mode = self.multiview_mode
        if mode not in ("full_prope", "full_rope_offset"):
            raise ValueError(f"Unsupported multiview_mode: {mode}")

        for layer_idx, block in enumerate(self.blocks):
            # Re-class the block itself so ``block.forward`` dispatches to the
            # MultiView branch. All parameters (self_attn, cross_attn, ffn, norms,
            # modulation) are kept — no copy, no re-alloc.
            block.__class__ = MultiViewWanAttentionBlock
            block.cross_view_mode = mode
            block.prope_attention_variant = self.prope_attention_variant

            old_sa = block.self_attn
            if mode == "full_prope":
                # Promote self_attn -> FullSelfAttentionWithPrope and add the small-init
                # o_prope projection. With std≈1/sqrt(dim) × 1e-3 the branch
                # contribution stays near-zero functionally, finite-grad numerically.
                old_sa.__class__ = FullSelfAttentionWithPrope
                old_sa.prope_attention_variant = self.prope_attention_variant
                device = old_sa.q.weight.device
                dtype = old_sa.q.weight.dtype
                o_prope = nn.Linear(old_sa.dim, old_sa.dim).to(device=device, dtype=dtype)
                _o_prope_init_std = 1e-3 / (old_sa.dim ** 0.5)
                nn.init.normal_(o_prope.weight, mean=0.0, std=_o_prope_init_std)
                if o_prope.bias is not None:
                    nn.init.zeros_(o_prope.bias)
                old_sa.o_prope = o_prope
            else:
                old_sa.__class__ = FullSelfAttentionWithRopeOffset
                if hasattr(old_sa, "o_prope"):
                    delattr(old_sa, "o_prope")
            old_sa.layer_idx = layer_idx
            old_sa.num_layers = self.num_layers
        print(f"✅ MultiView block promotion done ({total} blocks)", flush=True)

    @staticmethod
    def _normalize_multiview_image_sizes(
        *,
        image_sizes: Optional[torch.Tensor],
        view_grid_sizes: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-view image sizes as [B, V, 2] in (H, W) pixel units.

        MultiViewPropeDataset provides target video sizes directly. If callers do
        not pass them, derive the Wan token footprint using VAE stride 8 and patch
        size 2, i.e. 16 input pixels per transformer token.
        """
        if image_sizes is None:
            return (view_grid_sizes[:, :, 1:3].to(device=device, dtype=torch.float32) * 16.0).contiguous()
        image_sizes = image_sizes.to(device=device, dtype=torch.float32)
        if image_sizes.dim() == 2:
            image_sizes = image_sizes.view(1, image_sizes.shape[0], 2).expand(batch_size, -1, -1)
        if image_sizes.dim() != 3 or image_sizes.shape[0] != batch_size or image_sizes.shape[-1] != 2:
            raise ValueError(
                f"image_sizes must be [B,V,2] or [V,2] as (H,W), got {tuple(image_sizes.shape)}"
            )
        if image_sizes.shape[1] != view_grid_sizes.shape[1]:
            raise ValueError(
                f"image_sizes view count {image_sizes.shape[1]} != view_grid_sizes {view_grid_sizes.shape[1]}"
            )
        return image_sizes.contiguous()

    def _unpatchify_multiview_hetero(
        self,
        x: torch.Tensor,
        view_grid_sizes: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Unpatchify a per-view-concatenated token sequence back into a list of
        ``[B, C_out, F, H, W]`` tensors, allowing different spatial sizes per view.
        """
        batch_size, num_views = int(view_grid_sizes.shape[0]), int(view_grid_sizes.shape[1])
        per_view: list[torch.Tensor] = []
        # view_grid_sizes must be identical across the batch for this routing.
        for v_idx in range(num_views):
            grids = view_grid_sizes[:, v_idx]  # [B, 3]
            view_len = int(grids[0].prod().item())
            offset = int(view_grid_sizes[:, :v_idx].prod(dim=-1).sum(dim=-1)[0].item())
            view_tokens = x[:, offset : offset + view_len]
            # ``unpatchify`` returns a list of per-batch tensors; stack back into [B, C, F, H, W].
            per_batch = self.unpatchify(view_tokens, grids)
            per_view.append(torch.stack(per_batch, dim=0))
        return per_view

    def _forward_multiview_token_concat(
        self,
        x_views: list[torch.Tensor],  # each [B, C_in, F, H_i, W_i]
        t,
        context,
        seq_len,
        clip_fea=None,
        y_views: Optional[list[torch.Tensor]] = None,
        condition_latents_views: Optional[list[Optional[torch.Tensor]]] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Full-self-attention forward for heterogeneous-resolution N-view inputs.

        ``x_views`` is a list of per-view latents (already VAE-encoded and channel-concatenated
        with the I2V padding if needed). ``y_views`` and ``condition_latents_views`` follow the
        same layout as ``x_views``; they are channel-concatenated per view before
        ``patch_embedding``. Returns a list of per-view output latents with the same spatial
        shape as each input view.
        """
        if self.sp_world_size > 1:
            raise NotImplementedError(f"{self.multiview_mode} does not support sequence-parallel training yet.")
        if not isinstance(x_views, (list, tuple)) or len(x_views) == 0:
            raise TypeError(f"{self.multiview_mode} expects `x_views` as a non-empty list of per-view tensors.")
        if self.multiview_mode == "full_prope" and (viewmats is None or Ks is None):
            raise ValueError("full_prope requires viewmats and Ks.")

        num_views = len(x_views)
        batch_size = x_views[0].shape[0]
        frames_in = x_views[0].shape[2]
        if any(v.shape[0] != batch_size or v.shape[2] != frames_in for v in x_views):
            raise ValueError("All views must share batch size and frame count.")

        device = self.patch_embedding.weight.device
        dtype = x_views[0].dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # Channel-concat y / condition per view (mirrors the single-view DWM concat path).
        fused_views: list[torch.Tensor] = []
        for v_idx, x_v in enumerate(x_views):
            merged = x_v
            if y_views is not None:
                merged = torch.cat([merged, y_views[v_idx]], dim=1)
            if condition_latents_views is not None and isinstance(
                condition_latents_views[v_idx], torch.Tensor
            ):
                merged = torch.cat([merged, condition_latents_views[v_idx]], dim=1)
            fused_views.append(merged)

        # Patchify each view; collect per-view grid (F_tok, H_tok, W_tok) and token streams.
        token_streams: list[torch.Tensor] = []
        per_view_grid: list[tuple[int, int, int]] = []
        for v_idx, merged in enumerate(fused_views):
            patched = self.patch_embedding(merged)  # [B, dim, F_tok, H_tok, W_tok]
            f_tok, h_tok, w_tok = int(patched.shape[2]), int(patched.shape[3]), int(patched.shape[4])
            per_view_grid.append((f_tok, h_tok, w_tok))
            tokens = patched.flatten(2).transpose(1, 2).contiguous()  # [B, F*H*W, dim]
            token_streams.append(tokens)

        view_grid_sizes = (
            torch.tensor(per_view_grid, device=device, dtype=torch.long)
            .view(1, num_views, 3)
            .expand(batch_size, num_views, 3)
            .contiguous()
        )
        view_seq_lens = view_grid_sizes.prod(dim=-1, dtype=torch.long)  # [B, V]
        total_seq_lens = view_seq_lens.sum(dim=1)  # [B]
        image_sizes = self._normalize_multiview_image_sizes(
            image_sizes=image_sizes,
            view_grid_sizes=view_grid_sizes,
            batch_size=batch_size,
            device=device,
        )

        x = torch.cat(token_streams, dim=1)  # [B, sum(F*H*W), dim]
        total_seq = int(total_seq_lens.max().item())
        seq_len_padded = max(int(seq_len), total_seq)
        if x.size(1) < seq_len_padded:
            x = torch.cat(
                [x, x.new_zeros(batch_size, seq_len_padded - x.size(1), x.size(2))],
                dim=1,
            )
        seq_lens = total_seq_lens

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            if t.dim() != 1:
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len_padded)).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            e0 = e0.to(dtype)
            e = e.to(dtype)

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, _view_grid, _total_seq_lens, _viewmats, _Ks, _image_sizes):
                    def custom_forward(x_, e_, seq_lens_, context_, context_lens_, dtype_, t_):
                        return module(
                            x_,
                            e=e_,
                            seq_lens=seq_lens_,
                            grid_sizes=_view_grid[:, 0],
                            view_grid_sizes=_view_grid,
                            total_seq_lens=_total_seq_lens,
                            freqs=self.freqs,
                            context=context_,
                            context_lens=context_lens_,
                            dtype=dtype_,
                            t=t_,
                            viewmats=_viewmats,
                            Ks=_Ks,
                            image_sizes=_image_sizes,
                        )
                    return custom_forward

                ckpt_kwargs: Dict[str, object] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, view_grid_sizes, total_seq_lens, viewmats, Ks, image_sizes),
                    x,
                    e0,
                    seq_lens,
                    context,
                    context_lens,
                    dtype,
                    t,
                    **ckpt_kwargs,
                )
            else:
                x = block(
                    x,
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=view_grid_sizes[:, 0],
                    view_grid_sizes=view_grid_sizes,
                    total_seq_lens=total_seq_lens,
                    freqs=self.freqs,
                    context=context,
                    context_lens=context_lens,
                    dtype=dtype,
                    t=t,
                    viewmats=viewmats,
                    Ks=Ks,
                    image_sizes=image_sizes,
                )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward_head(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward_head(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        return self._unpatchify_multiview_hetero(x, view_grid_sizes)

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        condition_latents=None,
        multiview_batch_size: Optional[int] = None,
        num_views: Optional[int] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
    ):
        del y_camera, full_ref, subject_ref, cond_flag, multiview_batch_size, num_views
        if self.multiview_mode not in ("full_prope", "full_rope_offset"):
            raise ValueError(f"Unsupported multiview_mode: {self.multiview_mode}")
        return self._forward_multiview_token_concat(
            x_views=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y_views=y,
            condition_latents_views=condition_latents,
            viewmats=viewmats,
            Ks=Ks,
            image_sizes=image_sizes,
        )
