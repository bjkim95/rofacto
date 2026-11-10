# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_transformer3d import WanTransformer3DModel


class WanTransformer3DModelWithStateAdaLN(WanTransformer3DModel):
    """Wan transformer variant that adds per-frame robot state into block AdaLN."""

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
        state_adaln_dim: int = 7,
        state_adaln_hidden_dim: Optional[int] = None,
        state_adaln_zero_init: bool = True,
        state_adaln_scale: float = 1.0,
        state_adaln_zero_first_frame: bool = False,
        condition_channels: int = 0,
        is_wan2_2: bool = False,
    ):
        if is_wan2_2:
            cross_attn_type = "cross_attn"
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
        )
        if is_wan2_2 and hasattr(self, "img_emb"):
            del self.img_emb
        if int(condition_channels or 0) != 0:
            raise ValueError("WanTransformer3DModelWithStateAdaLN does not use concat condition channels")

        self.condition_channels = 0
        self.is_wan2_2 = bool(is_wan2_2)
        self.state_adaln_dim = int(state_adaln_dim)
        self.state_adaln_hidden_dim = int(state_adaln_hidden_dim or dim)
        self.state_adaln_scale = float(state_adaln_scale)
        self.state_adaln_zero_first_frame = bool(state_adaln_zero_first_frame)
        self.state_adaln_norm = nn.LayerNorm(self.state_adaln_dim)
        self.state_adaln_mlp = nn.Sequential(
            nn.Linear(self.state_adaln_dim, self.state_adaln_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.state_adaln_hidden_dim, dim * 6),
        )
        self._init_state_adaln(zero_init=bool(state_adaln_zero_init))
        self.register_to_config(
            state_adaln_dim=self.state_adaln_dim,
            state_adaln_hidden_dim=self.state_adaln_hidden_dim,
            state_adaln_zero_init=bool(state_adaln_zero_init),
            state_adaln_scale=self.state_adaln_scale,
            state_adaln_zero_first_frame=self.state_adaln_zero_first_frame,
            condition_channels=0,
            is_wan2_2=self.is_wan2_2,
        )

    def _init_state_adaln(self, *, zero_init: bool) -> None:
        for module in self.state_adaln_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if zero_init:
            final = self.state_adaln_mlp[-1]
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def get_condition_info(self) -> Dict[str, Any]:
        return {
            "approach": "state_adaln",
            "has_conditions": True,
            "state_adaln_dim": self.state_adaln_dim,
            "state_adaln_hidden_dim": self.state_adaln_hidden_dim,
            "state_adaln_zero_first_frame": self.state_adaln_zero_first_frame,
            "condition_channels": 0,
            "is_wan2_2": self.is_wan2_2,
        }

    def _grid_sizes_from_latents(self, x) -> list[tuple[int, int, int]]:
        patch_t, patch_h, patch_w = [int(v) for v in self.patch_size]
        samples = [x[i] for i in range(x.shape[0])] if isinstance(x, torch.Tensor) and x.dim() == 5 else x
        if not isinstance(samples, list):
            raise ValueError("state_adaln expects x as [B,C,F,H,W] tensor or list of [C,F,H,W] tensors")
        grid_sizes = []
        for sample in samples:
            if sample.dim() != 4:
                raise ValueError(f"Expected latent sample [C,F,H,W], got {tuple(sample.shape)}")
            _, frames, height, width = sample.shape
            grid_sizes.append(
                (
                    max(1, (int(frames) - patch_t) // patch_t + 1),
                    max(1, (int(height) - patch_h) // patch_h + 1),
                    max(1, (int(width) - patch_w) // patch_w + 1),
                )
            )
        return grid_sizes

    def _resize_state_frames(self, state_adaln: torch.Tensor, target_frames: int) -> torch.Tensor:
        if state_adaln.shape[1] == target_frames:
            return state_adaln
        if state_adaln.shape[1] <= 1:
            return state_adaln.repeat(1, target_frames, 1)
        state_bdt = state_adaln.transpose(1, 2).float()
        resized = F.interpolate(state_bdt, size=target_frames, mode="linear", align_corners=True)
        return resized.transpose(1, 2).to(dtype=state_adaln.dtype)

    def _build_state_adaln_modulation(self, state_adaln: torch.Tensor, x, seq_len: int) -> torch.Tensor:
        if state_adaln.dim() == 2:
            state_adaln = state_adaln[:, None, :]
        if state_adaln.dim() != 3:
            raise ValueError(f"state_adaln must be [B,T,D] or [B,D], got {tuple(state_adaln.shape)}")
        if state_adaln.shape[-1] != self.state_adaln_dim:
            raise ValueError(
                f"state_adaln dim mismatch: expected {self.state_adaln_dim}, got {state_adaln.shape[-1]}"
            )

        grid_sizes = self._grid_sizes_from_latents(x)
        if len(grid_sizes) != state_adaln.shape[0]:
            raise ValueError(f"state_adaln batch size {state_adaln.shape[0]} does not match x batch {len(grid_sizes)}")

        max_frames = max(grid[0] for grid in grid_sizes)
        state_adaln = self._resize_state_frames(state_adaln, max_frames)
        encoded = self.state_adaln_mlp(self.state_adaln_norm(state_adaln))
        encoded = encoded.view(encoded.shape[0], encoded.shape[1], 6, self.dim)
        encoded = encoded * self.state_adaln_scale
        if self.state_adaln_zero_first_frame and encoded.shape[1] > 0:
            encoded[:, :1] = 0

        packed = encoded.new_zeros((encoded.shape[0], int(seq_len), 6, self.dim))
        for batch_idx, (frames, height, width) in enumerate(grid_sizes):
            per_frame_tokens = height * width
            tokens = encoded[batch_idx, :frames].repeat_interleave(per_frame_tokens, dim=0)
            token_count = min(tokens.shape[0], int(seq_len))
            packed[batch_idx, :token_count] = tokens[:token_count]
        return packed

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
        state_adaln=None,
    ):
        if condition_latents is not None:
            raise ValueError("State-AdaLN transformer does not accept condition_latents")
        adaln_modulation = None
        if state_adaln is not None:
            state_adaln = state_adaln.to(device=x.device if isinstance(x, torch.Tensor) else x[0].device)
            adaln_modulation = self._build_state_adaln_modulation(state_adaln, x, int(seq_len))

        return super().forward(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            subject_ref=subject_ref,
            cond_flag=cond_flag,
            adaln_modulation=adaln_modulation,
        )
