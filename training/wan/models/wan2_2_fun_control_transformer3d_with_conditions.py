# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
"""Wan2.2 Fun-Control transformer with optional extra concat conditions.

The Wan2.2-Fun-5B-Control checkpoint already expects the VideoX-Fun control
layout in ``patch_embedding``:

    x noisy latent (48) + y control/inpaint channels (100) = 148

This wrapper keeps that pretrained layout intact and only expands the patch
embedding when additional condition streams are explicitly requested.
"""

from .wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat


class Wan2_2FunControlTransformer3DModelWithConcat(WanTransformer3DModelWithConcat):
    """Semantic wrapper for Wan2.2 Fun-Control + optional extra conditions."""

    uses_fun_control_y = True
