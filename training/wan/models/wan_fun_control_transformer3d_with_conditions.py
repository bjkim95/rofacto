# RoFacto adaptation; upstream attribution and licenses: THIRD_PARTY_NOTICES.md.
"""Wan2.1 Fun-Control transformer with optional extra concat conditions.

The Wan2.1-Fun-14B-Control checkpoint already includes the VideoX-Fun
control lane in ``patch_embedding``:

    x noisy latent (16) + y control/static channels (32) = 48

This wrapper keeps that pretrained layout intact. ``condition_channels`` means
only streams beyond the built-in control/static lanes, such as depth latents.
"""

from .wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat


class WanFunControlTransformer3DModelWithConcat(WanTransformer3DModelWithConcat):
    """Semantic wrapper for Wan2.1 Fun-Control + optional extra conditions."""

    uses_fun_control_y = True
