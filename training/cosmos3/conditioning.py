"""Static, mesh, target streams share temporal/spatial positions in joint attention."""
from contextlib import contextmanager

STREAMS = ('static', 'mesh', 'target')


def make_inference_batch(model, caption, frames, fps, negative_prompt):
    import torch
    from cosmos_framework.data.generator.sequence_packing import SequencePlan
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
    ids = tokenize_caption(caption, model.vlm_tokenizer, is_video=True,
                           use_system_prompt=model.vlm_config.use_system_prompt)
    if len(ids) > 4096:
        raise ValueError(f'Prompt exceeds 4096 tokens: {len(ids)}')
    # These tensors carry shape metadata; the cached native x0 path supplies pixels.
    videos = [torch.zeros((1, 3, frames, 480, 832), device='cuda', dtype=torch.bfloat16) for _ in STREAMS]
    return {model.input_video_key: videos, model.input_caption_key: [caption],
        'text_token_ids': [torch.tensor([ids], dtype=torch.long, device='cuda')],
        'num_vision_items_per_sample': [3],
        'sequence_plan': [SequencePlan(has_text=True, has_vision=True,
            condition_frame_indexes_vision=[], share_vision_temporal_positions=True)],
        'image_size': [torch.tensor([[480,832,480,832]], dtype=torch.float32, device='cuda') for _ in STREAMS],
        'fps': torch.tensor([fps], device='cuda'), 'conditioning_fps': torch.tensor([fps], device='cuda'),
        'num_frames': torch.tensor([frames], device='cuda'), 'is_preprocessed': True,
        'neg_' + model.input_caption_key: [negative_prompt]}


def audit_packing(packed, frames):
    import torch
    latent_frames = 1 + (frames - 1) // 4
    assert packed.vision.token_shapes == [(latent_frames, 15, 26)] * 3
    masks = packed.vision.condition_mask
    assert len(masks) == 3
    assert all(torch.all(m == 1) for m in masks[:2]) and torch.all(masks[-1] == 0)
    indexes = packed.vision.sequence_indexes
    n = latent_frames * 15 * 26
    assert indexes.numel() == 3 * n
    pos = packed.position_ids
    assert all(torch.equal(pos[..., indexes[:n]], pos[..., indexes[i*n:(i+1)*n]]) for i in (1, 2))


@contextmanager
def frozen_latents(model, latents, master_noise, frames, seed):
    """Match the reference's three identical seeded noise draws, restore all hooks."""
    import torch
    from cosmos_framework.utils import misc
    original_encode, original_rand = model._encode_vision_x0_tokens, misc.arch_invariant_rand
    calls = [0]
    def encode(raw_state_vision, *args, **kwargs):
        assert len(raw_state_vision) == len(latents) == 3
        assert all(x.shape[2] == frames for x in raw_state_vision)
        return [x.to(device='cuda', dtype=torch.float32).contiguous() for x in latents]
    def noise(shape, dtype, device, seed=None):
        if len(shape) == 5 and tuple(shape[:2]) == (1, 48) and tuple(shape[3:]) == (30, 52):
            assert seed == expected_seed and tuple(shape) == tuple(master_noise.shape)
            calls[0] += 1
            return master_noise.to(device=device, dtype=dtype).contiguous()
        return original_rand(shape, dtype, device, seed)
    expected_seed = seed
    model._encode_vision_x0_tokens = encode
    misc.arch_invariant_rand = noise
    try:
        yield calls
    finally:
        model._encode_vision_x0_tokens = original_encode
        misc.arch_invariant_rand = original_rand
