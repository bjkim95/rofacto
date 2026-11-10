"""Native multi-sample packing: each sample owns static, mesh, and target items."""
import random
from .latent_cache import pixel_inputs, attach


def make_batch(samples, positions, model, tokenize, plan_type, device='cuda', height=480, width=832, seed=42, text_dropout=.1):
    import torch
    assert len(samples) == len(positions) > 0
    captions, ids, videos = [], [], []
    for sample, absolute in zip(samples, positions):
        caption = sample['row']['caption']
        if random.Random(seed + 1000003 * (absolute + 1)).random() < text_dropout:
            caption = ''
        tokens = tokenize(caption, model.vlm_tokenizer, is_video=True,
                          use_system_prompt=model.vlm_config.use_system_prompt)
        assert len(tokens) <= 2048 and len(sample['stream_names']) == 3
        captions.append(caption)
        ids.append(torch.tensor([tokens], dtype=torch.long, device=device))
        videos.extend(pixel_inputs(sample, device, height, width))
    batch = len(samples)
    result = {
        model.input_video_key: videos, model.input_caption_key: captions,
        'text_token_ids': ids, 'num_vision_items_per_sample': [3] * batch,
        'sequence_plan': [plan_type(has_text=True, has_vision=True,
            condition_frame_indexes_vision=[], share_vision_temporal_positions=True) for _ in samples],
        'image_size': [torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)
                       for _ in range(3 * batch)],
        'fps': torch.tensor([s['fps'] for s in samples], device=device),
        'conditioning_fps': torch.tensor([s['fps'] for s in samples], device=device),
        'num_frames': torch.tensor([81] * batch, device=device), 'is_preprocessed': True,
    }

    return attach(result, samples)
