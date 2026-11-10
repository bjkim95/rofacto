"""Native Wan2.2 BF16 x0 caches; cached training retains native noise/packing."""
from functools import lru_cache
from pathlib import Path
import json

from .io import file_hash, object_hash

KEYS = ('static', 'mesh', 'target')
FORMAT = 'rofacto_wan22_bf16_x0_rgb_v1'
PREPROCESS = {'height':480, 'width':832, 'frames':81, 'resize':'ffmpeg_bilinear_direct',
              'normalization':'bf16_pixels_div_127.5_minus_1', 'streams':list(KEYS)}


def cache_path(root, row):
    identity = {k: row.get(k) for k in ('target','mesh','static','static_mode')}
    key = object_hash(identity)
    return Path(root) / key[:2] / f'{key}.pt'


def source_paths(root, row):
    keys = ['mesh','target'] + (['static'] if row['static_mode'] == 'video' else [])
    return {key: Path(root) / row[key] for key in keys}


@lru_cache(maxsize=200000)
def checked_file_hash(path, size, mtime_ns):
    return file_hash(path)


def sources(root, row):
    result = {}
    for key, path in source_paths(root,row).items():
        stat = path.stat()
        result[key] = checked_file_hash(str(path),stat.st_size,stat.st_mtime_ns)
    return result


def load_cached(cache, data_root, row, height, width, vae_sha256):
    import torch
    if (height,width) != (480,832):
        raise ValueError('Cached RGB training requires 832x480')
    payload = torch.load(cache_path(cache,row),map_location='cpu',weights_only=True)
    if (payload['format'] != FORMAT or payload['preprocess'] != PREPROCESS
        or payload['vae_sha256'] != vae_sha256 or payload['source_sha256'] != sources(data_root,row)
        or payload['static_mode'] != row['static_mode']):
        raise ValueError(f"Stale/incompatible latent cache: {row['target']}")
    values=payload['latents']
    assert len(values)==3 and all(tuple(v.shape)==(1,48,21,30,52) and v.dtype==torch.bfloat16 and torch.isfinite(v).all() for v in values)
    return {'row':row,'latents':values,'fps':payload['fps'],'stream_names':KEYS}


def pixel_inputs(sample, device, height, width):
    import torch
    if 'latents' in sample:
        return [torch.zeros((),device=device,dtype=torch.bfloat16).expand(1,3,81,height,width) for _ in KEYS]
    return [v.to(device=device,dtype=torch.bfloat16)/127.5-1 for v in sample['videos']]


def attach(batch,samples):
    have=['latents' in sample for sample in samples]
    if any(have):
        assert all(have), 'Do not mix cached and pixel inputs in a packed batch'
        batch['_rofacto_x0_cache']=[v for s in samples for v in s['latents']]
    return batch


def install(model):
    original_data=model.get_data_and_condition
    original_encode=model._encode_vision_x0_tokens
    active=[]
    def encode(raw,*args,**kwargs):
        if not active:
            return original_encode(raw,*args,**kwargs)
        values=active[-1]
        assert len(values)==len(raw) and all(tuple(p.shape)==(1,3,81,480,832) for p in raw)
        import torch
        return [v.to(device=raw[0].device,dtype=torch.float32).contiguous() for v in values]
    def get_data(batch,*args,**kwargs):
        values=batch.pop('_rofacto_x0_cache',None)
        if values is None:
            return original_data(batch,*args,**kwargs)
        assert not active
        active.append(values)
        try:
            return original_data(batch,*args,**kwargs)
        finally:
            active.pop()
    model._encode_vision_x0_tokens=encode
    model.get_data_and_condition=get_data


def verify_native_parity(model, raw_batch, cached_batch, output, rank):
    import torch
    from .io import write_json
    with torch.no_grad():
        online=model.get_data_and_condition(raw_batch)
        cached=model.get_data_and_condition(cached_batch)
        a,b=online.x0_tokens_vision,cached.x0_tokens_vision
        assert len(a)==len(b)==3
        differences=[float((x-y).abs().max()) for x,y in zip(a,b)]
        assert all(torch.equal(x,y) for x,y in zip(a,b)), differences
        assert online.num_vision_items_per_sample==cached.num_vision_items_per_sample
    report=dict(status='PASS',exact_latent_equality=True,max_abs_errors=differences,rank=rank)
    write_json(Path(output)/f'cache_parity_rank{rank}.json',report)
    return report
