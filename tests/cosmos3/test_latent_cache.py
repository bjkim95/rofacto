from pathlib import Path

import pytest
import torch

from training.cosmos3.latent_cache import FORMAT, PREPROCESS, cache_path, load_cached, sources


def test_cache_rejects_changed_vae_or_source_and_preserves_native_dtype(tmp_path):
    row=dict(target='target.mp4',mesh='mesh.mp4',static=None,static_mode='first_frame_repeat')
    for key in ('target','mesh'):
        (tmp_path/row[key]).write_bytes(b'source fixture')
    cache=tmp_path/'cache'
    dst=cache_path(cache,row)
    dst.parent.mkdir(parents=True)
    latent=torch.zeros((1,48,21,30,52),dtype=torch.bfloat16)
    payload=dict(format=FORMAT,preprocess=PREPROCESS,vae_sha256='vae-a',
        source_sha256=sources(tmp_path,row),static_mode=row['static_mode'],fps=15.,latents=[latent]*3)
    torch.save(payload,dst)
    sample=load_cached(cache,tmp_path,row,480,832,'vae-a')
    assert 'videos' not in sample and sample['latents'][0].dtype==torch.bfloat16
    with pytest.raises(ValueError,match='Stale/incompatible'):
        load_cached(cache,tmp_path,row,480,832,'vae-b')
    (tmp_path/'mesh.mp4').write_bytes(b'changed motion')
    with pytest.raises(ValueError,match='Stale/incompatible'):
        load_cached(cache,tmp_path,row,480,832,'vae-a')
