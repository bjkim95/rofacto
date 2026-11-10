"""One-time RGB VAE encoding. Run directly on one GPU or with torchrun."""
import argparse
import json
import os
from pathlib import Path
import time

from .data import load_sample, read_manifest, vision_keys
from .io import emit, file_hash, write_json
from .latent_cache import FORMAT, PREPROCESS, cache_path, load_cached, sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--source-config',type=Path,required=True,help='Exported model_config.json')
    p.add_argument('--vae',type=Path,required=True)
    args=p.parse_args()
    import torch
    import hydra
    import omegaconf
    from cosmos_framework.inference.common.config import structure_config
    rank=int(os.environ.get('RANK',0))
    world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config=json.loads(args.source_config.read_text())['config']['tokenizer']
    config.update(vae_path=str(args.vae),bucket_name='',object_store_credential_path_pretrained='')
    tokenizer=hydra.utils.instantiate(structure_config(config,omegaconf.DictConfig))
    tokenizer.reset_dtype()
    digest=file_hash(args.vae)
    rows=[row for split in ('train','test') for row in read_manifest(args.manifest_dir/f'{split}.jsonl')[0]]
    if any(vision_keys(row) != ('static','mesh','target') for row in rows):
        raise ValueError('This preencoder supports RGB static + mesh + target only')
    identities=[str(cache_path(args.cache,row)) for row in rows]
    assert len(set(identities))==len(rows), 'Duplicate manifest inputs'
    args.cache.mkdir(parents=True,exist_ok=True)
    created=skipped=0
    for row in rows[rank::world]:
        dst=cache_path(args.cache,row)
        if dst.exists():
            load_cached(args.cache,args.data_root,row,480,832,digest)
            skipped+=1
            continue
        started=time.monotonic()
        sample=load_sample(args.data_root,row,480,832)
        latents=[]
        for video in sample['videos']:
            normalized=video.to(device='cuda',dtype=torch.bfloat16)/127.5-1
            with torch.no_grad():
                latent=tokenizer.encode(normalized)
            assert tuple(latent.shape)==(1,48,21,30,52) and latent.dtype==torch.bfloat16 and torch.isfinite(latent).all()
            latents.append(latent.cpu())
            del normalized,latent
        payload=dict(format=FORMAT,preprocess=PREPROCESS,vae_sha256=digest,
            source_sha256=sources(args.data_root,row),static_mode=row['static_mode'],fps=sample['fps'],latents=latents)
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_suffix(f'.rank{rank}.tmp')
        torch.save(payload,tmp)
        tmp.replace(dst)
        created+=1
        emit('preencoded',target=row['target'],seconds=time.monotonic()-started)
    report=dict(status='PASS',rank=rank,world=world,created=created,skipped=skipped,vae_sha256=digest)
    write_json(args.cache/f'complete_rank{rank}.json',report)
    emit('preencode_complete',**report)


if __name__=='__main__':
    main()
