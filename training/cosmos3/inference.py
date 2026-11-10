"""One GPU, one model load, ordered JSON/JSONL controlled-video generation."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

from .inputs import read_inputs, validate_media
from .io import emit, file_hash, object_hash, probe_video, read_video, write_json, write_video
from .model import inspect_export, load_inference_model
from .conditioning import make_inference_batch, audit_packing, frozen_latents
from .checkpoint import tensor_hash


def load_sampler(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    required = {'steps', 'guidance', 'shift', 'sigma_max', 'normalize_cfg', 'has_negative_prompt',
                'use_batched_cfg', 'negative_prompt', 'seed'}
    if set(config) != required:
        raise ValueError(f'Sampler keys must be exactly {sorted(required)}')
    if config['steps'] <= 0 or config['guidance'] < 1 or config['use_batched_cfg']:
        raise ValueError('Require positive steps, CFG >= 1, unbatched CFG')
    return config


def completed(folder, protocol_hash, media):
    rp, vp = folder / 'result.json', folder / 'generated.mp4'
    if not rp.exists():
        return False
    result = json.loads(rp.read_text())
    if result.get('protocol_hash') != protocol_hash:
        raise ValueError(f'Existing output has a different protocol: {folder}. Choose a new output directory.')
    return (result.get('status') == 'PASS' and vp.is_file()
            and file_hash(vp) == result['video_sha256']
            and probe_video(vp) == {k: media[k] for k in ('frames', 'fps', 'width', 'height')})


def encode_inputs(model, row):
    import numpy as np
    import torch
    from PIL import Image
    mesh = read_video(row['robot_mesh_video'])
    if 'static_image' in row:
        static = np.asarray(Image.open(row['static_image']).convert('RGB')).copy()
        static = torch.from_numpy(static).permute(2, 0, 1)[None, :, None].expand(1, 3, row['frames'], 480, 832)
    else:
        static = torch.from_numpy(read_video(row['static_video'])).permute(3, 0, 1, 2)[None]
    mesh = torch.from_numpy(mesh).permute(3, 0, 1, 2)[None]
    model.tokenizer_vision_gen.model.model.to('cuda')
    latents = []
    for pixels in (static, mesh, None):
        video = (pixels.to(device='cuda', dtype=torch.bfloat16) / 127.5 - 1 if pixels is not None
                 else torch.zeros((1, 3, row['frames'], 480, 832), device='cuda', dtype=torch.bfloat16))
        latents.append(model._encode_vision_item(video, num_views=1).cpu())
        del video
    return latents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('nano', 'super'), required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--vae', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resident-layers', type=int, default=None)
    parser.add_argument('--pin-cpu-memory', action='store_true')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true', help='Check inputs/export without CUDA or model loading')
    parser.add_argument('--save-latents', action='store_true', help='Save tensors for reference comparisons')
    args = parser.parse_args()
    sampler = load_sampler(args.config)
    _, export = inspect_export(args.adapter, args.model)
    rows = [validate_media(r) for r in read_inputs(args.inputs, num_shards=args.num_shards, shard_index=args.shard_index)]
    if not args.base.is_dir() or not args.vae.is_file():
        raise FileNotFoundError('Local base model directory and VAE file are required')
    vae_hash = file_hash(args.vae)
    configs = dict(sampler=sampler, model=args.model, base_revision=export['metadata']['signature']['base'],
        adapter_hash=export['adapter_hash'], vae_sha256=vae_hash,
        model_config_sha256=file_hash(args.adapter / 'model_config.json'),
        streams=['static','mesh','target'], shared_positions=True, first_target_anchor=False)
    pending = []
    for row in rows:
        spec = dict(configs, inputs=row['input_hashes'], caption=row['caption'],
                    frames=row['frames'], fps=row['fps'], resolution=[480,832])
        digest = object_hash(spec)
        folder = args.output / row['id']
        if completed(folder, digest, row):
            emit('already_complete', sample=row['id'])
        else:
            pending.append((row, spec, digest, folder))
    emit('inputs_validated', selected=len(rows), pending=len(pending))
    if args.validate_only or not pending:
        return
    import torch
    import torch.distributed as dist
    from cosmos_framework.utils import misc
    resident = args.resident_layers if args.resident_layers is not None else (36 if args.model == 'nano' else 24)
    model, _ = load_inference_model(args.base, args.vae, args.adapter, args.model,
        max(row['frames'] for row in rows), resident, args.pin_cpu_memory)
    vae = model.tokenizer_vision_gen.model.model
    assert model.tokenizer_vision_gen.is_causal
    original_denoise = model.denoise
    try:
        for row, spec, digest, folder in pending:
            folder.mkdir(parents=True, exist_ok=True)
            import fcntl
            with (folder / '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if completed(folder, digest, row):
                    continue
                write_json(folder / 'protocol.json', spec)
                tick = time.monotonic()
                latents = encode_inputs(model, row)
                noise = misc.arch_invariant_rand((1,48,1+(row['frames']-1)//4,30,52), torch.float32, 'cpu', sampler['seed'])
                vae.to('cpu')
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                calls = []
                def denoise(*a, **kw):
                    if not calls:
                        audit_packing(kw['data_batch_packed'], row['frames'])
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    result = original_denoise(*a, **kw)
                    torch.cuda.synchronize()
                    calls.append(time.monotonic() - start)
                    return result
                model.denoise = denoise
                batch = make_inference_batch(model, row['caption'], row['frames'], row['fps'], sampler['negative_prompt'])
                generation_args = {k: v for k, v in sampler.items() if k not in ('steps','seed','negative_prompt')}
                emit('generation_start', sample=row['id'], frames=row['frames'], steps=sampler['steps'])
                sampling_start = time.monotonic()
                with frozen_latents(model, latents, noise, row['frames'], sampler['seed']) as noise_calls:
                    result = model.generate_samples_from_batch(batch, seed=[sampler['seed']],
                        num_steps=sampler['steps'], **generation_args)
                torch.cuda.synchronize()
                sampling_seconds = time.monotonic() - sampling_start
                assert len(calls) == sampler['steps'] * (1 if sampler['guidance'] == 1 else 2)
                assert noise_calls[0] == 3
                generated = result['vision'][0]
                assert tuple(generated.shape) == tuple(noise.shape) and torch.isfinite(generated).all()
                del batch
                gc.collect()
                torch.cuda.empty_cache()
                vae.to('cuda')
                decoded = model.decode(generated)
                assert tuple(decoded.shape) == (1,3,row['frames'],480,832) and torch.isfinite(decoded).all()
                pixels = ((decoded[0].float().clamp(-1,1)+1)*127.5).byte().permute(1,2,3,0).cpu().numpy()
                video_path = folder / 'generated.mp4'
                write_video(video_path, pixels, row['fps'])
                if args.save_latents:
                    torch.save(dict(conditions=latents, noise=noise, generated=generated.cpu(), decoded=decoded.cpu()), folder / 'tensors.pt')
                report = dict(status='PASS', protocol_hash=digest, frames=row['frames'], fps=row['fps'],
                    seconds=time.monotonic()-tick, sampling_seconds=sampling_seconds,
                    denoise_calls=len(calls), denoise_seconds=calls,
                    condition_cache_hash=tensor_hash([(str(i), x) for i,x in enumerate(latents)]),
                    master_noise_hash=tensor_hash([('master',noise)]), video_sha256=file_hash(video_path),
                    peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                    gpu=torch.cuda.get_device_name(), resident_layers=resident)
                write_json(folder / 'result.json', report)
                emit('generation_complete', sample=row['id'], seconds=report['seconds'], denoise_calls=len(calls))
                del latents, noise, result, generated, decoded, pixels
                vae.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        model.denoise = original_denoise
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
