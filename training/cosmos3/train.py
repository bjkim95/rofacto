"""Cosmos3-Super aligned RGB LoRA on the exact DROID + DiT4DiT paper split.

FSDP4 or FSDP8 / CP2, deterministic CP-window data consumption, optimizer accumulation,
resumable adapter/AdamW/RNG/sampling/schedule, and held-out domain validation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from .checkpoint import (local, tensor_hash, optimizer_hash, canonical_parameters,
                         legacy_full_parameters, checkpoints_to_prune)
from .io import emit, write_json, file_hash
from .model import resolve_framework_assets
from .latent_cache import attach, pixel_inputs, install, load_cached, verify_native_parity
RANK = int(os.environ.get('RANK', '0'))
from .data import WindowSampler, lr_factor, load_sample, read_manifest, vision_keys
from .batching import make_batch as make_packed_batch
from .batch_policy import positions
from .validation import evaluate_loss, isolated_evaluation, select_validation_rows, validation_protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model-variant', choices=('super', 'nano'), default='super')
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--stop-step', type=int, default=1000)
    p.add_argument('--schedule-steps', type=int, default=15000)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--gradient-audit', action='store_true')
    p.add_argument('--cp', type=int, default=2)
    p.add_argument('--lora-rank', type=int, default=64)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--text-dropout', type=float, default=.1)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--keep-checkpoints-every', type=int, default=1000,
                   help='Export adapter-only weights every N steps; 0 disables periodic exports')
    p.add_argument('--keep-checkpoint-steps', type=int, nargs='*', default=[200, 299, 500],
                   help='Additional adapter-only milestone steps')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--sample-steps', type=int, default=12)
    p.add_argument('--val-loss-samples-per-domain', type=int, default=16,
                   help='Fixed held-out clips per domain; 0 disables validation loss')
    p.add_argument('--val-loss-noise-draws', type=int, default=2)
    p.add_argument('--val-loss-seed', type=int, default=1729)
    p.add_argument('--resume', type=Path)
    p.add_argument('--activation-checkpointing', choices=('full', 'selective', 'none'), default='full')
    p.add_argument('--benchmark-only', action='store_true',
                   help='Disposable resumed updates; never save checkpoints or run validation')
    p.add_argument('--benchmark-warmup', type=int, default=2)
    p.add_argument('--smoke', action='store_true', help='Eight distinct clips, balanced domains, only for integration checks')
    p.add_argument('--latent-cache', type=Path)
    p.add_argument('--verify-cache-parity', action='store_true')
    p.add_argument('--config', type=Path, help='YAML defaults; explicit CLI flags take precedence')
    known, _ = p.parse_known_args()
    if known.config:
        import yaml
        defaults = yaml.safe_load(known.config.read_text())
        valid = {a.dest for a in p._actions}
        if set(defaults) - valid:
            p.error(f'Unknown config options: {set(defaults) - valid}')
        p.set_defaults(**defaults)
    args = p.parse_args()
    assert (args.height, args.width) == (480, 832)
    vae_sha = file_hash(args.vae)
    def sample_loader(root, row, height, width):
        if args.latent_cache:
            return load_cached(args.latent_cache, root, row, height, width, vae_sha)
        return load_sample(root, row, height, width)

    assert args.microbatch > 0
    assert args.microbatch == 1 or args.cp == 1
    assert not args.gradient_audit or args.cp == 1
    assert args.accum > 0 and args.accum % args.cp == 0
    assert 0 < args.stop_step <= args.schedule_steps and 0 <= args.text_dropout <= 1
    assert args.save_every > 0 and args.eval_every > 0
    assert args.keep_checkpoints_every >= 0 and all(s > 0 for s in args.keep_checkpoint_steps)
    assert args.sample_steps >= 0 and args.val_loss_samples_per_domain >= 0 and args.val_loss_noise_draws > 0
    out = args.output.resolve()
    if args.benchmark_only:
        assert not args.resume or out != args.resume.resolve().parent
        assert not (out / 'config.json').exists(), 'Benchmark output must be unused'
        assert args.benchmark_warmup >= 1
        args.sample_steps = args.val_loss_samples_per_domain = 0
    if not args.resume and out.exists() and any(out.glob('checkpoint_step*/COMPLETE')):
        raise ValueError('Output already contains completed checkpoints; use --resume or a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict
    from cosmos_framework.inference.common.init import init_script
    from cosmos_framework.inference.common.config import structure_config
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel, _DiffusersHuggingFaceStorageReader, _DiffusersLoadPlanner
    from cosmos_framework.data.generator.sequence_packing import SequencePlan
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
    import hydra
    import omegaconf
    init_script(training=True)
    world = dist.get_world_size()
    assert world in (4, 8) and args.cp in (1, 2) and world % args.cp == 0
    profile = {
        'super': dict(revision='f543c56225b2e04d0ad141e29655be3a45d9c455',
                      frozen=63997928640, tensors=512, rank16=39845888),
        'nano': dict(revision='e59a53c25979a090fa8706c9acc0c254a6e89b92',
                     frozen=15156222144, tensors=288, rank16=15335424),
    }[args.model_variant]
    torch.manual_seed(args.seed)
    random.seed(args.seed + RANK)
    np.random.seed(args.seed + RANK)
    train, train_sha = read_manifest(args.manifest_dir / 'train.jsonl')
    test, test_sha = read_manifest(args.manifest_dir / 'test.jsonl')
    num_items = 3
    assert all(len(vision_keys(row)) == num_items for row in train + test), 'Manifest/condition mode mismatch'
    assert not {r['episode'] for r in train} & {r['episode'] for r in test}
    if args.smoke:
        train = [r for domain in ('droid', 'robocasa') for r in [x for x in train if x['domain'] == domain][:4]]
    sampler = WindowSampler(len(train), args.seed, world, RANK)
    validation = [next(r for r in test if r['domain'] == domain) for domain in ('droid', 'robocasa')]
    val_loss_rows = select_validation_rows(test, args.val_loss_samples_per_domain, args.val_loss_seed)
    val_protocol = validation_protocol(val_loss_rows, seed=args.val_loss_seed,
        draws=args.val_loss_noise_draws, per_domain=args.val_loss_samples_per_domain, test_sha256=test_sha)
    signature = {k: getattr(args, k) for k in ('schedule_steps', 'warmup', 'accum', 'cp', 'lora_rank',
        'lora_alpha', 'lr', 'text_dropout', 'height', 'width', 'seed', 'smoke')}
    if args.microbatch > 1:
        signature['microbatch'] = args.microbatch
    signature.update(world=world, train_sha256=train_sha, test_sha256=test_sha,
        framework='96303bb0bdd14d9efa18f24d8ef98c7a0bfb8412', base=profile['revision'])
    signature.update(condition_streams=['static','mesh'], vae_sha256=vae_sha)
    if RANK == 0:
        write_json(out / 'config.json', dict(signature, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in signature},
            clips=dict(Counter(r['domain'] for r in train)), effective_batch=world * args.microbatch * args.accum // args.cp,
            frozen_base_parameters=profile['frozen'], depth=False, native_source_fps=True))
        write_json(out / 'validation_manifest.json', validation)
        if val_loss_rows:
            # Evaluation settings do not invalidate old training-state checkpoints.
            # A new protocol ID separates metrics if the evaluation subset/seed changes.
            write_json(out / f"validation_loss_manifest_{val_protocol['id']}.json", val_protocol)
    config = Cosmos3OmniConfig.from_pretrained(str(args.checkpoint))
    c = config.model['config']
    c.update(action_gen=False, sound_gen=False, vision_gen=True, sound_tokenizer=None,
        precision='bfloat16', resolution='480', lora_enabled=True, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, correct_cp_gradients=True,
        max_num_tokens_after_packing=args.microbatch * (num_items * 21 * (args.height // 32) * (args.width // 32) + 2048))
    c['ema']['enabled'] = False
    # Execution policy, deliberately outside the data/optimizer resume signature.
    # Canonical adapter/AdamW hashes still have to match when changing AC mode.
    c['activation_checkpointing']['mode'] = args.activation_checkpointing
    if args.activation_checkpointing == 'selective':
        # The pretrained preset only lists "fmha" (NATTEN). H200 can choose
        # FlashAttention3, so cover the actual supported attention operators.
        c['activation_checkpointing']['save_ops_regex'] = [
            'fmha', '_flash_attn_forward', '_flash_attn_varlen_forward']
    c['parallelism'].update(data_parallel_shard_degree=world, data_parallel_replicate_degree=1,
        context_parallel_shard_degree=args.cp, cfg_parallel_shard_degree=1,
        enable_inference_mode=False, fsdp_master_dtype='bfloat16')
    c['compile']['enabled'] = False
    c['tokenizer'].update(vae_path=str(args.vae), bucket_name='', object_store_credential_path_pretrained='')
    c['vlm_config']['tokenizer'].update(config_variant='hf', pretrained_model_name=str(args.checkpoint / 'text_tokenizer'))
    if RANK == 0:
        write_json(out / 'model_config.json', config.model)
        shutil.copyfile(Path(__file__), out / 'trainer_source.py')
    Cosmos3OmniModel.before_load_model()
    emit('model_build_start')
    started = time.monotonic()
    model = hydra.utils.instantiate(structure_config(resolve_framework_assets(config.model), omegaconf.DictConfig))
    base = {k: v for k, v in get_model_state_dict(model.net).items() if '.lora_' not in k}
    dcp.load(state_dict=base, storage_reader=_DiffusersHuggingFaceStorageReader(args.checkpoint),
             planner=_DiffusersLoadPlanner(args.checkpoint), no_dist=False)
    del base
    install(model)
    params = canonical_parameters((k, v) for k, v in model.net.named_parameters() if v.requires_grad)
    assert len(params) == profile['tensors'] and all('.lora_' in k for k in params)
    assert sum(v.numel() for v in params.values()) == profile['rank16'] * args.lora_rank // 16
    for v in params.values():
        assert v.dtype == torch.bfloat16 and local(v).numel() * world == v.numel()
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    emit('pretrained_loaded', seconds=time.monotonic() - started, lora_parameters=sum(v.numel() for v in params.values()))

    def adapter_state():
        return {k: v for k, v in get_model_state_dict(model.net).items() if '.lora_' in k}

    def rng_state():
        return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state())

    def restore_rng(saved):
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])

    step = 0
    if args.resume:
        assert (args.resume / 'COMPLETE').is_file()
        meta = json.loads((args.resume / 'resume_metadata.json').read_text())
        assert meta['signature'] == signature, 'Resume config/data/model signature mismatch'
        state = {'adapter': adapter_state(), 'optimizer': get_optimizer_state_dict(model.net, optimizer)}
        dcp.load(state, checkpoint_id=str(args.resume / 'dcp'))
        set_optimizer_state_dict(model.net, optimizer, state['optimizer'])
        del state
        saved = torch.load(args.resume / f'rank{RANK}_state.pt', map_location='cpu', weights_only=False)
        hash_format = meta.get('parameter_hash_format', 'legacy_full_wrapper_names')
        assert hash_format in ('canonical_v1', 'legacy_full_wrapper_names')
        hashed_params = params if hash_format == 'canonical_v1' else legacy_full_parameters(params)
        assert tensor_hash(hashed_params.items()) == saved['adapter_hash'], 'Adapter reload mismatch'
        assert optimizer_hash(optimizer, hashed_params) == saved['optimizer_hash'], 'Optimizer reload mismatch'
        del hashed_params
        step = meta['step']
        assert meta['microstep'] == step * args.accum
        assert meta['next_window'] == step * args.accum // args.cp
        assert saved['next_sample_index'] == (sampler.index(meta['next_window']) if args.microbatch == 1 else WindowSampler(len(train), args.seed, 1, 0).index(positions(meta['next_window'], world, RANK, args.microbatch)[0]))
        restore_rng(saved)
        emit('resume_exact_pass', step=step, next_sample_index=saved['next_sample_index'], adapter=True, optimizer=True, rng=True)
    else:
        torch.manual_seed(args.seed + RANK)
    assert step < args.stop_step
    initial_step = step
    flat_sampler = WindowSampler(len(train), args.seed, 1, 0)

    def first_sample_index(window):
        if args.microbatch == 1:
            return sampler.index(window)
        return flat_sampler.index(positions(window, world, RANK, args.microbatch)[0])
    if args.benchmark_only:
        assert args.stop_step - initial_step > args.benchmark_warmup
    if args.smoke:
        initial_hash = tensor_hash(params.items())

    def make_batch(sample, generate=False, window=None):
        caption = sample['row']['caption']
        if window is not None:
            # Stateless per-sample dropout; independent of worker prefetch and resume.
            absolute = window * world + RANK
            if random.Random(args.seed + 1000003 * (absolute + 1)).random() < args.text_dropout:
                caption = ''
        ids = tokenize_caption(caption, model.vlm_tokenizer, is_video=True, use_system_prompt=model.vlm_config.use_system_prompt)
        assert len(ids) <= 2048
        videos = pixel_inputs(sample, 'cuda', args.height, args.width)
        assert len(videos) == num_items
        if generate:
            videos[-1] = torch.zeros_like(videos[-1])
        batch = {model.input_video_key: videos, model.input_caption_key: [caption],
            'text_token_ids': [torch.tensor([ids], dtype=torch.long, device='cuda')],
            'num_vision_items_per_sample': [num_items],
            'sequence_plan': [SequencePlan(has_text=True, has_vision=True,
                condition_frame_indexes_vision=[], share_vision_temporal_positions=True)],
            'image_size': [torch.tensor([[args.height, args.width, args.height, args.width]], dtype=torch.float32, device='cuda') for _ in range(num_items)],
            'fps': torch.tensor([sample['fps']], device='cuda'),
            'conditioning_fps': torch.tensor([sample['fps']], device='cuda'),
            'num_frames': torch.tensor([81], device='cuda'), 'is_preprocessed': True}
        return attach(batch, [sample])

    if args.verify_cache_parity:
        assert args.latent_cache
        row = train[RANK % len(train)]
        raw_sample = load_sample(args.data_root, row, args.height, args.width)
        cached_sample = sample_loader(args.data_root, row, args.height, args.width)
        verify_native_parity(model, make_batch(raw_sample), make_batch(cached_sample), out, RANK)
        del raw_sample, cached_sample
        gc.collect()
        torch.cuda.empty_cache()

    if args.gradient_audit:
        from .gradient_audit import audit
        audit_samples = [sample_loader(args.data_root, train[flat_sampler.index(i * world + RANK)], args.height, args.width) for i in range(4)]
        audit(model, params, optimizer, lambda i: make_batch(audit_samples[i], window=i), out)
        del audit_samples
        gc.collect()
        torch.cuda.empty_cache()

    original_denoise = model.denoise
    audited = False

    def audit_denoise(*a, **kw):
        nonlocal audited
        if not audited:
            packed = kw['data_batch_packed']
            masks = packed.vision.condition_mask
            assert len(masks) == num_items * args.microbatch
            pos, idx = packed.position_ids.detach().cpu(), packed.vision.sequence_indexes.detach().cpu()
            n = len(idx) // (num_items * args.microbatch)
            assert n * num_items * args.microbatch == len(idx)
            for b in range(args.microbatch):
                group = masks[b*num_items:(b+1)*num_items]
                assert all(torch.all(m == 1) for m in group[:-1]) and torch.all(group[-1] == 0)
                start = b * num_items * n
                assert all(torch.equal(pos[..., idx[start:start+n]], pos[..., idx[start+i*n:start+(i+1)*n]]) for i in range(1, num_items))
            emit('packing_pass', microbatch=args.microbatch, masks=[1] * (num_items - 1) + [0], aligned=True, tokens_per_stream=n)
            audited = True
        return original_denoise(*a, **kw)

    model.denoise = audit_denoise

    def save_checkpoint(completed):
        assert model._cp_window_slot == 0
        folder = out / f'checkpoint_step{completed:06d}'
        if (folder / 'COMPLETE').exists():
            raise FileExistsError(f'Refusing to overwrite completed checkpoint: {folder}')
        folder.mkdir(exist_ok=True)
        dcp.save({'adapter': adapter_state(), 'optimizer': get_optimizer_state_dict(model.net, optimizer)},
                 checkpoint_id=str(folder / 'dcp'))
        torch.save(dict(rng_state(), adapter_hash=tensor_hash(params.items()), optimizer_hash=optimizer_hash(optimizer, params),
            next_sample_index=first_sample_index(completed * args.accum // args.cp)), folder / f'rank{RANK}_state.pt')
        dist.barrier()
        if RANK == 0:
            write_json(folder / 'resume_metadata.json', dict(signature=signature, step=completed,
                microstep=completed * args.accum, next_window=completed * args.accum // args.cp,
                activation_checkpointing=args.activation_checkpointing, parameter_hash_format='canonical_v1'))
            (folder / 'COMPLETE').write_text('All ranks saved.\n')
            tmp = out / 'latest.tmp'
            tmp.write_text(str(folder) + '\n')
            tmp.replace(out / 'latest.txt')
            # Export light evaluation weights synchronously before pruning full states.
            if (completed % args.eval_every == 0 or completed == args.stop_step or completed in args.keep_checkpoint_steps
                or (args.keep_checkpoints_every and completed % args.keep_checkpoints_every == 0)):
                subprocess.run([sys.executable, '-m', 'training.cosmos3.checkpoint',
                    '--checkpoint', str(folder), '--output', str(out / 'evaluation_weights' / f'step{completed:06d}')],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True, timeout=600)
            for old in checkpoints_to_prune(out, keep_recent=1, keep_every=0, keep_steps=()):
                shutil.rmtree(old)
        dist.barrier()
        emit('checkpoint_saved', step=completed, path=str(folder))

    def validate(completed):
        if args.latent_cache:
            model.tokenizer_vision_gen.model.model.to('cuda')
        saved_denoise = model.denoise
        folder = out / 'validation' / f'step{completed:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with isolated_evaluation(model, 'cuda'):
                model.denoise = original_denoise
                if val_loss_rows:
                    def val_inputs(row):
                        # Full caption and real GT target for denoising loss, not generation.
                        return make_batch(sample_loader(args.data_root, row, args.height, args.width))

                    if args.smoke:
                        before_adapter = tensor_hash(params.items())
                        before_optimizer = optimizer_hash(optimizer, params)
                    summary, records = evaluate_loss(model, val_loss_rows, val_inputs,
                        seed=args.val_loss_seed, draws=args.val_loss_noise_draws)
                    if args.smoke:
                        _, repeated = evaluate_loss(model, val_loss_rows, val_inputs,
                            seed=args.val_loss_seed, draws=args.val_loss_noise_draws)
                        import math
                        assert len(records) == len(repeated)
                        for first, second in zip(records, repeated):
                            assert all(first[k] == second[k] for k in ('domain', 'target', 'draw', 'seed', 'sigma'))
                            assert all(math.isclose(first[k], second[k], rel_tol=1e-5, abs_tol=1e-7)
                                       for k in ('loss', 'target_mse'))
                        assert tensor_hash(params.items()) == before_adapter
                        assert optimizer_hash(optimizer, params) == before_optimizer
                        emit('validation_loss_repeat_pass', step=completed, evaluations=len(records))
                    record = dict(event='validation_loss', step=completed, protocol_id=val_protocol['id'],
                        val_loss_droid=summary['domains']['droid']['loss'],
                        val_loss_robocasa=summary['domains']['robocasa']['loss'], **summary)
                    if RANK == 0:
                        write_json(folder / 'loss.json', record)
                        write_json(folder / 'loss_per_sample.json', records)
                        with (out / 'validation_loss.jsonl').open('a') as stream:
                            stream.write(json.dumps(record) + '\n')
                        emit(**record)
                if args.sample_steps > 0:
                    for row in validation:
                        sample = load_sample(args.data_root, row, args.height, args.width)
                        inputs = make_batch(sample, generate=True)
                        assert torch.count_nonzero(inputs[model.input_video_key][-1]).item() == 0
                        tick = time.monotonic()
                        result = model.generate_samples_from_batch(inputs, seed=[args.seed], guidance=1., num_steps=args.sample_steps, shift=5.)
                        decoded = model.decode(result['vision'][0])
                        assert tuple(decoded.shape) == (1, 3, 81, args.height, args.width) and torch.isfinite(decoded).all()
                        if RANK == 0:
                            pixels = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
                            refs = [v[0].permute(1, 2, 3, 0).numpy() for v in sample['videos']]
                            # Save at the native source cadence, including DROID's 15 FPS.
                            from subprocess import run
                            for suffix, frames in [('generated', pixels), ('comparison', np.concatenate([*refs[:-1], pixels, refs[-1]], axis=2))]:
                                frames = np.ascontiguousarray(frames)
                                h, w = frames.shape[1:3]
                                run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                                     '-r', str(sample['fps']), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
                                     '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(folder / f"{row['domain']}_{suffix}.mp4")],
                                    input=frames.tobytes(), check=True)
                            write_json(folder / f"{row['domain']}.json", dict(row, source_fps=sample['fps'], target_input='zeros',
                                steps=args.sample_steps, guidance=1., seconds=time.monotonic() - tick))
                        emit('validation_pass', step=completed, domain=row['domain'], source_fps=sample['fps'])
                        del inputs, result, decoded, sample
        finally:
            model.denoise = saved_denoise
            if args.latent_cache:
                model.tokenizer_vision_gen.model.model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
        dist.barrier()

    stopping = False

    def handle_stop(signum, frame):
        nonlocal stopping
        stopping = True
        emit('stop_requested', signal=signum)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    model.net.train()
    if args.latent_cache:
        model.tokenizer_vision_gen.model.model.to('cpu')
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    executor = ThreadPoolExecutor(max_workers=1)

    def prefetch(window):
        if args.microbatch == 1:
            return executor.submit(sample_loader, args.data_root, train[sampler.index(window)], args.height, args.width)
        def load_many():
            ps = positions(window, world, RANK, args.microbatch)
            return [sample_loader(args.data_root, train[flat_sampler.index(i)], args.height, args.width) for i in ps], ps
        return executor.submit(load_many)

    window = step * args.accum // args.cp
    pending = prefetch(window)
    metrics = (out / f'metrics_rank{RANK}.jsonl').open('a', buffering=1)
    consumed = (out / f'samples_rank{RANK}.jsonl').open('a', buffering=1)
    domain_counts = Counter()
    while step < args.stop_step:
        if args.benchmark_only:
            dist.barrier()
            torch.cuda.synchronize()
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * lr_factor(step, args.warmup, args.schedule_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses = []
        for offset in range(args.accum):
            microstep = step * args.accum + offset
            if microstep % args.cp == 0:
                window = microstep // args.cp
                loaded = pending.result()
                if args.microbatch == 1:
                    sample = loaded
                    row = sample['row']
                    domain_counts[row['domain']] += 1
                    consumed.write(json.dumps(dict(window=window, index=sampler.index(window), target=row['target'],
                        domain=row['domain'], fps=sample['fps'])) + '\n')
                    inputs = make_batch(sample, window=window)
                    del sample
                else:
                    samples, ps = loaded
                    for sample, absolute in zip(samples, ps):
                        row = sample['row']
                        domain_counts[row['domain']] += 1
                        consumed.write(json.dumps(dict(window=window, index=flat_sampler.index(absolute), target=row['target'],
                            domain=row['domain'], fps=sample['fps'])) + '\n')
                    inputs = make_packed_batch(samples, ps, model, tokenize_caption, SequencePlan, seed=args.seed, text_dropout=args.text_dropout)
                    del samples, sample
                del loaded
                pending = prefetch(window + 1) if microstep + args.cp < args.stop_step * args.accum else None
            else:
                inputs = {}  # Native CP cache owns this slot; never advance the loader here.
            auxiliary, loss = model.training_step(inputs, iteration=microstep)
            assert torch.isfinite(loss).all(), f'Non-finite loss at {microstep}'
            (loss / args.accum).backward()
            losses.append(loss.detach().float())
            del auxiliary, loss, inputs
        grads = [local(v.grad).float() for v in params.values() if v.grad is not None]
        assert len(grads) == len(params) and all(torch.isfinite(g).all() for g in grads)
        norm2 = torch.stack([g.square().sum() for g in grads]).sum()
        dist.all_reduce(norm2)
        norm = norm2.sqrt().item()
        assert norm > 0
        if args.benchmark_only and step == initial_step:
            # Sample every trainable tensor before clipping for cross-policy
            # gradient agreement. This IO is only in the excluded first warmup.
            sampled = {}
            for name, value in params.items():
                flat = local(value.grad).detach().flatten()
                stride = max(1, (flat.numel() + 255) // 256)
                sampled[name] = flat[::stride].float().cpu().clone()
            torch.save(sampled, out / f'first_gradients_rank{RANK}.pt')
            del sampled, flat
        for v in params.values():
            v.grad.mul_(min(1., 1. / (norm + 1e-6)))
        optimizer.step()
        mean_loss = torch.stack(losses).mean()
        dist.all_reduce(mean_loss)
        mean_loss /= world
        step += 1
        if args.benchmark_only:
            torch.cuda.synchronize()
        record = dict(step=step, microstep=step * args.accum, next_window=step * args.accum // args.cp,
            loss=mean_loss.item(), lr=lr, grad_norm=norm, seconds=time.monotonic() - tick,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            device_capacity_gib=torch.cuda.get_device_properties(0).total_memory / 2**30, local_domains=dict(domain_counts))
        if args.benchmark_only:
            record['measured'] = step > initial_step + args.benchmark_warmup
        metrics.write(json.dumps(record) + '\n')
        if RANK == 0:
            emit('train_step', **record)
        del grads, norm2, losses, mean_loss
        stop = torch.tensor(int(stopping), device='cuda')
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        stopping = bool(stop.item())
        checkpoint_due = (step % args.save_every == 0 or step == args.stop_step or stopping
                          or step in args.keep_checkpoint_steps
                          or (args.keep_checkpoints_every > 0 and step % args.keep_checkpoints_every == 0))
        eval_due = (args.sample_steps > 0 or bool(val_loss_rows)) and (step % args.eval_every == 0 or step == args.stop_step)
        if args.benchmark_only:
            checkpoint_due = eval_due = False
        if checkpoint_due or eval_due:
            save_checkpoint(step)
        if eval_due and not stopping:
            validate(step)
        if stopping:
            break
    executor.shutdown(wait=True, cancel_futures=True)
    metrics.close()
    consumed.close()
    if args.smoke:
        assert tensor_hash(params.items()) != initial_hash
        all_counts = [None] * world
        dist.all_gather_object(all_counts, dict(domain_counts))
        total_counts = sum((Counter(x) for x in all_counts), Counter())
        assert set(total_counts) == {'droid', 'robocasa'}, total_counts
        emit('mixed_smoke_pass', global_domains=dict(total_counts))
    write_json(out / f'result_rank{RANK}.json', dict(status='CHECKPOINTED_STOP' if stopping else 'PASS',
        step=step, smoke=args.smoke, benchmark_only=args.benchmark_only,
        activation_checkpointing=args.activation_checkpointing,
        local_domains=dict(domain_counts), last_metrics=record))
    if RANK == 0:
        (out / ('STOPPED' if stopping else 'COMPLETE')).write_text(f'{step}\n')
    emit('training_complete', step=step, stopped=stopping)
    dist.barrier()


if __name__ == '__main__':
    main()
