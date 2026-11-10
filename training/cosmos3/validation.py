"""Deterministic held-out flow-matching evaluation for the native CP-window trainer.

No optimizer/backward calls. Every CP window is completed, including dummy
padding forwards, but each real clip/draw is counted exactly once globally.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import random
import time
from .data import vision_keys

DOMAINS = ('droid', 'robocasa')
PROTOCOL_VERSION = 1


def stable_seed(seed, *parts):
    raw = json.dumps([seed, *parts], ensure_ascii=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little')


def select_validation_rows(rows, per_domain, seed):
    """Stable clip subset, independent of manifest ordering and training RNG."""
    if per_domain < 0:
        raise ValueError('Validation clips per domain must be nonnegative')
    if per_domain == 0:
        return []
    selected = []
    for domain in DOMAINS:
        candidates = sorted((r for r in rows if r['domain'] == domain), key=lambda r: r['target'])
        if not candidates:
            raise ValueError(f'No held-out clips for {domain}')
        if len({r['target'] for r in candidates}) != len(candidates):
            raise ValueError(f'Duplicate validation target in {domain}')
        if any(r.get('split') != 'test' for r in candidates):
            raise ValueError('Validation loss requires held-out test rows')
        random.Random(stable_seed(seed, domain, 'subset')).shuffle(candidates)
        selected.extend(candidates[:per_domain])
    return selected


def validation_protocol(rows, *, seed, draws, per_domain, test_sha256):
    if draws < 1:
        raise ValueError('Validation noise draws must be positive')
    protocol = dict(version=PROTOCOL_VERSION, seed=seed, draws=draws,
                    requested_clips_per_domain=per_domain, test_sha256=test_sha256,
                    rows=rows)
    raw = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
    return dict(protocol, id=hashlib.sha256(raw).hexdigest())


def _seed_noise(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed(seed)


@contextmanager
def isolated_evaluation(model, device):
    """Restore RNG, exact module modes, native counters and CP cache on exit."""
    import numpy as np
    import torch
    if model._cp_window_slot != 0 or model._cp_local_training_payload is not None:
        raise RuntimeError('Validation must start at a completed CP-window boundary')
    py_rng, np_rng = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.net.modules()]
    counters = {}
    for name in ('accum_image_sample_counter', 'accum_video_sample_counter'):
        if hasattr(model.net, name):
            value = getattr(model.net, name)
            counters[name] = (value, value.clone() if torch.is_tensor(value) else value)
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        try:
            model.net.eval()
            yield
        finally:
            model._cp_window_slot = 0
            model._cp_local_training_payload = None
            for name, (original, saved) in counters.items():
                if torch.is_tensor(original):
                    original.copy_(saved)
                setattr(model.net, name, original)
            for module, was_training in modes:
                module.training = was_training
            random.setstate(py_rng)
            np.random.set_state(np_rng)


@contextmanager
def _fixed_noise(model, active_seed, device):
    """Seed at the native noise boundaries, independent of VAE RNG consumption.

    iteration=None bypasses upstream's optional (iteration, global-rank) RNGs.
    This keeps a clip/draw's sigma and epsilon identical across CP assignments
    as well as across checkpoints. The native training noise distribution,
    condition masks, resolution shift and loss weighting remain in use.
    """
    names = ('_get_train_noise_level_vision', '_add_noise_to_input')
    originals = {name: getattr(model, name) for name in names}
    instance_values = {name: model.__dict__[name] for name in names if name in model.__dict__}

    def wrapper(name):
        def call(*args, **kwargs):
            _seed_noise(stable_seed(active_seed[0], name), device)
            kwargs['iteration'] = None
            return originals[name](*args, **kwargs)
        return call

    try:
        for name in names:
            setattr(model, name, wrapper(name))
        yield
    finally:
        for name in names:
            if name in instance_values:
                setattr(model, name, instance_values[name])
            else:
                delattr(model, name)


def summarize_records(records, rows, draws):
    expected = {(r['domain'], r['target'], draw) for r in rows for draw in range(draws)}
    keys = [(r['domain'], r['target'], r['draw']) for r in records]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise RuntimeError('Validation coverage mismatch: missing, duplicate, or unexpected clip/draw')
    if not all(math.isfinite(r[k]) for r in records for k in ('loss', 'target_mse', 'sigma')):
        raise RuntimeError('Non-finite validation loss/noise level')
    domains = {}
    for domain in DOMAINS:
        values = [r for r in records if r['domain'] == domain]
        domains[domain] = dict(
            loss=sum(r['loss'] for r in values) / len(values),
            target_mse=sum(r['target_mse'] for r in values) / len(values),
            clips=len({r['target'] for r in values}), evaluations=len(values))
    return dict(
        val_loss=sum(d['loss'] for d in domains.values()) / len(domains),
        val_target_mse=sum(d['target_mse'] for d in domains.values()) / len(domains),
        domains=domains, averaging='equal_domain_mean',
        evaluations=len(records), clips=len(rows))


def evaluate_loss(model, rows, make_inputs, *, seed, draws, device='cuda'):
    """Evaluate one clip/rank/CP window using the existing native training loss.

    make_inputs(row) must retain the real target and disable text dropout.
    All ranks call this function. CP rank 0 reports its group's result once;
    dummy rows only align collectives and never enter the metrics.
    """
    import torch
    import torch.distributed as dist
    if draws < 1 or not rows:
        raise ValueError('Validation needs clips and positive noise draws')
    distributed = dist.is_initialized()
    world, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
    pd = model.parallel_dims
    if pd is not None and pd.cp_enabled:
        cp = pd.cp_mesh.size()
        cp_rank = pd.cp_rank
        cp_group = pd.cp_mesh.get_group()
        owners = [dist.get_global_rank(cp_group, slot) for slot in range(cp)]
    else:
        cp, cp_rank, owners = 1, 0, [rank]
    active_seed = [seed]
    records = []
    started = time.monotonic()
    with isolated_evaluation(model, device), _fixed_noise(model, active_seed, device):
        for domain in DOMAINS:
            domain_rows = [r for r in rows if r['domain'] == domain]
            if not domain_rows:
                raise ValueError(f'Missing validation domain: {domain}')
            for draw in range(draws):
                for start in range(0, len(domain_rows), world):
                    local_index = start + rank
                    local_row = domain_rows[local_index] if local_index < len(domain_rows) else domain_rows[0]
                    inputs = make_inputs(local_row)
                    for slot in range(cp):
                        owner_index = start + owners[slot]
                        valid = owner_index < len(domain_rows)
                        row = domain_rows[owner_index] if valid else domain_rows[0]
                        active_seed[0] = stable_seed(seed, domain, row['target'], draw)
                        auxiliary, loss = model.training_step(inputs if slot == 0 else {}, iteration=0)
                        if slot == 0:
                            del inputs
                        if cp_rank == 0 and valid:
                            per_item = auxiliary['flow_matching_loss_vision_per_instance'].detach().reshape(-1)
                            masks = auxiliary['condition_mask_vision']
                            num_items = len(vision_keys(row))
                            if (per_item.numel() != num_items or len(masks) != num_items
                                    or not all(bool(torch.all(m == 1)) for m in masks[:-1])
                                    or not bool(torch.all(masks[-1] == 0))):
                                raise RuntimeError('Expected clean condition streams and fully generated target')
                            records.append(dict(domain=domain, target=row['target'], draw=draw,
                                seed=active_seed[0], sigma=auxiliary['sigma'].detach().reshape(-1)[-1].item(),
                                loss=loss.detach().item(), target_mse=per_item[-1].item()))
                        del auxiliary, loss
                    if model._cp_window_slot != 0 or model._cp_local_training_payload is not None:
                        raise RuntimeError('Validation left an incomplete CP window')
        if distributed:
            gathered = [None] * world
            dist.all_gather_object(gathered, records)
            records = [r for rank_records in gathered for r in rank_records]
        records.sort(key=lambda r: (r['domain'], r['target'], r['draw']))
        summary = summarize_records(records, rows, draws)
        elapsed = torch.tensor(time.monotonic() - started, dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        summary.update(seconds=elapsed.item(), world_size=world, cp_size=cp,
                       seed=seed, noise_draws=draws)
    return summary, records
