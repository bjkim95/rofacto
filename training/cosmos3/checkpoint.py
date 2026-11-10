"""Adapter integrity, complete-state retention and CPU-only DCP export."""
from contextlib import contextmanager
import tempfile
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def checkpoints_to_prune(output, *, keep_recent=2, keep_every=1000, keep_steps=()):
    """Completed local checkpoints removable after the new COMPLETE commit.

    Keep comparison milestones and recent resume states. Unknown names,
    incomplete directories and symlinks are never removal candidates.
    """
    import re
    if keep_recent < 1 or keep_every < 0 or any(step <= 0 for step in keep_steps):
        raise ValueError('Invalid checkpoint retention settings')
    explicit = set(keep_steps)
    completed = []
    for folder in Path(output).iterdir():
        match = re.fullmatch(r'checkpoint_step(\d+)', folder.name)
        if (match and folder.is_dir() and not folder.is_symlink()
                and (folder/'COMPLETE').is_file()):
            completed.append((int(match.group(1)), folder))
    completed.sort(key=lambda item: item[0])
    recent = {folder for _, folder in completed[-keep_recent:]}
    return [folder for step, folder in completed
            if folder not in recent and step not in explicit
            and not (keep_every and step % keep_every == 0)]


def local(t):
    return t.to_local() if hasattr(t, 'to_local') else t


def canonical_parameters(named):
    """Keep integrity hashes independent of PyTorch's checkpoint wrapper names."""
    values = list(named)
    result = {name.replace('_orig_mod.', '').replace('_checkpoint_wrapped_module.', ''): value for name, value in values}
    assert len(result) == len(values), 'Canonical parameter-name collision'
    return result


def legacy_full_parameters(canonical):
    """Recreate the names used by our pre-AC-toggle full-checkpoint LoRA saves."""
    import re
    result = {}
    for name, value in canonical.items():
        legacy, count = re.subn(r'(^|\.)(layers\.\d+\.)',
                               r'\1\2_checkpoint_wrapped_module.', name)
        assert count == 1, f'Unexpected legacy LoRA location: {name}'
        result[legacy] = value
    return result


def tensor_hash(named):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(named):
        tensor = local(value).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_hash(optimizer, trainable):
    tensors = []
    for name, param in trainable.items():
        for key, value in optimizer.state[param].items():
            if hasattr(value, 'shape'):
                tensors.append((name + '/' + key, value))
    return tensor_hash(tensors)



@contextmanager
def atomic_export_directory(output):
    """Publish a new export only after all files and its COMPLETE marker exist."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Export destination already exists; choose a new path: {output}')
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    try:
        yield temporary
        if not (temporary / 'COMPLETE').is_file():
            raise RuntimeError('Export writer did not finish')
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file
    checkpoint = args.checkpoint
    assert (checkpoint / 'COMPLETE').is_file()
    meta = json.loads((checkpoint / 'resume_metadata.json').read_text())
    reader = dcp.FileSystemReader(checkpoint / 'dcp')
    metadata = reader.read_metadata()
    adapter = {k.removeprefix('adapter.'): torch.empty(v.size, dtype=v.properties.dtype)
               for k, v in metadata.state_dict_metadata.items() if k.startswith('adapter.')}
    from . import PROFILES
    rank = meta['signature']['lora_rank']
    expected = {v['revision']: (v['tensors'], v['rank16'] * rank // 16) for v in PROFILES.values()}
    assert (len(adapter), sum(t.numel() for t in adapter.values())) == expected[meta['signature']['base']]
    dcp.load({'adapter': adapter}, storage_reader=reader, no_dist=True)
    world = meta['signature']['world']
    for rank in range(world):
        saved = torch.load(checkpoint / f'rank{rank}_state.pt', map_location='cpu', weights_only=False)
        assert all(v.shape[0] % world == 0 for v in adapter.values())
        shards = canonical_parameters((k, v.chunk(world, dim=0)[rank]) for k, v in adapter.items())
        if meta.get('parameter_hash_format', 'legacy_full_wrapper_names') == 'legacy_full_wrapper_names':
            shards = legacy_full_parameters(shards)
        assert tensor_hash(shards.items()) == saved['adapter_hash'], rank
    adapter = canonical_parameters(adapter.items())
    digest = tensor_hash(adapter.items())
    with atomic_export_directory(args.output) as destination:
        save_file(adapter, destination / 'adapter.safetensors', metadata={'step': str(meta['step']), 'sha256': digest})
        shutil.copy2(checkpoint.parent / 'model_config.json', destination / 'model_config.json')
        (destination / 'export.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
            checkpoint_step=meta['step'], adapter_hash=digest, metadata=meta, verified_ranks=world), indent=2))
        (destination / 'COMPLETE').write_text(digest+'\n')
    print('ADAPTER_EXPORT_VERIFIED', meta['step'], digest, flush=True)


if __name__ == '__main__':
    main()
