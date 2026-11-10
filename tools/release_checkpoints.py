"""Prepare allowlisted inference weights, then publish verified HF model repos.

Run on a CPU worker. No base weights, optimizer state, dataset, or credentials
are collected. Source directories are read-only; destinations must be new.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WAN_FILES = ('lora_diffusion_pytorch_model.safetensors', 'non_lora_weights.safetensors')
COSMOS_FILES = ('adapter.safetensors', 'model_config.json', 'export.json', 'COMPLETE')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def public_config(value, key=''):
    """Remove installation-specific locations, preserving portable package assets."""
    if isinstance(value, dict):
        return {k: public_config(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [public_config(v, key) for v in value]
    if isinstance(value, str):
        if any(word in key.lower() for word in ('credential', 'secret', 'password')):
            return ''
        if value.startswith('/'):
            if '/cosmos_framework/' in value:
                return 'cosmos_framework/' + value.split('/cosmos_framework/', 1)[1]
            return Path(value).name
    return value


def validate_wan(source, channels):
    from safetensors import safe_open
    if channels not in (16, 48):
        raise ValueError('Wan release supports 16 or 48 condition channels')
    with safe_open(source / WAN_FILES[0], framework='pt', device='cpu') as f:
        names = list(f.keys())
        if not names or not all(name.startswith('lora_') for name in names):
            raise ValueError('Expected LoRA-only Wan weights')
        if not any('lora_down' in name or 'lora_A' in name for name in names):
            raise ValueError('Missing Wan LoRA projection tensors')
    with safe_open(source / WAN_FILES[1], framework='pt', device='cpu') as f:
        if set(f.keys()) != {'patch_embedding.weight', 'patch_embedding.bias'}:
            raise ValueError('Unexpected non-LoRA tensors in Wan export')
        if f.get_slice('patch_embedding.weight').get_shape() != [5120, 36 + channels, 1, 2, 2]:
            raise ValueError('Wan patch embedding does not match requested condition channels')


def validate_cosmos(source, variant, full=False):
    from training.cosmos3.model import inspect_export
    from training.cosmos3 import PROFILES
    from safetensors import safe_open
    config, export = inspect_export(source, variant)
    digest = export['adapter_hash']
    if (source / 'COMPLETE').read_text().strip() != digest:
        raise ValueError('COMPLETE does not match adapter hash')
    with safe_open(source / 'adapter.safetensors', framework='pt', device='cpu') as f:
        names = list(f.keys())
        if len(names) != PROFILES[variant]['tensors']:
            raise ValueError('Unexpected Cosmos adapter tensor count')
        if not all('.lora_' in name for name in names):
            raise ValueError('Expected LoRA-only Cosmos weights')
    if full:
        from safetensors.torch import load_file
        from training.cosmos3.checkpoint import tensor_hash
        if tensor_hash(load_file(source / 'adapter.safetensors').items()) != digest:
            raise ValueError('Cosmos tensor hash mismatch')
    return config, export


def validate_manifest(manifest):
    seen = set()
    for model in manifest['models']:
        repo = model['repo_id']
        if not re.fullmatch(r'[\w-]+/[\w.-]+', repo) or repo in seen:
            raise ValueError(f'Invalid or duplicate repo: {repo}')
        seen.add(repo)
        for key in ('card', 'license', 'notice'):
            if not (ROOT / model[key]).is_file():
                raise FileNotFoundError(model[key])
        if model['backend'] == 'wan':
            subdirs = set()
            for variant in model['variants']:
                subdir = variant['subdir']
                if not re.fullmatch(r'[\w-]+', subdir) or subdir in subdirs:
                    raise ValueError('Invalid or duplicate variant directory')
                subdirs.add(subdir)
                validate_wan(Path(variant['source']), variant['condition_channels'])
                if not (ROOT / variant['config']).is_file():
                    raise FileNotFoundError(variant['config'])
        elif model['backend'] in ('nano', 'super'):
            validate_cosmos(Path(model['source']), model['backend'])
        else:
            raise ValueError(f"Unsupported backend: {model['backend']}")
    if not seen:
        raise ValueError('Empty release')


def prepare(manifest, output):
    validate_manifest(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Release destination exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.rofacto-release-', dir=output.parent))
    records = []
    try:
        for model in manifest['models']:
            folder = temporary / model['repo_id'].replace('/', '--')
            folder.mkdir()
            for key, name in (('card', 'README.md'), ('license', 'LICENSE'), ('notice', 'NOTICE')):
                shutil.copyfile(ROOT / model[key], folder / name)
            provenance = {'backend': model['backend'], 'source_code': manifest['source_code'],
                          'weights_only': True, 'optimizer_state_included': False}
            if model['backend'] == 'wan':
                provenance['variants'] = []
                for variant in model['variants']:
                    dest = folder / variant['subdir']
                    dest.mkdir()
                    for name in WAN_FILES:
                        shutil.copyfile(Path(variant['source']) / name, dest / name)
                    shutil.copyfile(ROOT / variant['config'], dest / 'inference.yaml')
                    provenance['variants'].append({k: v for k, v in variant.items() if k not in ('source', 'config')})
            else:
                source = Path(model['source'])
                config, export = validate_cosmos(source, model['backend'], full=True)
                shutil.copyfile(source / 'adapter.safetensors', folder / 'adapter.safetensors')
                write_json(folder / 'model_config.json', public_config(config))
                # Training signature and tensor identity are preserved; no private mount path.
                export = {k: v for k, v in export.items() if k != 'checkpoint'}
                write_json(folder / 'export.json', public_config(export))
                shutil.copyfile(source / 'COMPLETE', folder / 'COMPLETE')
                validate_cosmos(folder, model['backend'], full=True)
                signature = export['metadata']['signature']
                provenance.update(checkpoint_step=export['checkpoint_step'], adapter_hash=export['adapter_hash'],
                    effective_batch=signature['world'] * signature.get('microbatch', 1) * signature['accum'] // signature['cp'])
            write_json(folder / 'release_metadata.json', provenance)
            hashes = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
            (folder / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
            hashes['SHA256SUMS'] = sha256(folder / 'SHA256SUMS')
            records.append({'repo_id': model['repo_id'], 'folder': folder.name, 'sha256': hashes})
        write_json(temporary / 'release.json', {'models': records, 'source_code': manifest['source_code']})
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({'status': 'PREPARED', 'path': str(output), 'repos': [r['repo_id'] for r in records]}))


def verified_payload(root, record):
    root = Path(root).resolve()
    folder = (root / record['folder']).resolve()
    if folder.parent != root:
        raise ValueError('Invalid prepared folder')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != set(record['sha256']):
        raise ValueError('Prepared directory changed; unexpected or missing files')
    for name, digest in record['sha256'].items():
        path = folder / name
        if path.is_symlink() or root not in path.resolve().parents or sha256(path) != digest:
            raise ValueError(f'Prepared file changed: {name}')
    return folder


def publish(root, namespace):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    root = Path(root)
    records = json.loads((root / 'release.json').read_text())['models']
    # Verify every local package before creating any remote repository.
    for record in records:
        if record['repo_id'].split('/')[0] != namespace:
            raise ValueError('Repo owner differs from --namespace')
        verified_payload(root, record)
    api = HfApi()
    identity = api.whoami()
    if identity['name'] != namespace:
        raise ValueError('This release command requires the authenticated personal namespace')
    receipts = []
    for record in records:
        repo = record['repo_id']
        folder = root / record['folder']
        api.create_repo(repo, repo_type='model', private=True, exist_ok=True)
        info = api.model_info(repo, files_metadata=True)
        existing = {f.rfilename for f in info.siblings} - {'.gitattributes'}
        if existing:
            if existing != set(record['sha256']):
                raise ValueError(f'{repo} already has different content; refusing to overwrite')
        else:
            api.create_commit(repo, repo_type='model', parent_commit=info.sha,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(folder / name))
                            for name in record['sha256']],
                commit_message='Release RoFacto inference adapters and reproduction metadata')
            info = api.model_info(repo, files_metadata=True)
        remote = {f.rfilename: f for f in info.siblings}
        if set(remote) - {'.gitattributes'} != set(record['sha256']):
            raise ValueError('Remote file set differs from release')
        for name, expected in record['sha256'].items():
            entry = remote[name]
            lfs = entry.lfs
            digest = getattr(lfs, 'sha256', None) if lfs else None
            if digest is None:
                digest = sha256(hf_hub_download(repo, name, revision=info.sha))
            if digest != expected:
                raise ValueError(f'Remote content mismatch: {repo}/{name}')
        api.update_repo_settings(repo, repo_type='model', private=False)
        if api.model_info(repo, token=False).private:
            raise ValueError('Anonymous public-access verification failed')
        receipts.append({'repo_id': repo, 'revision': info.sha, 'url': f'https://huggingface.co/{repo}', 'status': 'PUBLIC_VERIFIED'})
        write_json(root / 'publication.json', receipts)
        print(json.dumps(receipts[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'prepare'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--output', type=Path, required=True)
    command = commands.add_parser('publish')
    command.add_argument('--prepared', type=Path, required=True)
    command.add_argument('--namespace', required=True)
    args = parser.parse_args()
    if args.command == 'publish':
        publish(args.prepared, args.namespace)
    else:
        manifest = json.loads(args.manifest.read_text())
        if args.command == 'prepare':
            prepare(manifest, args.output)
        else:
            validate_manifest(manifest)
            print('RELEASE_INPUTS_VALIDATED')


if __name__ == '__main__':
    main()
