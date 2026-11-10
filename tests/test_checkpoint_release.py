import hashlib
import json
from types import SimpleNamespace

import pytest

from tools import release_checkpoints as release


def prepared_fixture(tmp_path):
    folder = tmp_path / 'owner--model'
    folder.mkdir()
    (folder / 'README.md').write_text('model card\n')
    record = {'repo_id': 'owner/model', 'folder': folder.name,
              'sha256': {'README.md': release.sha256(folder / 'README.md')}}
    release.write_json(tmp_path / 'release.json', {'models': [record]})
    return folder, record


def test_prepare_uses_only_inference_allowlist(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    source = tmp_path / 'training'
    source.mkdir()
    save_file({'lora_unet_block.lora_down.weight': torch.zeros(2, 3)}, source / release.WAN_FILES[0])
    save_file({'patch_embedding.weight': torch.zeros(5120, 52, 1, 2, 2, dtype=torch.bfloat16),
               'patch_embedding.bias': torch.zeros(5120, dtype=torch.bfloat16)}, source / release.WAN_FILES[1])
    (source / 'optimizer.pt').write_bytes(b'private optimizer state')
    (source / 'private_prompt.json').write_text('private input')
    for name in ('card.md', 'LICENSE', 'NOTICE', 'inference.yaml'):
        (tmp_path / name).write_text('public example\n')
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    manifest = {'source_code': 'abc123', 'models': [{
        'repo_id': 'owner/model', 'backend': 'wan', 'card': 'card.md',
        'license': 'LICENSE', 'notice': 'NOTICE', 'variants': [{
            'subdir': 'rgb', 'source': str(source), 'condition_channels': 16,
            'config': 'inference.yaml', 'step': 8000}]}]}
    output = tmp_path / 'prepared'
    release.prepare(manifest, output)
    record = json.loads((output / 'release.json').read_text())['models'][0]
    assert not any('optimizer' in name or 'prompt' in name for name in record['sha256'])
    assert set(record['sha256']) == {
        'README.md', 'LICENSE', 'NOTICE', 'release_metadata.json', 'SHA256SUMS',
        'rgb/inference.yaml', *('rgb/' + name for name in release.WAN_FILES)}
    assert release.verified_payload(output, record).is_dir()
    assert (source / 'optimizer.pt').read_bytes() == b'private optimizer state'
    with pytest.raises(FileExistsError):
        release.prepare(manifest, output)
    with pytest.raises(ValueError, match='condition channels'):
        release.validate_wan(source, 48)


def test_public_config_preserves_portable_assets_and_removes_private_paths():
    original = {'vae_path': '/private/models/Wan2.2_VAE.pth',
                'credentials_path': 'credentials/gcp_checkpoint.secret',
                'json_file': '/private/code/cosmos_framework/configs/model.json',
                'lora_rank': 64}
    public = release.public_config(original)
    assert public == {'vae_path': 'Wan2.2_VAE.pth', 'credentials_path': '',
                      'json_file': 'cosmos_framework/configs/model.json', 'lora_rank': 64}
    assert original['vae_path'].startswith('/private/')


def test_modified_payload_is_rejected_before_hub_access(tmp_path, monkeypatch):
    import huggingface_hub
    folder, _ = prepared_fixture(tmp_path)
    (folder / 'extra-secret.txt').write_text('must not upload')
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: pytest.fail('Hub must not be touched'))
    with pytest.raises(ValueError, match='unexpected or missing'):
        release.publish(tmp_path, 'owner')


@pytest.mark.parametrize('match', [False, True])
def test_remote_hash_must_match_before_publication(tmp_path, monkeypatch, match):
    import huggingface_hub
    _, record = prepared_fixture(tmp_path)
    calls = []

    class FakeAPI:
        def whoami(self):
            return {'name': 'owner'}

        def create_repo(self, repo, **kwargs):
            assert kwargs['private'] is True
            calls.append('private')

        def model_info(self, repo, **kwargs):
            if kwargs.get('token') is False:
                assert 'public' in calls
                return SimpleNamespace(private=False)
            digest = record['sha256']['README.md'] if match else hashlib.sha256(b'wrong').hexdigest()
            return SimpleNamespace(sha='abc123', siblings=[SimpleNamespace(
                rfilename='README.md', lfs=SimpleNamespace(sha256=digest))])

        def update_repo_settings(self, repo, **kwargs):
            assert kwargs['private'] is False
            calls.append('public')

    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeAPI)
    if match:
        release.publish(tmp_path, 'owner')
        assert calls == ['private', 'public']
        assert json.loads((tmp_path / 'publication.json').read_text())[0]['status'] == 'PUBLIC_VERIFIED'
    else:
        with pytest.raises(ValueError, match='Remote content mismatch'):
            release.publish(tmp_path, 'owner')
        assert calls == ['private']
