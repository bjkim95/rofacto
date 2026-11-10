import json
import pytest
from training.cosmos3 import PROFILES, FRAMEWORK_REVISION
from training.cosmos3.model import inspect_export
from training.cosmos3.checkpoint import checkpoints_to_prune


def export_fixture(path):
    for name in ('COMPLETE','adapter.safetensors'):
        (path/name).touch()
    signature=dict(base=PROFILES['nano']['revision'],framework=FRAMEWORK_REVISION,lora_rank=64,lora_alpha=64)
    export=dict(metadata=dict(signature=signature))
    (path/'export.json').write_text(json.dumps(export))
    (path/'model_config.json').write_text(json.dumps(dict(config=dict(lora_enabled=True,lora_rank=64,lora_alpha=64))))
    return export


def test_variant_and_condition_mismatch_are_rejected(tmp_path):
    export=export_fixture(tmp_path)
    inspect_export(tmp_path,'nano')
    with pytest.raises(ValueError,match='super base'):
        inspect_export(tmp_path,'super')
    export['metadata']['signature']['condition_streams']=['static','mesh','scene_depth','eef_depth']
    (tmp_path/'export.json').write_text(json.dumps(export))
    with pytest.raises(ValueError,match='Only static RGB'):
        inspect_export(tmp_path,'nano')


def test_latest_full_state_retains_exports_and_incomplete(tmp_path):
    for step in (4,8):
        folder=tmp_path/f'checkpoint_step{step:06d}'
        folder.mkdir()
        (folder/'COMPLETE').touch()
    incomplete=tmp_path/'checkpoint_step000012'
    incomplete.mkdir()
    exports=tmp_path/'evaluation_weights'
    exports.mkdir()
    assert checkpoints_to_prune(tmp_path,keep_recent=1,keep_every=0) == [tmp_path/'checkpoint_step000004']


def test_upstream_assets_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from training.cosmos3.model import resolve_framework_assets
    package=tmp_path/'installed/cosmos_framework'
    (package/'configs').mkdir(parents=True)
    (package/'configs/model.json').write_text('{}')
    monkeypatch.setitem(sys.modules,'cosmos_framework',SimpleNamespace(__file__=str(package/'__init__.py')))
    original={'model':{'json_file':'cosmos_framework/configs/model.json'}}
    monkeypatch.chdir(tmp_path)
    resolved=resolve_framework_assets(original)
    assert resolved['model']['json_file']==str(package/'configs/model.json')
    assert original['model']['json_file']=='cosmos_framework/configs/model.json'


def test_failed_export_is_not_published_and_completed_export_is_immutable(tmp_path):
    from training.cosmos3.checkpoint import atomic_export_directory
    output=tmp_path/'export'
    with pytest.raises(RuntimeError,match='simulated'):
        with atomic_export_directory(output) as folder:
            (folder/'adapter.safetensors').write_bytes(b'partial')
            raise RuntimeError('simulated interruption')
    assert not output.exists() and list(tmp_path.iterdir())==[]
    with atomic_export_directory(output) as folder:
        (folder/'adapter.safetensors').write_bytes(b'weights')
        (folder/'COMPLETE').write_text('done')
    with pytest.raises(FileExistsError):
        with atomic_export_directory(output):
            pass
    assert (output/'adapter.safetensors').read_bytes()==b'weights'
