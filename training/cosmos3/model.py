"""Pinned upstream loading and bounded mixed GPU/CPU FSDP residency."""
from contextlib import contextmanager
import gc
import importlib
import json
from pathlib import Path
import time

from . import PROFILES, FRAMEWORK_REVISION
from .checkpoint import local, tensor_hash, canonical_parameters
from .io import emit, file_hash


def resolve_framework_assets(config):
    """Resolve upstream JSON assets without requiring the upstream repository CWD."""
    import cosmos_framework
    root = Path(cosmos_framework.__file__).resolve().parent
    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and (value.startswith('cosmos_framework/')
                or (value.startswith('/') and '/cosmos_framework/' in value)):
            suffix = value.split('cosmos_framework/', 1)[1]
            path = root / suffix
            if not path.is_file():
                raise FileNotFoundError(f'Missing upstream package asset: {path}')
            return str(path)
        return value
    return resolve(config)


def inspect_export(source, variant):
    source = Path(source)
    for name in ('COMPLETE', 'export.json', 'model_config.json', 'adapter.safetensors'):
        if not (source / name).is_file():
            raise ValueError(f'Incomplete inference export: {source / name}')
    export = json.loads((source / 'export.json').read_text())
    config = json.loads((source / 'model_config.json').read_text())
    signature = export['metadata']['signature']
    if signature.get('framework') != FRAMEWORK_REVISION:
        raise ValueError('Adapter uses a different Cosmos framework revision')
    if signature['base'] != PROFILES[variant]['revision']:
        raise ValueError(f'Adapter was not trained from the pinned {variant} base')
    c = config['config']
    if not c.get('lora_enabled') or c['lora_rank'] != signature['lora_rank'] or c['lora_alpha'] != signature['lora_alpha']:
        raise ValueError('LoRA rank/alpha/config/export mismatch')
    if signature.get('condition_streams', ['static', 'mesh']) != ['static', 'mesh']:
        raise ValueError('Only static RGB + mesh RGB adapters are supported')
    return config, export


@contextmanager
def mixed_offload(resident, pin_memory):
    from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy
    module = importlib.import_module('cosmos_framework.model.generator.mot.parallelize_unified_mot')
    original = module.CPUOffloadPolicy
    count = [0]
    def policy():
        index = count[0]
        count[0] += 1
        return OffloadPolicy() if index < resident else CPUOffloadPolicy(pin_memory=pin_memory)
    module.CPUOffloadPolicy = policy
    try:
        yield count
    finally:
        module.CPUOffloadPolicy = original


def load_inference_model(base, vae, source, variant, max_frames, resident, pin_memory):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from safetensors.torch import load_file
    from cosmos_framework.inference.common.init import init_script, get_free_port
    from cosmos_framework.inference.common.config import structure_config
    from cosmos_framework.inference.model import Cosmos3OmniModel, _DiffusersHuggingFaceStorageReader, _DiffusersLoadPlanner
    import hydra
    import omegaconf
    config, export = inspect_export(source, variant)
    layers = PROFILES[variant]['layers']
    if not 0 <= resident <= layers:
        raise ValueError(f'resident-layers must be in [0, {layers}]')
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    init_script(training=False)
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{get_free_port()}', rank=0, world_size=1)
    if dist.get_world_size() != 1:
        raise ValueError('This inference entrypoint requires world_size=1')
    c = config['config']
    c['parallelism'].update(data_parallel_shard_degree=1, data_parallel_replicate_degree=1,
        context_parallel_shard_degree=1, cfg_parallel_shard_degree=1, enable_inference_mode=True,
        fsdp_cpu_offload=True, fsdp_master_dtype='bfloat16')
    c['activation_checkpointing']['mode'] = 'none'
    c['compile']['enabled'] = False
    c['max_num_tokens_after_packing'] = 3 * (1 + (max_frames - 1) // 4) * 15 * 26 + 8192
    c['tokenizer']['vae_path'] = str(vae)
    c['vlm_config']['tokenizer']['pretrained_model_name'] = str(Path(base) / 'text_tokenizer')
    Cosmos3OmniModel.before_load_model()
    tick = time.monotonic()
    emit('model_build_start', model=variant, resident_layers=resident, gpu=torch.cuda.get_device_name())
    with mixed_offload(resident, pin_memory) as count:
        model = hydra.utils.instantiate(structure_config(resolve_framework_assets(config), omegaconf.DictConfig))
    assert count[0] == len(model.net.language_model.model.layers) == layers
    state = {k: v for k, v in get_model_state_dict(model.net).items() if '.lora_' not in k}
    dcp.load(state, storage_reader=_DiffusersHuggingFaceStorageReader(Path(base)),
             planner=_DiffusersLoadPlanner(Path(base)), no_dist=False)
    del state
    adapter = load_file(Path(source) / 'adapter.safetensors', device='cpu')
    assert tensor_hash(adapter.items()) == export['adapter_hash'], 'Corrupt adapter'
    params = canonical_parameters((k, v) for k, v in model.net.named_parameters() if v.requires_grad)
    assert set(params) == set(adapter), 'Unexpected LoRA target names'
    for name, value in params.items():
        local(value).copy_(adapter[name])
    assert tensor_hash(params.items()) == export['adapter_hash'], 'Adapter copy mismatch'
    del adapter, params
    model.net.eval()
    gc.collect()
    torch.cuda.empty_cache()
    emit('model_loaded', seconds=time.monotonic() - tick, gpu_gib=torch.cuda.memory_allocated()/2**30)
    return model, export
