"""Strict, GT-free inference manifest contract."""
import json
import math
from pathlib import Path
import re

from .io import file_hash, probe_video


def read_inputs(path, *, num_shards=1, shard_index=0):
    path = Path(path).resolve()
    if not 0 <= shard_index < num_shards:
        raise ValueError('Require 0 <= shard-index < num-shards')
    rows = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == '.jsonl' else [json.loads(path.read_text())])
    if not rows:
        raise ValueError('Empty input manifest')
    seen = set()
    results = []
    for row in rows:
        allowed = {'id', 'static_image', 'static_video', 'robot_mesh_video', 'prompt', 'prompt_file', 'fps'}
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f'Unknown input fields: {set(row) - allowed if isinstance(row, dict) else row}')
        sid = row.get('id', '')
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sid):
            raise ValueError(f'Unsafe or missing sample id: {sid!r}')
        if sid in seen:
            raise ValueError(f'Duplicate sample id: {sid}')
        seen.add(sid)
        if ('static_image' in row) == ('static_video' in row):
            raise ValueError(f'{sid}: supply exactly one of static_image/static_video')
        if ('prompt' in row) == ('prompt_file' in row):
            raise ValueError(f'{sid}: supply exactly one of prompt/prompt_file')
        if 'robot_mesh_video' not in row:
            raise ValueError(f'{sid}: robot_mesh_video is required')
        row = dict(row)
        for key in ('static_image', 'static_video', 'robot_mesh_video', 'prompt_file'):
            if key in row:
                value = (path.parent / row[key]).resolve()
                if not value.is_file():
                    raise FileNotFoundError(f'{sid}: {key}: {value}')
                row[key] = str(value)
        prompt = row.get('prompt')
        if 'prompt_file' in row:
            source = Path(row['prompt_file'])
            prompt = json.loads(source.read_text()) if source.suffix == '.json' else source.read_text().strip()
        if not isinstance(prompt, (str, dict)) or not prompt:
            raise ValueError(f'{sid}: prompt must be a nonempty string or structured JSON object')
        row['caption'] = (json.dumps(prompt, ensure_ascii=False, separators=(',', ':'))
                          if isinstance(prompt, dict) else prompt)
        results.append(row)
    return results[shard_index::num_shards]


def validate_media(row):
    from PIL import Image
    mesh = probe_video(row['robot_mesh_video'])
    if (mesh['width'], mesh['height']) != (832, 480):
        raise ValueError(f"{row['id']}: prepare mesh at 832x480, got {mesh}")
    if mesh['frames'] < 5 or (mesh['frames'] - 1) % 4:
        raise ValueError(f"{row['id']}: frame count must be 4n+1 (>=5), got {mesh['frames']}")
    if 'fps' in row and (not math.isfinite(row['fps']) or abs(row['fps'] - mesh['fps']) > 1e-4):
        raise ValueError(f"{row['id']}: requested FPS does not match mesh FPS")
    keys = ['robot_mesh_video']
    if 'static_image' in row:
        with Image.open(row['static_image']) as image:
            if image.size != (832, 480):
                raise ValueError(f"{row['id']}: static image must be 832x480")
        keys.append('static_image')
    else:
        static = probe_video(row['static_video'])
        if static != mesh:
            raise ValueError(f"{row['id']}: static/mesh frame count, FPS and resolution must match")
        keys.append('static_video')
    return dict(row, **mesh, input_hashes={key: file_hash(row[key]) for key in keys})
