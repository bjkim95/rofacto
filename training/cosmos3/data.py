"""Exact paper splits, sequential frame decoding, and resumable CP-window sampling."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess


def read_manifest(path):
    raw = Path(path).read_bytes()
    return [json.loads(x) for x in raw.splitlines() if x], hashlib.sha256(raw).hexdigest()


class WindowSampler:
    """One global shuffled permutation; every CP window consumes world_size rows.

    Each rank reads exactly one row per window. The native model rotates these
    rows across cp_size forwards. No rows are skipped at nonzero CP slots.
    Epoch boundaries do not pad/drop rows; sampler uses its own CPU RNG.
    """
    def __init__(self, length, seed, world_size, rank):
        assert length > 0 and 0 <= rank < world_size
        self.length, self.seed, self.world_size, self.rank = length, seed, world_size, rank
        self.epoch = None
        self.order = []

    def index(self, window):
        absolute = window * self.world_size + self.rank
        epoch, offset = divmod(absolute, self.length)
        if self.epoch != epoch:
            self.epoch = epoch
            self.order = list(range(self.length))
            random.Random(self.seed + epoch).shuffle(self.order)
        return self.order[offset]


def lr_factor(completed_steps, warmup, horizon):
    # Same single-cycle cosine-with-hard-restarts shape as the paper; no restart
    # before horizon. Use the update index so the first optimizer update is nonzero.
    update = completed_steps + 1
    if update <= warmup:
        return update / max(1, warmup)
    progress = min(1., (update - warmup) / max(1, horizon - warmup))
    return .5 * (1 + math.cos(math.pi * progress))


def decode_video(path, height, width):
    import numpy as np
    import torch
    meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate,width,height', '-of', 'json', str(path)]))['streams'][0]
    num, den = map(int, meta['r_frame_rate'].split('/'))
    fps = num / den
    assert 0 < fps <= 60 and int(meta['nb_frames']) == 81, (str(path), meta)
    # Preserve all 81 source frames in order: no temporal resampling or FPS filter.
    # Match the paper's direct spatial resize (no crop); bilinear FFmpeg kernel.
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
        '-vf', f'scale={width}:{height}:flags=bilinear', '-threads', '1', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', 'pipe:1'])
    arr = np.frombuffer(raw, np.uint8).reshape(81, height, width, 3).copy()
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0), fps


def vision_keys(row):
    depth = [bool(row.get(key)) for key in ('scene_depth', 'eef_depth')]
    if any(depth) and not all(depth):
        raise ValueError('Scene depth and EEF depth must both be present')
    return ('static', 'mesh', 'scene_depth', 'eef_depth', 'target') if all(depth) else ('static', 'mesh', 'target')


def load_sample(root, row, height, width):
    root = Path(root)
    keys = vision_keys(row)
    target, fps = decode_video(root / row['target'], height, width)
    mesh, mesh_fps = decode_video(root / row['mesh'], height, width)
    assert abs(fps - mesh_fps) < 1e-5, row['target']
    if row['static_mode'] == 'video':
        static, static_fps = decode_video(root / row['static'], height, width)
        assert abs(fps - static_fps) < 1e-5, row['target']
    else:
        assert row['domain'] == 'droid' and row['static_mode'] == 'first_frame_repeat'
        static = target[:, :, :1].expand_as(target)
    values = dict(static=static, mesh=mesh, target=target)
    for key in keys:
        if key not in values:
            video, depth_fps = decode_video(root / row[key], height, width)
            assert abs(fps - depth_fps) < 1e-5, (key, row['target'], fps, depth_fps)
            values[key] = video
    return {'row': row, 'videos': [values[key] for key in keys], 'fps': fps, 'stream_names': keys}
