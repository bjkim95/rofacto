"""Atomic metadata and FFmpeg video IO shared by CLI entrypoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from fractions import Fraction


def emit(event, **fields):
    print(json.dumps({'event': event, 'rank': int(os.environ.get('RANK', 0)), **fields}), flush=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def probe_video(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,avg_frame_rate',
        '-of', 'json', str(path)], timeout=180))
    if not data.get('streams'):
        raise ValueError(f'No video stream: {path}')
    stream = data['streams'][0]
    fps = float(Fraction(stream['avg_frame_rate']))
    nominal = float(Fraction(stream['r_frame_rate']))
    if fps <= 0 or abs(fps - nominal) > 1e-4:
        raise ValueError(f'Constant positive FPS required: {path}: {stream}')
    return dict(width=stream['width'], height=stream['height'],
                frames=int(stream['nb_read_frames']), fps=fps)


def read_video(path, *, height=480, width=832, resize=False):
    import numpy as np
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(path)]
    if resize:
        command += ['-vf', f'scale={width}:{height}:flags=bilinear']
    command += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-']
    raw = subprocess.run(command, stdout=subprocess.PIPE, check=True, timeout=180).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()


def write_video(path, pixels, fps):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.mp4')
    frames = np.ascontiguousarray(pixels, dtype=np.uint8)
    _, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError('Expected RGB [frames, height, width, 3]')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)],
        input=frames.tobytes(), check=True)
    tmp.replace(path)
