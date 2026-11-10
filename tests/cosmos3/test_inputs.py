import json
from pathlib import Path
import subprocess

import pytest

from training.cosmos3.inputs import read_inputs, validate_media
from training.cosmos3.inference import completed, load_sampler
from training.cosmos3.io import file_hash, write_json


def manifest(tmp_path, count=3):
    from PIL import Image
    Image.new('RGB', (832, 480), 'gray').save(tmp_path / 'static.png')
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=gray:s=832x480:r=15',
        '-frames:v','5','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(tmp_path/'mesh.mp4')], check=True)
    rows = [dict(id=f'case{i}', static_image='static.png', robot_mesh_video='mesh.mp4',
                 prompt={'task':'Pick up the box'}, fps=15) for i in range(count)]
    path = tmp_path / 'inputs.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path, rows


def test_sharding_relative_paths_and_prompt(tmp_path):
    path, rows = manifest(tmp_path)
    shards = [read_inputs(path,num_shards=2,shard_index=i) for i in range(2)]
    assert [r['id'] for r in shards[0]] == ['case0','case2']
    assert [r['id'] for r in shards[1]] == ['case1']
    assert shards[0][0]['static_image'] == str(tmp_path/'static.png')
    assert shards[0][0]['caption'] == '{"task":"Pick up the box"}'
    row = validate_media(shards[0][0])
    assert row['frames'] == 5 and row['fps'] == 15


def test_bad_inputs_and_fps(tmp_path):
    path, rows = manifest(tmp_path)
    row = read_inputs(path)[0]
    with pytest.raises(ValueError, match='FPS'):
        validate_media(dict(row, fps=16))
    rows[0]['id'] = '../escape'
    path.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError, match='Unsafe'):
        read_inputs(path)


def test_resume_rejects_protocol_change_and_corrupt_video(tmp_path):
    path, _ = manifest(tmp_path)
    row = validate_media(read_inputs(path)[0])
    (tmp_path/'generated.mp4').write_bytes((tmp_path/'mesh.mp4').read_bytes())
    write_json(tmp_path/'result.json', dict(status='PASS',protocol_hash='same',video_sha256=file_hash(tmp_path/'generated.mp4')))
    assert completed(tmp_path,'same',row)
    with pytest.raises(ValueError, match='different protocol'):
        completed(tmp_path,'different',row)
    (tmp_path/'generated.mp4').write_bytes(b'bad')
    assert not completed(tmp_path,'same',row)


def test_full_h2r_sampler_is_explicit():
    root=Path(__file__).resolve().parents[2]
    config=load_sampler(root/'exps_infer/cosmos3/h2r_super.yaml')
    assert (config['steps'],config['guidance'],config['shift']) == (35,6,10)


def test_event_logging_accepts_report_rank(capsys):
    from training.cosmos3.io import emit
    emit('complete', rank=2, status='PASS')
    assert json.loads(capsys.readouterr().out) == {'event':'complete','rank':2,'status':'PASS'}
