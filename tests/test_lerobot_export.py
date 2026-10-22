import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip('pyarrow')
import pyarrow.parquet as pq
from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter
from piper_sim.lerobot_export import export_lerobot_v2, _image_transform
from piper_sim.lerobot_alignment import CONNECTOR_PLUGIN_NAMES

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg is an optional export dependency')


def episode(path, task='test task', cameras=True, complete=True):
    k = [[40.,0,31.5],[0,40.,15.5],[0,0,1]]
    with EpisodeWriter(path, {'fps':30,'task':task,'task_scene':{'layout':{'seed':42}}}) as writer:
        for i in range(3):
            q = np.array([i*.01,.3,-.2,0,0,0,.04]*2)
            rgb = np.full((32,64,3), [[255,0,0],[0,255,0],[0,0,255]][i], dtype=np.uint8)
            images = {'overhead':rgb} if cameras else {}
            before = Observation(q,100+i/30,5+i/30,images,{'overhead':5+i/30} if cameras else {},
                {'cameras':{'overhead':{'intrinsics':k}},'objects':{'board':{'position':[.3,0,.025]}}})
            # Different after-image, to detect accidental one-step visual shift.
            after = Observation(q+.001,100+(i+1)/30,5+(i+1)/30,
                {'overhead':255-rgb} if cameras else {},{'overhead':5+(i+1)/30} if cameras else {},{})
            target=q.copy(); target[0]=.2+i*.01; target[[6,13]]=.095
            writer.append(before,Command(target,100+i/30,i),after)
        if not complete: writer.failed=True
    return path


@pytest.mark.parametrize('image_size', [None,[64,64]])
def test_v21_conversion_actions_video_stats_and_provenance(tmp_path,image_size):
    first=episode(tmp_path/'a.h5'); second=episode(tmp_path/'b.h5',task='second task')
    original=first.read_bytes()
    dest=tmp_path/'dataset'
    report=export_lerobot_v2([first,second],dest,image_size=image_size)
    assert report['frames']==6 and report['episodes']==2 and report['videos']==2
    assert first.read_bytes()==original
    info=json.loads((dest/'meta/info.json').read_text())
    assert info['codebase_version']=='v2.1' and info['total_tasks']==2
    assert info['features']['action']['names']==list(CONNECTOR_PLUGIN_NAMES)
    height=64 if image_size else 32
    assert info['features']['observation.images.cam_external_rgb']['shape']==[3,height,64]
    table=pq.read_table(dest/'data/chunk-000/episode_000000.parquet')
    actions=np.array(table['action'].to_pylist()); states=np.array(table['observation.state'].to_pylist())
    np.testing.assert_allclose(actions[:,0],np.rad2deg([.2,.21,.22]),rtol=1e-6)
    np.testing.assert_allclose(states[:,0],np.rad2deg([0,.01,.02]),rtol=1e-6)
    np.testing.assert_allclose(actions[:,6],1,atol=1e-7)
    np.testing.assert_allclose(table['timestamp'].to_numpy(),np.arange(3)/30,atol=1e-7)
    np.testing.assert_allclose(table['sim_timestamp'].to_numpy(),5+np.arange(3)/30)
    other=pq.read_table(dest/'data/chunk-000/episode_000001.parquet')
    assert other['index'].to_pylist()==[3,4,5] and other['task_index'].to_pylist()==[1]*3
    extra=json.loads((dest/'extras/chunk-000/episode_000000.jsonl').read_text().splitlines()[0])
    assert extra['output_camera_calibration']['overhead']['intrinsics'][1][2]==(31.5 if image_size else 15.5)
    assert extra['observation_extras']['cameras']['overhead']['intrinsics'][1][2]==15.5
    assert json.loads((dest/'meta/sim_episodes.jsonl').read_text().splitlines()[0])['metadata']['task_scene']['layout']['seed']==42
    stats=json.loads((dest/'meta/stats.json').read_text())
    assert stats['action']['count']==[6]
    assert stats['action']['mean'][0]==pytest.approx(actions[:,0].mean())
    np.testing.assert_allclose(np.array(stats['observation.images.cam_external_rgb']['mean']).ravel(),
                               [1/6 if image_size else 1/3]*3,atol=1e-8)
    video=dest/'videos/chunk-000/observation.images.cam_external_rgb/episode_000000.mp4'
    decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
    frames=np.frombuffer(decoded,np.uint8).reshape(3,height,64,3)
    assert frames[0,height//2,32,0]>230 and frames[0,height//2,32,1]<20
    assert frames[1,height//2,32,1]>230 and frames[1,height//2,32,0]<20
    assert frames[2,height//2,32,2]>230 and frames[2,height//2,32,0]<20
    if image_size: assert frames[:,0].max()<20  # Black letterbox bars, never stretched.


def test_native_resolution_and_state_only(tmp_path):
    src=episode(tmp_path/'source.h5',cameras=False)
    report=export_lerobot_v2(src,tmp_path/'data')
    assert report['videos']==0
    assert json.loads((tmp_path/'data/meta/episodes.jsonl').read_text())['length']==3
    transform=_image_transform((1080,1920,3),(640,480))
    assert transform['resized_size']==[640,360] and transform['pad_left_top']==[0,60]
    native=_image_transform((32,64,3),None)
    np.testing.assert_array_equal(native['pixel_transform'],np.eye(3))


def test_refuses_overwrite_partial_duplicate_and_mismatched_episodes(tmp_path):
    src=episode(tmp_path/'good.h5')
    dest=tmp_path/'existing';dest.mkdir(); (dest/'keep').write_text('keep')
    with pytest.raises(FileExistsError):export_lerobot_v2(src,dest)
    assert (dest/'keep').read_text()=='keep'
    episode(tmp_path/'unfinished.h5',complete=False)
    with pytest.raises(ValueError):export_lerobot_v2(tmp_path/'unfinished.h5.partial',tmp_path/'bad')
    with pytest.raises(ValueError,match='Duplicate'):export_lerobot_v2([src,src],tmp_path/'duplicate')
    other=episode(tmp_path/'no_cameras.h5',cameras=False)
    with pytest.raises(ValueError,match='identical'):export_lerobot_v2([src,other],tmp_path/'mismatch')


def test_rejects_timing_gaps_and_wrong_units(tmp_path):
    src=episode(tmp_path/'source.h5')
    with h5py.File(src,'r+') as f:f['observation/sim_time'][1]=9
    with pytest.raises(ValueError,match='continuous'):export_lerobot_v2(src,tmp_path/'gap')
    src=episode(tmp_path/'other.h5')
    with h5py.File(src,'r+') as f:f.attrs['units']=json.dumps(['degree']*14)
    with pytest.raises(ValueError,match='units'):export_lerobot_v2(src,tmp_path/'units')


def test_encoder_failure_leaves_hdf5_and_no_partial_dataset(tmp_path,monkeypatch):
    import piper_sim.lerobot_export as module
    src=episode(tmp_path/'source.h5')
    def fail(*args,**kwargs):raise RuntimeError('Encoder failed')
    monkeypatch.setattr(module,'_video',fail)
    with pytest.raises(RuntimeError,match='Encoder failed'):export_lerobot_v2(src,tmp_path/'result')
    assert src.exists() and not (tmp_path/'result').exists()
    assert not list(tmp_path.glob('result.partial-*'))
