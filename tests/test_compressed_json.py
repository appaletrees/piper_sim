import json

import h5py
import numpy as np
import pytest

from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter
from piper_sim.recording_json import CompressedJsonStreams
from piper_sim.dataset import validate_episode


def observation(i, extras, image):
    return Observation(np.full(14, i*.001), i, i, {'overhead':image}, {'overhead':i}, extras)


@pytest.mark.parametrize('images', ['duplicated','shared'])
def test_compressed_json_preserves_bytes_dtype_slices_and_move(tmp_path, images):
    # Cross bucket boundaries repeatedly; long values must never truncate.
    sizes = [3,250,255,256,257,12000,70000,10,70000]
    extras = [{'label':'纸杯😀\0', 'nested':{'array':[1.2,-3,None,True]}, 'text':'x'*n} for n in sizes]
    pixels = np.random.default_rng(3).integers(0,256,(len(extras),8,9,3),dtype=np.uint8)
    old,new = tmp_path/'old.h5',tmp_path/'new.h5'
    for mode,path in [('vlen',old),('compressed',new)]:
        with EpisodeWriter(path,{},image_storage=images,extras_storage=mode) as w:
            for i in range(len(extras)-1):
                w.append(observation(i,extras[i],pixels[i]),Command(np.full(14,.1),i,i),
                         observation(i+1,extras[i+1],pixels[i+1]))
    moved = tmp_path/'renamed'/'episode.h5';moved.parent.mkdir();new.rename(moved)
    assert validate_episode(moved)['transitions'] == len(extras)-1
    with h5py.File(old) as a,h5py.File(moved) as b:
        for prefix in ('observation','next_observation'):
            field=prefix+'/extras_json'
            assert h5py.check_string_dtype(a[field].dtype)==h5py.check_string_dtype(b[field].dtype)
            assert b[field].is_virtual
            assert all(s.file_name == '.' for s in b[field].virtual_sources())
            np.testing.assert_array_equal(a[field][:],b[field][:])
            np.testing.assert_array_equal(a[field].asstr()[1::2],b[field].asstr()[1::2])
            for name in ('state','sim_time','timestamp','images/overhead','image_timestamps/overhead'):
                np.testing.assert_array_equal(a[prefix+'/'+name][:],b[prefix+'/'+name][:])
        np.testing.assert_array_equal(a['action/position'][:],b['action/position'][:])
        assert json.loads(b['observation/extras_json'][0])==extras[0]


def test_json_finalize_failure_never_publishes_complete_episode(tmp_path, monkeypatch):
    path=tmp_path/'failed.h5';w=EpisodeWriter(path,{},extras_storage='compressed')
    image=np.zeros((2,2,3),np.uint8)
    w.append(observation(0,{},image),Command(np.zeros(14),0,0),observation(1,{},image))
    def fail(*args):raise RuntimeError('Injected finalization failure')
    monkeypatch.setattr(w._json_streams,'finalize',fail)
    with pytest.raises(RuntimeError):w.close(success=True)
    assert not path.exists()
    with h5py.File(w.partial) as f: assert not f.attrs['complete']


def test_nonfinite_json_is_rejected_even_if_caller_catches_error(tmp_path):
    path=tmp_path/'failed.h5';image=np.zeros((2,2,3),np.uint8)
    with EpisodeWriter(path,{},extras_storage='compressed') as w:
        w.append(observation(0,{},image),Command(np.zeros(14),0,0),observation(1,{},image))
        with pytest.raises(ValueError):
            w.append(observation(1,{'bad':float('nan')},image),Command(np.zeros(14),1,1),observation(2,{},image))
    assert not path.exists()


def test_streams_require_complete_contiguous_pairs(tmp_path):
    with h5py.File(tmp_path/'streams.h5','w') as f:
        streams=CompressedJsonStreams(f)
        streams.append('observation/extras_json','{}',0)
        with pytest.raises(ValueError,match='Noncontiguous'):streams.append('observation/extras_json','{}',2)
        with pytest.raises(ValueError,match='Incomplete'):streams.finalize(1)
        streams.append('next_observation/extras_json','{}',0)
        streams.finalize(1)
        assert f['observation/extras_json'][0]==b'{}'


def test_offline_compression_keeps_source_clears_acceptance_and_exports(tmp_path):
    import hashlib
    from piper_sim.recording_storage import compress_episode_json
    from piper_sim.lerobot_export import export_lerobot_v2
    path=tmp_path/'original.h5';image=np.zeros((8,8,3),np.uint8)
    with EpisodeWriter(path,{'fps':1},image_storage='shared') as w:
        for i in range(4):
            w.append(observation(i,{'text':'筷子'*5000},image),Command(np.zeros(14),i,i),
                     observation(i+1,{'text':'纸杯'*6000},image))
    with h5py.File(path,'r+') as f:
        f.attrs['accepted']=f.attrs['training_ready']=True
        f.attrs['dataset_audit']='stale'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    output=tmp_path/'compressed.h5'
    result=compress_episode_json(path,output)
    assert result['output_bytes']<result['source_bytes']
    assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_episode(output)['transitions']==4
    with h5py.File(output) as f:
        assert not f.attrs['accepted'] and not f.attrs['training_ready']
        assert 'dataset_audit' not in f.attrs
        assert f['next_observation/images/overhead'].is_virtual
        assert f['next_observation/extras_json'].is_virtual
    with pytest.raises(ValueError,match='Already compressed'):compress_episode_json(output,tmp_path/'again.h5')
    pytest.importorskip('pyarrow')
    import shutil
    if shutil.which('ffmpeg'):
        exported=export_lerobot_v2(output,tmp_path/'lerobot')
        assert exported['frames']==4 and exported['videos']==1


def test_compressed_json_survives_rgb_compaction(tmp_path):
    from piper_sim.recording_storage import compact_episode
    source, destination = tmp_path/'source.h5', tmp_path/'compact.h5'
    images=np.random.default_rng(9).integers(0,256,(5,8,8,3),dtype=np.uint8)
    with EpisodeWriter(source,{},extras_storage='compressed') as w:
        for i in range(4):
            w.append(observation(i,{'text':'杯子'*i*300},images[i]),Command(np.zeros(14),i,i),
                     observation(i+1,{'text':'杯子'*(i+1)*300},images[i+1]))
    compact_episode(source,destination)
    assert validate_episode(destination)['transitions']==4
    with h5py.File(source) as a,h5py.File(destination) as b:
        for prefix in ('observation','next_observation'):
            for field in ('extras_json','images/overhead'):
                np.testing.assert_array_equal(a[prefix+'/'+field][:],b[prefix+'/'+field][:])


def test_compression_cli_rejects_report_overwriting_recording(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    spec=importlib.util.spec_from_file_location('compress_episode_json',
        Path(__file__).parents[1]/'scripts/compress_episode_json.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    source, output = tmp_path/'source.h5', tmp_path/'output.h5'
    def must_not_run(*args):raise AssertionError('Should reject before touching recordings')
    monkeypatch.setattr(cli,'compress_episode_json',must_not_run)
    for report in (source,output,output.with_name(output.name+'.partial')):
        monkeypatch.setattr(sys,'argv',['compress',str(source),'--output',str(output),'--report',str(report)])
        with pytest.raises(SystemExit) as exc:cli.main()
        assert exc.value.code==2
    assert not list(tmp_path.iterdir())
