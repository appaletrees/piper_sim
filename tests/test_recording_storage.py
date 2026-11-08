import json
import h5py
import numpy as np
import pytest
from piper_sim.recording import EpisodeWriter
from piper_sim.recording_storage import compact_episode
from piper_sim.contracts import Command, Observation
from piper_sim.dataset import validate_episode


def source(path):
    images = np.random.default_rng(8).integers(0,256,(9,20,30,3),dtype=np.uint8)
    def obs(i): return Observation(np.zeros(14),i,i,{'overhead':images[i]},{'overhead':i})
    with EpisodeWriter(path, {'fps':1}) as w:
        for i in range(8): w.append(obs(i),Command(np.zeros(14),i,i),obs(i+1))
    return images


def test_lossless_virtual_frames_survive_move(tmp_path):
    src, dst = tmp_path/'source.h5', tmp_path/'compact.h5'
    images = source(src)
    with h5py.File(src,'r+') as f:
        f['observation/images/overhead'].attrs['color_order'] = 'RGB'
    compact_episode(src,dst)
    moved = tmp_path/'other'/'moved.h5'; moved.parent.mkdir(); dst.rename(moved)
    assert validate_episode(moved)['transitions'] == 8
    with h5py.File(moved) as f:
        np.testing.assert_array_equal(f['observation/images/overhead'][:],images[:-1])
        np.testing.assert_array_equal(f['next_observation/images/overhead'][:],images[1:])
        assert f['next_observation/images/overhead'].is_virtual
        assert f['observation/images/overhead'].attrs['color_order'] == 'RGB'
        assert not f.attrs['accepted'] and not f.attrs['training_ready']
    with pytest.raises(ValueError,match='Already compacted'):
        compact_episode(moved,tmp_path/'again.h5')


def test_noncontiguous_input_is_never_published(tmp_path):
    src, dst = tmp_path/'source.h5', tmp_path/'compact.h5'
    source(src)
    with h5py.File(src,'r+') as f:f['next_observation/images/overhead'][0]=0
    with pytest.raises(ValueError,match='Noncontiguous'):
        compact_episode(src,dst)
    assert not dst.exists() and not (tmp_path/'compact.h5.partial').exists()


def test_compacted_frames_export_through_existing_lerobot_reader(tmp_path):
    import shutil
    pytest.importorskip('pyarrow')
    if not shutil.which('ffmpeg'): pytest.skip('ffmpeg required')
    from piper_sim.lerobot_export import export_lerobot_v2
    src, dst = tmp_path/'source.h5', tmp_path/'compact.h5'
    source(src)
    compact_episode(src,dst)
    result = export_lerobot_v2(dst,tmp_path/'lerobot')
    assert result['frames'] == 8 and result['videos'] == 1
