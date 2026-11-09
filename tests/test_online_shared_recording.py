import h5py
import numpy as np
import pytest

from piper_sim.contracts import Command, Observation
from piper_sim.dataset import validate_episode
from piper_sim.recording import EpisodeWriter


@pytest.mark.parametrize('compression', ['lzf', 'gzip'])
@pytest.mark.parametrize('count', [1, 7])
def test_online_shared_frames_are_exact_self_contained_and_portable(tmp_path, compression, count):
    pixels = np.random.default_rng(99).integers(0, 256, (count+1, 24, 32, 3), dtype=np.uint8)
    def obs(i):
        return Observation(np.full(14, i/100), i, i/30,
                           {'overhead': pixels[i]}, {'overhead': i/30})
    path = tmp_path/'episode.h5'
    with EpisodeWriter(path, {'fps': 30}, image_compression=compression,
                       image_storage='shared', flush_every=3) as writer:
        for i in range(count):
            writer.append(obs(i), Command(np.zeros(14), i+.1, i), obs(i+1))
    renamed = tmp_path/'new'/'renamed.h5'
    renamed.parent.mkdir(); path.rename(renamed)
    assert validate_episode(renamed)['transitions'] == count
    with h5py.File(renamed) as f:
        assert f['observation/images/overhead'].compression == compression
        assert f['next_observation/images/overhead'].is_virtual
        np.testing.assert_array_equal(f['observation/images/overhead'][:], pixels[:-1])
        np.testing.assert_array_equal(f['next_observation/images/overhead'][:], pixels[1:])
        assert f['recording_terminal_images/overhead'].id.get_num_chunks() == 1
        assert not f.attrs.get('training_ready', False)


@pytest.mark.parametrize('failure', ['pixels', 'capture_time', 'shape'])
def test_shared_continuity_failure_never_publishes_even_if_caught(tmp_path, failure):
    def obs(i):
        return Observation(np.zeros(14), i, i, {'overhead': np.full((8, 8, 3), i, np.uint8)}, {'overhead': i})
    path = tmp_path/'broken.h5'
    with EpisodeWriter(path, {}, image_storage='shared') as writer:
        writer.append(obs(0), Command(np.zeros(14), .1, 0), obs(1))
        before, after = obs(1), obs(2)
        if failure == 'pixels': before.images['overhead'][0, 0] = 23
        if failure == 'capture_time': before.image_timestamps['overhead'] = 1.01
        if failure == 'shape': after.images['overhead'] = np.zeros((10, 8, 3), np.uint8)
        with pytest.raises(ValueError):
            writer.append(before, Command(np.zeros(14), 1.1, 1), after)
    assert not path.exists()
    with h5py.File(str(path)+'.partial') as f:
        assert not f.attrs['complete']


def test_after_image_is_owned_not_a_mutable_caller_reference(tmp_path):
    pixels = np.zeros((8, 8, 3), np.uint8)
    def obs(t): return Observation(np.zeros(14), t, t, {'overhead': pixels}, {'overhead': t})
    path = tmp_path/'owned.h5'
    with EpisodeWriter(path, {}, image_storage='shared') as writer:
        writer.append(obs(0), Command(np.zeros(14), .1, 0), obs(1))
        pixels[:] = 255
    with h5py.File(path) as f:
        assert not np.any(f['next_observation/images/overhead'][:])


def test_finalize_io_failure_stays_partial(tmp_path, monkeypatch):
    path = tmp_path/'finalize.h5'
    def obs(t): return Observation(np.zeros(14), t, t, {'overhead': np.zeros((8,8,3), np.uint8)}, {'overhead': t})
    writer = EpisodeWriter(path, {}, image_storage='shared')
    writer.append(obs(0), Command(np.zeros(14), .1, 0), obs(1))
    def fail(*args, **kwargs): raise OSError('disk full')
    monkeypatch.setattr(writer, '_finalize_shared_images', fail)
    with pytest.raises(OSError, match='disk full'): writer.close(success=True)
    assert not path.exists()
    with h5py.File(str(path)+'.partial') as f: assert not f.attrs['complete']


def test_stop_recording_reports_existing_partial_after_caught_error(tmp_path):
    from types import SimpleNamespace
    from piper_sim.simulation import SimulationSession
    session = SimulationSession(SimpleNamespace(read=lambda: Observation(np.zeros(14), 0)))
    path = tmp_path/'failed.h5'
    session.writer = EpisodeWriter(path, {})
    session.writer.append(Observation(np.zeros(14), 0), Command(np.zeros(14), .1, 0), Observation(np.zeros(14), 1))
    with pytest.raises(ValueError):
        session.writer.append(Observation(np.zeros(14), 3), Command(np.zeros(14), 3, 1), Observation(np.zeros(14), 2))
    result = session.stop_recording()
    assert result['path'] == str(path)+'.partial'
    assert not path.exists()
