import numpy as np
import pytest

from piper_sim.camera_cache import CameraSnapshotCache


def test_cache_owns_source_and_returned_pixels_and_poses():
    cache = CameraSnapshotCache()
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    expected = pixels.copy()
    poses = {'wrist': {'position': [0., 0., 1.]}}
    calls = []
    def capture():
        calls.append(1)
        return {'wrist': pixels}, poses
    image, pose = cache.read((0, 1), capture)
    pixels[:] = 4
    poses['wrist']['position'][0] = 5
    image['wrist'][:] = 8
    pose['wrist']['position'][0] = 9
    image, pose = cache.read((0, 1), capture)
    np.testing.assert_array_equal(image['wrist'], expected)
    assert pose['wrist']['position'] == [0., 0., 1.]
    assert len(calls) == 1
    # Static capture revision changes while control tick remains zero.
    image, pose = cache.read((0, 2), capture)
    np.testing.assert_array_equal(image['wrist'], pixels)
    assert pose['wrist']['position'][0] == 5
    assert len(calls) == 2
    cache.clear()  # Episode reset can reuse a timestamp/key.
    cache.read((0, 2), capture)
    assert len(calls) == 3


def test_failed_capture_cannot_publish_a_new_key():
    cache = CameraSnapshotCache()
    source = lambda: ({'rgb': np.ones((1, 1, 3), np.uint8)}, {})
    cache.read(1, source)
    def broken():
        raise RuntimeError('Readback failed')
    with pytest.raises(RuntimeError, match='Readback failed'):
        cache.read(2, broken)
    assert cache.key == 1
    assert cache.read(2, source)[0]['rgb'].sum() == 3
