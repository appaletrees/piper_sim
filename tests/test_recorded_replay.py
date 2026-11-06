"""Source-contract and kinematic replay checks, independent of private datasets."""
import json

import numpy as np
import pytest

from piper_sim.recorded_replay import (
    CAMERAS, load_rad15_episode, place_kinematic_observation, rad15_to_si, limit_report,
)


def test_rad15_conversion_is_distinct_from_degree_interface():
    raw = np.array([-.7, 1.6, -1.3, .1, 1.2, -.6, 1., .5, 1.7, -1.4, -.1, 1.1, .4, .5, 1.])
    original = raw.copy()
    result = rad15_to_si(raw)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(result[np.r_[0:6, 7:13]], raw[np.r_[0:6, 7:13]])
    np.testing.assert_allclose(result[[6, 13]], [.095, .0475])
    assert result.shape == (14,)
    assert rad15_to_si(np.array([raw, raw])).shape == (2, 14)
    for bad in (raw[:14], np.full(15, np.nan), np.zeros((2, 14))):
        with pytest.raises(ValueError):
            rad15_to_si(bad)
    raw[6] = 1.01
    with pytest.raises(ValueError, match='normalized'):
        rad15_to_si(raw)


def fixture_dataset(tmp_path, *, timestamps=None, frame_index=None, video_shape=None):
    pq = pytest.importorskip('pyarrow.parquet')
    import pyarrow as pa
    root = tmp_path / 'source'
    (root / 'meta').mkdir(parents=True)
    info = {'codebase_version': 'v2.1', 'robot_type': 'piper', 'total_episodes': 1, 'chunks_size': 1000, 'fps': 30,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': {k: {'shape': [15], 'dtype': 'float32'} for k in ('state', 'actions')}}
    info['features'].update({k: {'shape': video_shape or [480, 640, 3], 'dtype': 'video'} for k in CAMERAS})
    (root / 'meta/info.json').write_text(json.dumps(info))
    for key in CAMERAS:
        path = root / f'videos/chunk-000/{key}/episode_000000.mp4'
        path.parent.mkdir(parents=True)
        path.touch()  # Numeric reader only; CLI separately validates/decodes videos.
    states = [[.1]*14 + [0.], [.2]*14 + [0.], [.3]*14 + [0.]]
    actions = [[.4]*14 + [0.], [.5]*14 + [0.], [.6]*14 + [1.]]
    table = pa.table({'state': states, 'actions': actions, 'timestamp': timestamps if timestamps is not None else [0, 1/30, 2/30],
                      'frame_index': frame_index if frame_index is not None else [0, 1, 2], 'episode_index': [0]*3})
    path = root / 'data/chunk-000/episode_000000.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    return root


def test_reader_preserves_auxiliary_and_frame_correspondence(tmp_path):
    root = fixture_dataset(tmp_path)
    ep = load_rad15_episode(root)
    assert len(ep) == 3 and ep.resolution == (640, 480)
    assert ep.action_raw[-1, 14] == 1 and ep.state_raw[-1, 14] == 0
    assert ep.state_si[-1, 0] == .3 and ep.action_si[-1, 0] == .6
    assert ep.state_si[0, 6] == pytest.approx(.0095)
    with pytest.raises(ValueError, match='Episode index'):
        load_rad15_episode(root, -1)


@pytest.mark.parametrize('kwargs,match', [
    ({'timestamps': [0, .1, .2]}, 'timestamp'),
    ({'frame_index': [0, 2, 3]}, 'indices'),
    ({'video_shape': [3, 480, 640]}, 'HWC'),
])
def test_reader_rejects_unsynchronized_or_different_schema(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_rad15_episode(fixture_dataset(tmp_path, **kwargs))


def test_exact_pose_replay_never_advances_physics_or_relaxes_control_limits():
    pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    from piper_sim.contracts import Command
    backend = MujocoBackend(load('configs/clean_table_workcell.yaml'))
    try:
        initial = backend.read()
        ranges = backend.model.jnt_range.copy()
        q = np.array([-.7, 1.6, -1.3, .1, 1.2200026512146, -.6, .095,
                      .5, 1.7, -1.4, -.1, 1.1, .4, .0475])
        place_kinematic_observation(backend, q)
        after = backend.read()
        np.testing.assert_array_equal(after.position, q)
        assert after.sim_time == initial.sim_time == 0
        assert after.extras['object'] == initial.extras['object']
        np.testing.assert_array_equal(backend.model.jnt_range, ranges)
        np.testing.assert_allclose(backend.data.qpos[backend.indices['left']['qpos'][-2:]], [.0475, .0475])
        assert limit_report(q[None])['features'][0]['max_excess'] > 0
        with pytest.raises(ValueError, match='bounds'):
            backend.apply(Command(q, 1., 0))
    finally:
        backend.close()
