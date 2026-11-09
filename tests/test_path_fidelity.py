import numpy as np
import pytest

pytest.importorskip('mujoco')
from piper_sim.experts.path_fidelity import loaded_path_limits, path_pose_deviation
from piper_sim.experts.trajectory import path_trajectory


def forward(q):
    c, s = np.cos(q[3]), np.sin(q[3])
    return q[:3].copy(), np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def test_same_parameter_pose_check_includes_translation_and_geodesic_rotation():
    points = np.array([[0., 0., 0., np.deg2rad(170)], [1., 0., 0., np.deg2rad(190)]])
    sampling = {'points': points, 'knots': [0., 1.], 'samples': [.5, 1.]}
    path = np.array([[.5, .003, 0., np.pi+.02], points[-1]])
    result = path_pose_deviation(path, sampling, forward)
    assert result['position_deviation_m'] == pytest.approx(.003)
    assert result['rotation_deviation_rad'] == pytest.approx(.02)
    assert result['position_peak_sample'] == result['rotation_peak_sample'] == 0


def test_nearby_wrong_progress_is_not_accepted_as_a_nearest_point_match():
    points = np.array([[0., 0., 0., 0.], [1., 0., 0., 0.], [0., .001, 0., 0.]])
    # The sample is ON the polyline, but at the wrong traversal parameter.
    sampling = {'points': points, 'knots': [0., .5, 1.], 'samples': [.5]}
    error = path_pose_deviation(points[:1], sampling, forward)
    assert error['position_deviation_m'] == pytest.approx(1.)


@pytest.mark.parametrize('interpolation', ['natural', 'convex'])
def test_sampling_does_not_change_path_and_matches_filtered_points(interpolation):
    points = np.array([[0., .1], [0., .1], [.02, .12], [.04, .1]])
    kwargs = dict(velocity=1., acceleration=2., jerk=10., interpolation=interpolation)
    original = path_trajectory(points, 1/30, **kwargs)
    sampling = {}
    actual = path_trajectory(points, 1/30, sampling=sampling, **kwargs)
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(sampling['points'], points[[0, 2, 3]])
    assert len(sampling['samples']) == len(actual)
    assert sampling['knots'][0] == 0. and sampling['knots'][-1] == 1.
    t = np.arange(1, len(actual)+1)/len(actual)
    np.testing.assert_array_equal(sampling['samples'], t**4*(35+t*(-84+t*(70-20*t))))
    sampling['points'][0] = 123.
    np.testing.assert_array_equal(points[0], [0., .1])


def test_stationary_fidelity_and_nonfinite_rejection():
    sampling = {}
    q = np.zeros((3, 4))
    path = path_trajectory(q, 1/30, velocity=1., acceleration=2., jerk=10., sampling=sampling)
    assert path_pose_deviation(path, sampling, forward)['position_deviation_m'] == 0.
    with pytest.raises(ValueError, match='Invalid forward'):
        path_pose_deviation(path, sampling, lambda q: (q[:3]*np.nan, np.eye(3)))
    with pytest.raises(ValueError, match='Invalid path sampling'):
        path_pose_deviation(path, {**sampling, 'samples': [np.nan]}, forward)


@pytest.mark.parametrize('key,value', [
    ('max_position_deviation_m', 0), ('max_position_deviation_m', float('nan')),
    ('max_position_deviation_m', True), ('max_rotation_deviation_rad', '0.1'),
    ('max_rotation_deviation_rad', -1), ('max_refinements', 4),
    ('max_refinements', -1), ('max_refinements', 1.5), ('max_refinements', True),
])
def test_invalid_loaded_profiles_reject(key, value):
    config = {'max_position_deviation_m': .002, 'max_rotation_deviation_rad': .01, 'max_refinements': 2}
    assert loaded_path_limits(config) == config
    assert loaded_path_limits(None) is None
    with pytest.raises(ValueError): loaded_path_limits({**config, key: value})
    with pytest.raises(ValueError): loaded_path_limits({**config, 'typo': 1})
