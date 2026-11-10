import numpy as np
import pytest

from piper_sim.experts.supported_tilt import hand_path, validate_profile


PROFILE = {'angle_deg': 75., 'angle_step_deg': 10., 'support_gap_m': .0005,
           'clear_lift_m': .06, 'path_speed_m_s': .04, 'max_grasp_rotation_rad': .15}


def test_path_preserves_grasp_and_surface_height_without_mutating_inputs():
    p = np.array([.4, -.1, .025]);r = np.eye(3)
    hp = p+[-.08, -.04, .014];hr = np.diag([-1., 1., -1.])
    points = np.array([[x, y, z] for x in [-.095, .095] for y in [-.095, .095] for z in [0., .016]])
    inputs = [x.copy() for x in (p, r, hp, hr, points)]
    poses, evidence = hand_path(p, r, hp, hr, points, validate_profile(PROFILE))
    for (hand_p, hand_r), record in zip(poses[:-1], evidence['planned_object_poses']):
        obj_p, obj_r = np.array(record['object_position']), np.array(record['object_rotation'])
        assert (obj_p+points@obj_r.T)[:, 2].min() == pytest.approx(.0255)
        np.testing.assert_allclose(obj_r.T@(hand_p-obj_p), hp-p, atol=1e-15)
        np.testing.assert_allclose(obj_r.T@hand_r, hr, atol=1e-15)
    np.testing.assert_allclose(poses[-1][0]-poses[-2][0], [0., 0., .06], atol=1e-15)
    assert poses[-2][0][2] > hp[2]  # raise the grasped edge, not the opposite edge
    for x, y in zip(inputs, (p, r, hp, hr, points)): np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('key,value', [('angle_deg', 100), ('support_gap_m', -.01),
    ('clear_lift_m', float('nan')), ('path_speed_m_s', True), ('max_grasp_rotation_rad', 1.)])
def test_invalid_profile(key, value):
    with pytest.raises(ValueError): validate_profile({**PROFILE, key: value})
