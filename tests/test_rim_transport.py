from types import SimpleNamespace

import numpy as np
import pytest

from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.experts.rim_transport import frame_coordinates, orientation_error, search, tilt_rotation, validate_profile, yaw_rotation


PROFILE = {'kinds': ['plate'], 'yaw_offsets_deg': [0], 'xy_step_m': .025,
           'wall_margin_m': .012, 'support_gap_m': .01, 'rim_clearance_m': .035,
           'transfer_speed_m_s': .15, 'lower_speed_m_s': .04,
           'max_path_orientation_error_rad': .02, 'max_candidates': 32}


def test_orientation_reference_accepts_yaw_but_detects_pitch_sway():
    reference = grasp_rotation(45, 37, 90)
    for yaw in (-30, 0, 30):
        error, actual = orientation_error(yaw_rotation(yaw) @ reference, reference)
        assert error < 1e-12
        assert actual == pytest.approx(np.deg2rad(yaw))
    a = .1
    pitch = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    assert orientation_error(pitch @ reference, reference)[0] > .09


def test_measured_frame_detects_tool_roll_while_allowing_declared_yaw_and_tilt():
    reference = grasp_rotation(45, 37, 90)
    expected = yaw_rotation(-30) @ tilt_rotation(reference, 10) @ reference
    error, yaw, tilt = frame_coordinates(expected, reference)
    assert error < 1e-12
    assert yaw == pytest.approx(np.deg2rad(-30))
    assert tilt == pytest.approx(np.deg2rad(10))
    # A quarter turn about local tool Z leaves its approach axis unchanged,
    # but changes the jaw-opening direction and must remain detectable.
    assert frame_coordinates(expected @ yaw_rotation(90), reference)[0] == pytest.approx(np.pi/2)


@pytest.mark.parametrize('key,value', [('kinds', ['cup']), ('yaw_offsets_deg', [0, 0]),
    ('yaw_offsets_deg',[0,136]),('tilt_offsets_deg',[0,61]),('reorient_during_lift',1),
    ('support_gap_m', 0), ('wall_margin_m', float('nan')), ('max_candidates', True)])
def test_invalid_profile_is_rejected(key, value):
    with pytest.raises(ValueError):
        validate_profile({**PROFILE, key: value})


def fixture_teacher():
    points = np.array([[x, y, z] for x in (-.095, .095) for y in (-.095, .095) for z in (0., .016)])
    floor = {'points': [[-.18, -.155, 0], [.18, -.155, 0], [.18, .155, 0], [-.18, .155, 0]],
             'faces': [[0, 1, 2], [0, 2, 3]], 'collision': True}
    task = SimpleNamespace(active_names=['basket', 'plate'], parts={'basket': [floor]},
        specs={'basket': {'size_m': [.36, .31, .14], 'wall_m': .003}, 'plate': {'kind': 'plate'}},
        surface_points={'plate': points})
    bodies = {'task/basket': SimpleNamespace(xpos=np.zeros(3), xmat=np.eye(3).ravel()),
              'task/plate': SimpleNamespace(xpos=np.array([-.3, 0, .2]), xmat=np.eye(3).ravel())}
    q = np.zeros(14);q[7:10] = [-.3, 0, .2]
    teacher = SimpleNamespace(b=SimpleNamespace(task=task, data=SimpleNamespace(
        body=bodies.__getitem__, qpos=np.array([0., 0., 0., 0., 0., 0., .005, .005])),
        indices={'right': {'qpos': np.arange(8)}}), q=q,
        tcp=lambda side: (np.array([-.3, 0, .2]), np.eye(3)), carried_objects=lambda: ['plate'],
        solve=lambda side, p, r, *args: np.r_[p, np.zeros(3)],
        plan_grasp=lambda name: ('right', np.zeros(3), np.eye(3), .025),
        ik={'right': SimpleNamespace(forward=lambda q: (q[:3], np.eye(3)))})
    calls = []
    def plan(side, poses, *, start=None, speed=None, bounded=False):
        before = teacher.q if start is None else start
        rows = []
        for position, rotation in poses:
            command = before.copy();command[7:10] = position;rows.append(command)
        calls.append(('path', poses))
        if bounded: calls.append(('bounded',True))
        return np.array(rows)
    teacher.plan_pose_path = plan
    teacher.check_joint_path = lambda *a, **k: calls.append(('opening', k['gripper']))
    return teacher, calls, bodies


def test_search_reserves_level_footprint_checks_opening_and_keeps_state():
    t, calls, bodies = fixture_teacher()
    before = t.q.copy();positions = {k: v.xpos.copy() for k, v in bodies.items()}
    result = search(t, 'right', 'plate', 'basket', PROFILE)
    slot = np.array(result['evidence']['slot_xy_m'])
    assert np.all(abs(slot) + .095 + PROFILE['wall_margin_m'] <= [.177, .152])
    assert result['low'][2] == pytest.approx(.01)
    assert result['high'][2] == pytest.approx(.175)
    assert [name for name, _ in calls].count('path') == 2
    opening = [widths for name, widths in calls if name == 'opening']
    assert opening[0][0] == pytest.approx(.01) and opening[-1][-1] == pytest.approx(.045)
    assert max(b - a for a, b in opening) <= .002
    np.testing.assert_array_equal(t.q, before)
    for k, p in positions.items():
        np.testing.assert_array_equal(bodies[k].xpos, p)


def test_search_rejects_opening_collision_instead_of_returning_a_candidate():
    t, _, _ = fixture_teacher()
    def blocked(*args, **kwargs):
        raise SortFailure('opening would hit basket')
    t.check_joint_path = blocked
    with pytest.raises(SortFailure, match='opening would hit basket'):
        search(t, 'right', 'plate', 'basket', PROFILE)


def test_reorientation_route_requests_fidelity_bounded_paths():
    t,calls,_=fixture_teacher()
    search(t,'right','plate','basket',{**PROFILE,'reorient_during_lift':True})
    assert [name for name,_ in calls].count('bounded')==2
