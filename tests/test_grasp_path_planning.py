from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher


def teacher_fixture():
    teacher = CleanTableTeacher.__new__(CleanTableTeacher)
    teacher.q = np.array([0., .2, -.3, 0., .2, 0., .04]*2)
    teacher.fps = 30.
    teacher.carried = None
    teacher.bounded_loaded_path = None
    teacher.pacing = {'continuous_paths': True, 'cartesian_speed_m_s': .25,
                      'contact_speed_m_s': .06, 'grasp_height_offsets_m': [0., .002, .004]}
    teacher.motion_limits = lambda: {'velocity': [1.]*14, 'acceleration': [2.]*14, 'jerk': [10.]*14}
    teacher.ik = {'right': SimpleNamespace(forward=lambda q: (q[:3].copy(), np.eye(3)))}
    def solve(side, p, r, seed=None):
        q = teacher.q[7:13].copy() if seed is None else seed.copy()
        q[:3] = p
        return q
    teacher.solve = solve
    return teacher


def test_complete_cartesian_plan_is_checked_without_mutating_targets():
    teacher = teacher_fixture()
    before = teacher.q.copy()
    checked = []
    teacher.check_joint_path = lambda side, a, b: checked.append(b.copy())
    trajectory = teacher.plan_pose_path('right', [(before[7:10]+[.02, 0, 0], np.eye(3))])
    np.testing.assert_array_equal(teacher.q, before)
    assert len(checked) == len(trajectory)
    np.testing.assert_allclose(trajectory[-1, 7:10], before[7:10]+[.02, 0, 0])
    def reject(*args):
        raise SortFailure('obstructed')
    teacher.check_joint_path = reject
    with pytest.raises(SortFailure, match='obstructed'):
        teacher.plan_pose_path('right', [(before[7:10]+[.02, 0, 0], np.eye(3))])
    np.testing.assert_array_equal(teacher.q, before)


def test_ik_reachable_but_colliding_grasp_tries_another_candidate():
    teacher = teacher_fixture()
    teacher.b = SimpleNamespace(task=SimpleNamespace(specs={'object': {'kind': 'pen'}}), sim_time=1.)
    teacher.plan_grasp = lambda name: ('right', np.array([.2, .1, .03]), np.eye(3), .025)
    teacher.current = 'object'
    teacher.events = []
    calls = []
    def check(*args):
        calls.append(1)
        if len(calls) == 1:
            raise SortFailure('first IK solution collides')
    teacher.check_joint_path = check
    planned = []
    teacher.plan_pose_path = lambda *args, **kwargs: planned.append(kwargs['start'].copy())
    before = teacher.q.copy()
    teacher.reachable_grasp('object', check_approach=True)
    assert len(calls) == 2 and len(planned) == 1
    assert teacher.events[-1]['local_tilt_deg'] == 5
    assert teacher.events[-1]['rejected_candidates'] == 1
    np.testing.assert_array_equal(teacher.q, before)


def test_unloaded_limit_overshoot_replans_and_rechecks_but_loaded_motion_rejects():
    teacher = teacher_fixture()
    teacher.q[11] = .9
    before = teacher.q.copy()
    def solve(side, p, r, seed=None):
        q = teacher.q[7:13].copy()
        q[:3] = p
        q[4] = np.interp(p[0], [0., .01, .02, .03], [.9, 1.22, 1.22, 1.19])
        return q
    teacher.solve = solve
    checked = []
    teacher.check_joint_path = lambda side, a, b: checked.append(b.copy())
    target = [(before[7:10]+[.03, 0, 0], np.eye(3))]
    diagnostics = {}
    trajectory = teacher.plan_pose_path('right', target, diagnostics=diagnostics)
    assert diagnostics['interpolation'] == 'convex'
    assert len(checked) == len(trajectory) and trajectory[:, 11].max() <= 1.22
    np.testing.assert_array_equal(teacher.q, before)
    teacher.carried = ('right', 'plate', None)
    with pytest.raises(SortFailure, match='joint limits'):
        teacher.plan_pose_path('right', target)
    teacher.carried = None
    def obstructed(*args):
        raise SortFailure('bounded curve collides')
    teacher.check_joint_path = obstructed
    with pytest.raises(SortFailure, match='bounded curve collides'):
        teacher.plan_pose_path('right', target)


def test_opt_in_loaded_fallback_checks_pose_fidelity_and_original_geometry():
    teacher = teacher_fixture()
    teacher.q[11] = 1.21
    before = teacher.q.copy()
    teacher.carried = ('right', 'plate', None)
    teacher.bounded_loaded_path = {'max_position_deviation_m': .002,
                                   'max_rotation_deviation_rad': .01, 'max_refinements': 2}
    def solve(side, p, r, seed=None):
        q = teacher.q[7:13].copy()
        q[:3] = p
        q[4] = np.interp(p[0], [0., .01, .02, .03], [1.21, 1.22, 1.22, 1.19])
        return q
    teacher.solve = solve
    checked = []
    teacher.check_joint_path = lambda side, a, b: checked.append(b.copy())
    target = [(before[7:10]+[.03, 0, 0], np.eye(3))]
    details = {}
    path = teacher.plan_pose_path('right', target, diagnostics=details)
    assert details['interpolation'] == 'convex'
    assert details['fidelity']['attempts'][-1]['position_deviation_m'] <= .002
    assert len(checked) == len(path)
    assert path[:, 11].max() <= 1.22
    np.testing.assert_array_equal(path[-1, 7:13], solve('right', target[0][0], np.eye(3)))
    np.testing.assert_array_equal(teacher.q, before)
    def reject(*args): raise SortFailure('payload obstacle')
    teacher.check_joint_path = reject
    with pytest.raises(SortFailure, match='payload obstacle'):
        teacher.plan_pose_path('right', target)
    # A too-loose approximation must fail before any live state/geometry step.
    teacher.bounded_loaded_path['max_position_deviation_m'] = 1e-12
    teacher.bounded_loaded_path['max_refinements'] = 0
    with pytest.raises(SortFailure, match='pose fidelity'):
        teacher.plan_pose_path('right', target)
    np.testing.assert_array_equal(teacher.q, before)


def test_loaded_refinement_keeps_original_reference_and_retimes_commands():
    from piper_sim.experts.trajectory import path_trajectory
    from piper_sim.experts.path_fidelity import path_pose_deviation
    teacher = teacher_fixture()
    teacher.bounded_loaded_path = {'max_position_deviation_m': .002,
                                   'max_rotation_deviation_rad': .01, 'max_refinements': 2}
    points = np.tile(teacher.q, (5, 1))
    points[:, 7:9] = [[0., 0.], [.01, 0.], [.02, .01], [.03, .01], [.04, .01]]
    path, evidence = teacher.bounded_loaded_trajectory('right', points, 0.)
    assert len(evidence['attempts']) > 1
    assert evidence['reference_ik_knots'] == len(points)
    assert evidence['attempts'][0]['position_deviation_m'] > .002
    assert evidence['attempts'][-1]['position_deviation_m'] <= .002
    sampling = {}
    path_trajectory(points, 1/30, **teacher.motion_limits(), sampling=sampling)
    t = np.arange(1, len(path)+1)/len(path)
    sampling.update(points=sampling['points'][:, 7:13], samples=t**4*(35+t*(-84+t*(70-20*t))))
    check = path_pose_deviation(path[:, 7:13], sampling, teacher.ik['right'].forward)
    assert check['position_deviation_m'] == evidence['attempts'][-1]['position_deviation_m']
    padded = np.vstack([np.tile(points[0], (4, 1)), path, np.tile(points[-1], (4, 1))])
    for order, limit in enumerate(teacher.motion_limits().values(), 1):
        assert np.max(abs(np.diff(padded, n=order, axis=0))*30**order/limit) <= 1+1e-9


def test_slender_object_candidates_follow_long_axis_without_tilting_pads():
    teacher = RigidTableTeacher.__new__(RigidTableTeacher)
    teacher.pacing={}
    teacher.b = SimpleNamespace(task=SimpleNamespace(specs={'stick': {'kind': 'chopstick', 'size_m': [.225, .006, .006]}}))
    teacher.object_pose = lambda name: {'position': [.3, -.2, .03],
        'quaternion_wxyz': [2**-.5, 0., 0., 2**-.5]}
    p, r = np.array([.3, -.2, .035]), np.eye(3)
    candidates = list(teacher.grasp_candidates('stick', p, r, check_approach=True))
    assert len(candidates) == 5
    for target, rotation, details in candidates:
        np.testing.assert_allclose(target-p, [0., details['long_axis_offset_m'], 0.], atol=1e-15)
        np.testing.assert_array_equal(rotation, r)
        assert abs(details['long_axis_offset_m']) <= .225/4


def test_slender_pregrasp_opening_follows_cross_section_not_total_length():
    teacher = RigidTableTeacher.__new__(RigidTableTeacher)
    teacher.b = SimpleNamespace(task=SimpleNamespace(specs={'stick': {'kind': 'chopstick', 'size_m': [.225, .006, .006]}}))
    teacher.pacing = {'slender_pregrasp_clearance_m': .004}
    for roll in (0., np.pi/4, np.pi/2):
        teacher.object_pose = lambda name: {'position': [.3, -.2, .03],
            'quaternion_wxyz': [np.cos(roll/2), np.sin(roll/2), 0., 0.]}
        side, p, r, width = teacher.plan_grasp('stick')
        assert width == pytest.approx(.004+.006*(abs(np.cos(roll))+abs(np.sin(roll))))
        assert side == 'right' and width < .013
