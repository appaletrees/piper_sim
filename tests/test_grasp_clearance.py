from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from piper_sim.experts.grasp_clearance import validate_profile
from piper_sim.experts.rigid_table import RigidTableTeacher
from test_grasp_path_planning import teacher_fixture


def profile():
    return {'kinds': ['chopstick'], 'margin_m': .001,
            'extra_long_axis_fractions': [-.3, .3, -.4, .4]}


@pytest.mark.parametrize('key,value', [
    ('margin_m', 0), ('margin_m', float('nan')), ('margin_m', True),
    ('margin_m', '0.001'), ('kinds', ['cup']),
    ('extra_long_axis_fractions', [.5]), ('extra_long_axis_fractions', [float('inf')]),
    ('extra_long_axis_fractions', [-.3, -.3]), ('extra_long_axis_fractions', [True]),
])
def test_invalid_profile_fails_before_physics(key, value):
    config = profile(); config[key] = value
    with pytest.raises(ValueError): validate_profile(config)


def geometry_teacher():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="right/link6"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".001" mass="1" contype="0" conaffinity="0"/>
        <body name="right/link7" pos=".1 0 0"><joint type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".02" mass=".1"/>
        </body>
      </body>
      <body name="task/target" pos=".1 0 .039"><geom type="sphere" size=".02"/></body>
      <body name="task/neighbor" pos=".145 0 0"><geom name="neighbor" type="sphere" size=".02"/></body>
      <body name="task/inactive" pos=".1 .039 0"><geom type="sphere" size=".02" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    teacher = CleanTableTeacher.__new__(CleanTableTeacher)
    teacher.b = SimpleNamespace(model=model, data=data,
        indices={'right': {'qpos': np.array([0]*6+[1])}},
        task=SimpleNamespace(active_names=['target', 'neighbor'], specs={}))
    teacher.ik = {'right': SimpleNamespace(data=mujoco.MjData(model), qids=np.array([0]))}
    teacher.carried_objects = lambda: []
    teacher.current, teacher.pacing = 'target', {'continuous_paths': True}
    teacher.settings = {'quality': {'max_robot_penetration_m': .004}}
    return teacher


def test_target_contact_allowed_but_near_miss_and_subthreshold_neighbor_contact_rejected():
    teacher = geometry_teacher()
    start = np.array([0.])
    before = teacher.b.data.qpos.copy()
    teacher.check_joint_path('right', start, start, neighbor_clearance=.003)
    with pytest.raises(SortFailure, match='grasp neighbor clearance'):
        teacher.check_joint_path('right', start, start, neighbor_clearance=.006)
    # One mm contact used to pass the four mm penetration gate.
    teacher.b.model.body_pos[teacher.b.model.body('task/neighbor').id, 0] = .139
    teacher.check_joint_path('right', start, start)
    with pytest.raises(SortFailure, match='grasp neighbor clearance'):
        teacher.check_joint_path('right', start, start, neighbor_clearance=.001)
    g = teacher.b.model.geom('neighbor').id
    teacher.b.model.geom_contype[g] = teacher.b.model.geom_conaffinity[g] = 0
    teacher.check_joint_path('right', start, start, neighbor_clearance=.001)
    np.testing.assert_array_equal(teacher.b.data.qpos, before)


def test_planned_aperture_and_interior_sweep_are_checked_without_mutating_live_state():
    teacher = geometry_teacher()
    teacher.b.data.qpos[1] = -.02
    before = teacher.b.data.qpos.copy()
    teacher.check_joint_path('right', np.array([0.]), np.array([0.]), neighbor_clearance=.001)
    with pytest.raises(SortFailure):
        teacher.check_joint_path('right', np.array([0.]), np.array([0.]),
                                 gripper=(.04, .04), neighbor_clearance=.001)
    # Both endpoints clear the neighbor, but the intervening sweep does not.
    with pytest.raises(SortFailure):
        teacher.check_joint_path('right', np.array([-.1]), np.array([.15]), neighbor_clearance=.001)
    np.testing.assert_array_equal(teacher.b.data.qpos, before)


def test_candidate_rejection_precedes_motion_and_keeps_reasons():
    teacher = teacher_fixture()
    teacher.grasp_clearance = profile()
    teacher.b = SimpleNamespace(task=SimpleNamespace(specs={'stick': {'kind': 'chopstick'}}), sim_time=1.)
    teacher.plan_grasp = lambda name: ('right', np.array([.2, .1, .03]), np.eye(3), .01)
    teacher.current, teacher.events = 'stick', []
    teacher.step = lambda: pytest.fail('Candidate search must not execute')
    checked = []
    def check(*args, **kwargs):
        checked.append(kwargs)
        if len(checked) == 1: raise SortFailure('neighbor too close')
    teacher.check_joint_path = check
    plans = []
    teacher.plan_pose_path = lambda *args, **kwargs: plans.append(kwargs)
    before = teacher.q.copy()
    teacher.reachable_grasp('stick', check_approach=True)
    assert all(c['neighbor_clearance'] == .001 and c['gripper'] == (.04, .04) for c in checked)
    assert plans[0]['neighbor_clearance'] == .001 and plans[0]['start'][13] == .04
    assert teacher.events[0]['rejected_candidate_details'][0]['error'] == 'neighbor too close'
    def blocked(*args, **kwargs): raise SortFailure('all neighbors blocked')
    teacher.check_joint_path = blocked
    with pytest.raises(SortFailure, match='No reachable grasp'):
        teacher.reachable_grasp('stick', check_approach=True)
    np.testing.assert_array_equal(teacher.q, before)


def test_spline_checks_planned_aperture_instead_of_unrelated_native_aperture():
    teacher = teacher_fixture()
    checked = []
    teacher.check_joint_path = lambda *args, **kwargs: checked.append(kwargs)
    start = teacher.q.copy(); start[13] = .01
    path = teacher.plan_pose_path('right', [(start[7:10]+[.02, 0, 0], np.eye(3))],
                                 start=start, neighbor_clearance=.001)
    assert len(checked) == len(path)
    for call in checked:
        assert call['neighbor_clearance'] == .001
        np.testing.assert_allclose(call['gripper'], [.01, .01])
    assert teacher.q[13] == .04


def test_extra_candidates_follow_measured_long_axis_with_end_margin():
    teacher = RigidTableTeacher.__new__(RigidTableTeacher)
    teacher.pacing={'grasp_height_offsets_m':[0.,.002,.004]}
    teacher.grasp_clearance = profile()
    teacher.b = SimpleNamespace(task=SimpleNamespace(specs={'stick': {'kind': 'chopstick', 'size_m': [.225, .006, .006]}}))
    teacher.object_pose = lambda name: {'position': [.3, -.2, .03],
        'quaternion_wxyz': [2**-.5, 0., 0., 2**-.5]}
    p, r = np.array([.3, -.2, .035]), np.eye(3)
    candidates = list(teacher.grasp_candidates('stick', p, r, check_approach=True))
    assert len(candidates) == 27
    for target, rotation, detail in candidates:
        offset = detail['long_axis_offset_m']
        np.testing.assert_allclose(target-p, [0., offset, detail['height_offset_m']], atol=1e-15)
        np.testing.assert_array_equal(rotation, r)
        assert abs(offset) <= .4*.225
    assert len(list(teacher.grasp_candidates('stick', p, r, check_approach=False))) == 5
