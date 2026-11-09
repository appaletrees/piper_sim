from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from piper_sim.experts.collision import robot_obstacle_clearance


def test_positive_robot_container_distance_checks_masks_and_selected_arm():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="right/link6"><joint name="joint"/><geom type="sphere" size=".02" pos=".1 0 0" mass="1"/></body>
      <body name="left/link6" pos=".14 0 0"><geom type="sphere" size=".02"/></body>
      <body name="task/basket" pos=".145 0 0"><geom name="basket" type="sphere" size=".02"/></body>
      <body name="task/ignored" pos=".12 0 0"><geom type="sphere" size=".02"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    distance, pair = robot_obstacle_clearance(model, data, 'right', ['basket'], .01)
    assert distance == pytest.approx(.005) and pair == ['right/link6', 'task/basket']
    teacher = CleanTableTeacher.__new__(CleanTableTeacher)
    teacher.b = SimpleNamespace(model=model, data=data,
        task=SimpleNamespace(specs={'item': {'target': 'basket'}}))
    teacher.ik = {'right': SimpleNamespace(data=mujoco.MjData(model), qids=np.array([0]))}
    teacher.carried_objects = lambda: []
    teacher.current, teacher.pacing = 'ignored', {'continuous_paths': True}
    teacher.settings = {'quality': {'max_robot_penetration_m': .1}}
    teacher.check_joint_path('right', np.array([0.]), np.array([0.]), transit_clearance=.003)
    with pytest.raises(SortFailure, match='arm/container clearance'):
        teacher.check_joint_path('right', np.array([0.]), np.array([0.]), transit_clearance=.006)
    # World positions and collision masks must never be cached with topology.
    model.body_pos[model.body('task/basket').id, 0] = .138
    mujoco.mj_forward(model, data)
    assert robot_obstacle_clearance(model, data, 'right', ['basket'], .01)[0] == pytest.approx(-.002)
    model.geom_contype[model.geom('basket').id] = 0
    model.geom_conaffinity[model.geom('basket').id] = 0
    assert robot_obstacle_clearance(model, data, 'right', ['basket'], .01) == (.01, None)


def fixture_teacher():
    teacher = CleanTableTeacher.__new__(CleanTableTeacher)
    teacher.q = np.array([0., .2, -.4, 0., .1, 0., .03]*2)
    teacher.pacing = {'unloaded_transit_clearance_m': .002, 'unloaded_transit_rises_m': [.02, .04]}
    teacher.current, teacher.carried = 'item', None
    teacher.b = SimpleNamespace(sim_time=1.)
    teacher.ik = {'right': SimpleNamespace(forward=lambda q: (np.array([.3, -.3, .2]), np.eye(3)))}
    teacher.events, teacher.executed = [], []
    teacher.step = lambda: teacher.executed.append(teacher.q.copy())
    teacher.settle = lambda: None
    teacher.move_joints = lambda *a, **k: teacher.executed.append('final_move')
    return teacher


def test_clear_direct_transit_does_not_add_waypoints():
    teacher = fixture_teacher()
    checks = []
    teacher.check_joint_path = lambda *a, **k: checks.append(k)
    teacher.plan_pose_path = lambda *a, **k: pytest.fail('Unnecessary detour')
    teacher.transit_to_pregrasp('right', teacher.q[7:13]+.01, .02)
    assert teacher.executed == ['final_move'] and teacher.events == []
    assert checks[0] == {'gripper': (.03, .02), 'transit_clearance': .002}


def test_entire_detour_is_checked_before_any_motion_and_blocked_routes_do_not_execute():
    teacher = fixture_teacher()
    planned = []
    def plan(side, poses):
        assert not teacher.executed
        q = teacher.q.copy(); q[7] = poses[0][0][2]
        planned.append(q.copy())
        return np.array([q])
    def check(side, start, end, **kwargs):
        assert not teacher.executed
        if start[0] < .23: raise SortFailure('near container')
    teacher.plan_pose_path, teacher.check_joint_path = plan, check
    teacher.transit_to_pregrasp('right', teacher.q[7:13]+.01, .02)
    assert len(planned) == 2 and teacher.events[0]['vertical_rise_m'] == .04
    np.testing.assert_array_equal(teacher.executed[0], planned[-1])
    assert teacher.executed[-1] == 'final_move'
    teacher = fixture_teacher()
    def blocked(*args, **kwargs): raise SortFailure('blocked')
    teacher.check_joint_path = teacher.plan_pose_path = teacher.solve = blocked
    with pytest.raises(SortFailure, match='No checked pregrasp transit'):
        teacher.transit_to_pregrasp('right', teacher.q[7:13]+.01, .02)
    assert not teacher.executed
