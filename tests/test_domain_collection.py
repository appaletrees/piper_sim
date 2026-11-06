import copy
import numpy as np
import pytest
import yaml

from piper_sim.config import ROOT, load
from piper_sim.randomization import load_profile, sample, sensor_images
from piper_sim.tasks import create_task
from piper_sim.experts.trajectory import smooth_trajectory
from piper_sim.experts.quality import MotionQuality


def test_c3_trajectory_bounds_across_reversals_and_holds():
    dt = 1/30
    q0 = np.zeros(14)
    q1 = np.array([1, -.3, .6, 0, -.5, 2., .095]*2)
    v, a, j = np.array([.5]*6+[.04])*1., np.array([1.2]*6+[.12]), np.array([8.]*6+[1.])
    v, a, j = (np.tile(x, 2) for x in (v, a, j))
    first = smooth_trajectory(q0, q1, dt, velocity=v, acceleration=a, jerk=j)
    second = smooth_trajectory(q1, q0, dt, velocity=v, acceleration=a, jerk=j)
    path = np.concatenate([np.tile(q0, (5, 1)), first, np.tile(q1, (5, 1)), second, np.tile(q0, (5, 1))])
    for order, bound in enumerate((v, a, j), 1):
        assert np.all(np.abs(np.diff(path, n=order, axis=0))/dt**order <= bound+1e-7)
    np.testing.assert_array_equal(first[-1], q1)
    assert np.all(np.diff(first[:, 0]) >= 0)
    with pytest.raises(ValueError):
        smooth_trajectory(q0, q1, dt, velocity=0, acceleration=a, jerk=j)


def test_randomization_independent_streams_and_validation():
    profile = load_profile('configs/domain_randomization/clean_table.yaml')
    a = sample(profile, 42, ['paper'], ['paper'])
    b = sample(profile, 42, ['new', 'paper'], ['paper', 'new'])
    assert a['objects']['paper'] == b['objects']['paper']
    assert a['materials']['paper'] == b['materials']['paper']
    assert a['cameras'] == b['cameras']
    assert a != sample(profile, 43, ['paper'], ['paper'])
    for field in ('mass_scale', 'friction_scale'):
        bad = copy.deepcopy(profile); bad['physics'][field] = [-1, 1]
        with pytest.raises(ValueError): load_profile(bad)
    images = {'overhead': np.full((40, 50, 3), 128, dtype=np.uint8)}
    first = sensor_images(images, a, 10)['overhead']
    np.testing.assert_array_equal(first, sensor_images(images, a, 10)['overhead'])
    assert not np.array_equal(first, sensor_images(images, a, 11)['overhead'])
    assert images['overhead'].min() == 128


def test_mujoco_domain_reset_is_absolute_and_preserves_camera_geometry():
    pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    cfg = load('configs/clean_table_workcell.yaml')
    cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table.yaml').read_text())
    cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
    b = MujocoBackend(cfg)
    try:
        m = b.model
        nominal = {k: getattr(m, k).copy() for k in ('body_mass', 'body_inertia', 'geom_friction', 'mat_rgba', 'light_pos')}
        camera = [m.cam_pos.copy(), m.cam_quat.copy(), m.cam_fovy.copy()]
        b.reset(42, True)
        params = copy.deepcopy(b.task.domain_parameters)
        first = {k: getattr(m, k).copy() for k in nominal}
        body = m.body('task/paper').id
        assert m.body_mass[body] == pytest.approx(nominal['body_mass'][body]*params['objects']['paper']['mass_scale'])
        b.reset(7, True); b.reset(42, True)
        assert b.task.domain_parameters == params
        for k in first: np.testing.assert_array_equal(getattr(m, k), first[k])
        for _ in range(60): b.step()
        assert np.isfinite(b.data.qpos).all()
        b.reset(8, False)
        assert b.task.domain_parameters is None
        for k in nominal: np.testing.assert_array_equal(getattr(m, k), nominal[k])
        for value, original in zip((m.cam_pos, m.cam_quat, m.cam_fovy), camera):
            np.testing.assert_array_equal(value, original)
    finally: b.close()


def test_grasp_transforms_equivariantly_with_object_pose():
    pytest.importorskip('mujoco')
    from piper_sim.experts.clean_table import CleanTableTeacher
    from piper_sim.tasks.clean_table import yaw_quaternion
    from types import SimpleNamespace
    task = create_task('configs/tasks/clean_table.yaml', load())
    teacher = CleanTableTeacher.__new__(CleanTableTeacher)
    teacher.b = SimpleNamespace(task=task)
    for name in task.config['success']['required']:
        nominal = copy.deepcopy(task.layout['poses'][name])
        teacher.object_pose = lambda n: nominal
        _, p, r, w = teacher.plan_grasp(name)
        angle = .17
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        shifted = {'position': (rotation@nominal['position']+[.02, -.03, 0]).tolist(),
                   'quaternion_wxyz': yaw_quaternion(task.specs[name].get('yaw_deg', 0)+np.rad2deg(angle))}
        teacher.object_pose = lambda n: shifted
        _, p2, r2, w2 = teacher.plan_grasp(name)
        np.testing.assert_allclose(p2, rotation@p+[.02, -.03, 0], atol=1e-12)
        np.testing.assert_allclose(r2, rotation@r, atol=1e-12)
        assert w2 == w


def test_quality_rejects_motion_spike_and_timestamp_gap():
    settings = yaml.safe_load((ROOT/'configs/collection/clean_table.yaml').read_text())
    monitor = MotionQuality(settings, 1/30)
    for i in range(20): monitor.add(i/30, np.zeros(14), np.zeros(14), holding=True)
    assert monitor.report()['passed']
    bad = np.zeros(14); bad[0] = .1
    monitor.add(22/30, bad, bad)
    result = monitor.report()
    assert not result['passed']
    assert 'command_arm_jerk' in result['violations']
    assert 'nonuniform_simulation_timestamps' in result['violations']
