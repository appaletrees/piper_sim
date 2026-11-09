from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher


def make_teacher(poses, success=True):
    teacher = RigidTableTeacher.__new__(RigidTableTeacher)
    teacher.fps, teacher.holding, teacher.carried = 30., False, None
    teacher.settings = {'dataset': {'final_stable_hold_s': 1., 'final_object_speed_m_s': .01,
                                    'final_object_angular_speed_rad_s': .1}}
    teacher.pacing = {'final_settle_timeout_s': 5.}
    teacher.b = SimpleNamespace(task=SimpleNamespace(required=['object']))
    teacher.frames, teacher.events = 0, []
    def observation():
        return SimpleNamespace(extras={'objects': {'object': poses(teacher.frames)},
            'task': {'metrics': {'summary': {'success': success}}}})
    teacher.session = SimpleNamespace(read=observation)
    def step():
        assert teacher.holding
        teacher.frames += 1
        return observation()
    teacher.step = step
    teacher.event = lambda phase, **kwargs: teacher.events.append({'phase': phase, **kwargs})
    return teacher


def stationary(frame):
    # Quaternion sign changes represent identical orientation.
    return {'position': [0., 0., 0.], 'quaternion_wxyz': [(-1.)**frame, 0., 0., 0.]}


def test_already_stable_scene_preserves_two_second_recorded_minimum():
    teacher = make_teacher(stationary)
    teacher.wait_for_final_stability()
    assert teacher.frames == 60 and not teacher.events and not teacher.holding


def test_late_motion_extends_without_trimming_or_changing_thresholds():
    def pose(frame):
        # Last moving interval ends at frame 33, leaving 27 stable frames at 2 s.
        angle = min(frame, 33)*.2/30
        return {'position': [0., 0., 0.], 'quaternion_wxyz': [np.cos(angle/2), 0., 0., np.sin(angle/2)]}
    teacher = make_teacher(pose)
    teacher.wait_for_final_stability()
    assert teacher.frames == 63 and teacher.events[0]['phase'] == 'final_settle_extension'
    assert teacher.events[0]['stable_seconds'] == pytest.approx(.9)
    assert teacher.settings['dataset']['final_object_angular_speed_rad_s'] == .1


@pytest.mark.parametrize('success,drift', [(False, 0.), (True, .001)])
def test_uncontained_or_continuously_moving_scene_times_out_and_restores_holding(success, drift):
    def pose(frame):
        p = stationary(frame)
        p['position'][0] = frame*drift
        return p
    teacher = make_teacher(pose, success)
    with pytest.raises(SortFailure, match='did not remain stable'):
        teacher.wait_for_final_stability()
    assert teacher.frames == 150 and not teacher.holding
