from types import SimpleNamespace

import numpy as np
import pytest

from piper_sim.experts.clean_table import CleanTableTeacher


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@pytest.mark.parametrize('angle', [0., .6, np.pi])
def test_contact_anchor_can_stay_fixed_while_object_rotates(angle):
    teacher = CleanTableTeacher.__new__(CleanTableTeacher)
    teacher.carried = ('right', 'plate', np.zeros(3))
    teacher.carried_rotation = np.eye(3)
    bodies = {'right/link6': SimpleNamespace(xmat=np.eye(3).ravel()),
              'task/plate': SimpleNamespace(xmat=rotation_z(angle).ravel())}
    teacher.b = SimpleNamespace(data=SimpleNamespace(body=bodies.__getitem__))
    before = {name: body.xmat.copy() for name, body in bodies.items()}
    # A rotation about the origin leaves the contact anchor at the origin:
    # zero translational slip must not imply zero rotational slip.
    np.testing.assert_array_equal(rotation_z(angle)@np.zeros(3), np.zeros(3))
    assert teacher.grasp_rotation_change() == pytest.approx(angle)
    for name, body in bodies.items():np.testing.assert_array_equal(body.xmat, before[name])
    # Moving the entire grasp in the world cannot change relative rotation.
    global_r = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    for body in bodies.values():body.xmat = (global_r@body.xmat.reshape(3, 3)).ravel()
    assert teacher.grasp_rotation_change() == pytest.approx(angle)
