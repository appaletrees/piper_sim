import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_payload_wrench import wrenches_about_com


def test_world_wrench_keeps_force_lever_arm_and_direct_contact_torque():
    f, t = wrenches_about_com(np.eye(3)[None], np.array([[0., 2., 0., 0., 0., .5]]),
        [True], [[.1, 0., 0.]], [[0., 0., 0.]])
    np.testing.assert_allclose(f, [[0., 2., 0.]])
    np.testing.assert_allclose(t, [[0., 0., .7]])


def test_geom1_sign_and_rotated_contact_frame_apply_to_force_and_torque():
    frame = np.array([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])
    f, t = wrenches_about_com(frame, np.array([[2., 0., 0., .5, 0., 0.]]),
        [False], [[1., 0., 0.]], [[1., 0., 0.]])
    np.testing.assert_allclose(f, [[0., -2., 0.]])
    np.testing.assert_allclose(t, [[0., -.5, 0.]])


def test_translation_of_contact_and_com_preserves_wrench():
    local = np.array([[1., 2., 3., .1, .2, .3]])
    a = wrenches_about_com(np.eye(3)[None], local, [True], [[.1, .2, .3]], [[0., 0., 0.]])
    b = wrenches_about_com(np.eye(3)[None], local, [True], [[2.1, 3.2, 4.3]], [[2., 3., 4.]])
    for x, y in zip(a, b): np.testing.assert_allclose(x, y, atol=1e-14)


def test_nonfinite_com_is_rejected():
    with pytest.raises(ValueError, match='Finite'):
        wrenches_about_com(np.eye(3)[None], np.zeros((1, 6)), [True], [[0., 0., 0.]], [[np.nan, 0., 0.]])
