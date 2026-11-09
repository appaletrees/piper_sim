import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('native_payload_load',
    Path(__file__).resolve().parents[1]/'scripts/audit_native_payload_load.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_force_sign_respects_geometry_order_and_contact_frame():
    # Normal +Z, tangent +X/+Y; 2 N normal and 3/-4 N friction.
    frame = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    forces = diagnostic.contact_forces_on_payload(
        np.stack((frame, frame)), [[2, 3, -4, 0, 0, 0]] * 2, [True, False])
    np.testing.assert_allclose(forces, [[3, -4, 2], [-3, 4, -2]])


def test_contact_torque_is_not_added_to_linear_force():
    force = diagnostic.contact_forces_on_payload(
        [np.eye(3)], [[0, 0, 0, 100, 200, 300]], [True])
    np.testing.assert_array_equal(force, [[0, 0, 0]])


def test_nonfinite_force_is_rejected():
    with pytest.raises(ValueError):
        diagnostic.contact_forces_on_payload([np.eye(3)], [[np.nan, 0, 0, 0, 0, 0]], [True])
