import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('contact_cones',
    Path(__file__).resolve().parents[1]/'scripts/audit_grasp_contact_cones.py')
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_both_fingers_pressing_top_cannot_support_weight():
    # Both contacts have downward normals 30 degrees from -Z; even maximum
    # tangential friction at mu=0.82 cannot turn their forces upward.
    c,s=np.cos(np.pi/6),np.sin(np.pi/6)
    frame=np.array([[s,0,-c],[c,0,s],[0,1,0]])
    frames=np.stack((frame,frame));friction=np.full((2,2),.82)
    result=diagnostic.upward_force_capacity(frames,friction)
    assert result['all_finger_forces_strictly_downward']
    assert result['max_over_contacts']==pytest.approx(-c+.82*s)


def test_opposed_side_contacts_are_not_rejected_as_downward():
    frame=np.eye(3);opposed=frame.copy();opposed[0]*=-1
    frames=np.stack((frame,opposed));friction=np.full((2,2),.8)
    result=diagnostic.upward_force_capacity(frames,friction)
    assert not result['all_finger_forces_strictly_downward']
    assert result['max_over_contacts']==pytest.approx(.8)
    assert not diagnostic.horizontal_separation(frames,friction)['all_finger_cones_strictly_same_horizontal_halfspace']


def test_same_side_contact_forces_have_horizontal_separating_axis():
    result=diagnostic.horizontal_separation(np.stack((np.eye(3),np.eye(3))),np.full((2,2),.5))
    assert result['all_finger_cones_strictly_same_horizontal_halfspace']
    assert result['min_projection_per_unit_normal_force']==pytest.approx(1.)


def test_anisotropic_friction_uses_corresponding_tangent_axis():
    low,high=diagnostic.cone_force_bounds([np.eye(3)],[[.2,.8]],[[0,1,0],[0,0,1]])
    np.testing.assert_allclose(high,[[.2,.8]])
    np.testing.assert_allclose(low,[[-.2,-.8]])


def test_horizontal_rotation_preserves_upward_capacity_and_inputs():
    a=np.deg2rad(37);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    frames=np.array([[[.6,0,-.8],[.8,0,.6],[0,1,0]]]);friction=np.full((1,2),.5)
    before=frames.copy()
    assert diagnostic.upward_force_capacity(frames,friction)['max_over_contacts']==pytest.approx(
        diagnostic.upward_force_capacity(frames@r.T,friction)['max_over_contacts'])
    np.testing.assert_array_equal(frames,before)


@pytest.mark.parametrize('frames,friction,axes',[
    (np.zeros((1,3,3)),[[.5,.5]],[[0,0,1]]),
    ([np.eye(3)],[[-.5,.5]],[[0,0,1]]),
    ([np.eye(3)],[[.5,.5]],[[0,0,2]]),
])
def test_invalid_geometry_is_not_treated_as_evidence(frames,friction,axes):
    with pytest.raises(ValueError):diagnostic.cone_force_bounds(frames,friction,axes)
