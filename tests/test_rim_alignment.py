import numpy as np
import pytest

from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.experts.rim_alignment import hand_path


def example():
    a=np.deg2rad(60)
    r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    points=np.array([[x,y,z] for x in (-.095,.095) for y in (-.095,.095) for z in (0.,.016)])
    p=np.array([.3,0.,.025-(points@r.T)[:,2].min()])
    hp=p+r@[-.085,0.,.014]
    return p,r,hp,grasp_rotation(45,0,90),points


def test_measured_alignment_preserves_grasp_and_joins_without_pose_jump():
    args=example(); before=[v.copy() for v in args]
    poses, evidence=hand_path(*args,angle_deg=25,support_z_m=.025,gap_m=.008)
    p,r,hp,hr,points=args
    np.testing.assert_allclose(poses[0][0],hp,atol=1e-14)
    np.testing.assert_allclose(poses[0][1],hr,atol=1e-14)
    initial_offset=hr.T@(p-hp); initial_rotation=hr.T@r
    for (h,rot),planned in zip(poses,evidence['planned_object_poses']):
        op=np.asarray(planned['object_position']);ort=np.asarray(planned['object_rotation'])
        np.testing.assert_allclose(rot.T@(op-h),initial_offset,atol=1e-14)
        np.testing.assert_allclose(rot.T@ort,initial_rotation,atol=1e-14)
        assert (points@ort.T+op)[:,2].min() >= .025-1e-14
    last=evidence['planned_object_poses'][-1]
    ort=np.asarray(last['object_rotation']);op=np.asarray(last['object_position'])
    assert (points@ort.T+op)[:,2].min()==pytest.approx(.033)
    assert evidence['final_tilt_deg']==pytest.approx(85.)
    for a,b in zip(args,before):np.testing.assert_array_equal(a,b)


def test_alignment_refuses_detached_or_overturned_plate():
    args=list(example());args[0]=args[0]+[0,0,.01]
    with pytest.raises(ValueError,match='support'):hand_path(*args,angle_deg=20,support_z_m=.025)
    with pytest.raises(ValueError,match='final tilt'):hand_path(*example(),angle_deg=30,support_z_m=.025)
