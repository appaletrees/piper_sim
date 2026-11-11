"""Measured-frame, support-preserving rim alignment geometry.

Returns hand waypoints only. The caller must execute position commands through
normal physics and retain physical contact, slip and trajectory quality gates.
"""
import numpy as np


def hand_path(object_position, object_rotation, hand_position, hand_rotation,
              collision_points, *, angle_deg, support_z_m, gap_m=.0005):
    p, r, hp, hr, points = (np.asarray(v, float).copy() for v in (
        object_position, object_rotation, hand_position, hand_rotation, collision_points))
    if (p.shape != (3,) or hp.shape != (3,) or r.shape != (3,3) or hr.shape != (3,3)
            or points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not all(np.isfinite(v).all() for v in (p,r,hp,hr,points))):
        raise ValueError('Finite hand, object and collision vertices required')
    if any(not np.allclose(a.T@a,np.eye(3),atol=1e-6) or np.linalg.det(a)<0 for a in (r,hr)):
        raise ValueError('Proper rotations required')
    scalars=np.asarray([angle_deg,support_z_m,gap_m],float)
    if not np.isfinite(scalars).all() or not 1 <= angle_deg <= 35 or not 0 <= gap_m <= .01:
        raise ValueError('Alignment increment 1..35 degrees and gap 0..10 mm required')
    tilt=np.degrees(np.arccos(np.clip(r[2,2],-1,1)))
    if not 10 <= tilt or tilt+angle_deg > 88:
        raise ValueError('Measured tilt must be >=10 degrees and final tilt <=88 degrees')
    bottom=p[2]+(points@r.T)[:,2].min()
    if abs(bottom-support_z_m)> .003:
        raise ValueError('Alignment requires measured support within 3 mm')
    axis=np.cross([0.,0.,1.],r[:,2]); axis/=np.linalg.norm(axis)
    k=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    offset=hp-p; poses=[]; planned=[]
    for degrees in np.linspace(0.,angle_deg,int(np.ceil(angle_deg/5))+1):
        a=np.deg2rad(degrees); delta=np.eye(3)+np.sin(a)*k+(1-np.cos(a))*(k@k)
        ort=delta@r; op=p.copy()
        op[2]=support_z_m+gap_m-(points@ort.T)[:,2].min()
        # The first pose joins continuously to the measured hand rather than
        # instantaneously correcting its small contact penetration/gap.
        fraction=degrees/angle_deg
        op[2] += (1-fraction)*(bottom-support_z_m-gap_m)
        poses.append((op+delta@offset,delta@hr))
        planned.append({'angle_deg':float(degrees),'object_position':op.tolist(),
                        'object_rotation':ort.tolist()})
    return poses, {'hypothesis':'Rigid non-slipping measured grasp on a horizontal support; only hand poses are commanded.',
        'initial_tilt_deg':float(tilt),'final_tilt_deg':float(tilt+angle_deg),
        'support_z_m':float(support_z_m),'gap_m':float(gap_m),'axis_world':axis.tolist(),
        'planned_object_poses':planned}
