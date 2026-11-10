"""Rigid grasp geometry for tilting a vessel while retaining planar support.

This only proposes hand poses. The object is never assigned these poses during
execution; real contact, slip, collision and motion gates remain necessary.
"""
import copy
import math

import numpy as np


def validate_profile(value):
    if value is None: return None
    fields = {'angle_deg', 'angle_step_deg', 'support_gap_m', 'clear_lift_m',
              'path_speed_m_s', 'max_grasp_rotation_rad'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Supported tilt requires angle, sampling, gap, lift, speed and rotation budget')
    p = copy.deepcopy(value)
    for key, low, high in [('angle_deg', 20., 85.), ('angle_step_deg', 1., 15.),
                          ('support_gap_m', 0., .002), ('clear_lift_m', .04, .15),
                          ('path_speed_m_s', .01, .1), ('max_grasp_rotation_rad', .01, .2)]:
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError('Invalid supported tilt '+key)
    return p


def hand_path(object_position, object_rotation, hand_position, hand_rotation, surface_points, profile):
    """Return tilt/lift waypoints under an explicit no-slip planning hypothesis.

    At every tilt sample the lowest supplied object surface stays on its initial
    horizontal support plane plus the configured gap. XY of the object origin
    stays fixed; the supporting contact point may roll. The last waypoint raises
    the entire vessel clear of support. This is not a physical pivot constraint.
    """
    p, r, hp, hr, points = [np.asarray(x, float).copy() for x in (
        object_position, object_rotation, hand_position, hand_rotation, surface_points)]
    if (p.shape != (3,) or hp.shape != (3,) or r.shape != (3, 3) or hr.shape != (3, 3)
            or points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not all(np.isfinite(v).all() for v in (p, r, hp, hr, points))):
        raise ValueError('Finite object/hand poses and surface samples required')
    for rotation in (r, hr):
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
            raise ValueError('Proper rotation matrices required')
    if r[2, 2] < np.cos(np.deg2rad(5)):
        raise ValueError('Supported tilt requires an initially upright vessel')
    radial = hp-p;radial[2] = 0.
    if np.linalg.norm(radial) < .02:
        raise ValueError('An off-center rim grasp is required')
    axis = np.cross(radial, [0., 0., 1.]);axis /= np.linalg.norm(axis)
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    support_z = float(p[2]+(points@r.T)[:, 2].min())
    offset = hp-p
    count = int(math.ceil(profile['angle_deg']/profile['angle_step_deg']))
    poses, planned = [], []
    for degrees in np.linspace(0., profile['angle_deg'], count+1):
        a = np.deg2rad(degrees);delta = np.eye(3)+np.sin(a)*k+(1.-np.cos(a))*(k@k)
        obj_r = delta@r
        obj_p = p.copy();obj_p[2] = support_z+profile['support_gap_m']-(points@obj_r.T)[:, 2].min()
        poses.append((obj_p+delta@offset, delta@hr))
        planned.append({'angle_deg': float(degrees), 'object_position': obj_p.tolist(),
                        'object_rotation': obj_r.tolist(), 'bottom_gap_m': profile['support_gap_m']})
    last_p, last_r = poses[-1]
    poses.append((last_p+np.array([0., 0., profile['clear_lift_m']]), last_r.copy()))
    return poses, {'support_plane_z_m': support_z, 'world_tilt_axis': axis.tolist(),
                   'planned_object_poses': planned, 'clear_lift_m': profile['clear_lift_m'],
                   'hypothesis': 'Rigid non-slipping grasp and initially horizontal planar support; only hand targets are executed.'}
