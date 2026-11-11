"""Measured-frame placement planning for a physically held rim.

Only hand position targets are returned. Rigid grasp and vertical support are
planning hypotheses; the caller retains real contact, slip and motion checks.
"""
import copy
import math
from itertools import product

import numpy as np

from .mujoco_ik import rotation_error
from .placement_search import support_surface


def validate_profile(value):
    if value is None:
        return None
    keys = {'kinds', 'yaw_offsets_deg', 'xy_step_m', 'wall_margin_m', 'support_gap_m',
            'rim_clearance_m', 'transfer_speed_m_s', 'lower_speed_m_s',
            'max_path_orientation_error_rad', 'max_candidates'}
    if not isinstance(value, dict) or not keys <= set(value) or set(value)-keys-{'tilt_offsets_deg'}:
        raise ValueError('rim_transport requires exactly ' + ', '.join(sorted(keys)))
    p = copy.deepcopy(value)
    p.setdefault('tilt_offsets_deg', [0])
    if p['kinds'] != ['plate']:
        raise ValueError('Measured rim transport currently supports plate only')
    yaws = p['yaw_offsets_deg']
    if (not isinstance(yaws, list) or not yaws or yaws[0] != 0
            or len(set(yaws)) != len(yaws) or any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or abs(v) > 45 for v in yaws)):
        raise ValueError('Unique finite yaw offsets within 45 degrees, starting at zero, required')
    tilts = p['tilt_offsets_deg']
    if (not isinstance(tilts, list) or not tilts or tilts[0] != 0 or len(set(tilts)) != len(tilts)
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or abs(v) > 20 for v in tilts)):
        raise ValueError('Unique tilt offsets within 20 degrees, starting at zero, required')
    for key, lo, hi in [('xy_step_m', .015, .05), ('wall_margin_m', .005, .03),
                       ('support_gap_m', .003, .02), ('rim_clearance_m', .02, .06),
                       ('transfer_speed_m_s', .03, .2), ('lower_speed_m_s', .01, .06),
                       ('max_path_orientation_error_rad', .005, .035)]:
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError('Invalid rim transport ' + key)
    if type(p['max_candidates']) is not int or not 1 <= p['max_candidates'] <= 256:
        raise ValueError('rim_transport max_candidates must be in 1..256')
    return p


def yaw_rotation(angle):
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])


def orientation_error(rotation, reference):
    """Deviation from a pure world-up yaw of the measured hand frame."""
    delta = rotation @ reference.T
    yaw = np.arctan2(delta[1, 0], delta[0, 0])
    expected = yaw_rotation(np.rad2deg(yaw)) @ reference
    return float(np.linalg.norm(rotation_error(expected, rotation))), float(yaw)


def tilt_rotation(reference, angle):
    axis = np.cross([0., 0., 1.], reference[:, 2])
    if np.linalg.norm(axis) < .1:
        raise ValueError('Rim hand reference requires an oblique approach axis')
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.deg2rad(angle)
    return np.eye(3) + np.sin(a)*k + (1-np.cos(a))*(k@k)


def frame_coordinates(rotation, reference):
    """Separate intended yaw/tilt from unrequested roll about the tool axis."""
    original, actual = reference[:, 2], rotation[:, 2]
    yaw = np.arctan2(actual[1], actual[0]) - np.arctan2(original[1], original[0])
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    tilt = np.arctan2(np.linalg.norm(original[:2]), -original[2]) - np.arctan2(np.linalg.norm(actual[:2]), -actual[2])
    expected = yaw_rotation(np.rad2deg(yaw)) @ tilt_rotation(reference, np.rad2deg(tilt)) @ reference
    return float(np.linalg.norm(rotation_error(expected, rotation))), float(yaw), float(tilt)


def search(teacher, side, name, target, profile):
    from .clean_table import SortFailure
    p = validate_profile(profile)
    b = teacher.b
    tcp, hand_r = teacher.tcp(side)
    obj, basket = (b.data.body('task/' + n) for n in (name, target))
    relative_p = hand_r.T @ (obj.xpos - tcp)
    relative_r = hand_r.T @ obj.xmat.reshape(3, 3)
    bin_p, bin_r = basket.xpos.copy(), basket.xmat.reshape(3, 3).copy()
    spec = b.task.specs[target]
    half = np.asarray(spec['size_m'][:2]) / 2 - spec['wall_m']
    rim = bin_p[2] + spec['size_m'][2]
    points = b.task.surface_points[name]
    support, obstacles = support_surface(teacher, target, teacher.carried_objects())
    x = np.arange(-int(half[0]/p['xy_step_m']), int(half[0]/p['xy_step_m']) + 1) * p['xy_step_m']
    y = np.arange(-int(half[1]/p['xy_step_m']), int(half[1]/p['xy_step_m']) + 1) * p['xy_step_m']
    slots = [np.array([a, c, 0.]) for a in x for c in y]
    off = 0 if side == 'left' else 7
    failures, checked = [], 0
    tiers = sorted(product(p['yaw_offsets_deg'], p['tilt_offsets_deg']), key=lambda c: abs(c[1])+.2*abs(c[0]))
    for yaw, tilt in tiers:
        hand_goal = yaw_rotation(yaw) @ tilt_rotation(hand_r, tilt) @ hand_r if tilt else yaw_rotation(yaw) @ hand_r
        object_goal = hand_goal @ relative_r
        rotated = points @ object_goal.T
        candidates = []
        for slot in slots:
            local = rotated @ bin_r + slot
            margin = float(np.min(half - np.max(abs(local[:, :2]), axis=0)))
            # The released plate may settle flat. A narrow tilted projection
            # alone must not select a slot whose level footprint crosses a wall.
            level_margin = float(np.min(half - np.max(abs(points[:, :2] + slot[:2]), axis=0)))
            margin = min(margin, level_margin)
            if margin < p['wall_margin_m']:
                continue
            object_p = bin_p + bin_r @ slot
            heights = support.height(object_p[:2] + rotated[:, :2])
            valid = np.isfinite(heights)
            if not np.any(valid):
                continue
            low_obj = object_p.copy()
            low_obj[2] = np.max(heights[valid] - rotated[valid, 2]) + p['support_gap_m']
            high_obj = low_obj.copy()
            high_obj[2] = max(low_obj[2] + .045, rim + p['rim_clearance_m'] - rotated[:, 2].min())
            high, low = high_obj - hand_goal @ relative_p, low_obj - hand_goal @ relative_p
            candidates.append((float(np.linalg.norm(high[:2] - tcp[:2])), high, low, slot, margin))
        # Reserve a finite share for every yaw tier instead of spending the
        # entire IK budget on almost identical orientations.
        tier_budget = max(1, int(np.ceil(p['max_candidates'] / len(tiers))))
        for _, high, low, slot, margin in sorted(candidates, key=lambda c: c[0])[:tier_budget]:
            if checked >= p['max_candidates']:
                raise SortFailure('Measured rim placement candidate budget exhausted: ' + repr(failures[-3:]))
            checked += 1
            try:
                q_high = teacher.solve(side, high, hand_goal)
                teacher.solve(side, low, hand_goal, q_high)
                lift = tcp.copy();lift[2] = max(tcp[2], high[2])
                transfer = teacher.plan_pose_path(side, [(lift, hand_r), (high, hand_goal)],
                                                  speed=p['transfer_speed_m_s'])
                lower = teacher.plan_pose_path(side, [(low, hand_goal)], start=transfer[-1],
                                               speed=p['lower_speed_m_s'])
                aperture = float(b.data.qpos[b.indices[side]['qpos'][6:]].sum())
                release = min(.095, teacher.plan_grasp(name)[3] + .020)
                widths = np.linspace(aperture, release, max(2, int(np.ceil(abs(release-aperture)/.002))+1))
                for before, after in zip(widths[:-1], widths[1:]):
                    teacher.check_joint_path(side, lower[-1][off:off+6], lower[-1][off:off+6],
                                             gripper=(before, after))
                maximum = 0.
                for q in np.vstack((transfer, lower)):
                    _, actual = teacher.ik[side].forward(q[off:off+6])
                    if tilt:
                        error, actual_yaw, actual_tilt = frame_coordinates(actual, hand_r)
                    else:
                        error, actual_yaw = orientation_error(actual, hand_r);actual_tilt = 0.
                    maximum = max(maximum, error)
                    bound = p['max_path_orientation_error_rad']
                    if (error > bound or actual_yaw < min(0., np.deg2rad(yaw)) - bound
                            or actual_yaw > max(0., np.deg2rad(yaw)) + bound
                            or actual_tilt < min(0., np.deg2rad(tilt)) - bound
                            or actual_tilt > max(0., np.deg2rad(tilt)) + bound):
                        raise SortFailure('Measured rim hand-frame path deviates from its yaw/tilt reference')
            except SortFailure as exc:
                failures.append({'yaw_deg': yaw, 'tilt_offset_deg': tilt, 'slot_xy_m': slot[:2].tolist(), 'error': str(exc)})
                continue
            return {'transfer': transfer, 'lower': lower, 'high': high, 'low': low,
                    'rotation': hand_goal, 'evidence': {
                        'strategy': 'measured_rim_hand_frame', 'yaw_offset_deg': yaw,
                        'tilt_offset_deg': tilt,
                        'slot_xy_m': slot[:2].tolist(), 'wall_margin_m': margin,
                        'predicted_object_up_world': object_goal[:, 2].tolist(),
                        'predicted_tilt_from_container_up_rad': float(np.arccos(np.clip(
                            object_goal[:, 2] @ bin_r[:, 2], -1., 1.))),
                        'orientation_reference_error_rad': maximum,
                        'opening_sweep_checked_m': [aperture, release],
                        'footprint_gate': 'Measured tilted footprint and level plate footprint both retain the wall margin.',
                        'candidates_checked': checked, 'rejected_candidates': failures,
                        'support_obstacles': obstacles,
                        'scope': 'Measured rigid-grasp and sampled support prediction; physical slip/stability still required.'}}
    raise SortFailure('No measured-frame rim placement: ' + repr(failures[-3:]))
