"""Privileged geometric closure planning; only the caller issues servo targets.

Collision onset in an isolated MjData is not force/contact certification. The
normal physical contact, slip and trajectory checks remain the caller's job.
"""
import copy
import math

import mujoco
import numpy as np


def validate_profile(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'kinds', 'scan_step_m', 'overtravel_m'}:
        raise ValueError('Geometric capture requires exactly kinds, scan_step_m and overtravel_m')
    p = copy.deepcopy(value)
    if (not isinstance(p['kinds'], list) or not p['kinds']
            or any(not isinstance(k, str) or not k for k in p['kinds'])
            or len(set(p['kinds'])) != len(p['kinds'])):
        raise ValueError('Unique nonempty geometric capture kinds required')
    for key, lower, upper in [('scan_step_m', .00001, .001), ('overtravel_m', .00001, .002)]:
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lower <= v <= upper:
            raise ValueError(f'Invalid geometric capture {key}')
    return p


def closure_target(model, live, scratch, *, jaw_qpos, finger_bodies, object_body, profile):
    """Scan decreasing symmetric jaw widths against the current measured pose.

    Only scratch qpos/kinematics/collision buffers are written. Geometric
    penetration (dist <= 0), rather than an inactive contact margin or solver
    force, determines the first bilateral sample. No object motion is predicted.
    The endpoint is that sampled width minus explicitly declared overtravel.
    """
    if scratch is live or np.shares_memory(scratch.qpos, live.qpos):
        raise ValueError('Independent scratch MjData required')
    qids = np.asarray(jaw_qpos, dtype=int)
    fingers = tuple(int(b) for b in finger_bodies)
    if qids.shape != (2,) or len(set(qids)) != 2 or len(fingers) != 2 or len(set(fingers)) != 2:
        raise ValueError('Two independent jaws and finger bodies required')
    if (np.any(qids < 0) or np.any(qids >= model.nq)
            or any(b <= 0 or b >= model.nbody for b in (*fingers, object_body))
            or object_body in fingers):
        raise ValueError('Invalid geometric capture indices')
    original = live.qpos.copy()
    start = float(original[qids].sum())
    if not np.isfinite(original).all() or not 0 < start <= .096:
        raise RuntimeError('Invalid measured grasp opening')
    count = int(math.ceil(start/profile['scan_step_m']))
    for i, width in enumerate(np.linspace(start, 0., count+1)):
        scratch.qpos[:] = original
        scratch.qpos[qids] = width/2
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_collision(model, scratch)
        touched = set()
        for c in scratch.contact:
            if c.dist > 0:
                continue
            bodies = model.geom_bodyid[[c.geom1, c.geom2]]
            if object_body in bodies:
                touched.update(b for b in fingers if b in bodies)
        if len(touched) == 2:
            target = float(width-profile['overtravel_m'])
            if target <= 0:
                raise RuntimeError('Geometric capture requires a positive closure target')
            return {'measured_start_width_m': start, 'bilateral_width_m': float(width),
                    'target_width_m': target, 'scan_samples': i+1,
                    'actual_scan_step_m': start/count, 'overtravel_m': profile['overtravel_m'],
                    'already_bilateral_at_start': i == 0,
                    'scope': 'Isolated geometric prediction; actual solver contacts still required'}
    raise RuntimeError('No bilateral geometric capture within the gripper stroke')
