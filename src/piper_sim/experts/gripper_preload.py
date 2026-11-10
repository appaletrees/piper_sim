"""Optional position-target preload reduction after a physical pinch.

This is a privileged teacher policy, not a servo override. Only ordinary
recorded gripper position trajectories are issued. Width tracking error is a
configuration budget; it is not a calibrated normal-force measurement.
"""
import copy
import math


def validate_profile(value):
    if value is None:
        return None
    keys = {'kinds', 'target_tracking_error_m', 'max_tracking_error_m', 'max_iterations'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('Grasp preload requires exactly kinds, tracking-error budgets and max_iterations')
    p = copy.deepcopy(value)
    if (not isinstance(p['kinds'], list) or not p['kinds']
            or any(not isinstance(k, str) or not k for k in p['kinds'])
            or len(p['kinds']) != len(set(p['kinds']))):
        raise ValueError('Unique nonempty grasp preload kinds required')
    for key in ('target_tracking_error_m', 'max_tracking_error_m'):
        if (isinstance(p[key], bool) or not isinstance(p[key], (int, float))
                or not math.isfinite(p[key]) or not 0 < p[key] < .095):
            raise ValueError('Positive finite preload width budget below stroke required')
    if p['target_tracking_error_m'] >= p['max_tracking_error_m']:
        raise ValueError('Target preload must be strictly below its acceptance budget')
    if type(p['max_iterations']) is not int or not 1 <= p['max_iterations'] <= 20:
        raise ValueError('Grasp preload max_iterations must be an integer in [1, 20]')
    return p


def relax_preload(*, profile, read_width, read_target, has_bilateral_contact,
                  move, report):
    """Bounded, monotone opening of the position target while contact persists.

``move`` must finish a normally bounded trajectory and its settling interval.
No target is clipped, no state is assigned, and no controller parameter changes.
Failure is explicit; callers must not proceed with an unverified grasp.
"""
    for iteration in range(profile['max_iterations']+1):
        width, target = float(read_width()), float(read_target())
        if not math.isfinite(width) or not math.isfinite(target):
            raise RuntimeError('Nonfinite grasp preload measurement')
        if not has_bilateral_contact():
            raise RuntimeError('Bilateral contact lost while reducing grasp preload')
        error = width-target
        report({'iteration': iteration, 'measured_width_m': width,
                'command_width_m': target, 'tracking_error_m': error})
        if error <= profile['max_tracking_error_m']:
            return
        if iteration == profile['max_iterations']:
            break
        next_target = width-profile['target_tracking_error_m']
        if not target < next_target <= .095:
            raise RuntimeError('Preload reduction would require an invalid opening target')
        move(next_target)
    raise RuntimeError('Grasp preload did not converge within the declared iteration budget')
