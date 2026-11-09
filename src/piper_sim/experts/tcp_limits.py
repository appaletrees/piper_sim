"""Optional Cartesian planning budgets; independent of joint and collision gates."""
import numpy as np
from .mujoco_ik import rotation_error


def validate_limits(config):
    if config is None:
        return None
    fields = {'linear_speed_m_s', 'angular_speed_rad_s', 'planning_fraction'}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError('tcp_limits requires exactly '+', '.join(sorted(fields)))
    for key, value in config.items():
        if (isinstance(value, (bool, str)) or not np.isscalar(value)
                or not np.isfinite(value) or value <= 0):
            raise ValueError('Positive finite tcp_limits '+key+' required')
    if config['planning_fraction'] > 1:
        raise ValueError('TCP planning_fraction must not exceed one')
    return dict(config)


def pose_speed_peaks(start, path, forward, dt):
    previous_p, previous_r = forward(start)
    peaks = np.zeros(2)
    for q in path:
        p, r = forward(q)
        if (np.shape(p) != (3,) or np.shape(r) != (3, 3)
                or not np.isfinite(p).all() or not np.isfinite(r).all()):
            raise ValueError('Invalid TCP forward kinematics')
        peaks = np.maximum(peaks, [np.linalg.norm(p-previous_p)/dt,
                                  np.linalg.norm(rotation_error(r, previous_r))/dt])
        previous_p, previous_r = p, r
    return peaks


def speed_constraint(start, forward, dt, config):
    limits = validate_limits(config)
    if limits is None:
        return None
    budget = np.array([limits['linear_speed_m_s'], limits['angular_speed_rad_s']])*limits['planning_fraction']
    def ratio(path):
        return float(np.max(pose_speed_peaks(start, path, forward, dt)/budget))
    return ratio


def measured_violations(evidence, config):
    """Online rejection after the offending control interval has been recorded."""
    limits = validate_limits(config)
    if limits is None:
        return []
    if not isinstance(evidence, dict) or set(evidence.get('arms', {})) != {'left', 'right'}:
        return ['missing_native_tcp_evidence']
    errors = []
    for side, arm in evidence['arms'].items():
        for kind, key in (('linear', 'linear_speed_m_s'), ('angular', 'angular_speed_rad_s')):
            value = arm.get(kind+'_speed_max')
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or value > limits[key]*(1+1e-9):
                errors.append(f'{side}/{kind}: {value} > {limits[key]}')
    return errors
