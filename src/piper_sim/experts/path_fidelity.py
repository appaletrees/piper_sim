"""Read-only FK check of spline approximation, separate from collision checking."""
import numpy as np

from .mujoco_ik import rotation_error


def loaded_path_limits(config):
    """An absent profile keeps the original loaded-path rejection behavior."""
    if config is None:
        return None
    keys = {'max_position_deviation_m', 'max_rotation_deviation_rad', 'max_refinements'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError('bounded_loaded_path requires exactly '+', '.join(sorted(keys)))
    for key in ('max_position_deviation_m', 'max_rotation_deviation_rad'):
        value = config[key]
        if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError('Positive finite bounded_loaded_path '+key+' required')
    value = config['max_refinements']
    if type(value) is not int or not 0 <= value <= 3:
        raise ValueError('bounded_loaded_path max_refinements must be an integer from 0 to 3')
    return dict(config)


def path_pose_deviation(path, sampling, forward):
    """Maximum sample error from the parameter-matched solved-IK pose polyline.

    This bounds additional spline approximation, NOT IK error, continuous-time
    clearance, tracking error, or hardware calibration. Interior poses interpolate
    position linearly and orientation along the shortest SO(3) arc. Matching the
    same path parameter avoids accepting a nearby but wrong traversal segment.
    ``forward`` must operate on scratch data and return (position, rotation).
    """
    points, knots, samples = (np.asarray(sampling[k], float) for k in ('points', 'knots', 'samples'))
    path = np.asarray(path, float)
    if (path.ndim != 2 or points.ndim != 2 or path.shape[1] != points.shape[1]
            or len(path) == 0 or len(points) == 0 or knots.shape != (len(points),)
            or samples.shape != (len(path),) or any(not np.isfinite(a).all() for a in (path, points, knots, samples))
            or np.any(np.diff(knots) <= 0) or np.any(np.diff(samples) < 0)
            or samples.min() < knots[0]-1e-12 or samples.max() > knots[-1]+1e-12):
        raise ValueError('Invalid path sampling for pose fidelity check')
    poses = [tuple(np.asarray(x, float).copy() for x in forward(q)) for q in points]
    if any(np.asarray(p).shape != (3,) or np.asarray(r).shape != (3, 3)
           or not np.isfinite(p).all() or not np.isfinite(r).all() for p, r in poses):
        raise ValueError('Invalid forward kinematics')
    errors = []
    for q, parameter in zip(path, samples):
        i = int(np.clip(np.searchsorted(knots, parameter, side='right')-1, 0, max(0, len(knots)-2)))
        p, r = poses[i]
        if len(points) > 1:
            fraction = (parameter-knots[i])/(knots[i+1]-knots[i])
            p = p+fraction*(poses[i+1][0]-p)
            axis = rotation_error(poses[i+1][1], r)
            angle = np.linalg.norm(axis)
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            if angle > 1e-12:
                r = (np.eye(3)+np.sin(fraction*angle)/angle*skew
                     +(1-np.cos(fraction*angle))/angle**2*(skew@skew))@r
        actual_p, actual_r = forward(q)
        if not np.isfinite(actual_p).all() or not np.isfinite(actual_r).all():
            raise ValueError('Invalid forward kinematics')
        errors.append((np.linalg.norm(actual_p-p), np.linalg.norm(rotation_error(actual_r, r))))
    errors = np.asarray(errors)
    return {'position_deviation_m': float(errors[:, 0].max()),
            'rotation_deviation_rad': float(errors[:, 1].max()),
            'position_peak_sample': int(errors[:, 0].argmax()),
            'rotation_peak_sample': int(errors[:, 1].argmax()),
            'reference': 'parameter-matched solved-IK pose polyline',
            'scope': 'sampled additional approximation; excludes IK and tracking error'}
