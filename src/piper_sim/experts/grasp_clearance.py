"""Optional geometric clearance for unladen contact approaches."""
import numpy as np


def validate_profile(config):
    if config is None:
        return None
    fields = {'kinds', 'margin_m', 'extra_long_axis_fractions'}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError('grasp_approach_clearance requires exactly '+', '.join(sorted(fields)))
    kinds = config['kinds']
    if not isinstance(kinds, list) or kinds != ['chopstick']:
        raise ValueError('Grasp clearance currently supports kinds: [chopstick]')
    margin = config['margin_m']
    if (isinstance(margin, (bool, str)) or not np.isscalar(margin)
            or not np.isfinite(margin) or margin <= 0):
        raise ValueError('Grasp clearance margin must be positive and finite')
    fractions = config['extra_long_axis_fractions']
    if not isinstance(fractions, list) or any(
            isinstance(v, (bool, str)) or not np.isscalar(v)
            or not np.isfinite(v) or abs(v) > .4 for v in fractions):
        raise ValueError('Extra grasp fractions must be finite within +/-0.4 of object length')
    if len(set(fractions)) != len(fractions):
        raise ValueError('Extra grasp fractions must be distinct')
    return {'kinds': list(kinds), 'margin_m': float(margin),
            'extra_long_axis_fractions': list(fractions)}
