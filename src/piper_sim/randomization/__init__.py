"""Engine-independent episode sampling. Geometry and camera calibration stay fixed."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT


def load_profile(value):
    if isinstance(value, (str, Path)):
        value = yaml.safe_load((ROOT/Path(value)).read_text())
    value = copy.deepcopy(value)
    if value.get('schema_version') != 1:
        raise ValueError('Unsupported randomization profile')
    allowed = {'schema_version', 'name', 'layout', 'appearance', 'physics', 'sensor'}
    if set(value)-allowed:
        raise ValueError(f'Unknown randomization fields: {set(value)-allowed}')
    ranges = {'appearance': {'light_scale', 'light_position_delta_m', 'color_scale', 'roughness_delta'},
              'physics': {'mass_scale', 'friction_scale'},
              'sensor': {'gain', 'white_balance', 'noise_std_uint8'}}
    for section, keys in ranges.items():
        for name, bounds in value.get(section, {}).items():
            if name not in keys:
                raise ValueError(f'Unsupported {section} randomization: {name}')
            a = np.asarray(bounds, float)
            if name == 'light_position_delta_m':
                valid = a.shape == () and a >= 0
            else:
                valid = a.shape == (2,) and a[0] <= a[1]
                if name not in ('roughness_delta', 'noise_std_uint8'):
                    valid = valid and a[0] > 0
                if name == 'noise_std_uint8':
                    valid = valid and a[0] >= 0
            if not valid or not np.isfinite(a).all():
                raise ValueError(f'Invalid randomization range {section}.{name}')
    layout = value.get('layout', {})
    for key, val in layout.items():
        if key not in ('setting_xy_m', 'setting_yaw_deg', 'container_xy_m', 'container_yaw_deg'):
            raise ValueError(f'Unsupported layout field {key}')
        a = np.asarray(val, float)
        if a.shape != ((2,) if key.endswith('xy_m') else ()) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError(f'Invalid layout range {key}')
    return value


def stream_seed(seed, key):
    digest = hashlib.sha256(f'piper.dr.v1/{int(seed)}/{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def sample(profile, seed, object_names, material_names):
    """Each named property has its own stream, independent of iteration order."""
    def draw(section, key, default, entity='', size=None):
        bounds = profile.get(section, {}).get(key, default)
        rng = np.random.default_rng(stream_seed(seed, f'{section}/{key}/{entity}'))
        return rng.uniform(*bounds, size=size)
    appearance = profile.get('appearance', {})
    delta = appearance.get('light_position_delta_m', 0.)
    rng = np.random.default_rng(stream_seed(seed, 'appearance/light_position'))
    return {'profile': profile['name'], 'seed': int(seed),
            'config_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            'light_scale': float(draw('appearance', 'light_scale', [1, 1])),
            'light_position_delta_m': rng.uniform(-delta, delta, 3).tolist(),
            'materials': {n: {'color_scale': draw('appearance', 'color_scale', [1, 1], n, 3).tolist(),
                               'roughness_delta': float(draw('appearance', 'roughness_delta', [0, 0], n))}
                          for n in sorted(material_names)},
            'objects': {n: {'mass_scale': float(draw('physics', 'mass_scale', [1, 1], n)),
                             'friction_scale': float(draw('physics', 'friction_scale', [1, 1], n))}
                        for n in sorted(object_names)},
            'cameras': {n: {'gain': float(draw('sensor', 'gain', [1, 1], n)),
                             'white_balance': draw('sensor', 'white_balance', [1, 1], n, 3).tolist(),
                             'noise_std_uint8': float(draw('sensor', 'noise_std_uint8', [0, 0], n))}
                        for n in ('overhead', 'left_wrist', 'right_wrist')},
            'fixed': ['CAD geometry', 'camera intrinsics/extrinsics', 'gravity', 'control rate'],
            'sampling': 'episode reset; stable per-property streams; no timing perturbation'}


def sensor_images(images, parameters, tick):
    """Repeat reads at the same simulation tick return identical sensor noise."""
    if not parameters:
        return images
    out = {}
    for name, pixels in images.items():
        p = parameters['cameras'][name]
        rgb = pixels.astype(np.float32)*p['gain']*np.asarray(p['white_balance'], np.float32)
        if p['noise_std_uint8']:
            rng = np.random.default_rng(stream_seed(parameters['seed'], f'sensor/{name}/{tick}'))
            rgb += rng.normal(0, p['noise_std_uint8'], rgb.shape).astype(np.float32)
        out[name] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out
