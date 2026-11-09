"""Engine-independent episode sampling. Geometry and camera calibration stay fixed."""
import copy
from functools import lru_cache
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
    allowed = {'schema_version', 'name', 'layout', 'appearance', 'physics', 'sensor', 'content'}
    if set(value)-allowed:
        raise ValueError(f'Unknown randomization fields: {set(value)-allowed}')
    ranges = {'appearance': {'light_scale', 'light_position_delta_m', 'light_color_scale', 'color_scale', 'roughness_delta'},
              'physics': {'mass_scale', 'friction_scale'},
              'sensor': {'gain', 'white_balance', 'noise_std_uint8', 'blur_sigma_px', 'gamma'}}
    for section, keys in ranges.items():
        for name, bounds in value.get(section, {}).items():
            if section=='appearance' and name=='tabletop_materials':
                if not isinstance(bounds,list) or not bounds or any(not isinstance(n,str) for n in bounds):
                    raise ValueError('tabletop_materials must be a nonempty list of material names')
                continue
            if name not in keys:
                raise ValueError(f'Unsupported {section} randomization: {name}')
            a = np.asarray(bounds, float)
            if name == 'light_position_delta_m':
                valid = a.shape == () and a >= 0
            else:
                valid = a.shape == (2,) and a[0] <= a[1]
                if name not in ('roughness_delta', 'noise_std_uint8', 'blur_sigma_px'):
                    valid = valid and a[0] > 0
                if name in ('noise_std_uint8','blur_sigma_px'):
                    valid = valid and a[0] >= 0
            if not valid or not np.isfinite(a).all():
                raise ValueError(f'Invalid randomization range {section}.{name}')
    layout = value.get('layout', {})
    for key, val in layout.items():
        if key not in ('setting_xy_m', 'setting_yaw_deg', 'container_xy_m', 'container_yaw_deg', 'object_xy_m', 'object_yaw_deg'):
            raise ValueError(f'Unsupported layout field {key}')
        a = np.asarray(val, float)
        if a.shape != ((2,) if key.endswith('xy_m') else ()) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError(f'Invalid layout range {key}')
    content = value.get('content', {})
    if set(content)-{'cup_count', 'cup_palette'}:
        raise ValueError('Unsupported content randomization')
    count = content.get('cup_count', [1, 1])
    if len(count) != 2 or any(type(n) is not int for n in count) or not 1 <= count[0] <= count[1] <= 3:
        raise ValueError('cup_count must be integer bounds between 1 and 3')
    palette = np.asarray(content.get('cup_palette', [[1., 1., 1.]]), float)
    if palette.ndim != 2 or palette.shape[1] != 3 or not len(palette) or not np.isfinite(palette).all() or np.any((palette < 0)|(palette > 1)):
        raise ValueError('cup_palette must contain RGB triples in [0,1]')
    return value


def cup_content(profile, seed):
    """Scene inventory is independent of lighting, dynamics and sensor draws."""
    content = profile.get('content', {})
    lo, hi = content.get('cup_count', [2, 3])
    rng = np.random.default_rng(stream_seed(seed, 'content/cup_count'))
    count = int(rng.integers(lo, hi+1))
    palette = content.get('cup_palette', [[.94, .94, .90]])
    names = ('cup_00', 'cup_01', 'cup_02')[:hi]
    colors = {}
    for name in names:
        rng = np.random.default_rng(stream_seed(seed, 'content/cup_palette/'+name))
        colors[name] = list(palette[int(rng.integers(len(palette)))])
    return {'cup_count': count, 'cup_colors': colors}


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
    tables=[n for n in appearance.get('tabletop_materials',[]) if n in material_names]
    tabletop=(np.random.default_rng(stream_seed(seed,'appearance/tabletop')).choice(tables).item() if tables else None)
    warmth=float(draw('appearance','light_color_scale',[1,1]))
    return {'profile': profile['name'], 'seed': int(seed),
            'tabletop_material':tabletop,
            'config_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            'light_scale': float(draw('appearance', 'light_scale', [1, 1])),
            'light_color_scale': [warmth,1.,2.-warmth],
            'light_position_delta_m': rng.uniform(-delta, delta, 3).tolist(),
            'materials': {n: {'color_scale': [float(draw('appearance', 'color_scale', [1, 1], n))]*3,
                               'roughness_delta': float(draw('appearance', 'roughness_delta', [0, 0], n))}
                          for n in sorted(material_names)},
            'objects': {n: {'mass_scale': float(draw('physics', 'mass_scale', [1, 1], n)),
                             'friction_scale': float(draw('physics', 'friction_scale', [1, 1], n))}
                        for n in sorted(object_names)},
            'cameras': {n: {'gain': float(draw('sensor', 'gain', [1, 1], n)),
                             'white_balance': draw('sensor', 'white_balance', [1, 1], n, 3).tolist(),
                             'gamma': float(draw('sensor','gamma',[1,1],n)),
                             'blur_sigma_px': float(draw('sensor','blur_sigma_px',[0,0],n)),
                             'noise_std_uint8': float(draw('sensor', 'noise_std_uint8', [0, 0], n))}
                        for n in ('overhead', 'left_wrist', 'right_wrist')},
            'fixed': ['CAD geometry', 'camera intrinsics/extrinsics', 'gravity', 'control rate'],
            'sampling': 'episode reset; stable per-property streams; no timing perturbation'}


@lru_cache(maxsize=64)
def _gamma_curve(gamma):
    # RGB inputs have only 256 possible values. Preserve the exact float32
    # arithmetic of the full image expression; rounding still occurs after
    # white balance, gain and deterministic noise, never inside this lookup.
    values = 255.*(np.arange(256, dtype=np.float32)/255.)**gamma
    values.flags.writeable = False
    return values


def sensor_images(images, parameters, tick):
    """Repeat reads at the same simulation tick return identical sensor noise."""
    if not parameters:
        return images
    out = {}
    for name, pixels in images.items():
        p = parameters['cameras'][name]
        if p.get('blur_sigma_px',0):
            from PIL import Image,ImageFilter
            pixels=np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(p['blur_sigma_px'])))
        gamma = p.get('gamma', 1.)
        rgb = (_gamma_curve(gamma)[pixels] if pixels.dtype == np.uint8 else
               255.*(pixels.astype(np.float32)/255.)**gamma)
        rgb *= p['gain']*np.asarray(p['white_balance'], np.float32)
        if p['noise_std_uint8']:
            rng = np.random.default_rng(stream_seed(parameters['seed'], f'sensor/{name}/{tick}'))
            rgb += rng.normal(0, p['noise_std_uint8'], rgb.shape).astype(np.float32)
        out[name] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out
