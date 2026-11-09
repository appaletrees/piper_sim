"""Backend-owned, reset-time finger material; independent of expert policy.

These are explicitly uncalibrated MuJoCo contact parameters. A different material
is sampled for each arm at reset and remains fixed throughout that episode.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ROOT
from .randomization import stream_seed


def require_contact_engine(cfg, engine):
    if cfg.get('mujoco', {}).get('finger_contact') is not None and engine != 'mujoco':
        raise ValueError('mujoco.finger_contact requires MuJoCo; no equivalent Isaac contact profile is implemented')


def load_finger_profile(value):
    if isinstance(value, (str, Path)):
        import yaml
        path = (ROOT/Path(value)).resolve()
        raw = path.read_bytes()
        profile = yaml.safe_load(raw)
        source = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    else:
        profile = copy.deepcopy(value)
        source = {'path': None, 'sha256': hashlib.sha256(
            json.dumps(profile, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    fields = {'schema_version', 'name', 'friction', 'solref', 'solimp', 'friction_scale_range'}
    if not isinstance(profile, dict) or set(profile) != fields or profile['schema_version'] != 1:
        raise ValueError('Finger material requires exactly the supported schema v1 fields')
    if not isinstance(profile['name'], str) or not profile['name'].strip():
        raise ValueError('Finger material name must be nonempty')
    for key, count in (('friction', 3), ('solref', 2), ('solimp', 5), ('friction_scale_range', 2)):
        value = np.asarray(profile[key], float)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError('Invalid finite finger material vector: '+key)
        profile[key] = value.tolist()
    f, ref, imp, scale = (profile[k] for k in ('friction', 'solref', 'solimp', 'friction_scale_range'))
    if min(f) < 0 or min(ref) <= 0 or not 0 < scale[0] <= scale[1]:
        raise ValueError('Invalid friction, positive-format solref or friction scale bounds')
    if not .0001 <= imp[0] <= imp[1] <= .9999 or imp[2] <= 0 or not 0 < imp[3] < 1 or imp[4] < 1:
        raise ValueError('Invalid contact impedance bounds, width, midpoint or power')
    return profile, source


class MujocoFingerContact:
    def __init__(self, model, value):
        self.model = model
        self.profile, self.source = load_finger_profile(value)
        if self.profile['solref'][0] < 2*model.opt.timestep:
            raise ValueError('Contact profile requires a time constant of at least two MuJoCo substeps')
        self.indices = {}
        names = [model.body(int(b)).name for b in model.geom_bodyid]
        for side in ('left', 'right'):
            bodies = (side+'/link7', side+'/link8')
            ids = np.array([i for i, name in enumerate(names) if name in bodies], dtype=int)
            if not all(any(names[i] == body and (model.geom_contype[i] or model.geom_conaffinity[i])
                           for i in ids) for body in bodies):
                raise ValueError('Missing colliding finger geometry: '+side)
            self.indices[side] = ids
        self.all_indices = np.concatenate(list(self.indices.values()))
        # The explicit higher-priority finger material must actually determine
        # contacts; do not silently accept an overriding pair or other geom.
        self.other_indices = np.setdiff1d(np.arange(model.ngeom), self.all_indices)
        if model.npair or np.any(model.geom_priority[self.other_indices] >= 1):
            raise ValueError('Fixed finger profile does not support explicit pairs or overriding geom priorities')
        self.check_contact_override()
        self.expected = {}
        self.apply(0, False)

    def apply(self, seed, randomize):
        import mujoco
        self.check_contact_override()
        m, p = self.model, self.profile
        self.seed, self.randomized = int(seed), bool(randomize)
        self.scales = {}
        for side, ids in self.indices.items():
            scale = (float(np.random.default_rng(stream_seed(seed, 'mujoco/finger_material/'+side)).uniform(
                *p['friction_scale_range'])) if randomize else 1.)
            self.scales[side] = scale
            m.geom_friction[ids] = np.array(p['friction'])*scale
            m.geom_condim[ids] = 6
            m.geom_priority[ids] = 1
            m.geom_solref[ids] = p['solref']
            m.geom_solimp[ids] = p['solimp']
        m.opt.impratio = 10
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        self.expected = {key: getattr(m, key)[self.all_indices].copy()
            for key in ('geom_friction', 'geom_condim', 'geom_priority', 'geom_solref', 'geom_solimp')}

    def assert_unchanged(self):
        self.check_contact_override()
        if np.any(self.model.geom_priority[self.other_indices] >= 1):
            raise RuntimeError('Other geom priorities override the fixed finger material')
        if self.profile['solref'][0] < 2*self.model.opt.timestep:
            raise RuntimeError('Time step violates the fixed contact profile time constant')
        for key, expected in self.expected.items():
            if not np.array_equal(getattr(self.model, key)[self.all_indices], expected):
                raise RuntimeError('Finger contact material changed during episode: '+key)

    def check_contact_override(self):
        import mujoco
        if self.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
            raise ValueError('Global MuJoCo contact override would replace the fixed finger material')

    def metadata(self):
        self.assert_unchanged()
        return {'schema_version': 'piper.mujoco_finger_contact.v1',
            'parameter_origin': 'uncalibrated_configuration', 'constant_within_episode': True,
            'profile': copy.deepcopy(self.profile), 'source': copy.deepcopy(self.source),
            'seed': self.seed, 'randomized': self.randomized,
            'arms': {side: {'friction_scale': self.scales[side],
                'friction': (np.array(self.profile['friction'])*self.scales[side]).tolist(),
                'geom_names': [self.model.geom(int(i)).name for i in ids]} for side, ids in self.indices.items()},
            'condim': 6, 'priority': 1,
            'scope': 'Per-arm finger material. Object/table coefficients remain governed by their own profiles '
                     'and MuJoCo mixing rules. No per-target material switching or hardware calibration claim.'}
