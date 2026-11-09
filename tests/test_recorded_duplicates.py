import hashlib
import importlib.util
import json

import h5py
import numpy as np
import pytest

from piper_sim.config import ROOT

spec = importlib.util.spec_from_file_location('duplicate_audit', ROOT/'scripts/audit_recorded_duplicates.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def source(path, *, sign=1, shift=0., pixels=80, units=None):
    with h5py.File(path, 'w') as f:
        f.attrs['complete'] = True
        f.attrs['metadata'] = json.dumps({'units': units or (['rad']*6+['m'])*2,
            'task_scene': {'layout': {'active_objects': ['cup'], 'seed': 0, 'randomized': True,
                'poses': {'cup': {'position': [.3, .1, .025], 'quaternion_wxyz': [sign, 0, 0, 0]}}}}})
        q = np.zeros((12, 14), dtype=np.float32)
        q[:, 0] = np.linspace(0, .1, 12)+shift
        for key in ('observation/state', 'action/position', 'next_observation/state'):
            f.create_dataset(key, data=q)
        f.create_dataset('observation/sim_time', data=np.arange(12)/4)
        f.create_dataset('observation/images/overhead', data=np.full((12, 8, 8, 3), pixels, np.uint8))
    return {'path': str(path), 'split': 'train'}


def test_detects_cross_split_duplicates_and_quaternion_sign_without_writing_source(tmp_path):
    a = source(tmp_path/'a.h5')
    b = source(tmp_path/'b.h5', sign=-1)
    b['split'] = 'validation'
    before = hashlib.sha256((tmp_path/'a.h5').read_bytes()).hexdigest()
    pair = cli.compare(cli.fingerprint(a), cli.fingerprint(b))
    assert pair['exact_layout'] and pair['exact_trajectory'] and pair['cross_split']
    assert pair['near_visual_candidate'] and pair['normalized_joint_rms'] == 0
    assert hashlib.sha256((tmp_path/'a.h5').read_bytes()).hexdigest() == before


def test_different_motion_and_visuals_are_not_near_candidates_and_units_must_match(tmp_path):
    a = cli.fingerprint(source(tmp_path/'a.h5'))
    b = cli.fingerprint(source(tmp_path/'b.h5', shift=1., pixels=200))
    pair = cli.compare(a, b)
    assert not pair['exact_trajectory'] and not pair['near_trajectory_candidate']
    assert not pair['near_visual_candidate']
    bad = source(tmp_path/'bad.h5', units=(['deg']*6+['m'])*2)
    with pytest.raises(ValueError, match='radians'):
        cli.fingerprint(bad)
