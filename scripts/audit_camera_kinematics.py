#!/usr/bin/env python3
"""Read-only recorded camera pose versus measured-joint URDF FK diagnostic.

This tests simulator attachment freshness, not real camera calibration or
whether rendered pixels have the correct latency. No new acceptance is granted.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.config import ROOT
from piper_sim.kinematics import forward
from piper_sim.transforms import pose_matrix
from piper_sim.contracts import JOINT_NAMES


def inspect(path):
    with h5py.File(path, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        cfg = meta['config']
        rig = meta['camera_rig']['cameras']
        urdf = ROOT/cfg['robot']['urdf']
        count = int(f.attrs['num_transitions'])
        maxima = {name: {'matrix_max_abs_error': 0., 'position_error_m': 0.,
                         'position_span_m': 0.} for name in rig}
        initial = {}
        for index in range(count+1):
            prefix, i = ('observation', index) if index < count else ('next_observation', count-1)
            q = f[prefix+'/state'][i]
            extras = json.loads(f[prefix+'/extras_json'][i])
            actual = extras['cameras']
            hands = {}
            for side, offset in (('left', 0), ('right', 7)):
                base = pose_matrix(cfg['robot'][side]['position'], cfg['robot'][side]['rotation'])
                hands[side] = forward(urdf, dict(zip(JOINT_NAMES, q[offset:offset+6])), base)['link6']
            for name, camera in rig.items():
                expected = pose_matrix(camera['position'], camera['rotation_wxyz_ros'])
                if name != 'overhead':
                    expected = hands[name.split('_')[0]]@expected
                observed = pose_matrix(actual[name]['position_world'], actual[name]['quaternion_wxyz_ros'])
                entry = maxima[name]
                entry['matrix_max_abs_error'] = max(entry['matrix_max_abs_error'],
                                                    float(np.max(np.abs(expected-observed))))
                entry['position_error_m'] = max(entry['position_error_m'],
                                                float(np.linalg.norm(expected[:3, 3]-observed[:3, 3])))
                initial.setdefault(name, observed[:3, 3].copy())
                entry['position_span_m'] = max(entry['position_span_m'],
                    float(np.linalg.norm(initial[name]-observed[:3, 3])))
        return {'source': str(path.resolve()), 'backend': meta['backend'], 'samples': count+1,
                'cameras': maxima, 'numerical_fk_check_passed':
                all(v['matrix_max_abs_error'] < 2e-5 for v in maxima.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [inspect(path) for path in args.source]
    report = {'status': 'diagnostic_only', 'matrix_tolerance': 2e-5,
        'scope': 'All unique recorded states including terminal. URDF/Fabric attachment consistency only.',
        'physical_alignment_verified': False, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not all(r['numerical_fk_check_passed'] for r in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
