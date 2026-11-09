#!/usr/bin/env python3
"""Independently reconstruct privileged grasp angles from a closed recording.

Uses recorded native joints and object quaternions on an isolated kinematic
model. Checks numerical evidence, not grasp acceptability or hardware limits.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest = sha256(args.source)
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
              'script_sha256': sha256(__file__), 'training_ready': False,
              'tolerance_rad': 1e-6, 'segments': [], 'violations': [], 'checked_samples': 0}
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata']);quality = json.loads(f.attrs['quality'])
        cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
        b = MujocoBackend(cfg, cameras=False)
        key, anchor, segment = None, None, None
        try:
            for i in range(len(f['action/position'])):
                for prefix in ('observation', 'next_observation'):
                    extra = json.loads(f[prefix+'/extras_json'][i]);grasp = extra['teacher'].get('grasp')
                    if not grasp:
                        key = None;continue
                    new_key = (grasp['side'], grasp['object'])
                    reported = float(grasp['rotation_change_rad'])
                    if not np.isfinite(reported) or not 0 <= reported <= np.pi+1e-9:
                        raise ValueError('Invalid recorded rotation')
                    side, name = new_key
                    b.data.qpos[b.indices[side]['qpos']] = extra['teacher']['physics']['arms'][side]['joint_position']
                    mujoco.mj_kinematics(b.model, b.data)
                    object_r = np.empty(9)
                    mujoco.mju_quat2Mat(object_r, np.asarray(extra['objects'][name]['quaternion_wxyz']))
                    relative = b.data.body(side+'/link6').xmat.reshape(3, 3).T@object_r.reshape(3, 3)
                    if new_key != key:
                        key, anchor = new_key, relative.copy()
                        segment = {'side': side, 'object': name, 'anchor_index': i,
                                   'anchor_prefix': prefix, 'anchor_reported_rad': reported,
                                   'samples': 0, 'max_reconstructed_rad': 0., 'max_reported_rad': 0.,
                                   'max_abs_error_rad': 0.}
                        result['segments'].append(segment)
                        if reported > result['tolerance_rad']:
                            result['violations'].append('missing_zero_reference/'+str(i))
                    expected = float(np.arccos(np.clip((np.trace(anchor.T@relative)-1)/2, -1, 1)))
                    error = abs(expected-reported)
                    segment['samples'] += 1;result['checked_samples'] += 1
                    segment['max_abs_error_rad'] = max(segment['max_abs_error_rad'], error)
                    segment['max_reconstructed_rad'] = max(segment['max_reconstructed_rad'], expected)
                    segment['max_reported_rad'] = max(segment['max_reported_rad'], reported)
            declared = quality['physical_grasps']
            if len(declared)!=len(result['segments']):
                result['violations'].append('grasp_segment_count')
            for i, (segment, report) in enumerate(zip(result['segments'], declared)):
                if segment['object']!=report['name']:result['violations'].append('grasp_segment_order/'+str(i))
                segment['declared_max_rad'] = report['max_rotation_change_rad']
                segment['declared_max_error_rad'] = abs(segment['max_reported_rad']-segment['declared_max_rad'])
                if segment['max_abs_error_rad']>result['tolerance_rad'] or segment['declared_max_error_rad']>result['tolerance_rad']:
                    result['violations'].append('rotation_reconstruction/'+str(i))
        finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    result['consistent'] = not result['violations']
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result['consistent']:raise SystemExit('Recorded rotation evidence inconsistent')


if __name__ == '__main__':main()
