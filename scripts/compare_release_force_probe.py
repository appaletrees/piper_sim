#!/usr/bin/env python3
"""Compare every phase of an experimental release rollout with its baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


def load(directory):
    source = directory/'episode.h5'
    digest = sha256(source)
    with h5py.File(source, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed source required, including rejected attempts')
        arrays = {key: f[key][:] for key in ('observation/state', 'next_observation/state', 'action/position',
                                           'observation/sim_time', 'next_observation/sim_time')}
        prefix = hashlib.sha256()
        boundary = None
        groups = {}
        for i, raw in enumerate(f['next_observation/extras_json']):
            e = json.loads(raw)
            teacher, c = e['teacher'], e['controller_interval']
            if teacher['active_assist'] or teacher['grasp_mode'] != 'physical':
                raise ValueError('Nonphysical/assisted source')
            if boundary is None and teacher['object'] == 'bowl' and teacher['phase'] == 'unload_gripper':
                boundary = i
            if boundary is None:
                prefix.update(json.dumps({**{k: e[k] for k in ('objects', 'controller_interval', 'tcp_motion_interval')},
                                          'physics': teacher['physics']}, sort_keys=True).encode())
            group = groups.setdefault(str(teacher['object'])+'/'+teacher['phase'], {'frames': 0, 'gripper_acceleration': 0., 'gripper_jerk': 0.})
            group['frames'] += 1
            for field, key in (('native_acceleration_abs_max', 'gripper_acceleration'), ('native_jerk_abs_max', 'gripper_jerk')):
                group[key] = max(group[key], c[field][13])
        metadata = json.loads(f.attrs['metadata'])
    audit = json.loads((directory/'audit.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    if sha256(source) != digest or audit['source_sha256'] != digest:
        raise ValueError('Source changed or stale independent audit')
    return arrays, {'source': str(source), 'sha256': digest, 'prefix_boundary': boundary,
        'prefix_native_sha256': prefix.hexdigest(), 'summary': report.get('summary'), 'error': report.get('error'),
        # Rejected recordings can fail the success precondition before policy
        # evaluation. Preserve that absence rather than inventing an audit.
        'acceptance_policy': audit.get('acceptance_policy'), 'audit_passed': audit['passed'],
        'violations': audit['violations'], 'training_ready': audit['training_ready'], 'frames': len(arrays['action/position']),
        'online_quality': report['quality'], 'teacher': report['teacher'], 'groups': groups,
        'finger_contact': metadata['teacher']['finger_contact']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    a, baseline = load(args.baseline)
    b, candidate = load(args.candidate)
    boundary = baseline['prefix_boundary']
    if boundary is None or boundary != candidate['prefix_boundary']:
        raise ValueError('No common first intervention boundary')
    prefix = {key: np.array_equal(a[key][:boundary], b[key][:boundary]) for key in a}
    same_policy = (baseline['acceptance_policy']['sha256'] == candidate['acceptance_policy']['sha256']
                   if baseline['acceptance_policy'] and candidate['acceptance_policy'] else None)
    same_thresholds = baseline['online_quality']['thresholds'] == candidate['online_quality']['thresholds']
    same_material = baseline['finger_contact'] == candidate['finger_contact']
    ratios = {}
    for name in ('bowl/unload_gripper', 'bowl/release', 'plate/release', 'side_dish/release'):
        if name not in candidate['groups']:
            continue
        ratios[name] = {key: candidate['groups'][name][key]/baseline['groups'][name][key]
                       for key in ('gripper_acceleration', 'gripper_jerk')}
    result = {'scope': __doc__, 'script_sha256': sha256(__file__), 'baseline': baseline, 'candidate': candidate,
              'prefix_exact_public_streams': prefix,
              'prefix_exact_native_physics': baseline['prefix_native_sha256'] == candidate['prefix_native_sha256'],
              'same_evaluated_policy': same_policy, 'same_online_thresholds': same_thresholds,
              'same_finger_material': same_material,
              'candidate_to_baseline_native_ratios': ratios,
              'limitations': 'One known scene and uncalibrated force-limit control. Physical consequences after first modification are included. '
                             'Native differences are diagnostics, not a hardware or production acceptance threshold.'}
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('prefix_exact_public_streams', 'prefix_exact_native_physics',
        'same_evaluated_policy', 'same_online_thresholds', 'same_finger_material', 'candidate_to_baseline_native_ratios')}, indent=2))
    if not all(prefix.values()) or not result['prefix_exact_native_physics'] or same_policy is False or not same_thresholds or not same_material:
        raise SystemExit('Comparison integrity failed; report preserved')


if __name__ == '__main__':
    main()
