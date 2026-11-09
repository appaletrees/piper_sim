#!/usr/bin/env python3
"""Paired closed-recording audit for a grasp change at a named approach event.

Verifies the unchanged prefix, preserves failures, and reports both coarse
quality and native interval extrema. It never changes training acceptance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from audit_supported_release import scan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path);p.add_argument('candidate', type=Path)
    p.add_argument('--object', default='plate');p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    a, ah, baseline = scan(args.baseline);b, bh, candidate = scan(args.candidate)
    reports = [json.loads((directory/'report.json').read_text()) for directory in (args.baseline, args.candidate)]
    times = [next(e['sim_time'] for e in report['events']
                  if e['phase']=='approach' and e['object']==args.object) for report in reports]
    indices = [int(np.argmin(abs(arrays['observation/sim_time']-time)))
               for arrays, time in zip((a,b), times)]
    if abs(times[0]-times[1])>1e-8 or indices[0]!=indices[1]:
        raise ValueError('Intervention approach starts at different times/indices')
    boundary = indices[0]
    for arrays, time in zip((a,b), times):
        if abs(arrays['observation/sim_time'][boundary]-time)>1e-8:
            raise ValueError('No exact approach boundary')
    prefix = {key: np.array_equal(a[key][:boundary], b[key][:boundary]) for key in a}
    first_differences = {}
    for key in a:
        n = min(len(a[key]), len(b[key]));mask = a[key][:n]!=b[key][:n]
        if mask.ndim>1:mask = np.any(mask, axis=tuple(range(1,mask.ndim)))
        ids = np.flatnonzero(mask)
        first_differences[key] = int(ids[0]) if len(ids) else None
    native_diffs = [i for i,(x,y) in enumerate(zip(ah,bh)) if x!=y]
    same_thresholds = baseline['quality']['thresholds']==candidate['quality']['thresholds']
    policies = [s['acceptance_policy'] for s in (baseline,candidate)]
    same_policy = policies[0]['sha256']==policies[1]['sha256'] if all(policies) else None
    result = {'scope': __doc__, 'script_sha256': sha256(__file__),
        'baseline': baseline, 'candidate': candidate,
        'intervention': {'object': args.object, 'phase': 'approach', 'sim_time':times[0],
                         'transition_index':boundary}, 'prefix_public_streams_exact':prefix,
        'prefix_objects_physics_controller_tcp_exact': ah[:boundary]==bh[:boundary],
        'first_public_difference': first_differences,
        'first_objects_physics_controller_tcp_difference': native_diffs[0] if native_diffs else None,
        'same_online_thresholds':same_thresholds, 'same_evaluated_policy':same_policy,
        'target_grasp_reports':{label:[g for g in report['quality']['physical_grasps'] if g['name']==args.object]
                               for label,report in zip(('baseline','candidate'),reports)},
        'target_events':{label:[e for e in report['events'] if e['object']==args.object]
                         for label,report in zip(('baseline','candidate'),reports)},
        'training_ready':False,
        'limitations':'One known seed. Early independent audit failure may leave policy comparison unavailable; '
                      'a failed episode is not a successful robustness case. Native metrics are not calibrated hardware limits.'}
    atomic_json(args.output, result)
    print(json.dumps({key:result[key] for key in ('intervention','prefix_public_streams_exact',
          'prefix_objects_physics_controller_tcp_exact','same_online_thresholds','same_evaluated_policy',
          'first_objects_physics_controller_tcp_difference','target_grasp_reports')},indent=2))
    if not all(prefix.values()) or not result['prefix_objects_physics_controller_tcp_exact'] or not same_thresholds or same_policy is False:
        raise SystemExit('Paired comparison integrity failed')


if __name__ == '__main__':main()
