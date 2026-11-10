#!/usr/bin/env python3
"""Read-only component checks, including recordings that failed task success.

Normal collection acceptance stops immediately on an unsuccessful episode.
This diagnostic keeps that verdict and separately evaluates available evidence;
it never publishes a training audit or changes HDF5 acceptance attributes.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_audit import audit_streams, audit_task, load_acceptance_policy
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.position_execution import audit_position_execution
from piper_sim.tcp_audit import audit_tcp_motion
from piper_sim.training_acceptance import require_self_contained
from piper_sim.experts.quality import MotionQuality


def recorded_motion(f, metadata, settings):
    """Recompute policy metrics from closed-file state/action, even after failure."""
    fps = float(metadata['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive finite fps required')
    actions, states = f['action/position'][:], f['next_observation/state'][:]
    times = f['next_observation/sim_time'][:]
    n = len(actions)
    initial = np.asarray(metadata['collection_initial_target'])
    if (n == 0 or actions.shape != (n, 14) or states.shape != (n, 14)
            or times.shape != (n,) or initial.shape != (14,)
            or not all(np.isfinite(v).all() for v in (actions, states, times, initial))
            or len(f['next_observation/extras_json']) != n):
        raise ValueError('Complete finite motion streams required')
    quality = MotionQuality(settings, 1/fps)
    quality.add(float(f['observation/sim_time'][0]), initial, f['observation/state'][0],
                context={'phase': 'prepared'})
    for i, raw in enumerate(f['next_observation/extras_json']):
        teacher = json.loads(raw)['teacher']
        if type(teacher['holding']) is not bool:
            raise ValueError('Boolean holding evidence required')
        quality.add(times[i], actions[i], states[i], holding=teacher['holding'],
                    context={'phase': teacher['phase'], 'object': teacher['object']})
    return {**quality.report(), 'source_precision': 'Stored state/action arrays; no smoothing or native-substep claim.'}


def diagnose(source, policy):
    settings, identity = load_acceptance_policy(policy)
    digest = sha256(source)
    result = {'schema_version': 'piper.collection_diagnostics.v1', 'scope': __doc__,
        'source': str(Path(source).resolve()), 'source_sha256': digest,
        'acceptance_policy': identity, 'training_ready': False, 'checks': {}}
    with h5py.File(source, 'r') as f:
        require_self_contained(f)
        if not f.attrs.get('complete'):
            raise ValueError('Closed recording required')
        metadata = json.loads(f.attrs['metadata'])
        result['recorded_success'] = bool(f.attrs.get('success', False))
        checks = [('streams_without_rgb_requirement', lambda: audit_streams(f, metadata, settings, False)),
                  ('recorded_motion', lambda: recorded_motion(f, metadata, settings)),
                  ('task_and_motion', lambda: audit_task(f, metadata, settings)),
                  ('position_execution', lambda: audit_position_execution(f, metadata))]
        if 'tcp_motion' in settings:
            checks.append(('tcp_motion', lambda: audit_tcp_motion(f, metadata, settings['tcp_motion'])))
        for name, check in checks:
            try:
                result['checks'][name] = check()
            except Exception as exc:
                result['checks'][name] = {'passed': False, 'violations': [type(exc).__name__+': '+str(exc)]}
    if sha256(source) != digest:
        raise ValueError('Source changed during component diagnosis')
    result.update(source_unchanged=True, script_sha256=sha256(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = diagnose(args.source, args.policy)
    atomic_json(args.output, result)
    print(json.dumps({name: {'passed': check['passed'], 'violations': check['violations']}
                      for name, check in result['checks'].items()}, indent=2))


if __name__ == '__main__':
    main()
