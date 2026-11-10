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

from piper_sim.collection_audit import audit_streams, audit_task, load_acceptance_policy
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.position_execution import audit_position_execution
from piper_sim.tcp_audit import audit_tcp_motion
from piper_sim.training_acceptance import require_self_contained


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
