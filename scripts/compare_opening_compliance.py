#!/usr/bin/env python3
"""Compare complete opening-compliance experiment outcomes, without prefix claims."""
import argparse
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json, sha256


def load(directory, trace_analysis):
    source = directory/'episode.h5'
    digest = sha256(source)
    audit = json.loads((directory/'audit.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    trace = json.loads(trace_analysis.read_text())
    if audit['source_sha256'] != digest or trace['observed']['sha256'] != digest:
        raise ValueError('Stale audit/trace evidence')
    with h5py.File(source, 'r') as f:
        if not f.attrs.get('complete'):
            raise ValueError('Closed source required')
        metadata = json.loads(f.attrs['metadata'])
        frames = len(f['action/position'])
        duration = float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
    if sha256(source) != digest:
        raise ValueError('Source changed')
    return {'source': str(source), 'sha256': digest, 'transitions': frames, 'recorded_duration_s': duration,
            'task_summary': report['summary'], 'audit_passed': audit['passed'],
            'training_ready': audit['training_ready'], 'acceptance_policy': audit['acceptance_policy'],
            'online_quality': report['quality'], 'native_phase_extrema': trace['observed']['phase_native_extrema'],
            'selected_release_extrema': {k: {field: value[field] for field in ('acceleration_peak_m_s2', 'jerk_peak_m_s3')}
                                        for k, value in trace['groups'].items()},
            'finger_contact': metadata['teacher']['finger_contact'],
            'initial_scene': metadata['task_scene']['layout'],
            'controller': metadata['physics_model']['controller']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--baseline-analysis', type=Path, required=True)
    p.add_argument('--candidate-analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    a, b = load(args.baseline, args.baseline_analysis), load(args.candidate, args.candidate_analysis)
    checks = {'same_acceptance_policy': a['acceptance_policy']['sha256'] == b['acceptance_policy']['sha256'],
              'same_online_thresholds': a['online_quality']['thresholds'] == b['online_quality']['thresholds'],
              'same_finger_contact': a['finger_contact'] == b['finger_contact'],
              'same_initial_scene': a['initial_scene'] == b['initial_scene']}
    ratios = {name: {k: v/a['selected_release_extrema'][name][k] for k, v in peaks.items()}
              for name, peaks in b['selected_release_extrema'].items()}
    worsened = [name for name, peaks in ratios.items() if any(v > 1+1e-6 for v in peaks.values())]
    result = {'scope': __doc__, 'script_sha256': sha256(__file__), 'baseline': a, 'candidate': b,
              'comparison_checks': checks, 'candidate_to_baseline_native_ratios': ratios,
              'worsened_release_phases': worsened,
              'disposition': ('do_not_promote_native_release_worsens' if worsened else
                              'requires_further_validation_not_production_approval'),
              'limitations': 'One known seed. Compliance affects both arms throughout opening motions; '
              'no equal-prefix or isolated-release intervention claim. State-only; no RGB training data. '
              'Native extrema are simulation diagnostics, not calibrated hardware thresholds.'}
    atomic_json(args.output, result)
    print(json.dumps({'comparison_checks': checks, 'ratios': ratios, 'disposition': result['disposition']}, indent=2))
    if not all(checks.values()):
        raise SystemExit('Comparison integrity failed; report retained')


if __name__ == '__main__':
    main()
