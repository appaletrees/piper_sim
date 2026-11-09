#!/usr/bin/env python3
"""Check candidate internal targets over stored actions without running physics.

This is an offline feasibility check, not a closed-loop rollout or quality grant.
The controller receives only the current and previous accepted target. Existing
float32 source quantization and any additional interpolation overshoot are both
reported, without silently clipping either input or internal target.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.contracts import LOWER, UPPER


def check(source, interpolate, substeps=32, bounded=False):
    with h5py.File(source, 'r') as f:
        if not f.attrs.get('complete'):
            raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        targets = f['action/position'][:].astype(float)
        initial = np.asarray(meta['collection_initial_target'], dtype=float)
        before, after = f['observation/sim_time'][:], f['next_observation/sim_time'][:]
        dt = float(np.median(after-before))
        if (not len(targets) or targets.shape[1:] != (14,) or initial.shape != (14,)
                or not np.isfinite(targets).all() or not np.isfinite(initial).all()
                or not np.isfinite(dt) or dt <= 0
                or not np.allclose(after-before, dt, atol=1e-7, rtol=0)):
            raise ValueError('Invalid source actions/timing')
    previous, velocity = initial, np.zeros(14)
    maxima = np.zeros(14); violations = 0; examples = []
    limited_frames = 0; maximum_velocity_jump = np.zeros(14)
    for i, target in enumerate(targets):
        speed = (target-previous)/dt
        if bounded:
            clipped = np.clip(velocity, np.minimum(0,3*speed), np.maximum(0,3*speed))
            jump = np.abs(clipped-velocity)
            limited_frames += int(np.any(jump > 0))
            maximum_velocity_jump = np.maximum(maximum_velocity_jump, jump)
        q, v = interpolate(previous, target, velocity, speed, dt, substeps)
        excess = np.maximum(np.maximum(LOWER-q, q-UPPER), 0)
        maxima = np.maximum(maxima, excess.max(axis=0))
        if np.any(excess > 0):
            violations += 1
            if len(examples) < 12:
                step, channel = np.unravel_index(excess.argmax(), excess.shape)
                examples.append({'frame': i, 'substep': int(step+1), 'channel': int(channel),
                    'sim_time': float(before[i]+(step+1)*dt/substeps),
                    'target': float(q[step,channel]), 'excess_si': float(excess[step,channel]),
                    'current_accepted_target': float(target[channel])})
        previous, velocity = target, v[-1]
    input_excess = np.maximum(np.maximum(LOWER-targets, targets-UPPER), 0)
    return {'source': str(source.resolve()), 'source_sha256': sha256(source),
        'seed': meta['task_scene']['layout']['seed'], 'frames': len(targets), 'control_dt': dt,
        'substeps': substeps, 'stored_input_max_excess_per_channel': input_excess.max(axis=0).tolist(),
        'candidate_max_excess_per_channel': maxima.tolist(), 'frames_with_any_excess': violations,
        'first_examples': examples, 'all_internal_targets_within_bounds': violations == 0,
        'frames_with_derivative_limiter':limited_frames,
        'max_limiter_velocity_jump_per_channel':maximum_velocity_jump.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bounded', action='store_true', help='Explicit slope limiting; do not claim global C1 continuity')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh output file')
    script = Path(__file__).with_name('diagnose_substep_motion.py')
    spec = importlib.util.spec_from_file_location('controller_diagnostic', script)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result = {'scope': __doc__, 'training_ready': False, 'physics_run': False,
        'implementation_sha256': sha256(__file__), 'interpolator_sha256': sha256(script),
        'lower_bounds': LOWER.tolist(), 'upper_bounds': UPPER.tolist(),
        'candidate':'bounded_hermite_velocity' if args.bounded else 'hermite_velocity',
        'runs': [check(source, module.bounded_hermite_targets if args.bounded else module.hermite_targets,
                       bounded=args.bounded) for source in args.sources],
        'limitations': ['Stored actions are float32; quantify input excess separately.',
            'A new controller changes observations and future teacher commands in a closed-loop run.',
            'Passing targets does not validate state limits, forces, contacts, task success or a dataset.']}
    atomic_json(args.output, result)
    print(json.dumps([{'seed':r['seed'],'frames':r['frames'],'violating_frames':r['frames_with_any_excess']}
                      for r in result['runs']]))


if __name__ == '__main__':
    main()
