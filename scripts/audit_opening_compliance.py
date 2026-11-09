#!/usr/bin/env python3
"""Read-only reconstruction of experimental opening compliance from recordings.

Uses only reference history, native joint feedback and declared servo parameters;
never reads object geometry, labels, target placements or contact membership.
This verifies controller evidence, not task success or training acceptance.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


def reconstruct(before, after, physics, profile):
    old = before['opening_compliance']
    current = after['opening_compliance']
    dt, steps = float(after['integration_dt']), after['integration_steps']
    tau, floor = float(profile['time_constant_s']), float(profile['stiffness_floor'])
    if profile['mode'] != 'opening_negative_error_compliance_v1' or not 0 < floor <= 1 or not np.isfinite(tau) or tau <= 0:
        raise ValueError('Unsupported or invalid compliance profile')
    ids = np.array([6, 13])
    a, b = [np.asarray(after[key], float)[ids] for key in ('previous_command', 'command')]
    previous = np.asarray(before['command'], float)[ids]
    if not np.array_equal(previous, a):
        raise ValueError('Command history mismatch')
    secant = (b-a)/(steps*dt)
    va = np.asarray(before['last_velocity_reference'], float)[ids]
    va = np.clip(va, np.minimum(0, 3*secant), np.maximum(0, 3*secant))
    first, second, direction = [np.asarray(old[key], float).copy()
                                for key in ('filter_first', 'filter_second', 'opening_direction')]
    for value in (first, second, direction):
        if value.shape != (2,) or not np.isfinite(value).all() or np.any(value < -1e-12) or np.any(value > 1+1e-12):
            raise ValueError('Invalid compliance history')
    decay = np.exp(-dt/tau)
    c1, c2 = a+steps*dt*va/3, b-steps*dt*secant/3
    max_filter_error = 0.
    for i in range(1, steps+1):
        s = i/steps
        q = (1-s)**3*a+3*s*(1-s)**2*c1+3*s*s*(1-s)*c2+s**3*b
        q[a == b] = a[a == b]
        if i == steps:
            q = b.copy()
        delta = q-previous
        direction[delta > 1e-12] = 1
        direction[delta < -1e-12] = 0
        next_first = direction+(first-direction)*decay
        second = direction+(second-direction+(first-direction)*dt/tau)*decay
        first, previous = next_first, q
    for key, expected in (('filter_first', first), ('filter_second', second), ('opening_direction', direction)):
        actual = np.asarray(current[key], float)
        if actual.shape != (2,) or not np.isfinite(actual).all():
            raise ValueError('Invalid compliance state')
        max_filter_error = max(max_filter_error, float(np.max(np.abs(actual-expected))))
    if abs(float(current['sim_time'])-(after['end_sim_time']-dt)) > 1e-8:
        raise ValueError('Compliance clock mismatch')
    kp, kv = [np.asarray(after[key], float)[ids] for key in ('position_gain', 'velocity_gain')]
    bounds = np.asarray(after['actuator_force_range'], float)[ids]
    end_width, end_velocity, force = [], [], []
    for side in ('left', 'right'):
        arm = physics['arms'][side]
        end_width.append(sum(arm['joint_position'][6:]))
        end_velocity.append(sum(arm['joint_velocity'][6:]))
        force.append(arm['actuator_generalized_force'][6:])
    end_width, end_velocity = np.asarray(end_width), np.asarray(end_velocity)
    # Scalar slide coordinates integrate using the new velocity in implicitfast.
    # Thus post-step q,v recover the preceding last-step jaw opening exactly up
    # to floating-point rounding; no live simulator or state edits are involved.
    position_error = end_width-dt*end_velocity-b
    bias = (1-floor)*second*kp*np.maximum(position_error, 0)
    expected_force = np.clip(kp*(b-end_width)+kv*(np.asarray(after['last_velocity_reference'])[ids]-end_velocity)+bias,
                             bounds[:, 0], bounds[:, 1])
    errors = {'filter': max_filter_error,
              'position_error_m': float(np.max(np.abs(position_error-current['position_error_m']))),
              'bias_N': float(np.max(np.abs(bias-current['force_bias_N']))),
              'post_forward_force_N': float(np.max(np.abs(expected_force[:, None]-force)))}
    if not np.isfinite(list(errors.values())).all() or errors['filter'] > 1e-9 or errors['position_error_m'] > 1e-9 or max(errors['bias_N'], errors['post_forward_force_N']) > 1e-7:
        raise ValueError('Compliance reconstruction mismatch: '+str(errors))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    maximum, error, checked = {}, None, 0
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        profile = meta['physics_model']['controller']['experimental_opening_compliance']
        for i in range(len(f['action/position'])):
            try:
                before, after = [json.loads(f[p+'/extras_json'][i]) for p in ('observation', 'next_observation')]
                result = reconstruct(before['controller_interval'], after['controller_interval'], after['teacher']['physics'], profile)
                for key, value in result.items():
                    maximum[key] = max(maximum.get(key, 0.), value)
                checked += 1
            except Exception as exc:
                error = {'transition_index': i, 'error': str(exc)}
                break
    unchanged = digest == sha256(args.source)
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
              'script_sha256': sha256(__file__), 'source_unchanged': unchanged, 'checked_transitions': checked,
              'maximum_reconstruction_error': maximum, 'error': error, 'passed': error is None and unchanged,
              'training_ready': False, 'controller_profile': profile}
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('checked_transitions', 'maximum_reconstruction_error', 'error', 'passed')}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
