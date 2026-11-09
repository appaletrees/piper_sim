#!/usr/bin/env python3
"""Analyze a native force trace and prove its full rollout matches a source.

Read-only diagnostic; native acceleration/jerk are velocity differences across
actual integrator boundaries. This does not set a training acceptance threshold.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


def evidence(source):
    digest = sha256(source)
    streams = {}
    contexts = {}
    privileged = hashlib.sha256()
    with h5py.File(source, 'r') as f:
        if not f.attrs.get('complete'):
            raise ValueError('Closed recording required')
        for key in ('observation/state', 'next_observation/state', 'action/position',
                    'observation/sim_time', 'next_observation/sim_time'):
            data = f[key][:]
            streams[key] = {'shape': list(data.shape), 'dtype': str(data.dtype),
                            'sha256': hashlib.sha256(data.tobytes()).hexdigest()}
        for raw in f['next_observation/extras_json']:
            e = json.loads(raw)
            teacher, controller = e['teacher'], e['controller_interval']
            if teacher['active_assist'] or teacher['grasp_mode'] != 'physical':
                raise ValueError('Physical unassisted rollout required')
            privileged.update(json.dumps({**{k: e[k] for k in ('objects', 'controller_interval', 'tcp_motion_interval')},
                                          'physics': teacher['physics']}, sort_keys=True).encode())
            key = str(teacher['object'])+'/'+teacher['phase']
            value = contexts.setdefault(key, {'acceleration': 0., 'jerk': 0.})
            value['acceleration'] = max(value['acceleration'], controller['native_acceleration_abs_max'][13])
            value['jerk'] = max(value['jerk'], controller['native_jerk_abs_max'][13])
    if sha256(source) != digest:
        raise ValueError('Source changed during read-only analysis')
    return {'source': str(source), 'sha256': digest, 'streams': streams,
            'privileged_sha256': privileged.hexdigest(), 'phase_native_extrema': contexts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        help='Optional distinct recording for an exact noninterference comparison')
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.trace/'trace.json').read_text())
    if sha256(args.trace/'trace.npz') != meta['trace_sha256']:
        raise ValueError('Trace hash mismatch')
    identity = json.loads(args.identity.read_text())
    changed = [name for name, digest in identity['files'].items() if sha256(name) != digest]
    observed = evidence(args.recording)
    comparison = args.baseline is not None and args.baseline.resolve() != args.recording.resolve()
    baseline = evidence(args.baseline) if comparison else None
    same_streams = baseline['streams'] == observed['streams'] if comparison else None
    same_physics = baseline['privileged_sha256'] == observed['privileged_sha256'] if comparison else None
    with np.load(args.trace/'trace.npz', allow_pickle=False) as z:
        dt = meta['integration_dt']
        v0, v1 = z['qvel_before'][:, 6:].sum(axis=1), z['qvel_after'][:, 6:].sum(axis=1)
        acceleration = (v1-v0)/dt
        contiguous = np.isclose(z['time_before'][1:], z['time_after'][:-1], atol=1e-8, rtol=0)
        jerk = np.r_[np.nan, np.where(contiguous, np.diff(acceleration)/dt, np.nan)]
        # The numeric arrays are in dof order (6 arm + 2 jaw), but actuator
        # arrays contain only 6 arm + 1 full-opening tendon servo.
        force_error = np.max(np.abs(z['constraint_force_by_type'].sum(axis=1)-z['qfrc_constraint']))
        expected_force = z['gain'][:, :, 0]*z['control'] + z['bias'][:, :, 0] \
            + z['bias'][:, :, 1]*z['actuator_length'] + z['bias'][:, :, 2]*z['actuator_velocity']
        expected_force = np.clip(expected_force, z['force_range'][:, :, 0], z['force_range'][:, :, 1])
        actuator_error = float(np.max(np.abs(expected_force-z['actuator_force'])))
        groups = {}

        def snapshot(i):
            start, count = int(z['contact_offset'][i]), int(z['contact_count'][i])
            contacts = []
            for j in range(start, start+count):
                contacts.append({'geoms': [meta['geom_names'][int(z[key][j])]
                                           for key in ('contact_geom1', 'contact_geom2')],
                                 'distance_m': float(z['contact_distance'][j]),
                                 'force_local_N_Nm': z['contact_force_local'][j].tolist()})
            return {'row': int(i), 'solver_time': float(z['time_before'][i]),
                    'integrated_time': float(z['time_after'][i]),
                    'width_m': float(z['qpos_before'][i, 6:].sum()),
                    'command_m': float(z['control'][i, 6]),
                    'velocity_m_s': [float(v0[i]), float(v1[i])],
                    'acceleration_m_s2': float(acceleration[i]),
                    'jerk_m_s3': float(jerk[i]) if np.isfinite(jerk[i]) else None,
                    'position_gain_N_m': float(z['gain'][i, 6, 0]),
                    'affine_force_bias_N': float(z['bias'][i, 6, 0]),
                    'force_range_N': z['force_range'][i, 6].tolist(),
                    'actuator_force_N': float(z['actuator_force'][i, 6]),
                    'jaw_constraint_force_N': z['qfrc_constraint'][i, 6:].tolist(),
                    'jaw_constraint_by_type_N': z['constraint_force_by_type'][i, :, 6:].tolist(),
                    'contacts': contacts}

        for target, phase in sorted(set(zip(z['target'], z['phase']))):
            ids = np.flatnonzero((z['target'] == target) & (z['phase'] == phase))
            peak = int(ids[np.argmax(np.abs(acceleration[ids]))])
            finite = ids[np.isfinite(jerk[ids])]
            jp = int(finite[np.argmax(np.abs(jerk[finite]))])
            key = target+'/'+phase
            recorded = observed['phase_native_extrema'][key]
            accel_peak, jerk_peak = float(abs(acceleration[peak])), float(abs(jerk[jp]))
            groups[key] = {'steps': len(ids), 'acceleration_peak_m_s2': accel_peak,
                           'jerk_peak_m_s3': jerk_peak,
                           'recorded_interval_acceleration_matches': bool(np.isclose(accel_peak, recorded['acceleration'], atol=1e-8, rtol=0)),
                           'recorded_interval_jerk_matches': bool(np.isclose(jerk_peak, recorded['jerk'], atol=1e-6, rtol=0)),
                           'acceleration_peak': snapshot(peak), 'jerk_peak': snapshot(jp),
                           'acceleration_window': [snapshot(i) for i in range(max(ids[0], peak-5), min(ids[-1]+1, peak+6))]}
    result = {'schema_version': 1, 'scope': __doc__, 'script_sha256': sha256(__file__),
              'trace_sha256': meta['trace_sha256'], 'frozen_inputs_changed': changed,
              'comparison_scope': 'distinct_recording_noninterference' if comparison else 'single_recording_trace_consistency_only',
              'baseline': baseline, 'observed': observed, 'public_streams_exact': same_streams,
              'objects_native_physics_controller_tcp_exact': same_physics,
              'constraint_attribution_max_abs_error': float(force_error),
              'bounded_affine_actuator_max_abs_error': actuator_error,
              'groups': groups, 'training_ready': False}
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('frozen_inputs_changed', 'public_streams_exact',
        'objects_native_physics_controller_tcp_exact', 'constraint_attribution_max_abs_error',
        'bounded_affine_actuator_max_abs_error')}, indent=2))
    intervals_match = all(g['recorded_interval_acceleration_matches'] and g['recorded_interval_jerk_matches']
                          for g in groups.values())
    if changed or not intervals_match or (comparison and (not same_streams or not same_physics)) or force_error > 1e-8 or actuator_error > 1e-8:
        raise SystemExit('Diagnostic integrity check failed; report preserved')


if __name__ == '__main__':
    main()
