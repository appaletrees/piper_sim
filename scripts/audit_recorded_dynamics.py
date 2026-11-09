#!/usr/bin/env python3
"""Read-only diagnostics of recorded solver evidence; never a training verdict.

Native velocity, drive and TCP fields are snapshots at the recording frequency.
They do not recover internal integration-step extrema or identify real hardware.
No model is loaded, no RGB is decoded, and no source attributes are modified.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


JOINT_FIELDS = ('joint_position', 'joint_velocity', 'actuator_generalized_force',
                'applied_generalized_force', 'combined_drive_generalized_force',
                'constraint_generalized_force')
TCP_FIELDS = ('tcp_position_world_m', 'tcp_linear_velocity_world_m_s',
              'tcp_angular_velocity_world_rad_s')


def finite_vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('Invalid '+label)
    return result


def inspect_snapshot(raw, state, timestamp):
    extra = json.loads(raw)
    teacher = extra['teacher']
    physics = teacher['physics']
    if physics['schema_version'] != 'piper.physics_evidence.v1':
        raise ValueError('Unsupported native physics evidence schema')
    clock = float(physics['sim_time'])
    if not np.isfinite(clock) or abs(clock-timestamp) > 1e-6:
        raise ValueError('Native physics clock mismatch')
    if set(physics['arms']) != {'left', 'right'}:
        raise ValueError('Native arm inventory mismatch')
    arms = {}
    for side, offset in (('left', 0), ('right', 7)):
        arm = physics['arms'][side]
        arrays = {k: finite_vector(arm[k], 8, side+'/'+k) for k in JOINT_FIELDS}
        arrays.update({k: finite_vector(arm[k], 3, side+'/'+k) for k in TCP_FIELDS})
        q = arrays['joint_position']
        public = np.r_[q[:6], q[6]+q[7]]
        # Public state is stored as float32; native evidence keeps float64.
        if not np.allclose(public, state[offset:offset+7], rtol=0, atol=1e-6):
            raise ValueError('Native/public joint state mismatch: '+side)
        if not np.allclose(arrays['actuator_generalized_force']+arrays['applied_generalized_force'],
                           arrays['combined_drive_generalized_force'], rtol=1e-10, atol=1e-9):
            raise ValueError('Combined drive identity mismatch: '+side)
        arms[side] = arrays
    return arms, {'phase': teacher.get('phase'), 'object': teacher.get('object')}


def inspect_controller_intervals(f, meta):
    """Check optional substep summaries against both native boundary snapshots.

    The full 960 Hz trace is not recorded: this verifies summary consistency,
    not independent reconstruction of every internal peak.
    """
    from piper_sim.contracts import LOWER, UPPER, FEATURE_NAMES
    config=meta['physics_model']['controller']
    steps=config['integration_steps'];dt=float(config['integration_dt'])
    if (type(config['schema_version']) is not int or config['schema_version']!=1
            or config['mode'] not in ('monotone_cubic_velocity','zero_order_observed')
            or type(steps) is not int or steps<1
            or not np.isfinite(dt) or dt<=0):
        raise ValueError('Unsupported controller interval metadata')
    peaks={};previous=None;count=len(f['action/position']);limited=0
    fields=('previous_command','command','position_reference_min','position_reference_max',
            'velocity_reference_min','velocity_reference_max','last_velocity_reference',
            'limiter_start_velocity_jump','native_velocity_min','native_velocity_max',
            'native_acceleration_abs_max','native_jerk_abs_max','position_gain','velocity_gain')
    def native(extra):
        arms=extra['teacher']['physics']['arms']
        return np.concatenate([np.r_[finite_vector(arms[s]['joint_velocity'],8,s)[:6],
            np.asarray(arms[s]['joint_velocity'])[6:].sum()] for s in ('left','right')])
    for i in range(count):
        before,after=[json.loads(f[p+'/extras_json'][i]) for p in ('observation','next_observation')]
        value=after['controller_interval']
        arrays={key:finite_vector(value[key],14,'controller/'+key) for key in fields}
        t0,t1=[float(f[p+'/sim_time'][i]) for p in ('observation','next_observation')]
        clocks=np.array([value['start_sim_time'],value['end_sim_time'],value['integration_dt']],float)
        if (not np.isfinite(clocks).all() or abs(clocks[0]-t0)>1e-7 or abs(clocks[1]-t1)>1e-7
                or clocks[2]!=dt or type(value['integration_steps']) is not int or value['integration_steps']!=steps
                or abs(t1-t0-steps*dt)>1e-7 or value['mode']!=config['mode']):
            raise ValueError('Controller interval clock mismatch at '+str(i))
        if previous is not None and before.get('controller_interval')!=previous:
            raise ValueError('Controller interval continuity mismatch at '+str(i))
        prior=previous['command'] if previous is not None else meta['collection_initial_target']
        if (not np.array_equal(arrays['previous_command'],prior)
                or not np.array_equal(arrays['command'].astype(np.float32),f['action/position'][i])):
            raise ValueError('Controller endpoint/action mismatch at '+str(i))
        for key in ('position_reference','velocity_reference','native_velocity'):
            if np.any(arrays[key+'_min']>arrays[key+'_max']):
                raise ValueError('Reversed controller extrema at '+str(i))
        if (np.any(arrays['position_reference_min']<LOWER) or np.any(arrays['position_reference_max']>UPPER)
                or np.any(arrays['command']<arrays['position_reference_min'])
                or np.any(arrays['command']>arrays['position_reference_max'])):
            raise ValueError('Controller internal target bounds mismatch at '+str(i))
        if config['mode']=='zero_order_observed':
            if (any(not np.array_equal(arrays[key],arrays['command'])
                    for key in ('position_reference_min','position_reference_max'))
                    or any(np.any(arrays[key]!=0) for key in ('velocity_reference_min',
                        'velocity_reference_max','last_velocity_reference','limiter_start_velocity_jump'))):
                raise ValueError('Zero-order observation reference changed at '+str(i))
        v0,v1=native(before),native(after)
        for v in (v0,v1):
            if (np.any(v<arrays['native_velocity_min']-1e-12)
                    or np.any(v>arrays['native_velocity_max']+1e-12)):
                raise ValueError('Native boundary velocity outside substep extrema at '+str(i))
        if (np.any(arrays['native_acceleration_abs_max']<np.abs(v1-v0)/(steps*dt)-1e-9)
                or np.any(arrays['native_acceleration_abs_max']<0)
                or np.any(arrays['native_jerk_abs_max']<0)
                or np.any(arrays['position_gain']<=0) or np.any(arrays['velocity_gain']<0)
                or type(value['native_jerk_samples']) is not int
                or value['native_jerk_samples'] not in ([steps-1,steps] if i==0 else [steps])):
            raise ValueError('Invalid controller derivative/gain evidence at '+str(i))
        force_range=np.asarray(value['actuator_force_range'],float)
        if force_range.shape!=(14,2) or not np.isfinite(force_range).all() or np.any(force_range[:,0]>force_range[:,1]):
            raise ValueError('Invalid controller actuator force range at '+str(i))
        measurements={'native_velocity_abs_max':np.maximum(np.abs(arrays['native_velocity_min']),np.abs(arrays['native_velocity_max'])),
            **{key:np.abs(arrays[key]) for key in ('native_acceleration_abs_max','native_jerk_abs_max','limiter_start_velocity_jump')}}
        for name,values in measurements.items():
            if name not in peaks:peaks[name]=[None]*14
            for channel,x in enumerate(values):
                if peaks[name][channel] is None or x>peaks[name][channel]['absolute_max']:
                    peaks[name][channel]={'absolute_max':float(x),'transition_index':i,'interval_start_sim_time':t0,
                        'interval_end_sim_time':t1,'phase':after['teacher'].get('phase'),'object':after['teacher'].get('object')}
        limited+=int(np.any(arrays['limiter_start_velocity_jump']!=0));previous=value
    return {'integrity_status':'consistent_interval_summaries','mode':config['mode'],
        'intervals':count,'integration_steps':count*steps,
        'channels':list(FEATURE_NAMES),'integration_dt':dt,'intervals_with_slope_limiter':limited,'peaks':peaks,
        'peak_time_resolution':'Control interval containing the peak; exact substep time was not recorded.',
        'scope':'Recorded native substep summary consistency, not full-trace recomputation, contact validation, hardware calibration or training acceptance.'}


def analyze(source):
    started = time.perf_counter()
    source = Path(source).resolve()
    before = source.stat()
    result = {'schema_version': 'piper.recorded_dynamics.v1', 'source': str(source),
        'source_sha256': sha256(source), 'training_ready': False, 'scope': __doc__,
        'integrity_status': 'invalid', 'violations': [], 'arms': {},
        'limitations': [
            'Snapshots at the recording frequency cannot bound internal substep peaks.',
            'Native instantaneous velocity and position-difference interval velocity are different measurements.',
            'Forces are simulation generalized forces, not measured hardware torque or identified limits.',
            'Combined drive excludes passive and constraint forces; its two finger channels must not be mistaken for one actuator.',
            'This diagnostic does not audit images, task success, contact realism, or training acceptance.',
        ]}
    try:
        with h5py.File(source, 'r') as f:
            if not f.attrs.get('complete'):
                raise ValueError('Closed complete recording required')
            meta = json.loads(f.attrs['metadata'])
            if meta.get('backend') != 'MujocoBackend':
                raise ValueError('Native evidence requires a MuJoCo source')
            fps = float(meta['fps'])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid recording frequency')
            states = {p: f[p+'/state'][:] for p in ('observation', 'next_observation')}
            times = {p: f[p+'/sim_time'][:] for p in states}
            count = len(times['observation'])
            if count < 1:
                raise ValueError('Empty recording')
            for p in states:
                if states[p].shape != (count, 14) or not np.isfinite(states[p]).all():
                    raise ValueError('Invalid public states: '+p)
                if times[p].shape != (count,) or not np.isfinite(times[p]).all():
                    raise ValueError('Invalid simulation times: '+p)
                if len(f[p+'/extras_json']) != count:
                    raise ValueError('Native evidence frame count mismatch: '+p)
            if (not np.allclose(times['next_observation']-times['observation'], 1/fps, atol=1e-7, rtol=0)
                    or not np.allclose(times['observation'][1:], times['next_observation'][:-1], atol=1e-7, rtol=0)
                    or not np.array_equal(states['observation'][1:], states['next_observation'][:-1])):
                raise ValueError('Public transition discontinuity')
            peaks, rows = {'left': {}, 'right': {}}, {'left': [], 'right': []}
            def peak(side, name, value, prefix, index, context):
                value = np.abs(np.asarray(value))
                item = peaks[side].get(name)
                if item is None or float(value.max()) > item['absolute_max']:
                    peaks[side][name] = {'absolute_max': float(value.max()),
                        'component': int(value.argmax()), 'sim_time': float(times[prefix][index]),
                        'prefix': prefix, 'transition_index': index, **context}
            previous = None
            for index in range(count):
                for prefix in states:
                    try:
                        arms, context = inspect_snapshot(f[prefix+'/extras_json'][index],
                            states[prefix][index], times[prefix][index])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f'{prefix}[{index}]: {error}') from error
                    if prefix == 'observation' and previous is not None:
                        for side in arms:
                            for key in ('joint_position', 'joint_velocity', *TCP_FIELDS):
                                if not np.allclose(arms[side][key], previous[side][key], atol=1e-12, rtol=0):
                                    raise ValueError(f'Native transition discontinuity at {index}: {side}/{key}')
                    if prefix == 'next_observation':
                        previous = arms
                    for side, arm in arms.items():
                        for key in JOINT_FIELDS[1:]:
                            peak(side, key+'_arm', arm[key][:6], prefix, index, context)
                            peak(side, key+'_fingers', arm[key][6:], prefix, index, context)
                        peak(side, 'gripper_opening_velocity_m_s', [arm['joint_velocity'][6:].sum()], prefix, index, context)
                        peak(side, 'jaw_position_difference_m', [arm['joint_position'][6]-arm['joint_position'][7]], prefix, index, context)
                        peak(side, 'tcp_linear_speed_m_s', [np.linalg.norm(arm['tcp_linear_velocity_world_m_s'])], prefix, index, context)
                        peak(side, 'tcp_angular_speed_rad_s', [np.linalg.norm(arm['tcp_angular_velocity_world_rad_s'])], prefix, index, context)
                        if prefix == 'next_observation' or index == 0:
                            rows[side].append({**arm, '_sim_time': float(times[prefix][index]), '_context': context})
            for side, offset in (('left', 0), ('right', 7)):
                public = np.vstack([states['observation'][0], states['next_observation']])
                interval_v = np.diff(public[:, offset:offset+7].astype(float), axis=0)*fps
                q = np.array([row['joint_position'] for row in rows[side]])
                native_v = np.array([row['joint_velocity'] for row in rows[side]])
                tcp = np.array([row['tcp_position_world_m'] for row in rows[side]])
                tcp_interval_speed = np.linalg.norm(np.diff(tcp, axis=0)*fps, axis=1)
                tcp_peak_row = int(tcp_interval_speed.argmax())+1
                force_peaks = {key: np.max(np.abs([row[key] for row in rows[side]]), axis=0).tolist()
                    for key in JOINT_FIELDS[2:]}
                result['arms'][side] = {'peaks_all_observation_snapshots': peaks[side],
                    'unique_state_samples': len(q), 'native_position_min': q.min(axis=0).tolist(),
                    'native_position_max': q.max(axis=0).tolist(),
                    'native_velocity_abs_max_per_joint': np.abs(native_v).max(axis=0).tolist(),
                    'post_step_force_abs_max_per_joint': force_peaks,
                    'interval_position_difference_velocity_abs_max_per_channel': np.abs(interval_v).max(axis=0).tolist(),
                    'interval_tcp_speed_peak': {'m_s': float(tcp_interval_speed.max()),
                        'interval_end_sim_time': rows[side][tcp_peak_row]['_sim_time'],
                        **rows[side][tcp_peak_row]['_context']},
                    'units': {'joint_position': ['rad']*6+['m']*2,
                              'joint_velocity': ['rad/s']*6+['m/s']*2,
                              'generalized_force': ['N*m']*6+['N']*2},
                    'force_sample_note': 'Initial observation plus post-step snapshots; all-observation peaks also include pre-action feedforward updates.'}
            result.update(integrity_status='consistent_recorded_evidence', transitions=count,
                          unique_state_samples=count+1, checked_snapshots=2*count, fps=fps)
            if meta.get('physics_model',{}).get('controller'):
                result['controller_intervals']=inspect_controller_intervals(f,meta)
            result['declared_cartesian_path_speed_m_s'] = meta.get('teacher', {}).get('motion_and_quality', {}).get('pacing', {}).get('cartesian_speed_m_s')
            result['cartesian_speed_note'] = 'Declared planning speed is not a global TCP acceptance limit. '
            result['cartesian_speed_note'] += 'This report compares measurements without inventing a hardware or production-policy limit.'
    except (KeyError, TypeError, ValueError, OSError) as error:
        result['integrity_status']='invalid'
        result['violations'].append(type(error).__name__+': '+str(error))
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        result['integrity_status'] = 'invalid'
        result['violations'].append('Source changed during read-only inspection')
    result['wall_seconds'] = time.perf_counter()-started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; retain previous diagnostic')
    result = analyze(args.source)
    result['implementation_sha256'] = sha256(__file__)
    atomic_json(args.output, result)
    print(json.dumps({k: result.get(k) for k in ('integrity_status', 'transitions', 'violations', 'wall_seconds')}))
    return 0 if not result['violations'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
