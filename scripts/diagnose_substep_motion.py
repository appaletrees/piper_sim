#!/usr/bin/env python3
"""Independent pre-contact substep diagnostic, never training data or replay.

Initialize identical MuJoCo data copies from the first recorded approach.
Task-object velocities and solver warm starts were not recorded and start at zero.
Replay stored float32 action targets (not the original teacher's float64 values).
Compare batched mj_step with individually observed steps, without extra forwards
inside integration. Report native velocities separately from 30 Hz differences.
The optional linear controller changes the actuator targets and is not expected
to reproduce the zero-order controller's integration state.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


def integration_state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, value, kind)
    return value


def observed_steps(model, data, steps, qpos_indices, dof_indices):
    """Observe post-integration qpos/qvel only; no control or solver changes."""
    if type(steps) is not int or steps < 1:
        raise ValueError('Positive integer substep count required')
    qpos_indices, dof_indices = np.asarray(qpos_indices), np.asarray(dof_indices)
    for indices, size in ((qpos_indices, model.nq), (dof_indices, model.nv)):
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= size)):
            raise ValueError('Valid state indices required')
    dt, start = float(model.opt.timestep), float(data.time)
    times, positions, velocities = [], [], []
    for index in range(steps):
        mujoco.mj_step(model, data)
        if (abs(data.time-(start+(index+1)*dt)) > 1e-7
                or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()):
            raise RuntimeError('Nonfinite state or internal physics reset during substep capture')
        times.append(float(data.time))
        positions.append(data.qpos[qpos_indices].copy())
        velocities.append(data.qvel[dof_indices].copy())
    return np.asarray(times), np.asarray(positions), np.asarray(velocities)


def linear_targets(previous, target, steps):
    """Causal experimental servo ramp; endpoint input and control period unchanged."""
    a, b = np.asarray(previous, dtype=float), np.asarray(target, dtype=float)
    if type(steps) is not int or steps < 1 or a.ndim != 1 or a.shape != b.shape or not np.isfinite([a,b]).all():
        raise ValueError('Finite matching targets and positive substep count required')
    result = a+(b-a)*np.arange(1,steps+1)[:,None]/steps
    result[-1] = b
    return result


def public_positions(positions):
    return np.c_[positions[:, :6], positions[:, 6:8].sum(axis=1),
                 positions[:, 8:14], positions[:, 14:16].sum(axis=1)]


def peak(array, times):
    row, channel = np.unravel_index(np.abs(array).argmax(), array.shape)
    return {'absolute_max': float(abs(array[row, channel])), 'channel': int(channel),
            'sim_time': float(times[row])}


def summarize(times, positions, velocities, stride):
    dt = float(np.median(np.diff(times)))
    boundary = np.arange(0, len(times), stride)
    if boundary[-1] != len(times)-1:
        raise ValueError('Trace does not end at a control boundary')
    result = {}
    for side, off in (('left', 0), ('right', 8)):
        q = positions[:, off:off+8]
        v = velocities[:, off:off+8]
        public_q = np.c_[q[:, :6], q[:, 6:].sum(axis=1)]
        public_v = np.c_[v[:, :6], v[:, 6:].sum(axis=1)]
        interval_v = np.diff(public_q[boundary], axis=0)/(dt*stride)
        for label, indices in (('arm', range(6)), ('gripper', [6])):
            result[side+'/'+label] = {
                'native_substep_velocity': peak(public_v[:, indices], times),
                'native_control_boundary_velocity': peak(public_v[boundary][:, indices], times[boundary]),
                'control_position_difference_velocity': peak(interval_v[:, indices], times[boundary][1:]),
                'substep_velocity_difference_acceleration': peak(np.diff(public_v[:, indices], axis=0)/dt, times[1:]),
                'substep_velocity_second_difference_jerk': peak(np.diff(public_v[:, indices], n=2, axis=0)/dt**2, times[2:]),
                'units': 'rad/s, rad/s2, rad/s3' if label == 'arm' else 'm/s, m/s2, m/s3',
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=90, help='Initial pre-contact approach only')
    parser.add_argument('--compare-linear', action='store_true',
        help='Also test explicit internal target interpolation; never enables it in production')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh diagnostic directory')
    if not 2 <= args.frames <= 120: parser.error('Use 2..120 initial control frames')
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'): raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        if meta['backend'] != 'MujocoBackend' or meta['teacher']['grasp_mode'] != 'physical':
            raise ValueError('Physical MuJoCo source required')
        actions = f['action/position'][:args.frames].astype(float)
        if len(actions) != args.frames: raise ValueError('Source is too short')
        extras = json.loads(f['observation/extras_json'][0])
        targets = set()
        for raw in f['next_observation/extras_json'][:args.frames]:
            teacher = json.loads(raw)['teacher']
            if teacher['phase'] != 'approach' or teacher['grasp'] or teacher['active_assist']:
                raise ValueError('Select only the initial unloaded pre-contact approach')
            targets.add(teacher['object'])
        recorded = np.vstack([f['observation/state'][0], f['next_observation/state'][:args.frames]])
        start_time = float(f['observation/sim_time'][0])
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.experts.rigid_table import RigidTableTeacher
    from piper_sim.simulation import SimulationSession
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene']['domain_profile']
    backend = MujocoBackend(cfg, cameras=False)
    try:
        layout = meta['task_scene']['layout']
        backend.reset(layout['seed'], layout['randomized'])
        if len(targets) != 1 or any(backend.task.specs[name]['kind'] != 'cup' for name in targets):
            raise ValueError('Only the initial cup approach controller is reconstructed')
        RigidTableTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'])
        for side, arm in extras['teacher']['physics']['arms'].items():
            backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
            backend.data.qvel[backend.indices[side]['dof']] = arm['joint_velocity']
        for name, pose in extras['objects'].items():
            address = backend.model.joint('task/'+name+'/free').qposadr[0]
            backend.data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
        backend.data.time = start_time
        mujoco.mj_forward(backend.model, backend.data)
        model = backend.model
        baseline, traced = copy.copy(backend.data), copy.copy(backend.data)
        position_ids = [i for side in ('left','right') for i in backend.indices[side]['qpos']]
        velocity_ids = [i for side in ('left','right') for i in backend.indices[side]['dof']]
        control_ids = [i for side in ('left','right') for i in backend.indices[side]['ctrl']]
        stride = backend.decimation*backend.substeps
        baseline_states, walls, traces = [], {}, {}
        modes = [('batched', baseline), ('observed', traced)]
        if args.compare_linear: modes.append(('linear', copy.copy(backend.data)))
        for name, data in modes:
            times, positions, velocities = [[start_time]], [data.qpos[position_ids][None].copy()], [data.qvel[velocity_ids][None].copy()]
            previous = np.asarray(meta['collection_initial_target'], dtype=float)
            applied = [previous[None].copy()]
            started = time.perf_counter()
            for frame, action in enumerate(actions):
                for side, offset in (('left',0), ('right',7)):
                    ids = backend.indices[side]
                    data.qfrc_applied[ids['dof']] = data.qfrc_bias[ids['dof']]
                    if name != 'linear': data.ctrl[ids['ctrl']] = action[offset:offset+7]
                if name == 'batched':
                    mujoco.mj_step(model, data, nstep=stride)
                elif name == 'linear':
                    schedule = linear_targets(previous, action, stride)
                    for target in schedule:
                        data.ctrl[control_ids] = target
                        t, q, v = observed_steps(model, data, 1, position_ids, velocity_ids)
                        times.append(t); positions.append(q); velocities.append(v)
                    applied.append(schedule)
                else:
                    t, q, v = observed_steps(model, data, stride, position_ids, velocity_ids)
                    times.append(t); positions.append(q); velocities.append(v)
                    applied.append(np.tile(action,(stride,1)))
                previous = action
                # Match the existing backend's once-per-control-tick refresh.
                mujoco.mj_forward(model, data)
                state = integration_state(model, data)
                if name == 'batched': baseline_states.append(state)
                elif name == 'observed' and not np.array_equal(state, baseline_states[frame]):
                    raise ValueError('Substep observation changed the integration state at frame '+str(frame))
            walls[name] = time.perf_counter()-started
            if name != 'batched':
                traces[name] = {'time':np.concatenate(times),'joint_position':np.concatenate(positions),
                    'joint_velocity':np.concatenate(velocities),'actuator_targets':np.concatenate(applied)}
        t,q,v = [traces['observed'][key] for key in ('time','joint_position','joint_velocity')]
        boundary_q = q[::stride]
        public = public_positions(boundary_q)
        result = {'scope': __doc__, 'training_ready': False, 'status': 'diagnostic_complete',
            'source': str(args.source.resolve()), 'source_sha256': sha256(args.source),
            'implementation_sha256': sha256(__file__), 'mujoco_version': mujoco.__version__,
            'source_seed': layout['seed'], 'control_frames': args.frames, 'integration_samples': len(t),
            'control_hz': 1/(model.opt.timestep*stride), 'integration_hz': 1/model.opt.timestep,
            'observer_equivalence': {
                'methods': ['batched', 'observed'],
                'state': 'mjSTATE_INTEGRATION',
                'boundaries_compared': args.frames,
                'all_boundaries_bitwise_identical': True,
                'excludes_controller_candidates': True},
            'simulation_seconds': float(t[-1]-t[0]), 'window_wall_seconds': walls,
            'max_boundary_position_error_vs_stored_source_per_channel': np.abs(public-recorded).max(axis=0).tolist(),
            'metrics': summarize(t,q,v,stride),
            'limitations': ['Short pre-contact reconstruction, not exact historical physics or a complete task.',
                'Missing free-object velocities/warm starts and float32 action storage limit historical reconstruction.',
                'Native substep and 30 Hz finite-difference derivatives use different bandwidths; existing acceptance gates are not reinterpreted.',
                'Solver force/contact buffers are intentionally not assigned post-integration timestamps by this diagnostic.',
                'CPU-only low-priority work on the shared benchmark host; not end-to-end throughput.',
                'No real hardware response, force limit or calibration is established.']}
        args.output.mkdir(parents=True)
        trace = args.output/'trace.npz'
        np.savez_compressed(trace,**traces['observed'])
        result['trace'] = {'path':str(trace.resolve()),'sha256':sha256(trace)}
        if args.compare_linear:
            from piper_sim.collection_audit import load_acceptance_policy
            from piper_sim.experts.quality import MotionQuality
            settings, policy = load_acceptance_policy()
            result['controller_comparison'] = {'policy':policy, 'methods':{},
                'scope':'Short approach diagnostic with identical 30 Hz controller inputs and simulation duration. '
                        'Linear mode changes the internal actuator-target schedule and resulting physics. '
                        'No attachment, state assignment during motion, video retiming or quality-limit change.',
                'hardware_calibrated':False,'training_ready':False}
            for name, values in traces.items():
                monitor = MotionQuality(settings,model.opt.timestep*stride)
                actual = public_positions(values['joint_position'][::stride])
                for i, ts in enumerate(values['time'][::stride]):
                    monitor.add(ts, meta['collection_initial_target'] if i==0 else actions[i-1], actual[i])
                item = {'metrics':summarize(values['time'],values['joint_position'],values['joint_velocity'],stride),
                    'motion_diagnostic_30hz':monitor.report(), 'sim_seconds':float(values['time'][-1]-values['time'][0]),
                    'internal_target_schedule':f'Previous accepted target to current target over {stride} substeps' if name=='linear'
                        else f'Current accepted target held for all {stride} substeps',
                    'final_boundary_error_vs_zero_order':(actual[-1]-public[-1]).tolist()}
                if name == 'linear':
                    path = args.output/'linear_trace.npz';np.savez_compressed(path,**values)
                    item['trace']={'path':str(path.resolve()),'sha256':sha256(path)}
                result['controller_comparison']['methods'][name] = item
        atomic_json(args.output/'report.json',result)
        print(json.dumps({k:result[k] for k in ('status','control_frames','integration_samples','window_wall_seconds')}))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
