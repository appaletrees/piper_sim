#!/usr/bin/env python3
"""Static opening/withdrawal queries on isolated recorded plate geometry.

No dynamics replay, live pose edits or training acceptance. Geometry at each
snapshot stays fixed during queries; a moving released plate can still recontact.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.collision import robot_obstacle_clearance
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession
from probe_supported_lowering import integration_state
from probe_supported_release import opening_check


def distance_at(teacher, q, width):
    """Query the actual commanded jaw aperture on scratch geometry."""
    b, d = teacher.b, teacher.ik['right'].data
    d.qpos[:] = b.data.qpos
    d.qpos[b.indices['right']['qpos'][:6]] = q
    d.qpos[b.indices['right']['qpos'][6:]] = width/2
    mujoco.mj_forward(b.model, d)
    distance, pair = robot_obstacle_clearance(b.model, d, 'right', ['plate'], .02)
    return {'capped_signed_distance_m': distance, 'cap_m': .02, 'pair': pair}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact-time', type=float, required=True,
                        help='Recorded contact time from the independent native trace')
    parser.add_argument('--directions', action='store_true',
                        help='Also query tool-axis and outward withdrawal directions')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        events = meta['collection_outcome']['teacher_events']
        release = next(e for e in events if e['object'] == 'plate' and e['phase'] == 'release')
        times = f['observation/sim_time'][:]
        start = int(np.argmin(abs(times-release['sim_time'])))
        if abs(times[start]-release['sim_time']) > 1e-8:
            raise ValueError('No exact release snapshot')
        commands = f['action/position'][:]
        end = next(i for i in range(start+1, len(times))
                   if np.max(abs(commands[i, 7:13]-commands[start, 7:13])) > 1e-7)
        indices = [('opening_start', start), ('before_retreat', end-1),
                   ('retreat_contact', int(np.argmin(abs(times-args.contact_time))))]
        snapshots = [(label, i, float(times[i]), json.loads(f['observation/extras_json'][i]))
                     for label, i in indices]
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    results = []
    try:
        for label, i, time, extras in snapshots:
            layout = meta['task_scene']['layout']
            backend.reset(layout['seed'], layout['randomized'])
            for side, arm in extras['teacher']['physics']['arms'].items():
                backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
            for name, pose in extras['objects'].items():
                adr = backend.model.joint('task/'+name+'/free').qposadr[0]
                backend.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
            mujoco.mj_forward(backend.model, backend.data)
            teacher = RigidTableTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'])
            teacher.q = np.asarray(extras['controller_interval']['command'])
            teacher.current, teacher.phase = 'plate', 'release'
            teacher.configure_object_contact('plate')
            state = integration_state(backend.model, backend.data)
            tcp, rotation = teacher.ik['right'].forward(teacher.q[7:13])
            radial = tcp-backend.data.body('task/plate').xpos
            radial[2] = 0
            if np.linalg.norm(radial) < 1e-9:
                raise ValueError('Undefined outward direction')
            radial /= np.linalg.norm(radial)
            directions = {'vertical': np.array([0., 0., 1.])}
            if args.directions:
                directions.update(tool_axis=-rotation[:, 2], outward=radial,
                                  outward_up=(radial+[0, 0, 1])/np.sqrt(2))
            attempts = []
            for width in (.045, .06, .075, .09):
                row = {'width_m': width, 'opening_error': None, 'withdrawals': []}
                try:
                    opening_check(teacher, teacher.q[7:13], width)
                except Exception as exc:
                    row['opening_error'] = type(exc).__name__+': '+str(exc)
                row['initial_payload_distance'] = distance_at(teacher, teacher.q[7:13], width)
                start_q = teacher.q.copy();start_q[13] = width
                for direction, axis in directions.items():
                    attempt = {'direction': direction, 'axis_world': axis.tolist(), 'error': None}
                    try:
                        path = teacher.plan_pose_path('right', [(tcp+.03*axis, rotation)], start=start_q)
                        minimum = row['initial_payload_distance']['capped_signed_distance_m']
                        previous = start_q
                        for q in path:
                            # Explicitly check proposed, not measured, aperture.
                            teacher.check_joint_path('right', previous[7:13], q[7:13], gripper=(width, width))
                            minimum = min(minimum, distance_at(teacher, q[7:13], width)['capped_signed_distance_m'])
                            previous = q
                        attempt.update(minimum_payload_distance_m=minimum, command_samples=len(path))
                    except Exception as exc:
                        attempt['error'] = type(exc).__name__+': '+str(exc)
                    row['withdrawals'].append(attempt)
                row['isolated_state_unchanged'] = np.array_equal(state, integration_state(backend.model, backend.data))
                if not row['isolated_state_unchanged']:
                    raise RuntimeError('Scratch query changed the snapshot')
                attempts.append(row)
                print(label, json.dumps(row), flush=True)
            results.append({'label': label, 'source_index': i, 'sim_time': time, 'attempts': attempts})
    finally:
        backend.close()
    if sha256(args.source) != digest:
        raise ValueError('Source changed')
    atomic_json(args.output, {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
                             'script_sha256': sha256(__file__), 'contact_time_query': args.contact_time,
                             'results': results, 'training_ready': False})


if __name__ == '__main__':
    main()
