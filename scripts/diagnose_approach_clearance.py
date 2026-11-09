#!/usr/bin/env python3
"""Isolated physical approach trials from a recorded object-to-object transition.

The recorded initial state is restored once per trial. Thereafter every motion
uses the production controller and physics. Missing object velocities and solver
warm starts are zero, so this is not exact historical replay or training data.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession


class ApproachCompleted(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', required=True)
    parser.add_argument('--rises', type=float, nargs='+', default=[0., .02, .04])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output')
    if not np.isfinite(args.rises).all() or min(args.rises) < 0: parser.error('Invalid rises')
    report = json.loads(args.source.with_name('report.json').read_text())
    event = next(e for e in report['events'] if e['phase'] == 'approach' and e['object'] == args.object)
    with h5py.File(args.source, 'r') as source:
        metadata = json.loads(source.attrs['metadata'])
        index = int(np.argmin(abs(source['observation/sim_time'][:]-event['sim_time'])))
        extras = json.loads(source['observation/extras_json'][index])
        command = (source['action/position'][index-1].astype(float) if index else
                   np.asarray(metadata['collection_initial_target'], float))
    cfg = copy.deepcopy(metadata['config'])
    cfg['task_scene']['domain_randomization'] = metadata['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    trials = []
    try:
        for rise in args.rises:
            layout = metadata['task_scene']['layout']
            backend.reset(layout['seed'], layout['randomized'])
            for side, arm in extras['teacher']['physics']['arms'].items():
                backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
                backend.data.qvel[backend.indices[side]['dof']] = arm['joint_velocity']
            for name, pose in extras['objects'].items():
                address = backend.model.joint('task/'+name+'/free').qposadr[0]
                backend.data.qpos[address:address+7] = [*pose['position'], *pose['quaternion_wxyz']]
            for side, offset in (('left', 0), ('right', 7)):
                backend.data.ctrl[backend.indices[side]['ctrl']] = command[offset:offset+7]
            mujoco.mj_forward(backend.model, backend.data)
            teacher = RigidTableTeacher(SimulationSession(backend), settings=metadata['teacher']['motion_and_quality'])
            teacher.q = command.copy()
            teacher.quality.add(backend.sim_time, command, teacher.session.read().position,
                                context={'phase': 'initial_recorded_state'})
            original = teacher.reachable_grasp
            first = True
            def reachable(name, **kwargs):
                nonlocal first
                result = original(name, **kwargs)
                if first and rise:
                    side = result[0]
                    offset = 0 if side == 'left' else 7
                    tcp, rotation = teacher.ik[side].forward(teacher.q[offset:offset+6])
                    teacher.event('clear_basket_before_approach')
                    teacher.pose_path(side, [(tcp+[0, 0, rise], rotation)])
                first = False
                return result
            teacher.reachable_grasp = reachable
            def stop_before_descent(*_args, **_kwargs):
                raise ApproachCompleted()
            teacher.contact_approach = stop_before_descent
            result = {'vertical_rise_m': rise, 'completed_approach': False}
            try:
                teacher.sort_one(args.object)
            except ApproachCompleted:
                result['completed_approach'] = True
            except SortFailure as error:
                result['error'] = str(error)
            result.update(motion=teacher.quality.report(), collision_peak=teacher.collision_peak,
                          sim_seconds=backend.sim_time, events=teacher.events)
            trials.append(result)
            print(json.dumps({'rise': rise, 'completed': result['completed_approach'],
                'jerk': result['motion']['metrics']['measured_arm_jerk'],
                'collision_peak': result['collision_peak'], 'error': result.get('error')}), flush=True)
    finally:
        backend.close()
    result = {'status': 'isolated_physical_approach_diagnostic', 'source': str(args.source.resolve()),
        'scope': __doc__, 'source_frame': index, 'source_event': event, 'trials': trials,
        'training_ready': False, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
