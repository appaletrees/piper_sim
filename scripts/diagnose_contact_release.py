#!/usr/bin/env python3
"""Isolated release diagnostics, never training data or exact historical replay.

Restore a recorded release state once per trial. Unrecorded object velocities and
solver warm starts are zero. Thereafter only ordinary motor control and physics
run. Compare preload unloading and solver iteration continuity without changing
material, thresholds, CAD or production control policy.
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
from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', default='side_dish')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output')
    report = json.loads(args.source.with_name('report.json').read_text())
    event = next(e for e in report['events'] if e['phase'] == 'open_gripper' and e['object'] == args.object)
    with h5py.File(args.source) as source:
        if not source.attrs.get('complete'): raise ValueError('Closed source required')
        meta = json.loads(source.attrs['metadata'])
        times = source['observation/sim_time'][:]
        index = int(np.argmin(abs(times-event['sim_time'])))
        start_time = float(times[index])
        extras = json.loads(source['observation/extras_json'][index])
        command = source['action/position'][index-1].astype(float)
    if args.object != 'side_dish': raise ValueError('Currently only side_dish controller reconstruction is supported')
    cfg = copy.deepcopy(meta['config'])
    cfg['mujoco']['finger_contact'] = meta['physics_model']['finger_contact']['profile']
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    output = {'training_ready': False, 'scope': __doc__, 'source': str(args.source.resolve()),
        'source_sha256': sha256(args.source), 'source_frame': index, 'sim_time': start_time,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'expert_source_sha256': {str(p): sha256(p) for p in Path('src/piper_sim/experts').glob('*.py')},
        'trials': []}
    try:
        for unload, iterations in ((False, 0), (True, 0), (False, 10), (True, 10)):
            layout = meta['task_scene']['layout']
            backend.reset(layout['seed'], layout['randomized'])
            assert backend.finger_contact.metadata()['arms'] == meta['physics_model']['finger_contact']['arms']
            for side, arm in extras['teacher']['physics']['arms'].items():
                backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
                backend.data.qvel[backend.indices[side]['dof']] = arm['joint_velocity']
            for name, pose in extras['objects'].items():
                address = backend.model.joint('task/'+name+'/free').qposadr[0]
                backend.data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
            for side, offset in (('left', 0), ('right', 7)):
                backend.data.ctrl[backend.indices[side]['ctrl']] = command[offset:offset+7]
            backend.data.time = start_time
            mujoco.mj_forward(backend.model, backend.data)
            frames = []
            class ReleaseTeacher(RigidTableTeacher):
                def step(self):
                    self.b.model.opt.noslip_iterations = iterations
                    return CleanTableTeacher.step(self)
            def capture(teacher, observation):
                arm = teacher.telemetry()['physics']['arms']['right']
                frames.append({'sim_time': observation.sim_time, 'opening_m': float(observation.position[13]),
                    'command_m': float(teacher.q[13]), 'contacts': arm['target_finger_contacts']})
            teacher = ReleaseTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'],
                                     frame_hook=capture)
            teacher.current = args.object
            teacher.q = command.copy()
            aid = backend.model.actuator('right/gripper').id
            backend.model.actuator_forcerange[aid] = [-15., 15.]
            backend.model.actuator_gainprm[aid, 0] = 10000.
            backend.model.actuator_biasprm[aid, 1] = -10000.
            teacher.quality.add(start_time, command, teacher.session.read().position)
            if unload:
                teacher.event('unload_gripper')
                opening = float(teacher.session.read().position[13])
                teacher.grip('right', opening+.002)
            teacher.event('release')
            teacher.grip('right', .070)
            teacher.wait(.5)
            result = {'unload_preload': unload, 'noslip_iterations_during_release': iterations,
                'material': backend.finger_contact.metadata(), 'motion': teacher.quality.report(),
                'frames': frames, 'final_object': teacher.object_pose(args.object)}
            output['trials'].append(result)
            print(json.dumps({'unload': unload, 'iterations': iterations,
                              'passed': result['motion']['passed'], 'metrics': result['motion']['metrics']}), flush=True)
    finally:
        backend.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
