#!/usr/bin/env python3
"""Isolated contact rollouts from a recorded pregrasp; never training episodes.

Restores one recorded state once per trial. Object velocities and solver warm
starts are unavailable and reset to zero. Every subsequent movement uses normal
position control and contact physics. This is not a bitwise historical replay.
"""
import argparse
import copy
import hashlib
from itertools import product
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.simulation import SimulationSession
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.grasp_assist import finger_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', default='chopstick_b')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heights', type=float, nargs='+', default=[0., -.002, -.004])
    parser.add_argument('--tilts', type=float, nargs='+', default=[-15, -20, -10, 15])
    parser.add_argument('--axis-offsets', type=float, nargs='+', default=[0.])
    parser.add_argument('--closing-scales', type=float, nargs='+', default=[1.])
    parser.add_argument('--pregrasp-widths', type=float, nargs='+', default=[.012])
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.closing_scales):
        parser.error('Closing scales must be in (0, 1]')
    if any(not 0 < value <= .095 for value in args.pregrasp_widths):
        parser.error('Pregrasp widths must be in (0, .095] m')
    if args.output.exists(): raise FileExistsError(args.output)
    prior = json.loads(args.source.with_name('report.json').read_text())
    selected = next(e for e in prior['events'] if e['phase'] == 'grasp_candidate_selected'
                    and e['object'] == args.object)
    with h5py.File(args.source, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        index = int(np.argmin(abs(f['observation/sim_time'][:]-selected['sim_time'])))
        extras = json.loads(f['observation/extras_json'][index])
        command = (f['action/position'][index-1].astype(float) if index else
                   np.asarray(meta['collection_initial_target'], float))
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg)
    records = []
    try:
        for height in args.heights:
            for tilt, axis_offset, closing_scale, pregrasp_width in product(
                    args.tilts, args.axis_offsets, args.closing_scales, args.pregrasp_widths):
                backend.reset(meta['task_scene']['layout']['seed'], meta['task_scene']['layout']['randomized'])
                for side, arm in extras['teacher']['physics']['arms'].items():
                    backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
                    backend.data.qvel[backend.indices[side]['dof']] = arm['joint_velocity']
                for name, pose in extras['objects'].items():
                    address = backend.model.joint('task/'+name+'/free').qposadr[0]
                    backend.data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
                for side, offset in (('left', 0), ('right', 7)):
                    backend.data.ctrl[backend.indices[side]['ctrl']] = command[offset:offset+7]
                mujoco.mj_forward(backend.model, backend.data)
                teacher = RigidTableTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'])
                teacher.pacing['closing_limits'] = {k: v*closing_scale for k, v in teacher.pacing['closing_limits'].items()}
                teacher.current = args.object
                if backend.task.specs[args.object]['kind'] != 'chopstick':
                    raise ValueError('This diagnostic currently supports chopstick controller settings only')
                backend.model.actuator_forcerange[backend.model.actuator('right/gripper').id] = [-.30, .30]
                result = {'height_offset_m': height, 'local_tilt_deg': tilt, 'axis_offset_m': axis_offset,
                          'closing_scale': closing_scale, 'pregrasp_width_m': pregrasp_width}
                original = teacher.object_pose(args.object)['position']
                try:
                    if pregrasp_width != .012:
                        teacher.grip('right', pregrasp_width)
                    side, p, r, _ = teacher.plan_grasp(args.object)
                    p += [0, 0, height]
                    p += backend.data.body('task/'+args.object).xmat.reshape(3, 3)[:, 0]*axis_offset
                    a = np.deg2rad(tilt)
                    r = r@np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                    q = teacher.solve(side, p, r)
                    scratch = teacher.ik[side].data
                    scratch.qpos[:] = backend.data.qpos
                    scratch.qpos[backend.indices[side]['qpos']] = [*q, 0., 0.]
                    mujoco.mj_fwdPosition(backend.model, scratch)
                    predicted = set()
                    for contact in scratch.contact:
                        names = [backend.model.body(int(backend.model.geom_bodyid[g])).name for g in contact.geom]
                        if 'task/'+args.object in names and contact.dist <= 0:
                            predicted.update(n for n in names if n in (side+'/link7', side+'/link8'))
                    result['geometric_closed_fingers'] = sorted(predicted)
                    teacher.pose(side, teacher.approach_position(args.object, p, r, .035), r)
                    teacher.contact_approach(side, p, r, args.object)
                    teacher.grip(side, 0.)
                    contacts = finger_contacts(backend, side, args.object)
                    result['actual_contact_fingers'] = contacts
                    if len(contacts) != 2:
                        raise SortFailure('Missing physical bilateral contact')
                    teacher.track_grasp(side, args.object)
                    teacher.pose(side, p+[0, 0, .035], r)
                    result['lift_m'] = teacher.object_pose(args.object)['position'][2]-original[2]
                    result['lifted'] = result['lift_m'] >= .025
                except SortFailure as error:
                    result.update(lifted=False, error=str(error))
                result.update(sim_seconds=backend.sim_time, collision_peak=teacher.collision_peak,
                              final_object=teacher.object_pose(args.object), motion=teacher.quality.report())
                records.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ('motion', 'final_object')}), flush=True)
    finally:
        backend.close()
    report = {'status': 'isolated_contact_diagnostic', 'training_ready': False,
        'source': str(args.source.resolve()), 'source_frame': index, 'recorded_event': selected,
        'limitations': __doc__, 'trials': records,
        'implementation_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('src/piper_sim/experts/clean_table.py'), Path('src/piper_sim/experts/trajectory.py')]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
