#!/usr/bin/env python3
"""Inspect grasp candidates from a closed recording, without advancing physics.

This reconstructs native joint positions and object poses on diagnostic state.
It tests candidate geometry, not dynamic grasp success or force equivalence.
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
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.collision import robot_obstacle_clearance
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output')
    digest = sha256(args.source)
    report = json.loads(args.source.with_name('report.json').read_text())
    event = next(e for e in report['events'] if e['phase'] == 'grasp_candidate_selected'
                 and e.get('object') == args.object)
    with h5py.File(args.source, 'r') as source:
        if not source.attrs.get('complete'): raise ValueError('A closed recording is required')
        meta = json.loads(source.attrs['metadata'])
        times = source['observation/sim_time'][:]
        index = int(np.argmin(abs(times-event['sim_time'])))
        extras = json.loads(source['observation/extras_json'][index])
        command = source['action/position'][index-1].astype(float)
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    rows = []
    try:
        layout = meta['task_scene']['layout']
        backend.reset(layout['seed'], layout['randomized'])
        for side, arm in extras['teacher']['physics']['arms'].items():
            backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extras['objects'].items():
            address = backend.model.joint('task/'+name+'/free').qposadr[0]
            backend.data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
        backend.data.time = float(times[index])
        mujoco.mj_forward(backend.model, backend.data)
        teacher = RigidTableTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'])
        teacher.q = command.copy()
        teacher.current = args.object
        teacher.configure_object_contact(args.object)
        side, position, rotation, width = teacher.plan_grasp(args.object)
        offset = 0 if side == 'left' else 7
        obstacles = [n for n in backend.task.active_names if n != args.object]
        original_check = teacher.check_joint_path
        baseline = backend.data.qpos.copy()
        for p, r, details in teacher.grasp_candidates(args.object, position, rotation, check_approach=True):
            row = {**details, 'minimum_distance_m': .01, 'pair': None, 'checks': 0,
                   'original_gate_passed': False, 'planned_width_m': float(command[offset+6])}
            def check(*a, **kw):
                original_check(*a, **kw)
                distance, pair = robot_obstacle_clearance(backend.model, teacher.ik[side].data,
                                                         side, obstacles, .01)
                row['checks'] += 1
                if distance < row['minimum_distance_m']:
                    row.update(minimum_distance_m=distance, pair=pair, leg=leg)
            teacher.check_joint_path = check
            try:
                pregrasp = teacher.solve(side, teacher.approach_position(args.object, p, r, .035), r)
                leg = 'move_to_pregrasp'
                check(side, command[offset:offset+6], pregrasp)
                start = command.copy(); start[offset:offset+6] = pregrasp
                leg = 'contact_approach'
                teacher.plan_pose_path(side, [(p, r)], start=start, speed=teacher.pacing['contact_speed_m_s'])
                row['original_gate_passed'] = True
            except SortFailure as error:
                row['error'] = str(error)
            rows.append(row)
        np.testing.assert_array_equal(backend.data.qpos, baseline)
        np.testing.assert_array_equal(teacher.q, command)
    finally:
        backend.close()
    if sha256(args.source) != digest: raise RuntimeError('Source changed')
    atomic_json(args.output, {'training_ready': False, 'source': str(args.source.resolve()),
        'source_sha256': digest, 'source_unchanged': True, 'snapshot_frame': index,
        'snapshot_sim_time': float(times[index]), 'script_sha256': sha256(__file__),
        'scope': 'Static geometry at final checked sample of each original path segment; '
                 'same native finger positions as the legacy planner; no physics or force prediction.',
        'target': args.object, 'candidates': rows})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
