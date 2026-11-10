#!/usr/bin/env python3
"""Isolated approach queries at the final state of a failed recording.

Try narrower pregrasp targets while retaining all original path gates. This is
geometry-only screening; it does not prove grasping or generate training data.
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
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession


def state(backend):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(backend.model, kind))
    mujoco.mj_getState(backend.model, backend.data, value, kind)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', default='chopstick_a')
    parser.add_argument('--clearances', type=float, nargs='+', default=[.001, .00075, .0005])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not all(np.isfinite(v) and 0 < v <= .004 for v in args.clearances):
        parser.error('Positive finite opening clearance up to 4 mm required')
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']: raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        extras = json.loads(f['next_observation/extras_json'][-1])
        time = float(f['next_observation/sim_time'][-1])
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg)
    rows = []
    try:
        layout = meta['task_scene']['layout'];backend.reset(layout['seed'], layout['randomized'])
        for side, arm in extras['teacher']['physics']['arms'].items():
            backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extras['objects'].items():
            adr = backend.model.joint('task/'+name+'/free').qposadr[0]
            backend.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
        backend.data.time = time;mujoco.mj_forward(backend.model, backend.data)
        teacher = RigidTableTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'])
        teacher.current = args.object;teacher.phase = 'approach'
        if backend.task.specs[args.object]['kind'] != 'chopstick':
            raise ValueError('Only slender chopstick aperture queries supported')
        command = np.asarray(extras['controller_interval']['command'])
        original = state(backend)
        for clearance in args.clearances:
            teacher.q = command.copy()
            teacher.pacing['slender_pregrasp_clearance_m'] = clearance
            side, _, _, width = teacher.plan_grasp(args.object)
            offset = 0 if side == 'left' else 7
            row = {'opening_clearance_m': clearance, 'width_m': width, 'passed': False}
            try:
                q = teacher.q[offset:offset+6]
                teacher.check_joint_path(side, q, q, gripper=(teacher.q[offset+6], width),
                                         **teacher.grasp_path_options(args.object))
                teacher.q[offset+6] = width
                _, p, r, planned_width = teacher.reachable_grasp(args.object, check_approach=True)
                row.update(passed=True, position_m=p.tolist(), rotation=r.tolist(), returned_width_m=planned_width)
            except SortFailure as error:
                row['error'] = str(error)
            if not np.array_equal(original, state(backend)):
                raise RuntimeError('Isolated geometry query changed integration state')
            rows.append(row)
    finally:
        backend.close()
    if sha256(args.source) != digest: raise ValueError('Source changed')
    atomic_json(args.output, {'scope': __doc__, 'training_ready': False,
        'source': str(args.source.resolve()), 'source_sha256': digest, 'source_unchanged': True,
        'script_sha256': sha256(__file__), 'snapshot_sim_time': time, 'object': args.object,
        'isolated_integration_state_unchanged': True, 'trials': rows})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
