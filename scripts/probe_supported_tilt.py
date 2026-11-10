#!/usr/bin/env python3
"""Isolated IK/collision queries for a table-supported plate tilt.

Reconstructs a recorded lift-start pose only in a separate diagnostic backend.
It does not step physics or certify force closure, support or training quality.
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
from piper_sim.experts.supported_tilt import hand_path, validate_profile
from piper_sim.simulation import SimulationSession
from probe_grasp_aperture import state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']: raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate' and e['phase']=='lift')
        times = f['observation/sim_time'][:];i = int(np.argmin(abs(times-event['sim_time'])))
        if abs(times[i]-event['sim_time']) > 1e-8: raise ValueError('Exact lift boundary required')
        extra = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False);trials = []
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items(): b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extra['objects'].items():
            a = b.model.joint('task/'+name+'/free').qposadr[0]
            b.data.qpos[a:a+7] = pose['position']+pose['quaternion_wxyz']
        mujoco.mj_forward(b.model, b.data)
        t = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
        t.q = np.asarray(extra['controller_interval']['command']);t.current = 'plate';t.phase = 'lift'
        t.track_grasp('right', 'plate')
        initial = state(b);command = t.q.copy()
        hp, hr = t.tcp('right');obj = b.data.body('task/plate')
        for angle in (45., 60., 75.):
            profile = validate_profile({'angle_deg': angle, 'angle_step_deg': 10., 'support_gap_m': .0005,
                'clear_lift_m': .06, 'path_speed_m_s': .04, 'max_grasp_rotation_rad': .15})
            poses, plan = hand_path(obj.xpos, obj.xmat.reshape(3, 3), hp, hr, b.task.surface_points['plate'], profile)
            row = {'profile': profile, 'plan': plan, 'passed': False}
            try:
                path = t.plan_pose_path('right', poses, speed=profile['path_speed_m_s'])
                row.update(passed=True, command_samples=len(path), planned_duration_s=len(path)/t.fps)
            except SortFailure as exc: row['error'] = str(exc)
            if not np.array_equal(initial, state(b)) or not np.array_equal(command, t.q):
                raise RuntimeError('Diagnostic query changed integration state or command')
            trials.append(row);print(json.dumps({k:v for k,v in row.items() if k!='plan'}), flush=True)
    finally: b.close()
    if sha256(args.source) != digest: raise ValueError('Source changed')
    atomic_json(args.output, {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
        'script_sha256': sha256(__file__), 'sim_time': event['sim_time'], 'training_ready': False,
        'isolated_state_and_commands_unchanged': True, 'trials': trials})


if __name__ == '__main__': main()
