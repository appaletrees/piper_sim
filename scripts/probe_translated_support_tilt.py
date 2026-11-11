#!/usr/bin/env python3
"""Isolated geometry search for supported tilts with horizontal hand motion.

Object poses reconstructed from HDF5 exist only in this separate backend. The
path assumes a rigid grasp and sliding/rolling support; it does not certify
friction, force closure, execution stability or training quality.
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


def translated_path(position, rotation, hand_position, hand_rotation, points, profile, xy):
    poses, plan = hand_path(position, rotation, hand_position, hand_rotation, points, profile)
    xy = np.asarray(xy, float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or np.linalg.norm(xy) > .1:
        raise ValueError('Finite support translation within 10 cm required')
    for i, item in enumerate(plan['planned_object_poses']):
        t = item['angle_deg'] / profile['angle_deg']
        fraction = 10*t**3 - 15*t**4 + 6*t**5
        shift = np.r_[fraction * xy, 0.]
        poses[i] = (poses[i][0] + shift, poses[i][1])
        item['object_position'] = (np.asarray(item['object_position']) + shift).tolist()
    poses[-1] = (poses[-1][0] + np.r_[xy, 0.], poses[-1][1])
    plan['support_translation_xy_m'] = xy.tolist()
    plan['translation_hypothesis'] = 'Quintic horizontal displacement while tilting; support must slide or roll physically.'
    return poses, plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events']
                     if e['object'] == 'plate' and e['phase'] == 'lift')
        times = f['observation/sim_time'][:]
        i = int(np.argmin(abs(times - event['sim_time'])))
        if abs(times[i] - event['sim_time']) > 1e-8:
            raise ValueError('Exact lift boundary required')
        extra = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    trials = []
    try:
        layout = meta['task_scene']['layout']
        b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items():
            b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extra['objects'].items():
            adr = b.model.joint('task/' + name + '/free').qposadr[0]
            b.data.qpos[adr:adr+7] = pose['position'] + pose['quaternion_wxyz']
        mujoco.mj_forward(b.model, b.data)
        t = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
        t.q = np.asarray(extra['controller_interval']['command'])
        t.current, t.phase = 'plate', 'lift'
        t.track_grasp('right', 'plate')
        initial, command = state(b), t.q.copy()
        hp, hr = t.tcp('right')
        obj = b.data.body('task/plate')
        translations = [(0., 0.)] + [(x, y) for d in (.03, .06)
            for x, y in ((d, 0.), (-d, 0.), (0., d), (0., -d))]
        translations += [(x, y) for x in (-.045, .045) for y in (-.045, .045)]
        for angle in (45., 60., 75.):
            profile = validate_profile({'angle_deg': angle, 'angle_step_deg': 10., 'support_gap_m': .0005,
                'clear_lift_m': .06, 'path_speed_m_s': .04, 'max_grasp_rotation_rad': .15})
            for xy in translations:
                poses, plan = translated_path(obj.xpos, obj.xmat.reshape(3, 3), hp, hr,
                                              b.task.surface_points['plate'], profile, xy)
                row = {'profile': profile, 'plan': plan, 'passed': False, 'stage': 'final_pose_ik'}
                try:
                    t.solve('right', *poses[-1])
                    row['stage'] = 'full_path'
                    diagnostics = {}
                    path = t.plan_pose_path('right', poses, speed=profile['path_speed_m_s'], diagnostics=diagnostics)
                    row.update(passed=True, command_samples=len(path), duration_s=len(path)/t.fps,
                               diagnostics=diagnostics)
                except SortFailure as exc:
                    row['error'] = str(exc)
                if not np.array_equal(initial, state(b)) or not np.array_equal(command, t.q):
                    raise RuntimeError('Isolated geometric query changed integration state or command')
                trials.append(row)
                print(json.dumps({'angle_deg': angle, 'translation_xy_m': xy,
                                  **{k: v for k, v in row.items() if k not in ('profile', 'plan')}}), flush=True)
    finally:
        b.close()
    if sha256(args.source) != digest:
        raise ValueError('Source changed')
    atomic_json(args.output, {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
        'script_sha256': sha256(__file__), 'training_ready': False,
        'isolated_state_and_commands_unchanged': True, 'trials': trials})


if __name__ == '__main__':
    main()
