#!/usr/bin/env python3
"""Isolated placement trial initialized from a recorded transfer snapshot.

Initialization restores native arm state and object poses once. Object velocities
and solver warm-start are unavailable and reset, so this is not exact replay.
Subsequent motion uses motor control and physical contact. No dataset is written
and no result grants training acceptance; full fresh rollouts remain required.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import yaml

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.simulation import SimulationSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--object', required=True, choices=['chopstick_a', 'chopstick_b', 'side_dish'])
    p.add_argument('--profile', default='configs/collection/clean_table_placement_search.yaml', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.output.exists(): p.error('Retain existing diagnostics; select a new output')
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'): raise ValueError('Closed complete recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events']
                     if e['phase']=='transfer' and e['object']==args.object)
        times = f['observation/sim_time'][:]
        i = int(np.argmin(abs(times-event['sim_time'])))
        extra = json.loads(f['observation/extras_json'][i])
        time = float(times[i])
        interval = extra.get('controller_interval')
        command = np.array(interval['command']) if interval else f['action/position'][i-1].astype(float)
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    settings = yaml.safe_load(args.profile.read_text())
    b = MujocoBackend(cfg, cameras=False)
    result = {'scope': __doc__, 'training_ready': False, 'source': str(args.source.resolve()),
              'source_sha256': sha256(args.source), 'source_frame': i, 'snapshot_sim_time': time,
              'profile_sha256': sha256(args.profile), 'script_sha256': sha256(__file__),
              'executed': args.execute, 'error': None,
              'expert_source_sha256': {str(path): sha256(path) for path in Path('src/piper_sim/experts').glob('*.py')},
              'candidate_settings': settings}
    teacher = None
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items():
            b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
            b.data.qvel[b.indices[side]['dof']] = arm['joint_velocity']
        for name, pose in extra['objects'].items():
            adr = b.model.joint('task/'+name+'/free').qposadr[0]
            b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
        for side, off in [('left', 0), ('right', 7)]: b.data.ctrl[b.indices[side]['ctrl']] = command[off:off+7]
        b.data.time = time
        if b.reference_controller:
            b.reference_controller.reset(command)
            if interval: b.reference_controller.previous_velocity = np.array(interval['last_velocity_reference'])
        mujoco.mj_forward(b.model, b.data)
        teacher = RigidTableTeacher(SimulationSession(b), settings=settings)
        teacher.q = command.copy();teacher.current = args.object;teacher.phase = 'transfer'
        teacher.configure_object_contact(args.object)
        teacher.track_grasp('right', args.object)
        # Reproduce the production intermediate pose for a rim-held small dish.
        if args.object=='side_dish': teacher.pose('right', [.38, -.20, .32], grasp_rotation(45, 15))
        try:
            choice = teacher.placement('right', args.object, 'basket')
        except SortFailure:
            if args.object=='side_dish': raise
            teacher.pose('right', [.38, -.20, .32], grasp_rotation(45, 15))
            choice = teacher.placement('right', args.object, 'basket')
        score, high, low, rotation, q_high, q_low = choice
        result['choice'] = {'score': float(score), 'high': high.tolist(), 'low': low.tolist()}
        if args.execute:
            teacher.move_joints('right', q_high)
            teacher.event('lower');teacher.move_joints('right', q_low)
            teacher.releasing = ('right', args.object);teacher.carried = teacher.armed = None
            release_width = .04 if args.object.startswith('chopstick') else .07
            measured = float(b.data.qpos[b.indices['right']['qpos'][6:]].sum())
            if measured-teacher.q[13] > teacher.release_preload_m:
                teacher.event('unload_gripper');teacher.grip('right', min(release_width, measured+.002))
            teacher.event('release');teacher.grip('right', release_width);teacher.releasing = None
            teacher.wait(teacher.pacing['release_hold_s'])
            teacher.pose_path('right', [(high, rotation)])
            teacher.wait(teacher.pacing['placement_hold_s'])
            result['placement_metric'] = teacher.session.read().extras['task']['metrics'][args.object]
            result['final_pose'] = teacher.object_pose(args.object)
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)
    finally:
        if teacher:
            result.update(events=teacher.events, motion=teacher.quality.report(), grasps=teacher.grasps,
                          final_sim_time=b.sim_time)
        b.close()
        atomic_json(args.output, result)
    print(json.dumps({k: result.get(k) for k in ('error', 'placement_metric', 'choice')}, indent=2))
    return 0 if result['error'] is None else 2


if __name__ == '__main__': raise SystemExit(main())
