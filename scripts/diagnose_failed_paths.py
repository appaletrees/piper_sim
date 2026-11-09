#!/usr/bin/env python3
"""Plan from a failed recorded state in a separate kinematic scratch simulator.

Only this diagnostic's scratch joint coordinates are advanced. No physics rollout,
hardware access, grasp proof, source modification, or training acceptance occurs.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.simulation import SimulationSession
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.experts.clean_table import SortFailure
import piper_sim.experts.clean_table as teacher_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.source, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        extras = json.loads(f['next_observation/extras_json'][-1])
        command = f['action/position'][-1].astype(float)
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg)
    try:
        backend.reset(meta['task_scene']['layout']['seed'], meta['task_scene']['layout']['randomized'])
        session = SimulationSession(backend)
        teacher = RigidTableTeacher(session, settings=meta['teacher']['motion_and_quality'])
        teacher.current = extras['teacher']['object']
        for side, arm in extras['teacher']['physics']['arms'].items():
            backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extras['objects'].items():
            address = backend.model.joint('task/'+name+'/free').qposadr[0]
            backend.data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
        backend.data.qvel[:] = 0
        mujoco.mj_forward(backend.model, backend.data)
        initial = backend.data.qpos.copy()
        records = []
        peak = {}
        original = teacher_module.robot_penetration
        def penetration(*args, **kwargs):
            depth, pair = original(*args, **kwargs)
            if depth > peak.get('depth_m', 0):
                peak.update(depth_m=depth, pair=pair)
            return depth, pair
        teacher_module.robot_penetration = penetration
        count = [0]
        def advance_scratch():
            for side, offset in (('left', 0), ('right', 7)):
                q = teacher.q[offset:offset+7]
                backend.data.qpos[backend.indices[side]['qpos']] = [*q[:6], q[6]/2, q[6]/2]
            mujoco.mj_fwdPosition(backend.model, backend.data)
            count[0] += 1
        teacher.step = advance_scratch
        teacher.settle = lambda: None
        def trial(label, action):
            backend.data.qpos[:] = initial
            mujoco.mj_fwdPosition(backend.model, backend.data)
            teacher.q = command.copy()
            peak.clear(); count[0] = 0
            result = {'candidate': label}
            try:
                action()
                result['planned_clear'] = True
            except SortFailure as error:
                result.update(planned_clear=False, error=str(error))
            result.update(peak=copy.deepcopy(peak), scratch_command_samples=count[0])
            records.append(result)
        name = teacher.current
        if name == 'plate':
            def current():
                side, p, r, width = teacher.reachable_grasp(name)
                teacher.contact_approach(side, p, r, name)
            trial('recorded_command_contact_approach', current)
            for height in (0., .002, .004):
                for tilt in (0, -5, 5, -10, 10):
                    def candidate():
                        side, p, r, width = teacher.plan_grasp(name)
                        p[2] += height
                        a = np.deg2rad(tilt)
                        r = r@np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                        teacher.pose(side, teacher.approach_position(name, p, r, .035), r)
                        teacher.contact_approach(side, p, r, name)
                    trial({'height_offset_m': height, 'local_tilt_deg': tilt}, candidate)
        else:
            for rise in (0., .02, .04, .06, .08):
                for via_center in (False, True):
                    def candidate():
                        side, p, r, width = teacher.reachable_grasp(name)
                        if rise:
                            tcp, hand = teacher.ik[side].forward(teacher.q[7:13])
                            teacher.pose_path(side, [(tcp+[0, 0, rise], hand)])
                        if via_center:
                            teacher.pose(side, [.27, -.30, .27], grasp_rotation(30, 0))
                        yaw = np.rad2deg(np.arctan2(p[1]+.3, p[0]))
                        above = teacher.solve(side, p+[0, 0, .15], grasp_rotation(30, yaw))
                        teacher.move_joints(side, above, gripper_width=width)
                        teacher.pose(side, teacher.approach_position(name, p, r, .035), r)
                        teacher.contact_approach(side, p, r, name)
                    trial({'vertical_rise_m': rise, 'via_center': via_center}, candidate)
            for tilt in (0, -5, 5, -10, 10, -15, 15, -20, 20, -30, 30):
                for roll in (0, 180):
                    def candidate():
                        side, p, r, width = teacher.plan_grasp(name)
                        a, z = np.deg2rad([tilt, roll])
                        r = r@np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                        r = r@np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
                        teacher.pose(side, teacher.approach_position(name, p, r, .035), r)
                        teacher.contact_approach(side, p, r, name)
                    trial({'local_tilt_deg': tilt, 'roll_deg': roll}, candidate)
        report = {'status': 'scratch_planning_only', 'source': str(args.source.resolve()),
            'object': name, 'initial_state': 'recorded native eight-joint positions and actual commanded target',
            'assumptions': 'Ideal joint tracking after initial recorded state; other objects remain stationary. No physical contact dynamics.',
            'candidates': records, 'training_ready': False}
    finally:
        if 'original' in locals(): teacher_module.robot_penetration = original
        backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
