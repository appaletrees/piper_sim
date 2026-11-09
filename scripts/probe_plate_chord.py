#!/usr/bin/env python3
"""Isolated top-down chord-grasp screening, never a physical or training replay.

Uses the source's original plate, gripper and basket collision geometry. A
bilateral collision is only a geometric condition, not evidence of force or
stable carrying. A rigid, non-slipping lift is an explicit planning hypothesis.
Restores every isolated integration state after each candidate.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.collision import robot_penetration
from piper_sim.experts.grasp_assist import finger_contacts
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession
from probe_supported_lowering import integration_state
import probe_flat_plate as flat


PROFILE = {'radii_m': [.090, .095], 'azimuths_deg': [0, 45, -45],
           'tilts_deg': [30, 45], 'heights_m': [.012],
           'open_width_m': .095, 'lift_m': .15,
           'flat_placement_budget': 96, 'max_placement_queries': 4}


def restore(b, state):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(b.model, b.data, state, kind)
    mujoco.mj_forward(b.model, b.data)
    mujoco.mj_setState(b.model, b.data, state, kind)
    if not np.array_equal(state, integration_state(b.model, b.data)):
        raise RuntimeError('Failed to restore isolated integration state')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events']
                     if e['object']=='plate' and e['phase']=='approach')
        times = f['observation/sim_time'][:];i = int(np.argmin(abs(times-event['sim_time'])))
        if abs(times[i]-event['sim_time'])>1e-8:raise ValueError('Exact approach snapshot missing')
        extra = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
              'source_index': i, 'script_sha256': sha256(__file__), 'profile': PROFILE,
              'training_ready': False, 'candidates': [], 'placement_queries': 0}
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extra['objects'].items():
            a = b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[a:a+7] = pose['position']+pose['quaternion_wxyz']
        mujoco.mj_forward(b.model, b.data)
        t = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
        t.q = np.asarray(extra['controller_interval']['command']);t.current = 'plate';t.phase = 'grasp'
        t.configure_object_contact('plate')
        state = integration_state(b.model, b.data);command = t.q.copy()
        object_p = b.data.body('task/plate').xpos.copy();object_r = b.data.body('task/plate').xmat.reshape(3, 3).copy()
        original = t.plan_grasp
        qids = b.indices['right']['qpos'];adr = b.model.joint('task/plate/free').qposadr[0]
        for azimuth in PROFILE['azimuths_deg']:
            for tilt in PROFILE['tilts_deg']:
                for height in PROFILE['heights_m']:
                    for radius in PROFILE['radii_m']:
                        row = {'azimuth_deg': azimuth, 'tilt_deg': tilt, 'height_m': height,
                               'radius_m': radius, 'stage': 'grasp_ik', 'error': None}
                        result['candidates'].append(row)
                        a = np.deg2rad(azimuth)
                        local = np.array([-radius*np.cos(a), -radius*np.sin(a), height])
                        position = object_p+object_r@local
                        rotation = object_r@grasp_rotation(tilt, azimuth)
                        t.plan_grasp = lambda name: ('right', position, rotation, PROFILE['open_width_m']) if name=='plate' else original(name)
                        try:
                            # Cheap geometry first; do not spend path budgets on
                            # grasps that cannot meet two fingers or hit the desk.
                            q = t.solve('right', position, rotation)
                            d = t.ik['right'].data;row['stage'] = 'closing_geometry'
                            bilateral = None
                            for width in np.linspace(PROFILE['open_width_m'], 0., 191):
                                d.qpos[:] = b.data.qpos;d.qpos[qids[:6]] = q;d.qpos[qids[6:]] = width/2
                                mujoco.mj_forward(b.model, d)
                                penetration, pair = robot_penetration(b.model, d, 'plate')
                                if penetration>t.settings['quality']['max_robot_penetration_m']:
                                    raise SortFailure(f'Closing non-target obstruction {penetration:.6f} m: {pair}')
                                if len(finger_contacts(SimpleNamespace(model=b.model, data=d), 'right', 'plate'))==2:
                                    bilateral = float(width);break
                            if bilateral is None:raise SortFailure('No bilateral geometry in opening range')
                            row['first_bilateral_width_m'] = bilateral
                            row['stage'] = 'approach_path'
                            # Hold the named candidate fixed, without implicit
                            # local tilt/height substitutions from reachable_grasp.
                            yaw = np.rad2deg(np.arctan2(position[1]+.3, position[0]))
                            above = t.solve('right', position+[0, 0, .15], grasp_rotation(30, yaw))
                            t.check_joint_path('right', t.q[7:13], above,
                                gripper=(t.q[13], PROFILE['open_width_m']),
                                transit_clearance=t.pacing.get('unloaded_transit_clearance_m', 0.))
                            pre = t.solve('right', t.approach_position('plate', position, rotation, .035), rotation, above)
                            t.check_joint_path('right', above, pre, gripper=(PROFILE['open_width_m'], PROFILE['open_width_m']))
                            start = t.q.copy();start[7:13] = pre;start[13] = PROFILE['open_width_m']
                            t.plan_pose_path('right', [(position, rotation)], start=start, speed=t.pacing['contact_speed_m_s'])
                            row['stage'] = 'rigid_lift_hypothesis'
                            b.data.qpos[qids[:6]] = q;b.data.qpos[qids[6:]] = bilateral/2
                            mujoco.mj_forward(b.model, b.data)
                            t.q[7:13] = q;t.q[13] = bilateral;t.track_grasp('right', 'plate')
                            tcp, hand = t.tcp('right');off = hand.T@(object_p-tcp);rel = hand.T@object_r
                            # Match the existing teacher's two joint-space lift
                            # legs, including its full 150 mm target height.
                            short = t.solve('right', position+[0, 0, .035], rotation, q)
                            lifted = t.solve('right', position+[0, 0, PROFILE['lift_m']], rotation, short)
                            t.check_joint_path('right', q, short)
                            t.check_joint_path('right', short, lifted)
                            b.data.qpos[qids[:6]] = lifted
                            mujoco.mj_kinematics(b.model, b.data)
                            lp, lr = t.tcp('right')
                            b.data.qpos[adr:adr+3] = lp+lr@off
                            quat = np.empty(4);mujoco.mju_mat2Quat(quat, (lr@rel).reshape(9))
                            b.data.qpos[adr+3:adr+7] = quat
                            mujoco.mj_forward(b.model, b.data)
                            t.q[7:13] = lifted;t.phase = 'transfer'
                            row['stage'] = 'grasp_geometry_feasible'
                            row['ideal_lift'] = {'assumption': 'Rigid non-slipping grasp, not physical evidence',
                                                 'grasp_position_world': position.tolist(), 'grasp_rotation_world': rotation.tolist(),
                                                 'lift_joint_waypoints': [q.tolist(), short.tolist(), lifted.tolist()]}
                            if result['placement_queries']<PROFILE['max_placement_queries']:
                                result['placement_queries'] += 1
                                diag = {};row['placement'] = diag
                                flat.PROFILE = {**flat.PROFILE, 'max_candidates': PROFILE['flat_placement_budget'], 'release_width_m': .095}
                                try:
                                    flat.search_flat_plate(t, diag)
                                    row['stage'] = 'coupled_geometry_feasible'
                                except SortFailure as exc:diag['error'] = str(exc)
                        except SortFailure as exc:row['error'] = str(exc)
                        finally:
                            restore(b, state);t.q = command.copy();t.carried = None;t.phase = 'grasp'
                            row['isolated_state_restored'] = True
                        print(json.dumps({k:v for k,v in row.items() if k not in ('placement','ideal_lift')}), flush=True)
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    result['source_unchanged'] = True
    atomic_json(args.output, result)


if __name__ == '__main__':main()
