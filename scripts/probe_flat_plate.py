#!/usr/bin/env python3
"""Experimental flat plate placement search, retaining existing collision gates.

CLI performs isolated pose queries only. A candidate is not a physical success.
The search function can be used by an explicitly labelled fresh-rollout probe.
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
from piper_sim.experts.placement_search import collision_surface_points, support_surface, diverse_candidates
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession
from probe_supported_lowering import integration_state
from probe_supported_release import opening_check


PROFILE = {'mode': 'flat_plate_placement_probe_v1', 'support_gap_m': .003,
           'wall_margin_m': .012, 'surface_spacing_m': .005,
           'slot_radius_m': .075, 'slot_step_m': .025, 'yaw_step_deg': 15,
           'max_candidates': 256, 'release_width_m': .045,
           'calibration': 'unidentified experimental planning budgets'}


def search_flat_plate(teacher, diagnostics):
    if teacher.current != 'plate' or teacher.mode != 'physical' or teacher.carried[:2] != ('right', 'plate'):
        raise SortFailure('Flat placement requires an active physical plate grasp')
    b = teacher.b
    tcp, hand_r = teacher.tcp('right')
    obj, basket = b.data.body('task/plate'), b.data.body('task/basket')
    offset = hand_r.T@(obj.xpos-tcp)
    relative = hand_r.T@obj.xmat.reshape(3, 3)
    bin_r, bin_p = basket.xmat.reshape(3, 3).copy(), basket.xpos.copy()
    spec = b.task.specs['basket']
    half = np.asarray(spec['size_m'][:2])/2-spec['wall_m']
    rim = bin_p[2]+spec['size_m'][2]
    nominal = np.asarray(b.task.config['drop_slots_m']['plate'])
    points = collision_surface_points(b.task.parts['plate'], PROFILE['surface_spacing_m'])
    support, obstacles = support_surface(teacher, 'basket', ['plate'])
    candidates = []
    for yaw in range(-180, 180, PROFILE['yaw_step_deg']):
        a = np.deg2rad(yaw)
        rotation = bin_r@np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        cloud = points@rotation.T
        desired = rotation@relative.T
        for x in np.arange(-3, 4)*PROFILE['slot_step_m']:
            for y in np.arange(-3, 4)*PROFILE['slot_step_m']:
                slot = nominal+[x, y]
                margin = float(np.min(half-abs(cloud@bin_r+np.r_[slot, 0])[:, :2].max(0)))
                if margin < PROFILE['wall_margin_m']:
                    continue
                goal = bin_p+bin_r@np.r_[slot, 0]
                score = -margin+.2*np.linalg.norm(slot-nominal)
                candidates.append((score, slot, margin, goal, desired, cloud, yaw))
    diagnostics.update(profile=PROFILE, geometric_candidates=len(candidates), attempts=[], support_obstacles=obstacles)
    for score, slot, margin, goal, desired, cloud, yaw in diverse_candidates(candidates, nominal, tcp, PROFILE['max_candidates']):
        cloud = cloud+goal
        heights = support.height(cloud[:, :2])
        finite = np.isfinite(heights)
        attempt = {'slot_xy_m': slot.tolist(), 'yaw_deg': yaw, 'stage': 'support', 'error': None}
        diagnostics['attempts'].append(attempt)
        if not finite.any():
            attempt['error'] = 'No predicted support';continue
        shift = float(np.max(heights[finite]-cloud[finite, 2]))+PROFILE['support_gap_m']
        if ((cloud+[0, 0, shift]-bin_p)@bin_r)[:, 2].max() > spec['size_m'][2]:
            attempt['error'] = 'Predicted placement above basket rim';continue
        high_object = goal+[0, 0, rim+.035-cloud[:, 2].min()]
        low_object = goal+[0, 0, shift]
        high, low = high_object-desired@offset, low_object-desired@offset
        try:
            attempt['stage'] = 'high_IK'
            q_high = teacher.solve('right', high, desired)
            attempt['stage'] = 'low_IK'
            q_low = teacher.solve('right', low, desired, q_high)
            attempt['stage'] = 'transfer_path'
            teacher.check_joint_path('right', teacher.q[7:13], q_high)
            attempt['stage'] = 'lowering_path'
            teacher.check_joint_path('right', q_high, q_low)
            attempt['stage'] = 'opening_sweep'
            opening_check(teacher, q_low, PROFILE['release_width_m'])
        except SortFailure as exc:
            attempt['error'] = str(exc);continue
        attempt.update(stage='selected_geometry_only', wall_margin_m=margin,
                       high_tcp_m=high.tolist(), low_tcp_m=low.tolist(),
                       hand_rotation_world=desired.tolist(), predicted_object_rotation_world=(desired@relative).tolist())
        return score, high, low, desired, q_high, q_low
    raise SortFailure('No flat plate placement in bounded search; checked='+str(len(diagnostics['attempts'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate' and e['phase']=='transfer')
        times = f['observation/sim_time'][:]
        i = int(np.argmin(abs(times-event['sim_time'])))
        if abs(times[i]-event['sim_time'])>1e-8:raise ValueError('No exact transfer snapshot')
        extra = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
              'script_sha256': sha256(__file__), 'source_index': i, 'sim_time': event['sim_time'], 'error': None, 'training_ready': False}
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extra['objects'].items():
            adr = b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
        mujoco.mj_forward(b.model, b.data)
        t = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
        t.q = np.asarray(extra['controller_interval']['command']);t.current = 'plate';t.phase = 'transfer'
        t.configure_object_contact('plate');t.track_grasp('right', 'plate')
        before = integration_state(b.model, b.data)
        try:search_flat_plate(t, result)
        except SortFailure as exc:result['error'] = str(exc)
        result['isolated_state_unchanged'] = np.array_equal(before, integration_state(b.model, b.data))
        if not result['isolated_state_unchanged']:raise RuntimeError('Scratch search mutated state')
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    atomic_json(args.output, result)
    print(json.dumps({'error': result['error'], 'geometric_candidates': result.get('geometric_candidates'),
                      'checked':len(result['attempts']), 'last_attempt':result['attempts'][-1]}, indent=2))


if __name__ == '__main__':main()
