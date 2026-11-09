#!/usr/bin/env python3
"""Separate plate grasp rotation from the placement orientation actually selected.

Closed-recording pose evidence and an isolated query of the existing planner;
not a physical replay, calibrated grasp tolerance or training acceptance gate.
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
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession
from probe_supported_lowering import integration_state


def angle_degrees(rotation):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(rotation)-1)/2, -1., 1.))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        events = [e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate']
        times = f['observation/sim_time'][:]
        event_indices = {}
        for event in events:
            if event['phase'] in ('lift', 'transfer', 'lower', 'support_lower', 'release'):
                i = int(np.argmin(abs(times-event['sim_time'])))
                if abs(times[i]-event['sim_time'])>1e-8:raise ValueError('Missing exact event snapshot')
                event_indices[i] = event['phase']
        records = [(i, float(times[i]), json.loads(raw)) for i, raw in enumerate(f['observation/extras_json'])]
    cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
              'script_sha256': sha256(__file__), 'training_ready': False, 'events': [], 'held_samples': 0}
    anchor = None;peak = None
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        for i, time, extras in records:
            tdata = extras['teacher']
            if tdata['object']!='plate':continue
            grasp = tdata.get('grasp')
            if not grasp and i not in event_indices:continue
            for side, arm in tdata['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
            for name, pose in extras['objects'].items():
                adr = b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
            mujoco.mj_kinematics(b.model, b.data)
            object_r = b.data.body('task/plate').xmat.reshape(3, 3).copy()
            relative = b.data.body('right/link6').xmat.reshape(3, 3).T@object_r
            if anchor is None and grasp:
                anchor = relative.copy();result['anchor_source_index'] = i
                result['anchor_sim_time'] = time
            row = {'source_index': i, 'sim_time': time, 'phase': event_indices.get(i, tdata['phase']),
                   'object_up_axis_world': object_r[:, 2].tolist(),
                   'object_tilt_from_world_up_deg': float(np.rad2deg(np.arccos(np.clip(object_r[2, 2], -1, 1)))),
                   'relative_rotation_from_first_held_sample_deg': angle_degrees(anchor.T@relative) if anchor is not None else None,
                   'recorded_contact_anchor_slip_m': grasp['slip_m'] if grasp else None,
                   'recorded_contact_fingers': grasp['contact_fingers'] if grasp else None}
            if grasp:
                result['held_samples'] += 1
                if peak is None or row['relative_rotation_from_first_held_sample_deg']>peak['relative_rotation_from_first_held_sample_deg']:
                    peak = row
            if i in event_indices:result['events'].append(row)
            if event_indices.get(i)=='transfer':
                mujoco.mj_forward(b.model, b.data)
                teacher = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
                teacher.q = np.asarray(extras['controller_interval']['command']);teacher.current = 'plate';teacher.phase = 'transfer'
                teacher.configure_object_contact('plate');teacher.track_grasp('right', 'plate')
                state = integration_state(b.model, b.data)
                choice = teacher.placement('right', 'plate', 'basket')
                predicted = choice[3]@relative
                result['original_planner_query'] = {'score': choice[0], 'low_tcp_m': choice[2].tolist(),
                    'predicted_object_up_axis_world': predicted[:, 2].tolist(),
                    'predicted_object_tilt_from_world_up_deg': float(np.rad2deg(np.arccos(np.clip(predicted[2, 2], -1, 1)))),
                    'isolated_state_unchanged': bool(np.array_equal(state, integration_state(b.model, b.data)))}
                if not result['original_planner_query']['isolated_state_unchanged']:raise RuntimeError('Query mutated state')
        result['maximum_held_rotation'] = peak
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
