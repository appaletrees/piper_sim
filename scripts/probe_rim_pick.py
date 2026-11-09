#!/usr/bin/env python3
"""Isolated approach and geometric closing checks for plate rim inset candidates.

Contact geometry is not evidence of grasp force/stability. No physics stepping
or training recording. Original IK, robot and payload collision limits apply.
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
from piper_sim.experts.collision import robot_penetration
from piper_sim.experts.grasp_assist import finger_contacts
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession
from probe_supported_lowering import integration_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate' and e['phase']=='grasp')
        times = f['observation/sim_time'][:];i = int(np.argmin(abs(times-event['sim_time'])))
        if abs(times[i]-event['sim_time'])>1e-8:raise ValueError('No exact grasp snapshot')
        extra = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    result = {'scope': __doc__, 'script_sha256': sha256(__file__), 'source': str(args.source),
              'source_sha256': digest, 'source_index': i, 'sim_time': event['sim_time'],
              'training_ready': False, 'candidates': []}
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extra['objects'].items():
            adr = b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
        mujoco.mj_forward(b.model, b.data)
        t = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
        t.q = np.asarray(extra['controller_interval']['command']);t.current = 'plate';t.phase = 'grasp'
        t.configure_object_contact('plate')
        state = integration_state(b.model, b.data);original = t.plan_grasp
        radial = np.array([-.0687, -.0518, 0.]);unit = -radial/np.linalg.norm(radial)
        for inset in (0., .008, .016):
            def candidate(name):
                side, p, r, width = original(name)
                if name=='plate':p = p+b.data.body('task/plate').xmat.reshape(3, 3)@unit*inset
                return side, p, r, width
            t.plan_grasp = candidate
            row = {'inset_m': inset, 'error': None, 'first_bilateral_geometry': None}
            try:
                side, p, r, width = t.reachable_grasp('plate', check_approach=True)
                q = t.solve(side, p, r)
                row.update(position_m=p.tolist(), rotation=r.tolist(), open_width_m=width)
                d = t.ik[side].data
                for aperture in np.linspace(width, 0, 101):
                    d.qpos[:] = b.data.qpos;d.qpos[b.indices[side]['qpos'][:6]] = q
                    d.qpos[b.indices[side]['qpos'][6:]] = aperture/2
                    mujoco.mj_forward(b.model, d)
                    fingers = sorted(finger_contacts(SimpleNamespace(model=b.model, data=d), side, 'plate'))
                    depth, pair = robot_penetration(b.model, d, 'plate')
                    if len(fingers)==2:
                        row['first_bilateral_geometry'] = {'width_m': float(aperture), 'contact_fingers': fingers,
                            'non_target_penetration_m': depth, 'pair': pair,
                            'non_target_collision_gate_passed': depth<=t.settings['quality']['max_robot_penetration_m']}
                        break
            except Exception as exc:row['error'] = type(exc).__name__+': '+str(exc)
            row['isolated_state_unchanged'] = bool(np.array_equal(state, integration_state(b.model, b.data)))
            if not row['isolated_state_unchanged']:raise RuntimeError('Scratch pick query mutated state')
            result['candidates'].append(row);print(json.dumps(row), flush=True)
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    atomic_json(args.output, result)


if __name__ == '__main__':main()
