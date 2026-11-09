#!/usr/bin/env python3
"""Static lowering feasibility from recorded pre-release poses, never a rollout.

Restores pose geometry on an isolated backend, then queries scratch IK/collision
data only. No physical stepping, data collection, support-force claim or training
acceptance. Object velocity and solver history are not reconstructed.
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
from piper_sim.experts.placement_search import collision_surface_points, support_surface
from piper_sim.simulation import SimulationSession


def integration_state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, result, kind)
    return result


def clearance(teacher, name):
    body = teacher.b.data.body('task/'+name)
    cloud = collision_surface_points(teacher.b.task.parts[name], .005)
    cloud = cloud@body.xmat.reshape(3, 3).T+body.xpos
    support, obstacles = support_surface(teacher, teacher.b.task.specs[name]['target'], [name])
    heights = support.height(cloud[:, :2])
    finite = np.isfinite(heights)
    if not finite.any():
        raise ValueError('No geometric support under payload')
    return float(np.min(cloud[finite, 2]-heights[finite])), obstacles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'):
            raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])
        times = f['observation/sim_time'][:]
        snapshots = []
        for name in ('bowl', 'plate', 'side_dish'):
            event = next(e for e in meta['collection_outcome']['teacher_events']
                         if e['object'] == name and e['phase'] == 'open_gripper')
            index = int(np.argmin(abs(times-event['sim_time'])))
            if abs(times[index]-event['sim_time']) > 1e-8:
                raise ValueError('No exact pre-release snapshot')
            snapshots.append((name, index, float(times[index]), json.loads(f['observation/extras_json'][index])))
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    rows = []
    try:
        for name, index, time, extra in snapshots:
            layout = meta['task_scene']['layout']; b.reset(layout['seed'], layout['randomized'])
            for side, arm in extra['teacher']['physics']['arms'].items():
                b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
            for n, pose in extra['objects'].items():
                adr = b.model.joint('task/'+n+'/free').qposadr[0]
                b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
            command = np.asarray(extra['controller_interval']['command'])
            b.data.time = time
            mujoco.mj_forward(b.model, b.data)
            teacher = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
            teacher.q = command.copy();teacher.current = name;teacher.phase = 'lower'
            teacher.configure_object_contact(name);teacher.track_grasp('right', name)
            before = integration_state(b.model, b.data)
            gap, obstacles = clearance(teacher, name)
            tcp, rotation = teacher.tcp('right')
            width = min(.095, teacher.plan_grasp(name)[3]+.020)
            attempts = []
            for remaining in sorted(set([.001, .005, .01, .02, .04, .06, .08, .10, gap]), reverse=True):
                if remaining > gap+1e-8:
                    continue
                target = tcp+[0, 0, remaining-gap]
                stage, error = 'lowering_path', None
                try:
                    path = teacher.plan_pose_path('right', [(target, rotation)], speed=.06)
                    stage = 'opening_sweep'
                    q = path[-1, 7:13]
                    aperture = float(b.data.qpos[b.indices['right']['qpos'][6:]].sum())
                    widths = np.linspace(aperture, width, max(2, int(np.ceil(abs(width-aperture)/.001))+1))
                    for a, c in zip(widths[:-1], widths[1:]):
                        teacher.check_joint_path('right', q, q, gripper=(a, c))
                    stage = 'feasible_geometry_only'
                except Exception as exc:
                    error = type(exc).__name__+': '+str(exc)
                unchanged = np.array_equal(before, integration_state(b.model, b.data))
                if not unchanged:
                    raise RuntimeError('Scratch feasibility query changed isolated source state')
                row = {'remaining_sampled_gap_m': remaining, 'descent_m': gap-remaining,
                       'stage': stage, 'error': error, 'state_unchanged': unchanged}
                attempts.append(row);print(name, row, flush=True)
            rows.append({'object': name, 'source_index': index, 'sim_time': time,
                         'initial_sampled_gap_m': gap, 'obstacles': obstacles,
                         'release_aperture_m': width, 'attempts': attempts})
    finally:
        b.close()
    unchanged = sha256(args.source) == digest
    atomic_json(args.output, {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
                             'source_unchanged': unchanged, 'script_sha256': sha256(__file__),
                             'results': rows, 'training_ready': False})
    if not unchanged:
        raise ValueError('Source changed')


if __name__ == '__main__':
    main()
