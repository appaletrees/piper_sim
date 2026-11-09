#!/usr/bin/env python3
"""Static, collision-checked container placement offsets from recorded grasps.

No live rollout or data collection. Uses measured grasp geometry and an elevated
XY traverse followed by a vertical descent. A successful query is not proof of
grasp stability, physical support or final containment.
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
from piper_sim.experts.placement_search import collision_surface_points, support_surface, search
from piper_sim.simulation import SimulationSession
from probe_supported_lowering import integration_state
from probe_supported_release import opening_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--orientation-search', action='store_true',
                        help='Also query existing yaw/pitch/slot search with only a 1 mm support gap')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        if not f.attrs['complete']:
            raise ValueError('Closed recording required')
        snapshots = []
        for name in ('bowl', 'side_dish'):
            event = next(e for e in meta['collection_outcome']['teacher_events']
                         if e['object'] == name and e['phase'] == 'open_gripper')
            i = int(np.argmin(abs(f['observation/sim_time'][:]-event['sim_time'])))
            snapshots.append((name, i, json.loads(f['observation/extras_json'][i])))
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    results = []
    try:
        for name, index, extra in snapshots:
            layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
            for side, arm in extra['teacher']['physics']['arms'].items():
                b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
            for n, pose in extra['objects'].items():
                adr = b.model.joint('task/'+n+'/free').qposadr[0]
                b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
            mujoco.mj_forward(b.model, b.data)
            t = RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
            t.q = np.asarray(extra['controller_interval']['command']);t.current = name;t.phase = 'lower'
            t.configure_object_contact(name);t.track_grasp('right', name)
            state = integration_state(b.model, b.data)
            body = b.data.body('task/'+name)
            cloud = collision_surface_points(b.task.parts[name], .005)@body.xmat.reshape(3, 3).T+body.xpos
            support, obstacles = support_surface(t, 'basket', [name])
            tcp, r = t.tcp('right')
            basket = b.data.body('task/basket');br = basket.xmat.reshape(3, 3)
            spec = b.task.specs['basket'];half = np.asarray(spec['size_m'][:2])/2-spec['wall_m']
            rise = max(.04, basket.xpos[2]+spec['size_m'][2]+.025-cloud[:, 2].min())
            attempts = []
            for x, y in ((.04,.04),(.04,0),(0,.04),(-.04,.04),(-.04,0),(.08,0),(0,-.04),(-.08,0),(.04,-.04),(-.04,-.04)):
                shifted = cloud+[x, y, 0]
                local = (shifted-basket.xpos)@br
                stage, error = 'footprint', None
                gap = None
                try:
                    if np.any(abs(local[:, :2]).max(0) > half-.002):
                        raise ValueError('Payload footprint outside basket interior')
                    heights = support.height(shifted[:, :2]);finite = np.isfinite(heights)
                    if not finite.any():
                        raise ValueError('No predicted support')
                    gap = float(np.min(shifted[finite, 2]-heights[finite]))
                    if shifted[:, 2].max()-gap+.001 > basket.xpos[2]+spec['size_m'][2]:
                        raise ValueError('Predicted supported object above rim')
                    stage = 'path'
                    high = tcp+[0,0,rise]
                    low = tcp+[x,y,.001-gap]
                    trajectory = t.plan_pose_path('right', [(high,r),(high+[x,y,0],r),(low,r)], speed=.06)
                    stage = 'opening_sweep'
                    opening_check(t, trajectory[-1,7:13], min(.095,t.plan_grasp(name)[3]+.020))
                    stage = 'feasible_geometry_only'
                except Exception as exc:
                    error = type(exc).__name__+': '+str(exc)
                unchanged = np.array_equal(state, integration_state(b.model,b.data))
                if not unchanged:
                    raise RuntimeError('Scratch query mutated source geometry')
                attempt = {'offset_world_xy_m':[x,y], 'rise_m':rise, 'sampled_initial_gap_m':gap,
                           'stage':stage,'error':error,'state_unchanged':unchanged}
                attempts.append(attempt);print(name,attempt,flush=True)
                if error is None:
                    break
            orientation = None
            if args.orientation_search:
                profile = {'kinds': [b.task.specs[name]['kind']], 'xy_radius_m': .05, 'xy_step_m': .025,
                           'wall_margin_m': .012, 'support_gap_m': .001,
                           'extra_release_heights_m': [0.], 'max_candidates': 216}
                failures = []
                solve, check = t.solve, t.check_joint_path
                def tracked_solve(*a, **kw):
                    try:
                        return solve(*a, **kw)
                    except Exception as exc:
                        failures.append({'stage': 'IK', 'error': str(exc)})
                        raise
                def tracked_check(*a, **kw):
                    try:
                        return check(*a, **kw)
                    except Exception as exc:
                        failures.append({'stage': 'opening_sweep' if 'gripper' in kw else 'joint_path', 'error': str(exc)})
                        raise
                t.solve, t.check_joint_path = tracked_solve, tracked_check
                orientation = {'profile':profile, 'rejections':failures, 'error':None}
                try:
                    selected = search(t, 'right', name, 'basket', profile)
                    orientation['choice'] = {'high':selected[1].tolist(),'low':selected[2].tolist(),
                                             'rotation':selected[3].tolist(),'events':t.events}
                except Exception as exc:
                    orientation['error'] = type(exc).__name__+': '+str(exc)
                finally:
                    t.solve, t.check_joint_path = solve, check
                if not np.array_equal(state, integration_state(b.model,b.data)):
                    raise RuntimeError('Orientation query mutated source geometry')
                print(name,'orientation search',orientation['error'],flush=True)
            results.append({'object':name,'source_index':index,'support_obstacles':obstacles,
                            'attempts':attempts,'orientation_search':orientation})
    finally:
        b.close()
    atomic_json(args.output,{'scope':__doc__,'source':str(args.source),'source_sha256':digest,
                            'source_unchanged':digest==sha256(args.source),'script_sha256':sha256(__file__),
                            'training_ready':False,'results':results})


if __name__ == '__main__':
    main()
