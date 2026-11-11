#!/usr/bin/env python3
"""Isolated grasp screening inspired by the real demonstration, not dynamics replay.

The real video has no calibrated object poses. Candidate offsets remain explicit
hypotheses. Collision onset and a rigid lift assumption only screen geometry;
fresh physical grasps and unchanged quality checks are still required.
"""
import argparse
import copy
import json
from itertools import product
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.collision import robot_penetration
from piper_sim.experts.geometric_capture import closure_target
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession
from probe_grasp_aperture import state
from probe_supported_rim_grasps import reject_open_target_intersection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offset-x', type=float, nargs='+', default=[0., -.015, .015])
    p.add_argument('--offset-y', type=float, nargs='+', default=[-.075, -.085, -.065])
    p.add_argument('--heights', type=float, nargs='+', default=[.014, .010, .018])
    p.add_argument('--rotations', type=float, nargs='+', default=[28., 121., 125., 25., 90., 90., 35., 90., 90.],
                   help='Triples of tilt, yaw, roll in degrees')
    p.add_argument('--opening', type=float, default=.04)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if len(args.rotations) % 3 or not .015 <= args.opening <= .095: p.error('Invalid rotation triples or aperture')
    digest = sha256(args.source)
    with h5py.File(args.source) as f:
        if not f.attrs['complete']: raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate' and e['phase']=='approach')
        i = int(np.argmin(abs(f['observation/sim_time'][:]-event['sim_time'])))
        extra = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    rows = []
    try:
        layout=meta['task_scene']['layout']; b.reset(layout['seed'], layout['randomized'])
        for side, arm in extra['teacher']['physics']['arms'].items(): b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extra['objects'].items():
            a=b.model.joint('task/'+name+'/free').qposadr[0]; b.data.qpos[a:a+7] = pose['position']+pose['quaternion_wxyz']
        mujoco.mj_forward(b.model,b.data)
        t=RigidTableTeacher(SimulationSession(b), settings=meta['teacher']['motion_and_quality'])
        t.current, t.phase = 'plate', 'approach'
        initial=state(b); command=np.asarray(extra['controller_interval']['command']); t.q=command.copy()
        ids=b.indices['right']['qpos']; op=b.data.body('task/plate').xpos.copy(); ort=b.data.body('task/plate').xmat.reshape(3,3).copy()
        for offset, angles in product(product(args.offset_x,args.offset_y,args.heights), np.asarray(args.rotations).reshape(-1,3)):
            row={'offset_m':list(offset),'tilt_yaw_roll_deg':angles.tolist(),'opening_m':args.opening,'passed':False,'stage':'capture'}; rows.append(row)
            pos=op+ort@offset; rot=ort@grasp_rotation(*angles)
            try:
                q=t.solve('right',pos,rot)
                b.data.qpos[ids[:6]]=q; b.data.qpos[ids[6:]]=args.opening/2; mujoco.mj_forward(b.model,b.data)
                cap=closure_target(b.model,b.data,t.ik['right'].data,jaw_qpos=ids[6:],
                    finger_bodies=[b.model.body('right/link'+str(i)).id for i in (7,8)],object_body=b.model.body('task/plate').id,
                    profile={'scan_step_m':.0001,'overtravel_m':.0003})
                row['capture']=cap
                if cap['already_bilateral_at_start']: raise SortFailure('Already bilateral at open target')
                depth,pair=robot_penetration(b.model,t.ik['right'].data,'plate')
                if depth>t.settings['quality']['max_robot_penetration_m']: raise SortFailure(f'Closing obstruction {depth:.6f}: {pair}')
                mujoco.mj_setState(b.model,b.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION); mujoco.mj_forward(b.model,b.data)
                row['stage']='approach'
                yaw=np.rad2deg(np.arctan2(pos[1]+.3,pos[0])); above=t.solve('right',pos+[0,0,.15],grasp_rotation(30,yaw))
                t.check_joint_path('right',command[7:13],above,gripper=(command[13],args.opening),transit_clearance=t.pacing.get('unloaded_transit_clearance_m',0.))
                pre=t.solve('right',t.approach_position('plate',pos,rot,.035),rot,above)
                t.check_joint_path('right',above,pre,gripper=(args.opening,args.opening))
                start=command.copy(); start[7:13]=pre; start[13]=args.opening
                b.data.qpos[ids[6:]]=args.opening/2
                descent=t.plan_pose_path('right',[(pos,rot)],start=start,speed=t.pacing['contact_speed_m_s'])
                above_cmd=start.copy(); above_cmd[7:13]=above
                reject_open_target_intersection(t,np.vstack([above_cmd,start,descent]))
                row['stage']='rigid_lift_hypothesis'
                q=descent[-1,7:13]
                short=t.solve('right',pos+[0,0,.035],rot,q); high=t.solve('right',pos+[0,0,.15],rot,short)
                t.check_joint_path('right',q,short); t.check_joint_path('right',short,high)
                row.update(passed=True,grasp_joint=q.tolist(),position_world_m=pos.tolist(),rotation_world=rot.tolist())
            except (SortFailure,RuntimeError,ValueError) as exc: row['error']=str(exc)
            finally:
                mujoco.mj_setState(b.model,b.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION); mujoco.mj_forward(b.model,b.data)
                mujoco.mj_setState(b.model,b.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
                t.q=command.copy()
                if not np.array_equal(initial,state(b)): raise RuntimeError('Isolated state changed')
            print(json.dumps({k:v for k,v in row.items() if k not in ('rotation_world','grasp_joint','capture')}),flush=True)
    finally: b.close()
    if sha256(args.source)!=digest: raise ValueError('Source changed')
    atomic_json(args.output,{'scope':__doc__,'source':str(args.source),'source_sha256':digest,
        'script_sha256':sha256(__file__),'source_unchanged':True,'training_ready':False,'candidates':rows})


if __name__=='__main__': main()
