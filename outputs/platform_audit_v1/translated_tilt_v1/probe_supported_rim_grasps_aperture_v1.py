#!/usr/bin/env python3
"""Finite isolated rim-grasp/tilt preflight; no dynamics or training replay.

Each candidate uses a fixed plate pose for geometric contact onset, then an
explicit rigid-grasp hypothesis for the supported tilt. All isolated state is
restored between candidates. Passing is only permission for a physical trial.
"""
import argparse
import copy
import json
from pathlib import Path
from itertools import product

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
from piper_sim.experts.supported_tilt import hand_path, validate_profile
from piper_sim.simulation import SimulationSession
from probe_grasp_aperture import state


PROFILE = {'angle_deg': 75., 'angle_step_deg': 10., 'support_gap_m': .0005,
           'clear_lift_m': .06, 'path_speed_m_s': .04, 'max_grasp_rotation_rad': .15}


def reject_open_target_intersection(teacher, commands):
    """Extra preflight gate: no pushing the plate before deliberate closure."""
    b=teacher.b;d=teacher.ik['right'].data;ids=b.indices['right']['qpos']
    target=b.model.body('task/plate').id
    fingers={b.model.body('right/link'+str(i)).id for i in (7,8)}
    for before,after in zip(commands[:-1],commands[1:]):
        count=max(2,int(np.ceil(np.max(abs(after[7:13]-before[7:13]))/.01)),
                  int(np.ceil(abs(after[13]-before[13])/.001)))
        for fraction in np.linspace(0.,1.,count+1):
            q=(1-fraction)*before+fraction*after
            d.qpos[:]=b.data.qpos;d.qpos[ids[:6]]=q[7:13];d.qpos[ids[6:]]=q[13]/2
            mujoco.mj_kinematics(b.model,d);mujoco.mj_collision(b.model,d)
            for c in d.contact:
                bodies=set(b.model.geom_bodyid[c.geom])
                if c.dist < -.00001 and target in bodies and bodies & fingers:
                    raise SortFailure(f'Open approach intersects plate by {-c.dist:.6f} m')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--azimuths', type=float, nargs='+', default=[37., 0., -45., 90.])
    p.add_argument('--grasp-tilts', type=float, nargs='+', default=[45., 60., 75.])
    p.add_argument('--rolls', type=float, nargs='+', default=[90.])
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']: raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        event = next(e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate' and e['phase']=='approach')
        times=f['observation/sim_time'][:];i=int(np.argmin(abs(times-event['sim_time'])))
        if abs(times[i]-event['sim_time'])>1e-8: raise ValueError('Exact approach boundary required')
        extra=json.loads(f['observation/extras_json'][i])
    cfg=copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
    b=MujocoBackend(cfg,cameras=False);trials=[]
    try:
        layout=meta['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
        for side,arm in extra['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']]=arm['joint_position']
        for name,pose in extra['objects'].items():
            a=b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[a:a+7]=pose['position']+pose['quaternion_wxyz']
        mujoco.mj_forward(b.model,b.data)
        t=RigidTableTeacher(SimulationSession(b),settings=meta['teacher']['motion_and_quality'])
        command=np.asarray(extra['controller_interval']['command']);t.q=command.copy();t.current='plate'
        initial=state(b);qids=b.indices['right']['qpos'];op=b.data.body('task/plate').xpos.copy();orr=b.data.body('task/plate').xmat.reshape(3,3).copy()
        for azimuth in args.azimuths:
            for tilt, roll in product(args.grasp_tilts, args.rolls):
                row={'azimuth_deg':azimuth,'grasp_tilt_deg':tilt,'radius_m':float(np.hypot(.0687,.0518)),
                     'height_m':.014,'roll_deg':roll,'opening_m':.025,'stage':'capture_geometry','passed':False}
                trials.append(row)
                a=np.deg2rad(azimuth);pos=op+orr@[-row['radius_m']*np.cos(a),-row['radius_m']*np.sin(a),.014]
                rot=orr@grasp_rotation(tilt,azimuth,roll)
                try:
                    q=t.solve('right',pos,rot)
                    b.data.qpos[qids[:6]]=q;b.data.qpos[qids[6:]]=.0125;mujoco.mj_forward(b.model,b.data)
                    capture=closure_target(b.model,b.data,t.ik['right'].data,jaw_qpos=qids[6:],
                        finger_bodies=[b.model.body('right/link'+str(i)).id for i in (7,8)],
                        object_body=b.model.body('task/plate').id,
                        profile={'scan_step_m':.0001,'overtravel_m':.0003})
                    row['capture']=capture
                    depth,pair=robot_penetration(b.model,t.ik['right'].data,'plate')
                    if depth>t.settings['quality']['max_robot_penetration_m']:
                        raise SortFailure(f'Capture non-target penetration {depth} m: {pair}')
                    if capture['already_bilateral_at_start']:raise SortFailure('Pregrasp already intersects both fingers')
                    mujoco.mj_setState(b.model,b.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(b.model,b.data)
                    row['stage']='approach_path'
                    yaw=np.rad2deg(np.arctan2(pos[1]+.3,pos[0]));above=t.solve('right',pos+[0,0,.15],grasp_rotation(30,yaw))
                    t.check_joint_path('right',command[7:13],above,gripper=(command[13],.025),
                                       transit_clearance=t.pacing.get('unloaded_transit_clearance_m',0.))
                    pre=t.solve('right',t.approach_position('plate',pos,rot,.035),rot,above)
                    t.check_joint_path('right',above,pre,gripper=(.025,.025))
                    start=command.copy();start[7:13]=pre;start[13]=.025
                    # This is a future-state query in an isolated backend.
                    # The descent's scratch checks retain the backend jaw pose;
                    # use its planned 25 mm aperture, not the old bowl-release
                    # aperture at the reconstructed approach boundary.
                    b.data.qpos[qids[6:]]=.0125
                    descent=t.plan_pose_path('right',[(pos,rot)],start=start,speed=t.pacing['contact_speed_m_s'])
                    above_command=command.copy();above_command[7:13]=above;above_command[13]=.025
                    reject_open_target_intersection(t,np.vstack([command,above_command,start,descent]))
                    row['stage']='supported_tilt_hypothesis'
                    # Re-evaluate capture on the actual final IK branch rather
                    # than the independent initial pose-screen solution.
                    q=descent[-1,7:13]
                    b.data.qpos[qids[:6]]=q;mujoco.mj_forward(b.model,b.data)
                    capture=closure_target(b.model,b.data,t.ik['right'].data,jaw_qpos=qids[6:],
                        finger_bodies=[b.model.body('right/link'+str(i)).id for i in (7,8)],
                        object_body=b.model.body('task/plate').id,
                        profile={'scan_step_m':.0001,'overtravel_m':.0003})
                    if capture['already_bilateral_at_start']:raise SortFailure('Final open approach already intersects both fingers')
                    row['capture_after_approach']=capture
                    b.data.qpos[qids[:6]]=q;b.data.qpos[qids[6:]]=capture['bilateral_width_m']/2;mujoco.mj_forward(b.model,b.data)
                    t.q[7:13]=q;t.q[13]=0.;t.phase='lift';t.track_grasp('right','plate')
                    hp,hr=t.tcp('right');obj=b.data.body('task/plate')
                    poses,plan=hand_path(obj.xpos,obj.xmat.reshape(3,3),hp,hr,b.task.surface_points['plate'],validate_profile(PROFILE))
                    path=t.plan_pose_path('right',poses,speed=PROFILE['path_speed_m_s'])
                    row.update(passed=True,plan=plan,command_samples=len(path),duration_s=len(path)/t.fps)
                except (SortFailure,RuntimeError,ValueError) as exc:row['error']=str(exc)
                finally:
                    mujoco.mj_setState(b.model,b.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(b.model,b.data)
                    mujoco.mj_setState(b.model,b.data,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
                    t.q=command.copy();t.carried=None;t.phase='approach'
                    if not np.array_equal(initial,state(b)):raise RuntimeError('Isolated state restoration failed')
                print(json.dumps({k:v for k,v in row.items() if k not in ('plan','capture','capture_after_approach')}),flush=True)
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    atomic_json(args.output,{'scope':__doc__,'source':str(args.source),'source_sha256':digest,
        'script_sha256':sha256(__file__),'profile':PROFILE,'training_ready':False,'trials':trials})


if __name__=='__main__':main()
