#!/usr/bin/env python3
"""Isolated dual-arm regrasp geometry; no dynamics or training replay.

The plate stays at the recorded measured pose in scratch data. Checks candidate
left pinches near the COM while the original right rim grasp remains present.
Neither bilateral geometry nor clearance proves a stable handover.
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
from piper_sim.collection_jobs import atomic_json,sha256
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.collision import robot_penetration
from piper_sim.experts.geometric_capture import closure_target
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.rim_transport import yaw_rotation
from piper_sim.simulation import SimulationSession
from probe_grasp_aperture import state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shift-y',type=float,default=0.,help='Explicit isolated shared-workspace translation hypothesis')
    p.add_argument('--yaw',type=float,default=0.,help='Explicit isolated world-yaw hypothesis')
    p.add_argument('--snapshot',choices=('final','transfer'),default='final')
    p.add_argument('--orientation-neighborhood-deg',type=float,default=0.,
                   help='Propose achieved IK frames within this orientation neighborhood, then test actual geometry')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not np.isfinite(args.shift_y) or not 0 <= args.shift_y <= .16:raise ValueError('Expected shift within 0..160 mm')
    if not np.isfinite(args.yaw) or abs(args.yaw)>135:raise ValueError('Expected yaw within +/-135 degrees')
    if not np.isfinite(args.orientation_neighborhood_deg) or not 0 <= args.orientation_neighborhood_deg <= 15:
        raise ValueError('Expected candidate orientation neighborhood within 0..15 degrees')
    digest=sha256(args.source)
    with h5py.File(args.source) as f:
        if not f.attrs['complete']:raise ValueError('Closed recording required')
        m=json.loads(f.attrs['metadata'])
        if args.snapshot=='final':e=json.loads(f['next_observation/extras_json'][-1])
        else:
            ev=next(v for v in m['collection_outcome']['teacher_events'] if v['object']=='plate' and v['phase']=='transfer')
            i=int(np.argmin(abs(f['observation/sim_time'][:]-ev['sim_time'])))
            e=json.loads(f['observation/extras_json'][i])
    c=copy.deepcopy(m['config']);c['task_scene']['domain_randomization']=m['task_scene'].get('domain_profile')
    b=MujocoBackend(c,cameras=False);rows=[]
    try:
        layout=m['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
        for side,a in e['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']]=a['joint_position']
        for name,po in e['objects'].items():
            adr=b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7]=po['position']+po['quaternion_wxyz']
        mujoco.mj_forward(b.model,b.data)
        t=RigidTableTeacher(SimulationSession(b),settings=m['teacher']['motion_and_quality'])
        t.q=np.asarray(e['controller_interval']['command']);t.current='plate'
        if args.shift_y or args.yaw:
            hp,hr=t.tcp('right');po=b.data.body('task/plate');op,orr=po.xpos.copy(),po.xmat.reshape(3,3).copy()
            delta=yaw_rotation(args.yaw);new=op+[0,args.shift_y,0]
            q=t.solve('right',new+delta@(hp-op),delta@hr)
            b.data.qpos[b.indices['right']['qpos'][:6]]=q
            adr=b.model.joint('task/plate/free').qposadr[0];b.data.qpos[adr:adr+3]=new
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,(delta@orr).reshape(9));b.data.qpos[adr+3:adr+7]=quat
            mujoco.mj_forward(b.model,b.data);t.q[7:13]=q
        initial=state(b);command=t.q.copy();obj=b.data.body('task/plate')
        op,orr=obj.xpos.copy(),obj.xmat.reshape(3,3).copy();normal=orr[:,2]
        down=np.array([0.,0.,-1.]);down-=normal*(normal@down);down/=np.linalg.norm(down)
        tangent=np.cross(normal,down)
        ids=b.indices['left']['qpos'];scratch=t.ik['left'].data
        for (dx,dy),angle,sign in product([(0.,0.),(.025,0.),(-.025,0.),(0.,.025),(0.,-.025)], [0.,45.,90.,-90.],[1.,-1.]):
            theta=np.deg2rad(angle);z=np.cos(theta)*down+np.sin(theta)*tangent;y=normal*sign;x=np.cross(y,z)
            rot=np.column_stack((x,y,z));pos=op+orr@[dx,dy,.008]
            row={'local_xy_m':[dx,dy],'in_plane_approach_deg':angle,'jaw_normal_sign':sign,'passed':False,'stage':'ik'};rows.append(row)
            try:
                try:q=t.solve('left',pos,rot)
                except SortFailure:
                    if not args.orientation_neighborhood_deg:raise
                    seeds=[t.q[:6], b.task.config['reference']['joint14_rad_fraction'][:6]]
                    yaw=np.arctan2(pos[1]-.3,pos[0])
                    seeds += [[yaw,1.8,-1.,0.,-.8,roll] for roll in (-1.57,1.57,0.)]
                    candidates=[t.ik['left'].solve(pos,rot,s) for s in seeds]
                    q,ep,er=min(candidates,key=lambda a:a[1]+.1*a[2])
                    if ep>.003 or er>np.deg2rad(args.orientation_neighborhood_deg):raise
                    # This proposes a DIFFERENT achieved frame. It does not
                    # certify the unreachable requested frame, lower any live
                    # IK/quality tolerance, or execute a command. Future paths
                    # must target and check the actual proposed frame exactly.
                    pos,rot=t.ik['left'].forward(q)
                    row['neighbor_proposal']={'position_offset_m':ep,'orientation_offset_deg':float(np.rad2deg(er)),
                        'scope':'Achieved neighboring frame, not acceptance of the original requested pose'}
                # Only independent scratch is assigned. The right hand and all
                # object poses retain their original measured values.
                scratch.qpos[:]=b.data.qpos;scratch.qpos[ids[:6]]=q;scratch.qpos[ids[6:]]=.0125
                mujoco.mj_forward(b.model,scratch)
                row['stage']='open_collision'
                depth,pair=robot_penetration(b.model,scratch,'plate')
                if depth>t.settings['quality']['max_robot_penetration_m']:raise SortFailure(f'Obstruction {depth:.6f} m: {pair}')
                # closure_target requires a separate read-only source MjData.
                scan=mujoco.MjData(b.model)
                row['stage']='capture'
                cap=closure_target(b.model,scratch,scan,jaw_qpos=ids[6:],finger_bodies=[b.model.body('left/link'+str(j)).id for j in (7,8)],object_body=b.model.body('task/plate').id,profile={'scan_step_m':.0001,'overtravel_m':.0003})
                if cap['already_bilateral_at_start']:raise SortFailure('Already bilateral at 25 mm opening')
                depth,pair=robot_penetration(b.model,scan,'plate')
                if depth>t.settings['quality']['max_robot_penetration_m']:raise SortFailure(f'Closing obstruction {depth:.6f} m: {pair}')
                row.update(passed=True,joint=q.tolist(),position_world_m=pos.tolist(),rotation_world=rot.tolist(),capture=cap)
            except (SortFailure,RuntimeError) as exc:row['error']=str(exc)
            if not np.array_equal(initial,state(b)) or not np.array_equal(command,t.q):raise RuntimeError('Isolated source state changed')
            print(json.dumps({k:v for k,v in row.items() if k not in ('rotation_world','joint','capture')}),flush=True)
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    atomic_json(args.output,{'scope':__doc__,'training_ready':False,'source':str(args.source),'source_sha256':digest,
        'script_sha256':sha256(__file__),'shift_y_m':args.shift_y,
        'yaw_deg':args.yaw,
        'snapshot':args.snapshot,
        'orientation_neighborhood_deg':args.orientation_neighborhood_deg,
        'translation_scope':'Explicit rigid-grasp geometric hypothesis only; not a physically reached state or dynamics replay.',
        'candidates':rows})


if __name__=='__main__':main()
