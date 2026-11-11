#!/usr/bin/env python3
"""Isolated measured-grasp alignment preflight; never a physical/training replay."""
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
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.rim_alignment import hand_path
from piper_sim.simulation import SimulationSession
from probe_grasp_aperture import state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--angles',type=float,nargs='+',default=[10.,20.,25.])
    p.add_argument('--gap',type=float,default=.0005)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest=sha256(args.source)
    with h5py.File(args.source) as f:
        meta=json.loads(f.attrs['metadata'])
        if not f.attrs['complete']:raise ValueError('Closed recording required')
        event=next(e for e in meta['collection_outcome']['teacher_events'] if e['object']=='plate' and e['phase']=='transfer')
        i=int(np.argmin(abs(f['observation/sim_time'][:]-event['sim_time'])))
        e=json.loads(f['observation/extras_json'][i])
    c=copy.deepcopy(meta['config']); c['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
    b=MujocoBackend(c,cameras=False); rows=[]
    try:
        layout=meta['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
        for side,arm in e['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']]=arm['joint_position']
        for name,po in e['objects'].items():
            adr=b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7]=po['position']+po['quaternion_wxyz']
        mujoco.mj_forward(b.model,b.data)
        t=RigidTableTeacher(SimulationSession(b),settings=meta['teacher']['motion_and_quality'])
        t.q=np.asarray(e['controller_interval']['command']);t.current='plate';t.phase='lift';t.track_grasp('right','plate')
        initial=state(b);command=t.q.copy();obj=b.data.body('task/plate');hp,hr=t.tcp('right')
        points=np.concatenate([np.asarray(part['points']) for part in b.task.parts['plate'] if part.get('collision')])
        for angle in args.angles:
            row={'angle_deg':angle,'passed':False};rows.append(row)
            try:
                poses,plan=hand_path(obj.xpos,obj.xmat.reshape(3,3),hp,hr,points,angle_deg=angle,support_z_m=.025,gap_m=args.gap)
                row['plan']=plan
                t.solve('right',*poses[-1])
                trajectory=t.plan_pose_path('right',poses,speed=.04)
                row.update(passed=True,commands=len(trajectory),duration_s=len(trajectory)/t.fps)
            except (SortFailure,ValueError) as exc:row['error']=str(exc)
            if not np.array_equal(initial,state(b)) or not np.array_equal(command,t.q):raise RuntimeError('Isolated state changed')
            print(json.dumps(row),flush=True)
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    atomic_json(args.output,{'scope':__doc__,'training_ready':False,'source':str(args.source),'source_sha256':digest,
        'script_sha256':sha256(__file__),'geometry_sha256':sha256('src/piper_sim/experts/rim_alignment.py'),'candidates':rows})


if __name__=='__main__':main()
