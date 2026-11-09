#!/usr/bin/env python3
"""Read-only contact-geometry screening of recorded grasps on an isolated model.

Searches horizontal axes separating all reconstructed finger friction cones.
A positive bound excludes static horizontal force balance using fingers alone.
Failure to find an axis is inconclusive: it is not a force-closure certificate.
Geometry reconstruction is not the original solver force or integration trace.
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


def horizontal_separation(frames, friction):
    frames = np.asarray(frames, float);friction = np.asarray(friction, float)
    if (frames.ndim!=3 or frames.shape[1:]!=(3,3) or not len(frames)
            or friction.shape!=(len(frames),2) or not np.isfinite(frames).all()
            or not np.isfinite(friction).all() or np.any(friction<0)):
        raise ValueError('Finite contact frames and nonnegative tangential friction required')
    # Minimum axis projection of fn*n + f1*t1 + f2*t2 under the elliptic
    # tangential bound, divided by positive normal force fn. This ellipse also
    # contains MuJoCo's inscribed pyramidal force edges, so rejection is
    # conservative for that cone; friction torques cannot cancel net force.
    a = np.deg2rad(np.arange(360));axes = np.stack((np.cos(a),np.sin(a),np.zeros_like(a)),axis=1)
    normal = frames[:,0]@axes.T
    tangential = (frames[:,1:]@axes.T)*friction[:,:,None]
    bounds = normal-np.linalg.norm(tangential,axis=1)
    worst = bounds.min(axis=0);index = int(worst.argmax())
    return {'axis_world':axes[index].tolist(), 'min_projection_per_unit_normal_force':float(worst[index]),
            'all_finger_cones_strictly_same_horizontal_halfspace':bool(worst[index]>1e-8),
            'per_contact_bounds':bounds[:,index].tolist(), 'angular_search_step_deg':1}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--object',default='plate')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest=sha256(args.source)
    with h5py.File(args.source,'r') as f:
        if not f.attrs['complete']:raise ValueError('Closed recording required')
        meta=json.loads(f.attrs['metadata']);samples=[]
        for i,raw in enumerate(f['observation/extras_json']):
            e=json.loads(raw);g=e['teacher'].get('grasp')
            if g and g['object']==args.object:samples.append((i,float(f['observation/sim_time'][i]),e))
        i=len(f['action/position'])-1;e=json.loads(f['next_observation/extras_json'][i]);g=e['teacher'].get('grasp')
        if g and g['object']==args.object:samples.append((i+1,float(f['next_observation/sim_time'][i]),e))
    if not samples:raise ValueError('No recorded active grasp samples')
    # Keep contact transitions as well as evenly spaced observations.
    ids=set(np.linspace(0,len(samples)-1,min(12,len(samples)),dtype=int))
    for i in range(1,len(samples)):
        if samples[i][2]['teacher']['grasp']['contact_fingers']!=samples[i-1][2]['teacher']['grasp']['contact_fingers']:
            ids.update((i-1,i))
    cfg=copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
    b=MujocoBackend(cfg,cameras=False)
    out={'scope':__doc__,'script_sha256':sha256(__file__),'source':str(args.source),'source_sha256':digest,
         'training_ready':False,'cone_reference':'https://mujoco.readthedocs.io/en/3.3.7/computation/#contact',
         'sample_count':len(samples),'snapshots':[],
         'limitations':['No solver warm-start, velocity, or force replay; frames/friction are recomputed from recorded native poses.',
                       'A positive separation concerns finger-only static horizontal force equilibrium, not contact with the table or other objects.',
                       'A nonpositive bound is inconclusive; contact torques, COM and dynamic stability are not certified.']}
    try:
        layout=meta['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
        t=RigidTableTeacher(SimulationSession(b),settings=meta['teacher']['motion_and_quality']);t.configure_object_contact(args.object)
        out['model_cone']=int(b.model.opt.cone)
        target=b.model.body('task/'+args.object).id
        for j in sorted(ids):
            index,time,e=samples[j];side=e['teacher']['grasp']['side']
            for s,arm in e['teacher']['physics']['arms'].items():b.data.qpos[b.indices[s]['qpos']]=arm['joint_position']
            for name,pose in e['objects'].items():
                adr=b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7]=pose['position']+pose['quaternion_wxyz']
            mujoco.mj_forward(b.model,b.data)
            rows=[];other=[];frames=[];friction=[]
            for c in b.data.contact:
                bodies=[int(b.model.geom_bodyid[g]) for g in c.geom]
                if target not in bodies:continue
                obj_side=bodies.index(target);other_id=bodies[1-obj_side];body=b.model.body(other_id).name
                if body not in (side+'/link7',side+'/link8'):
                    other.append({'body':body,'distance_m':float(c.dist)});continue
                frame=c.frame.reshape(3,3).copy()
                frame[0]*=-1 if obj_side==0 else 1
                mu=c.friction[:2].copy() if c.dim>=3 else np.zeros(2)
                frames.append(frame);friction.append(mu)
                rows.append({'finger_body':body,'geoms':[b.model.geom(int(g)).name for g in c.geom],
                             'distance_m':float(c.dist),'position_world':c.pos.tolist(),
                             'force_normal_into_object_world':frame[0].tolist(),'sliding_friction':mu.tolist()})
            out['snapshots'].append({'source_state_index':index,'sim_time':time,'phase':e['teacher']['phase'],
                'recorded_grasp':e['teacher']['grasp'],
                'recorded_finger_forces':e['teacher']['physics']['arms'][side]['target_finger_contacts'],
                'reconstructed_contacts':rows,'reconstructed_external_contacts':other,
                'finger_only_separation':horizontal_separation(frames,friction) if rows else None})
    finally:b.close()
    if sha256(args.source)!=digest:raise ValueError('Source changed')
    out['source_unchanged']=True;atomic_json(args.output,out)
    print(json.dumps([{'time':s['sim_time'],'fingers':s['recorded_grasp']['contact_fingers'],
        'contacts':len(s['reconstructed_contacts']),'external_contacts':len(s['reconstructed_external_contacts']),
        'separation':s['finger_only_separation']} for s in out['snapshots']],indent=2))


if __name__=='__main__':main()
