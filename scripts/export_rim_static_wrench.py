#!/usr/bin/env python3
"""Export original native contact frames for a frozen-contact static query.

No contact solving/replay. Only a separate model's kinematics/Jacobians supply
the jaw virtual-work projection. Original recorded forces are comparison data,
not used to solve equilibrium. No training or dynamic grasp acceptance results.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples-per-phase',type=int,default=8)
    p.add_argument('--identity',type=Path,required=True,help='Frozen input manifest of the source rollout')
    p.add_argument('--orientation-sweep',action='store_true',help='Also query a rigidly rotated last-alignment contact hypothesis')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    identity=json.loads(args.identity.read_text());changed=[p for p,h in identity['files'].items() if sha256(p)!=h]
    if changed:raise ValueError('Original frozen inputs changed: '+repr(changed))
    tm=json.loads((args.trace/'trace.json').read_text());source=args.trace/'trace.npz';digest=sha256(source)
    if tm['trace_sha256']!=digest:raise ValueError('Original trace hash mismatch')
    rh=sha256(args.recording)
    with h5py.File(args.recording) as f:
        if not f.attrs['complete']:raise ValueError('Closed recording required')
        meta=json.loads(f.attrs['metadata'])
    c=copy.deepcopy(meta['config']);c['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
    b=MujocoBackend(c,cameras=False);cases=[]
    try:
        layout=meta['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
        if b.finger_contact is None or b.finger_contact.metadata()!=meta['physics_model'].get('finger_contact'):
            raise ValueError('Reset finger material differs from original recorded metadata')
        if [b.model.geom(i).name for i in range(b.model.ngeom)]!=tm['geom_names']:raise ValueError('Geometry map mismatch')
        ids=b.indices['right'];obj=b.model.body('task/plate').id
        payload=next(i for i,r in enumerate(tm['payloads']) if r['name']=='task/plate')
        mass=float(b.model.body_mass[obj])
        if mass!=tm['payloads'][payload]['mass_kg']:raise ValueError('Payload mass differs')
        fingers={b.model.body('right/link'+str(i)).id for i in (7,8)}
        with np.load(source,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
        for phase in ('lift','rim_alignment','transfer'):
            rows=np.flatnonzero((z['phase']==phase)&(z['target']=='plate'))
            if not len(rows):continue
            for i in rows[np.unique(np.linspace(0,len(rows)-1,min(len(rows),args.samples_per_phase),dtype=int))]:
                b.data.qpos[ids['qpos']]=z['qpos_before'][i]
                mujoco.mj_kinematics(b.model,b.data);mujoco.mj_comPos(b.model,b.data)
                com=z['payload_com_world'][i,payload];contacts=[];normal=0.
                start=int(z['contact_offset'][i]);stop=start+int(z['contact_count'][i])
                for j in range(start,stop):
                    geoms=[int(z['contact_geom1'][j]),int(z['contact_geom2'][j])];bodies=b.model.geom_bodyid[geoms]
                    if obj not in bodies:continue
                    obj_side=0 if bodies[0]==obj else 1;finger=int(bodies[1-obj_side])
                    if finger not in fingers:continue
                    fg=geoms[1-obj_side];other=geoms[obj_side]
                    if b.model.geom_priority[fg]<=b.model.geom_priority[other] or b.model.npair:
                        raise ValueError('Material precedence is not covered by this query')
                    mu=b.model.geom_friction[fg];friction=mu[[0,0,1,2,2]]
                    basis=z['contact_frame'][j].reshape(3,3).T*(1 if obj_side==1 else -1)
                    jp,jr=np.zeros((3,b.model.nv)),np.zeros((3,b.model.nv))
                    point=z['contact_position_world'][j]
                    mujoco.mj_jac(b.model,b.data,jp,jr,point,finger)
                    jaw=ids['dof'][6:]
                    # Force on the object is opposite the force on its finger.
                    # Equal jaw displacement dw/2 gives the full-opening tendon
                    # virtual-work projection. Coupling reaction cancels here.
                    projection=np.r_[.5*jp[:,jaw].sum(axis=1)@basis,.5*jr[:,jaw].sum(axis=1)@basis]
                    normal+=max(0.,float(z['contact_force_local'][j,0]))
                    contacts.append({'position_world_m':point.tolist(),'basis_on_object_world':basis.tolist(),
                        'friction':friction.tolist(),'motor_projection_local':projection.tolist(),
                        'geoms':[b.model.geom(g).name for g in geoms],
                        'recorded_distance_m':float(z['contact_distance'][j])})
                cases.append({'label':phase+'/'+str(i),'time_before':float(z['time_before'][i]),
                    'mass_kg':mass,'gravity_world_m_s2':b.model.opt.gravity.tolist(),'com_world_m':com.tolist(),
                    'motor_bias_residual_N':float(.5*(z['qfrc_bias'][i,6:]-z['qfrc_applied'][i,6:]).sum()),
                    'recorded_motor_force_N':float(z['actuator_force'][i,-1]),'recorded_normal_sum_N':normal,
                    'contacts':contacts})
                if args.orientation_sweep and phase=='rim_alignment' and i==rows[-1]:
                    rotation=z['payload_rotation_world'][i,payload].reshape(3,3)
                    up=rotation[:,2];axis=np.cross([0.,0.,1.],up);axis/=np.linalg.norm(axis)
                    k=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
                    for degrees in (-10.,-20.,-30.,-40.):
                        angle=np.deg2rad(degrees);delta=np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)
                        # Always rotate the original native contact set, never
                        # accumulate successive virtual transformations.
                        virtual=copy.deepcopy(next(v for v in cases if v['label']==phase+'/'+str(i)))
                        virtual['label']='rigid_contact_hypothesis/'+str(degrees)
                        virtual['hypothesis']='Rigidly rotate the last measured object/contact manifold; not native or physical evidence.'
                        virtual['tilt_change_deg']=degrees;virtual['time_before']=None
                        virtual.pop('recorded_motor_force_N');virtual.pop('recorded_normal_sum_N')
                        for co in virtual['contacts']:
                            co['position_world_m']=(com+delta@(np.asarray(co['position_world_m'])-com)).tolist()
                            co['basis_on_object_world']=(delta@np.asarray(co['basis_on_object_world'])).tolist()
                        cases.append(virtual)
    finally:b.close()
    if sha256(source)!=digest or sha256(args.recording)!=rh:raise ValueError('Read-only source changed')
    atomic_json(args.output,{'scope':__doc__,'training_ready':False,'source_trace_sha256':digest,
        'recording_sha256':rh,'script_sha256':sha256(__file__),'cases':cases,
        'frozen_input_manifest_sha256':sha256(args.identity),'frozen_inputs_changed':changed,
        'limitations':['Frozen-contact static feasibility only; original dynamic inertial wrench is not included.',
            'All recorded finger contacts, including inactive/margin contacts, may supply ideal reaction: optimistic.',
            'Compliance and force-penetration restrictions omitted; feasibility never certifies a physical grasp.',
            'Higher-priority finger friction reconstructed from the episode configuration/reset seed; original contact frames retained.']})
    print(json.dumps({'cases':len(cases),'output':str(args.output)}))


if __name__=='__main__':main()
