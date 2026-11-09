#!/usr/bin/env python3
"""Screen the OPEN approach as well as the final chord-grasp pose.

This experimental chord protocol requires open fingers to clear the target
before closing. The usual teacher intentionally allows finger/target contact;
this extra geometric condition is explicit and is not a new global gate.
It prevents relying on an unchanged plate pose after an intersecting descent.
"""
import mujoco

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
import probe_plate_chord as base
import probe_chord_force_feasibility as force


def main():
    original, publish=CleanTableTeacher.plan_pose_path,base.atomic_json
    checks=[]
    profile={'scope':__doc__,'script_sha256':sha256(__file__),
             'penetration_roundoff_tolerance_m':1e-5,
             'tolerance_scope':'Isolated approach geometry diagnostic; not a hardware or production collision limit.',
             'checks':checks}

    def path(t,side,poses,*args,**kwargs):
        trajectory=original(t,side,poses,*args,**kwargs)
        if t.current!='plate' or t.carried is not None or t.phase!='grasp':return trajectory
        model=t.b.model;data=t.ik[side].data;target=model.body('task/plate').id
        fingers={model.body(side+'/link'+str(i)).id for i in (7,8)}
        offset=0 if side=='left' else 7;indices=t.b.indices[side]['qpos']
        row={'command_samples':len(trajectory),'rejected':False,'first_intersection':None}
        checks.append(row)
        for i,command in enumerate(trajectory):
            data.qpos[:]=t.b.data.qpos
            data.qpos[indices[:6]]=command[offset:offset+6]
            data.qpos[indices[6:]]=command[offset+6]/2
            mujoco.mj_forward(model,data)
            for contact in data.contact:
                bodies=[int(model.geom_bodyid[g]) for g in contact.geom]
                if (target in bodies and any(b in fingers for b in bodies)
                        and contact.dist < -profile['penetration_roundoff_tolerance_m']):
                    row['rejected']=True
                    row['first_intersection']={'sample':i,'full_width_m':float(command[offset+6]),
                        'distance_m':float(contact.dist),
                        'geoms':[model.geom(int(g)).name for g in contact.geom],
                        'position_world_m':contact.pos.tolist()}
                    raise SortFailure('Open chord approach intersects target before closing: '
                                      f'{-contact.dist*1000:.3f} mm at sample {i}')
        return trajectory

    try:
        CleanTableTeacher.plan_pose_path=path
        base.atomic_json=lambda destination,value:publish(destination,{**value,'open_approach_screen':profile})
        return force.main()
    finally:
        CleanTableTeacher.plan_pose_path=original;base.atomic_json=publish


if __name__=='__main__':main()
