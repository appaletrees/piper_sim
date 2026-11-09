#!/usr/bin/env python3
"""Add necessary force-cone conditions to the isolated chord-grasp search.

This rejects provably impossible finger-only support in the reconstructed
geometry. Passing is inconclusive, never a grasp-stability or physics success.
No force limit, friction parameter, collision gate or live state is changed.
"""
import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from audit_grasp_contact_cones import horizontal_separation, upward_force_capacity
import audit_grasp_contact_cones as cones
import probe_plate_chord as base


def main():
    original, publish = CleanTableTeacher.track_grasp, base.atomic_json
    checks = []
    profile = {'scope':__doc__, 'script_sha256':sha256(__file__),
               'cone_diagnostic_sha256':sha256(cones.__file__), 'checks':checks}

    def track(t, side, name):
        original(t, side, name)
        model,data=t.b.model,t.b.data;target=model.body('task/'+name).id
        frames=[];friction=[]
        for c in data.contact:
            bodies=[int(model.geom_bodyid[g]) for g in c.geom]
            if target not in bodies:continue
            i=bodies.index(target)
            if model.body(bodies[1-i]).name not in (side+'/link7',side+'/link8'):continue
            frame=c.frame.reshape(3,3).copy();frame[0]*=-1 if i==0 else 1
            frames.append(frame);friction.append(c.friction[:2].copy() if c.dim>=3 else np.zeros(2))
        if not frames:raise SortFailure('No finger geometry for force-cone screening')
        up=upward_force_capacity(frames,friction);horizontal=horizontal_separation(frames,friction)
        row={'object':name,'gripper_width_m':float(data.qpos[t.b.indices[side]['qpos'][6:]].sum()),
             'tcp_world_m':t.tcp(side)[0].tolist(),'upward_capacity':up,'horizontal_separation':horizontal}
        row['rejected']=(up['all_finger_forces_strictly_downward'] or
                         horizontal['all_finger_cones_strictly_same_horizontal_halfspace'])
        checks.append(row)
        if row['rejected']:
            raise SortFailure('Finger-only static force balance impossible in this geometry: '
                f'max upward/Fn={up["max_over_contacts"]:.6f}, '
                f'horizontal separation={horizontal["min_projection_per_unit_normal_force"]:.6f}')

    try:
        CleanTableTeacher.track_grasp=track
        base.atomic_json=lambda path,value:publish(path,{**value,'necessary_force_screen':profile})
        return base.main()
    finally:
        CleanTableTeacher.track_grasp=original;base.atomic_json=publish


if __name__=='__main__':main()
