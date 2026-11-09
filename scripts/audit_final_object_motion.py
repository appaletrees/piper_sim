#!/usr/bin/env python3
"""Read-only stability diagnosis, including closed rejected episodes.

Uses recorded 30 Hz object pose differences and the independent production
stability policy. This does not change acceptance or infer native contact forces.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_audit import load_acceptance_policy
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.mujoco_ik import rotation_error
from piper_sim.transforms import pose_matrix


def analyze(source, window_s):
    policy, provenance = load_acceptance_policy()
    limits = policy['dataset']
    with h5py.File(source, 'r') as f:
        if not f.attrs.get('complete'): raise ValueError('A closed recording is required')
        metadata = json.loads(f.attrs['metadata'])
        times = np.r_[f['observation/sim_time'][:], f['next_observation/sim_time'][-1]]
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0): raise ValueError('Invalid source time')
        first = int(np.searchsorted(times, times[-1]-window_s))
        rows = [json.loads(x) for x in f['observation/extras_json'][first:]]+[json.loads(f['next_observation/extras_json'][-1])]
        times = times[first:];dt = np.diff(times)
        if len(dt)<1: raise ValueError('At least two states required')
        required = [name for name in metadata['task_scene']['config']['success']['required'] if name in rows[0]['objects']]
        if len(required)!=rows[-1]['task']['metrics']['summary']['required_count']:
            raise ValueError('Required-object inventory mismatch')
        global_stable = np.array([r['task']['metrics']['summary']['success'] for r in rows[1:]], bool)
        objects = {}
        for name in required:
            positions = np.array([r['objects'][name]['position'] for r in rows])
            quaternions = np.array([r['objects'][name]['quaternion_wxyz'] for r in rows])
            if (not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                    or not np.allclose(np.linalg.norm(quaternions,axis=1),1.,atol=1e-6,rtol=0)):
                raise ValueError('Invalid recorded object pose: '+name)
            velocity = np.linalg.norm(np.diff(positions,axis=0),axis=1)/dt
            angular = 2*np.arccos(np.clip(abs(np.sum(quaternions[:-1]*quaternions[1:],axis=1)),0,1))/dt
            rotations = [pose_matrix(p,q)[:3,:3] for p,q in zip(positions,quaternions)]
            vectors = np.array([rotation_error(b,a)/step for a,b,step in zip(rotations[:-1],rotations[1:],dt)])
            along = np.array([float(np.dot(w,r[:,0])) for w,r in zip(vectors,rotations[:-1])])
            stable = (velocity<=limits['final_object_speed_m_s'])&(angular<=limits['final_object_angular_speed_rad_s'])
            global_stable &= stable
            last_second = times[1:]>=times[-1]-1.
            objects[name] = {'linear_speed_max_m_s':float(velocity.max()),
                'angular_speed_max_rad_s':float(angular.max()),
                'local_x_angular_component_max_rad_s':float(np.max(abs(along))),
                'last_second_unstable_intervals':int(np.count_nonzero(~stable&last_second)),
                'unstable_interval_end_times_s':times[1:][~stable].tolist()}
    longest = current = 0.
    for stable, step in zip(global_stable,dt):
        current = current+step if stable else 0.;longest=max(longest,current)
    return {'scope': __doc__, 'training_ready': False, 'source':str(source.resolve()),
        'source_sha256':sha256(source),'acceptance_policy':provenance,
        'window_start_end_sim_time':[float(times[0]),float(times[-1])], 'samples':len(times),
        'thresholds':{k:limits[k] for k in ('final_stable_hold_s','final_object_speed_m_s','final_object_angular_speed_rad_s')},
        'objects':objects, 'longest_all_contained_stable_s':float(longest), 'ending_all_contained_stable_s':float(current),
        'limitations':['Pose differences at recording rate; native object velocities/substep extrema are not present.',
                       'A component along local X diagnoses axial rotation; it does not establish the force or cause.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--window-s',type=float,default=6.)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Preserve existing reports')
    if not np.isfinite(args.window_s) or args.window_s<=0:p.error('Positive finite window required')
    result=analyze(args.source,args.window_s);result['script_sha256']=sha256(__file__)
    atomic_json(args.output,result)
    print(json.dumps({'ending_stable_s':result['ending_all_contained_stable_s'],
        'last_second_unstable_objects':[n for n,r in result['objects'].items() if r['last_second_unstable_intervals']]}))


if __name__=='__main__':main()
