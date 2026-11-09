#!/usr/bin/env python3
"""Read-only paired evaluation of a supported-release experiment and its baseline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


def scan(directory):
    source = directory/'episode.h5'
    digest = sha256(source)
    report = json.loads((directory/'report.json').read_text())
    audit = json.loads((directory/'audit.json').read_text())
    if audit['source_sha256'] != digest:
        raise ValueError('Stale independent audit')
    with h5py.File(source, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed source required')
        arrays = {key: f[key][:] for key in ('observation/state', 'next_observation/state', 'action/position',
                                            'observation/sim_time', 'next_observation/sim_time')}
        hashes, groups, boundary, supports = [], {}, None, []
        for i, raw in enumerate(f['next_observation/extras_json']):
            e = json.loads(raw);t=e['teacher'];c=e['controller_interval']
            if t['active_assist'] or t['grasp_mode'] != 'physical':
                raise ValueError('Physical unassisted source required')
            if boundary is None and t['phase'] == 'support_lower':
                boundary = i
            hashes.append(hashlib.sha256(json.dumps({**{k:e[k] for k in ('objects','controller_interval','tcp_motion_interval')},
                                                     'physics':t['physics']},sort_keys=True).encode()).hexdigest())
            key = str(t['object'])+'/'+t['phase']
            g = groups.setdefault(key, {'right_gripper_acceleration':0.,'right_gripper_jerk':0.,'frames':0})
            g['frames']+=1
            g['right_gripper_acceleration']=max(g['right_gripper_acceleration'],c['native_acceleration_abs_max'][13])
            g['right_gripper_jerk']=max(g['right_gripper_jerk'],c['native_jerk_abs_max'][13])
            if t.get('payload_support'):
                s=t['payload_support']
                supports.append({'transition_index':i,'phase':t['phase'],'sim_time':s['sim_time'],
                                 'fraction':s['support_fraction_of_weight']})
    if sha256(source) != digest:
        raise ValueError('Source changed')
    summary = {'source':str(source),'source_sha256':digest,'frames':len(arrays['action/position']),
               'duration_s':float(arrays['next_observation/sim_time'][-1]-arrays['observation/sim_time'][0]),
               'summary':report['summary'],'error':report['error'],'quality':report['quality'],
               'audit_passed':audit['passed'],'training_ready':audit['training_ready'],
               'acceptance_policy':audit.get('acceptance_policy'),'audit_violations':audit['violations'],
               'first_intervention_transition':boundary,'native_phase_extrema':groups,
               'support_events':[e for e in report['events'] if e['phase'].startswith('support_')],
               'support_samples':supports}
    return arrays, hashes, summary


def native_support(trace_dir, events, geom_bodies):
    meta = json.loads((trace_dir/'trace.json').read_text())
    if not meta.get('includes_task_contacts') or sha256(trace_dir/'trace.npz') != meta['trace_sha256']:
        raise ValueError('Native task contact evidence missing or changed')
    results=[]
    with np.load(trace_dir/'trace.npz',allow_pickle=False) as archive:
        # NpzFile indexing re-decompresses a member on every access. Decode each
        # required column once, never once per contact in a million-row trace.
        z={key:archive[key] for key in ('time_before','time_after','target','contact_offset','contact_count',
                                       'contact_geom1','contact_geom2','contact_frame','contact_force_local')}
        names=np.asarray(meta['geom_names'])
        if len(geom_bodies)!=len(names):raise ValueError('Geometry/body map length mismatch')
        for event in events:
            if event['phase'] != 'support_verified':continue
            end=event['sim_time'];support=event['support'];name=event['object']
            ids=np.flatnonzero((z['target']==name)&(z['time_before']>=end-.1-1e-8)&(z['time_before']<end-1e-8))
            fractions=[];up=np.asarray(support['up_axis_world'])
            for i in ids:
                first=int(z['contact_offset'][i]);stop=first+int(z['contact_count'][i]);force=np.zeros(3)
                for j in range(first,stop):
                    a,b=names[int(z['contact_geom1'][j])],names[int(z['contact_geom2'][j])]
                    matched=[a.startswith('task/'+name+'/'),b.startswith('task/'+name+'/')]
                    if sum(matched)!=1:continue
                    other=b if matched[0] else a
                    other_id=int(z['contact_geom2'][j] if matched[0] else z['contact_geom1'][j])
                    if geom_bodies[other_id].startswith(('left/','right/')):continue
                    sign=-1 if matched[0] else 1
                    frame=z['contact_frame'][j].reshape(3,3)
                    if sign*frame[0]@up >= .5:
                        force+=sign*frame.T@z['contact_force_local'][j,:3]
                fractions.append(float(force@up/support['weight_N']))
            expected=round(.1/meta['integration_dt'])
            complete=(len(ids)==expected and np.allclose(z['time_after'][ids]-z['time_before'][ids],meta['integration_dt'],atol=1e-8,rtol=0)
                      and np.allclose(z['time_before'][ids[1:]],z['time_after'][ids[:-1]],atol=1e-8,rtol=0))
            results.append({'object':name,'event_time':end,'native_samples':len(ids),'expected_samples':expected,
                            'complete_window':bool(complete),'min_fraction':min(fractions) if fractions else None,
                            'max_fraction':max(fractions) if fractions else None,
                            'all_native_samples_at_least_half_weight':bool(complete and min(fractions)>=.5),
                            'fractions':fractions})
    return results


def geometry_body_map(source, trace_dir):
    # Robot geom names are not uniformly prefixed by the arm side. Rebuild
    # immutable topology from the frozen source config instead of guessing.
    from piper_sim.backends.mujoco import MujocoBackend
    with h5py.File(source,'r') as f:meta=json.loads(f.attrs['metadata'])
    cfg=copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
    backend=MujocoBackend(cfg,cameras=False)
    try:
        expected=json.loads((trace_dir/'trace.json').read_text())['geom_names']
        actual=[backend.model.geom(i).name for i in range(backend.model.ngeom)]
        if actual!=expected:raise ValueError('Rebuilt geom topology differs from recorded trace')
        return [backend.model.body(int(b)).name for b in backend.model.geom_bodyid]
    finally:backend.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    a,ah,base=scan(args.baseline);b,bh,candidate=scan(args.candidate)
    boundary=candidate['first_intervention_transition']
    if boundary is None:raise ValueError('No actual supported-lowering intervention')
    prefix={key:np.array_equal(a[key][:boundary],b[key][:boundary]) for key in a}
    same_native=ah[:boundary]==bh[:boundary]
    same_thresholds=base['quality']['thresholds']==candidate['quality']['thresholds']
    policies=[r['acceptance_policy'] for r in (base,candidate)]
    same_policy=policies[0]['sha256']==policies[1]['sha256'] if all(policies) else None
    result={'scope':__doc__,'script_sha256':sha256(__file__),'baseline':base,'candidate':candidate,
            'prefix_public_streams_exact':prefix,'prefix_native_physics_exact':same_native,
            'same_online_thresholds':same_thresholds,'same_evaluated_policy':same_policy,
            'native_support_hold':native_support(args.trace,candidate['support_events'],
                                                 geometry_body_map(args.candidate/'episode.h5',args.trace)),
            'training_ready':False,'limitations':'One seed, state-only; native diagnostics are not hardware limits. '
            'Trajectory after first intervention includes downstream physical consequences. '
            'A support reaction alone does not prove final stable placement or success.'}
    atomic_json(args.output,result)
    print(json.dumps({key:result[key] for key in ('prefix_public_streams_exact','prefix_native_physics_exact','same_online_thresholds','same_evaluated_policy','native_support_hold')},indent=2))
    if not all(prefix.values()) or not same_native or not same_thresholds or same_policy is False:
        raise SystemExit('Comparison integrity failed; evidence retained')


if __name__=='__main__':main()
