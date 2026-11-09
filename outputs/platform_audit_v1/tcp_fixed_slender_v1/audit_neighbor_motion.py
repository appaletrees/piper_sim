"""Measure collateral stick motion in a closed physical rollout; no new quality gate."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from piper_sim.collection_jobs import sha256,atomic_json
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
digest=sha256(a.source);rows=[];reference=None
with h5py.File(a.source,'r') as f:
    assert f.attrs['complete']
    for index,raw in enumerate(f['next_observation/extras_json']):
        e=json.loads(raw);t=e['teacher']
        if t['object']!='chopstick_a' or t['phase'] not in ('grasp','lift'):continue
        pose=e['objects']['chopstick_b'];pos=np.asarray(pose['position']);q=np.asarray(pose['quaternion_wxyz'])
        if reference is None:reference=(pos.copy(),q.copy())
        # Normalized, sign-invariant angle avoids arccos roundoff at zero.
        q=q/np.linalg.norm(q);q0=reference[1]/np.linalg.norm(reference[1]);q=q if q@q0>=0 else -q
        angle=4*np.arctan2(np.linalg.norm(q-q0),np.linalg.norm(q+q0))
        rows.append({'transition_index':index,'sim_time':e['physics_time'],'phase':t['phase'],
            'position_displacement_m':float(np.linalg.norm(pos-reference[0])),'rotation_displacement_rad':float(angle)})
assert rows and sha256(a.source)==digest
atomic_json(a.output,{'source':str(a.source),'source_sha256':digest,'source_unchanged':True,
    'script_sha256':sha256(__file__),'target':'chopstick_a','neighbor':'chopstick_b','phases':['grasp','lift'],
    'samples':len(rows),'start_transition':rows[0]['transition_index'],'end_transition':rows[-1]['transition_index'],
    'max_position_displacement_m':max(r['position_displacement_m'] for r in rows),
    'max_rotation_displacement_rad':max(r['rotation_displacement_rad'] for r in rows),
    'rows':rows,'training_ready':False,'scope':__doc__})
print(json.dumps({'samples':len(rows),'max_position_displacement_m':max(r['position_displacement_m'] for r in rows),'max_rotation_displacement_rad':max(r['rotation_displacement_rad'] for r in rows)}))
