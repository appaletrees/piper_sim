"""Freeze compact full-attempt evidence without changing original recordings."""
import argparse, copy, json, shutil
from pathlib import Path
import h5py, numpy as np
from piper_sim.collection_jobs import atomic_json,sha256
from piper_sim.collection_audit import load_acceptance_policy
from piper_sim.tcp_audit import audit_tcp_motion

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
if a.output.exists():raise FileExistsError(a.output)
identity=json.loads((a.root/'run.json').read_text())
changed=[p for p,digest in identity['files'].items() if not Path(p).is_file() or sha256(p)!=digest]
if changed:raise RuntimeError('Frozen inputs changed: '+repr(changed))
summary=json.loads((a.root/'summary.json').read_text())
rows=[]
policy_path=identity['options']['acceptance_policy']
settings,policy=load_acceptance_policy(policy_path)
for result in summary['episodes']:
    source=Path(result['path']);digest=sha256(source)
    r=json.loads(Path(result['report']).read_text())
    audit=json.loads(Path(result['audit']).read_text())
    if audit.get('source_sha256')!=digest:raise ValueError('Source audit hash mismatch')
    with h5py.File(source,'r') as f:
        assert f.attrs['complete'] and not f.attrs['training_ready']
        meta=json.loads(f.attrs['metadata'])
        frames=len(f['action/position'])
        material=meta['physics_model'].get('finger_contact')
        assert not any('images' in f[k] for k in ('observation','next_observation'))
        if material is not None:
            assert material==meta['teacher']['finger_contact']
            assert material['constant_within_episode'] and material['parameter_origin']=='uncalibrated_configuration'
            assert not meta['teacher']['per_target_contact_material_switching']
            material_path=material['source']['path']
            assert sha256(material_path)==material['source']['sha256']
        tcp=audit_tcp_motion(f,meta,settings['tcp_motion'])
        mode_counts={};assist_count=0
        for raw in f['next_observation/extras_json']:
            t=json.loads(raw)['teacher']
            key=t['grasp_mode'];mode_counts[key]=mode_counts.get(key,0)+1
            assist_count+=bool(t['active_assist'])
        assert mode_counts=={'physical':frames} and assist_count==0
        row={'seed':result['seed'],'source':str(source),'source_sha256':digest,'bytes':source.stat().st_size,
            'frames':frames,'recorded_seconds':frames/meta['fps'],'fps':meta['fps'],'summary':r['summary'],
            'error':r['error'],'diagnostic_passed':result['diagnostic_passed'],'training_ready':False,
            'audit_passed':audit['passed'],'audit_violations':audit['violations'],
            'online_quality':r['quality'],'tcp_diagnostic':tcp,
            'material':material,'per_target_material_switching':meta['teacher']['per_target_contact_material_switching'],
            'mode_counts':mode_counts,'assisted_frames':assist_count,
            'grasps':audit.get('checks',{}).get('task',{}).get('grasps'),
            'max_robot_penetration_m':r['robot_collision_peak'],
            'source_unchanged':sha256(source)==digest}
        assert row['source_unchanged']
        rows.append(row)
    print(json.dumps({'seed':row['seed'],'summary':row['summary'],'diagnostic_passed':row['diagnostic_passed'],'error':row['error']}),flush=True)
evidence=a.output.parent/(a.output.stem+'_evidence')
evidence.mkdir(exist_ok=False)
files={}
for p in sorted(a.root.rglob('*')):
    if not p.is_file() or p.suffix not in ('.json','.jsonl','.log'):continue
    if p.stat().st_size>2*1024**2:raise ValueError('Unexpected large evidence '+str(p))
    dest=evidence/p.relative_to(a.root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    files[str(dest)]=sha256(dest)
atomic_json(a.output,{'schema_version':1,'scope':'All attempted known-scene state-only episodes, including failures. No RGB or training-ready claim; no wall-time speed comparison across different concurrent host workloads.',
    'root':str(a.root),'frozen_input_count':len(identity['files']),'changed_inputs':changed,
    'acceptance_policy':policy,'rows':rows,'source_unchanged':all(r['source_unchanged'] for r in rows),
    'diagnostic_passed':sum(r['diagnostic_passed'] for r in rows),'attempted':len(rows),
    'training_ready':False,'durable_raw_archive':False,'evidence_files':files,'script_sha256':sha256(__file__)})
