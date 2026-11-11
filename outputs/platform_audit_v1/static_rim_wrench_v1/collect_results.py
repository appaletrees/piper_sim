"""Index completed rim experiments; source audit verdicts remain authoritative."""
import argparse,hashlib,json,shutil
from pathlib import Path
import h5py

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;rows=[]
families=['rim_force25_raw_v1','rim_capacity_raw_v1','rim_friction_raw_v1','rim_contained_raw_v1','rim_contained_scoped_raw_v1','rim_continuous_raw_v1','rim_fallback_raw_v1','rim_fallback_rgb_v1','rim_supported_raw_v1','rim_supported_rgb_v1','rim_supported20_raw_v1','rim_stiff_raw_v1','rim_damped_raw_v1','rim_damped_rgb_v1','rim_packed_raw_v1','rim_packed_rgb_v1','rim_bowl_clearance_raw_v1','rim_bowl_clearance_rgb_v2']
for family in families:
 for base in sorted((Path('outputs')/family).glob('seed*/episode')):
  source=base/'episode.h5';audit_path=base/'audit.json';report_path=base/'report.json'
  if not report_path.exists() or not audit_path.exists():continue
  try:
   with h5py.File(source,'r') as f:
    if not f.attrs.get('complete'):continue
    metadata=json.loads(f.attrs['metadata']);quality=json.loads(f.attrs['quality'])
   digest=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
  except (OSError,BlockingIOError):continue
  audit=json.loads(audit_path.read_text());report=json.loads(report_path.read_text());dest=root/'recorded_verdicts'/family/base.parent.name;dest.mkdir(parents=True,exist_ok=True)
  for file in (audit_path,report_path):
   target=dest/file.name
   if target.exists() and target.read_bytes()!=file.read_bytes():raise RuntimeError('Changed closed verdict: '+str(target))
   shutil.copyfile(file,target)
  rows.append({'family':family,'seed':metadata['task_scene']['layout']['seed'],'source':str(source.resolve()),'source_sha256':digest,'audit':str((dest/'audit.json').relative_to(root)),'report':str((dest/'report.json').relative_to(root)),'passed':audit['passed'],'training_ready':audit['training_ready'],'violations':audit['violations'],'physical_summary':report.get('summary'),'runtime_error':report.get('error'),'duration_s':audit.get('checks',{}).get('streams',{}).get('duration_s'),'frames':audit.get('checks',{}).get('streams',{}).get('frames'),'motion_violations':quality['violations'],'measured_gripper_jerk':quality['metrics']['measured_gripper_jerk'],'plate_grasp':next((r for r in quality.get('physical_grasps',[]) if r.get('name')=='plate'),None),'position_execution':metadata['config']['mujoco']['position_execution']})
a.output.write_text(json.dumps({'scope':__doc__,'not_a_new_acceptance_gate':True,'runs':rows},indent=2)+'\n');print('closed runs indexed',len(rows),'RGB accepted',sum(r['training_ready'] for r in rows))
