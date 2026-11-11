"""Read-only geometry query; no physics replay or training acceptance."""
import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import h5py,mujoco,numpy as np
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.simulation import SimulationSession
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.collision import robot_obstacle_clearance
from piper_sim.collection_jobs import atomic_json,sha256
from probe_grasp_aperture import state
source=Path('outputs/rim_contained_scoped_raw_v1/seed2/episode/episode.h5');digest=sha256(source)
with h5py.File(source,'r') as f:
 meta=json.loads(f.attrs['metadata']);ex=json.loads(f['next_observation/extras_json'][-1])
cfg=copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
b=MujocoBackend(cfg);rows=[]
try:
 layout=meta['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
 for side,arm in ex['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']]=arm['joint_position']
 for name,pose in ex['objects'].items():
  adr=b.model.joint('task/'+name+'/free').qposadr[0];b.data.qpos[adr:adr+7]=pose['position']+pose['quaternion_wxyz']
 mujoco.mj_forward(b.model,b.data);initial=state(b)
 t=RigidTableTeacher(SimulationSession(b),settings=meta['teacher']['motion_and_quality']);t.q=np.asarray(ex['controller_interval']['command']);t.current='chopstick_a';t.phase='approach'
 original_candidates=t.grasp_candidates
 for lateral in (-.00015,.00015,-.0003,.0003,-.00045,.00045):
  def candidates(name,position,rotation,*,check_approach):
   for p,r,detail in original_candidates(name,position,rotation,check_approach=check_approach):
    yield p+lateral*r[:,1],r,{**detail,'lateral_offset_m':lateral}
  t.grasp_candidates=candidates
  try:
   side,p,r,width=t.reachable_grasp(t.current,check_approach=True)
   rows.append({'lateral_m':lateral,'passed':True,'position_m':p.tolist(),'width_m':width,'candidate':t.events[-1]})
  except SortFailure as e:rows.append({'lateral_m':lateral,'passed':False,'error':str(e)})
 assert np.array_equal(initial,state(b))
finally:b.close()
assert sha256(source)==digest
atomic_json('outputs/platform_audit_v1/static_rim_wrench_v1/seed2_lateral_candidates.json',{'scope':__doc__,'source_sha256':digest,'query_only':True,'state_unchanged':True,'rows':rows})
print(json.dumps(rows,indent=2))
