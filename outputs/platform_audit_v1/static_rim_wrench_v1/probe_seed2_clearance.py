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
 actual=t.check_joint_path
 def check(side,start,end,**kwargs):
  try:return actual(side,start,end,**kwargs)
  except SortFailure as e:
   if 'neighbor clearance' in str(e):
    ik=t.ik[side];g=kwargs.get('gripper');neighbors=[n for n in b.task.active_names if n!=t.current];samples=[]
    for a in np.linspace(0,1,101):
     ik.data.qpos[:]=b.data.qpos;ik.data.qpos[ik.qids]=(1-a)*start+a*end
     if g is not None:ik.data.qpos[b.indices[side]['qpos'][6:]]=((1-a)*g[0]+a*g[1])/2
     mujoco.mj_kinematics(b.model,ik.data);mujoco.mj_collision(b.model,ik.data)
     d,p=robot_obstacle_clearance(b.model,ik.data,side,neighbors,.02);samples.append((float(d),float(a),p))
    row={'error':str(e),'start_q':np.asarray(start).tolist(),'end_q':np.asarray(end).tolist(),'options':kwargs,'min':min(samples,key=lambda x:x[0]),'first':samples[0],'last':samples[-1]};rows.append(row)
   raise
 t.check_joint_path=check
 try:t.reachable_grasp(t.current,check_approach=True)
 except SortFailure as e:error=str(e)
 assert np.array_equal(initial,state(b))
finally:b.close()
assert sha256(source)==digest
atomic_json('outputs/platform_audit_v1/static_rim_wrench_v1/seed2_clearance_detail.json',{'scope':__doc__,'source_sha256':digest,'query_only':True,'state_unchanged':True,'rows':rows})
print(json.dumps(rows,indent=2))
