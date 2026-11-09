"""Closed-record geometry diagnostic. No physics integration or source mutation."""
import argparse, copy, json
from pathlib import Path
import h5py, mujoco, numpy as np
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.clean_table import SortFailure
from piper_sim.simulation import SimulationSession

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
digest=sha256(a.source)
with h5py.File(a.source,'r') as f:
    assert f.attrs['complete']
    meta=json.loads(f.attrs['metadata'])
    extra=json.loads(f['next_observation/extras_json'][-1])
    time=float(f['next_observation/sim_time'][-1])
    command=np.asarray(extra['controller_interval']['command'])
cfg=copy.deepcopy(meta['config'])
cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
b=MujocoBackend(cfg,cameras=False)
rows=[]
try:
    layout=meta['task_scene']['layout']; b.reset(layout['seed'],layout['randomized'])
    t=RigidTableTeacher(SimulationSession(b),settings=meta['teacher']['motion_and_quality'])
    for side,arm in extra['teacher']['physics']['arms'].items():
        b.data.qpos[b.indices[side]['qpos']]=arm['joint_position']
        b.data.qvel[b.indices[side]['dof']]=arm['joint_velocity']
    for name,pose in extra['objects'].items():
        addr=b.model.joint('task/'+name+'/free').qposadr[0]
        b.data.qpos[addr:addr+7]=pose['position']+pose['quaternion_wxyz']
    b.data.time=time; mujoco.mj_forward(b.model,b.data)
    t.q=command.copy()
    frozen=b.data.qpos.copy(),b.data.qvel.copy(),b.data.time,t.q.copy()
    def forbidden(*args,**kwargs): raise AssertionError('No integration allowed')
    t.step=forbidden; b.step=forbidden
    for name in ('chopstick_a','chopstick_b'):
        t.current=name; t.phase='approach'; t.events=[]
        candidates=[]
        original=t.grasp_candidates
        def observe(*args,**kwargs):
            for p,r,details in original(*args,**kwargs):
                candidates.append(details)
                yield p,r,details
        t.grasp_candidates=observe
        checks=[]; original_check=t.check_joint_path
        def check(*args,**kwargs):
            try: return original_check(*args,**kwargs)
            except SortFailure as error:
                
                side,start,end=args[:3]
                delta=np.asarray(end)-start; mask=abs(delta)>1e-8
                fraction=float(np.median((t.ik[side].data.qpos[t.ik[side].qids][mask]-np.asarray(start)[mask])/delta[mask])) if mask.any() else 0.
                off=0 if side=='left' else 7
                checks.append({'error':str(error),'starts_at_command':bool(np.allclose(start,t.q[off:off+6],atol=1e-8,rtol=0)),'interpolation_fraction':fraction,'start':np.asarray(start).tolist(),'end':np.asarray(end).tolist()}); raise
        t.check_joint_path=check
        try:
            side,p,r,width=t.reachable_grasp(name,check_approach=True)
            result={'object':name,'reachable':True,'side':side,'position':p.tolist(),'rotation':r.tolist(),'width_m':width,'events':t.events}
        except SortFailure as error:
            result={'object':name,'reachable':False,'error':str(error)}
        finally:
            t.grasp_candidates=original; t.check_joint_path=original_check
        rows.append({**result,'candidates_attempted':candidates,'path_rejections':checks})
        for value,before in zip((b.data.qpos,b.data.qvel,b.data.time,t.q),frozen):
            assert np.array_equal(value,before)
finally: b.close()
assert sha256(a.source)==digest
atomic_json(a.output,{'source':str(a.source),'source_sha256':digest,'source_unchanged':True,
    'script_sha256':sha256(__file__),'sim_time':time,'cases':rows,'training_ready':False,
    'scope':'Read-only geometric candidate probe from final native pose/command of a rejected episode. No integration; no source changes. End-pose feasibility does not prove a live alternate order succeeds.'})
print(json.dumps([{'object':r['object'],'reachable':r['reachable'],'error':r.get('error')} for r in rows]))
