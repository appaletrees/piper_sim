"""Render selected native poses for inspection; not captured training RGB."""
import copy,json
from pathlib import Path
import h5py,mujoco,numpy as np
from PIL import Image,ImageDraw
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import sha256,atomic_json
source=Path('/dev/shm/piper_tcp_fixed_slender_v1/worker_00/episode_000000_seed_0/episode.h5')
out=Path('outputs/platform_audit_v1/tcp_fixed_slender_v1/state_view.jpg')
if out.exists():raise FileExistsError(out)
digest=sha256(source)
with h5py.File(source,'r') as f:
    assert f.attrs['complete']
    meta=json.loads(f.attrs['metadata']); rows={}
    for i,raw in enumerate(f['next_observation/extras_json']):
        e=json.loads(raw);t=e['teacher']
        if t['object']=='chopstick_a' and t['phase']=='grasp' and 'grasp' not in rows:rows['grasp']=(i,e)
        if t['object']=='chopstick_a' and t['phase']=='lift':rows['lift']=(i,e)
    rows['final']=(len(f['action/position'])-1,json.loads(f['next_observation/extras_json'][-1]))
cfg=copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
b=MujocoBackend(cfg,cameras=False);panels=[];details=[]
try:
    layout=meta['task_scene']['layout'];b.reset(layout['seed'],layout['randomized'])
    for phase,(index,e) in rows.items():
        for side,arm in e['teacher']['physics']['arms'].items():b.data.qpos[b.indices[side]['qpos']]=arm['joint_position']
        for name,pose in e['objects'].items():
            addr=b.model.joint('task/'+name+'/free').qposadr[0]
            b.data.qpos[addr:addr+7]=pose['position']+pose['quaternion_wxyz']
        b.data.time=e['physics_time'];mujoco.mj_forward(b.model,b.data)
        camera='overhead' if phase=='final' else 'right_wrist'
        before=b.data.qpos.copy();pixels=b._render(camera,640,480);assert np.array_equal(b.data.qpos,before)
        panel=Image.new('RGB',(640,515),(240,240,240));panel.paste(Image.fromarray(pixels),(0,35))
        label=f'{phase} | {camera} | t={b.data.time:.3f} s | transition {index}'
        ImageDraw.Draw(panel).text((12,10),label,fill=(15,15,15));panels.append(panel)
        details.append({'phase':phase,'camera':camera,'transition_index':index,'sim_time':b.data.time})
finally:b.close()
canvas=Image.new('RGB',(1920,550),(240,240,240))
for i,panel in enumerate(panels):canvas.paste(panel,(640*i,0))
ImageDraw.Draw(canvas).text((12,527),'STATE-ONLY GEOMETRY RECONSTRUCTION | original CAD/cameras | no physics replay | not captured RGB or training data',fill=(15,15,15))
canvas.save(out,quality=92)
assert sha256(source)==digest
atomic_json(out.with_suffix('.json'),{'source':str(source),'source_sha256':digest,'source_unchanged':True,'frames':details,'image_sha256':sha256(out),'script_sha256':sha256(__file__),'training_ready':False,'scope':__doc__,'sensor_postprocessing':False,'source_resolution':[640,480]})
print(str(out))
