#!/usr/bin/env python3
"""Render the asset-equipped MuJoCo workcell, from real simulation state."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import yaml, mujoco
from PIL import Image, ImageDraw
from piper_sim.config import load
from piper_sim.backends.mujoco import MujocoBackend
out=Path('outputs/blender_clean_table')
c=load('configs/clean_table_workcell.yaml');c['task_scene']=yaml.safe_load(open('configs/tasks/clean_table_rigid.yaml'))
b=MujocoBackend(c,cameras=True)
try:
    for _ in range(30):b.step()
    cam=mujoco.MjvCamera();cam.lookat[:]=[.40,-.06,.04];cam.distance,cam.azimuth,cam.elevation=1.05,155,-53
    Image.fromarray(b._render(cam,1280,960)).save(out/'mujoco_after.png')
    for n,im in b.read().images.items():Image.fromarray(im).save(out/('mujoco_'+n+'.png'))
    canvas=Image.new('RGB',(1600,640),'#18212b');draw=ImageDraw.Draw(canvas)
    for i,(filename,label) in enumerate([('mujoco_before.png','Before / MuJoCo native'),('mujoco_after.png','Blender assets / MuJoCo native')]):
        im=Image.open(out/filename).resize((800,600));canvas.paste(im,(i*800,40));draw.text((i*800+16,14),label,fill='white')
    canvas.save(out/'mujoco_comparison.jpg',quality=93)
    (out/'asset_runtime.json').write_text(json.dumps(b.task.metadata()['visual_assets'],indent=2))
finally:b.close()
