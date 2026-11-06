#!/usr/bin/env python3
"""Deterministic procedural laminate albedos; no baked shadows or reflections."""
from pathlib import Path
import numpy as np
from PIL import Image

out=Path(__file__).resolve().parents[1]/'assets/tasks/clean_table'
rng=np.random.default_rng(7301)
y,x=np.mgrid[:512,:512]
noise=rng.normal(0,1,(512,512))
for name,color in [('table_finegrain',[231,229,223]),('table_gray',[177,183,188])]:
    grain=noise*.8+np.sin(x*.7+y*.15)*.25
    rgb=np.clip(np.array(color)[None,None,:]+grain[:,:,None],0,255).astype('uint8')
    Image.fromarray(rgb).save(out/(name+'.png'))
# Seamless longitudinal grain, softly varying rather than checkerboard tiles.
phase=2*np.pi*x/512+0.045*np.sin(2*np.pi*y/512)+.02*np.sin(6*np.pi*y/512)
grain=3*np.sin(phase*21)+1.8*np.sin(phase*47)+noise*.8
rgb=np.clip(np.array([215,196,163])[None,None,:]+grain[:,:,None],0,255).astype('uint8')
Image.fromarray(rgb).save(out/'table_wood.png')
