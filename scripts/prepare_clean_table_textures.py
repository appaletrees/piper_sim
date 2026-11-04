#!/usr/bin/env python3
"""Reproducible original weave/terry textures; no source-video pixels are copied."""
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    out = Path(__file__).resolve().parents[1]/'assets/tasks/clean_table'
    out.mkdir(parents=True, exist_ok=True)
    n = 1024
    y, x = np.mgrid[:n, :n]
    rng = np.random.default_rng(2048)
    # Alternating over/under crossing threads, with two finer twisted strands.
    a = (np.sin(x*np.pi/4)+1)/2
    b = (np.sin(y*np.pi/4)+1)/2
    over = ((x//8+y//8) % 2).astype(bool)
    relief = np.where(over, a, b)*0.17 + 0.79
    relief += .018*np.sin((x+y)*np.pi/2) + rng.normal(0, .013, (n,n))
    woven = np.clip(relief[...,None]*[133, 103, 73],0,255).astype('uint8')
    Image.fromarray(woven).save(out/'woven.png')
    loop = np.sin(x*.9 + np.sin(y*.3))*np.sin(y*.85)
    cloth = np.clip(219 + 12*loop + rng.normal(0,4,(n,n)),0,255)
    Image.fromarray(np.stack([cloth,cloth,cloth*.98],axis=-1).astype('uint8')).save(out/'terry.png')


if __name__ == '__main__': main()
