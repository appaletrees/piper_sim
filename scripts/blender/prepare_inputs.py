#!/usr/bin/env python3
"""Snapshot authoritative dimensions and CAD transforms for Blender authoring.

Run with the project's MuJoCo Python. This is an asset/preview fixture, not a
recorded demonstration. No hardware is accessed and no live sim is modified.
"""
import json
import os
from pathlib import Path
import numpy as np
import yaml

os.environ.setdefault('MUJOCO_GL', 'egl')
from piper_sim.config import ROOT, load
from piper_sim.backends.mujoco import MujocoBackend

OUT = ROOT/'outputs/blender_clean_table'
OUT.mkdir(parents=True, exist_ok=True)
cfg = load('configs/clean_table_workcell.yaml')
cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
cfg['task_scene']['appearance'].pop('asset_library', None)
cfg['task_scene']['appearance'].pop('stage_library', None)
backend = MujocoBackend(cfg, cameras=True)
try:
    import mujoco
    from PIL import Image
    for _ in range(30): backend.step()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.40, -.06, .04]
    camera.distance, camera.azimuth, camera.elevation = 1.05, 155, -53
    Image.fromarray(backend._render(camera, 1280, 960)).save(OUT/'mujoco_before.png')
    task = backend.task
    inputs = {'specs': task.specs, 'parts': task.parts,
              'bounds': {n: v.tolist() for n,v in task.bounds.items()},
              'poses': backend.read().extras['objects'], 'materials': task.materials,
              'material_root': str(task.material_path.parent), 'environment': [], 'cameras': [], 'bodies': {}}
    m,d = backend.model,backend.data
    for i in range(m.nbody):
        inputs['bodies'][m.body(i).name] = {'position': d.xpos[i].tolist(),
            'rotation': d.xmat[i].reshape(3,3).tolist()}
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith('task/') or m.geom_group[g] == 3 or m.geom_rgba[g,3] == 0: continue
        mat = int(m.geom_matid[g]); material = m.material(mat).name if mat >= 0 else None
        p = dict(name=name, type=int(m.geom_type[g]), size=m.geom_size[g].tolist(),
                 position=d.geom_xpos[g].tolist(), rotation=d.geom_xmat[g].reshape(3,3).tolist(),
                 color=m.geom_rgba[g].tolist(), material=material, parent=m.body(int(m.geom_bodyid[g])).name)
        if mat >= 0: p['color'] = m.mat_rgba[mat].tolist()
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g]); start,count = m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
            p['points']=m.mesh_vert[start:start+count].tolist()
            start,count=m.mesh_faceadr[mesh],m.mesh_facenum[mesh]
            p['faces']=m.mesh_face[start:start+count].tolist()
        inputs['environment'].append(p)
    for i in range(m.ncam):
        inputs['cameras'].append(dict(name=m.camera(i).name, position=d.cam_xpos[i].tolist(),
            rotation=d.cam_xmat[i].reshape(3,3).tolist(), fovy=float(m.cam_fovy[i]),
            intrinsics=backend.rig['cameras'][m.camera(i).name]['rgb']['intrinsic_matrix']))
    (OUT/'inputs.json').write_text(json.dumps(inputs))
    print('Prepared',OUT/'inputs.json',flush=True)
finally: backend.close()
