#!/usr/bin/env python3
"""Export geometry edited in props.blend to the shared runtime library.

Use Blender --background assets/blender/clean_table/props.blend --python
scripts/blender/export_library.py. Export validates dimensions before writing.
PBR map changes are authored with build_clean_table.py; this command exports
geometry, per-corner normals/UVs and assigned existing material keys.
"""
import hashlib
import json
from pathlib import Path
import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'assets/blender/clean_table'
manifest=json.loads((OUT/'manifest.json').read_text())
arrays={};descriptors={};checks={}
depsgraph=bpy.context.evaluated_depsgraph_get()
for prototype in manifest['prototypes']:
    objects=sorted((o for o in bpy.data.objects if o.type=='MESH' and o.name.startswith(prototype+'__')),
                   key=lambda o:o.name)
    if not objects:raise ValueError('Missing prototype '+prototype)
    parts=[];all_points=[]
    for index,obj in enumerate(objects):
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            loops=np.array([loop for tri in mesh.loop_triangles for loop in tri.loops],dtype=np.int32)
            vertex=np.array([mesh.loops[int(i)].vertex_index for i in loops])
            transform=np.array(obj.matrix_world)
            points=np.array([v.co[:] for v in mesh.vertices],dtype=np.float64)[vertex]@transform[:3,:3].T+transform[:3,3]
            normals=np.array([n.vector[:] for n in mesh.corner_normals],dtype=np.float64)[loops]@np.linalg.inv(transform[:3,:3])
            length=np.linalg.norm(normals,axis=1)
            if (length<1e-6).any():raise ValueError(f'Invalid/degenerate normals: {obj.name}')
            normals/=length[:,None]
            if mesh.uv_layers.active is None:raise ValueError('Missing UVs: '+obj.name)
            uv=np.array([v.uv[:] for v in mesh.uv_layers.active.data],dtype=np.float64)[loops]
            packed=np.concatenate([points,normals,uv],axis=1).astype(np.float32)
            unique,inverse=np.unique(packed,axis=0,return_inverse=True)
            key=f'{prototype}_{index}'
            for suffix,array in [('points',unique[:,:3]),('normals',unique[:,3:6]),('uv',unique[:,6:]),('faces',inverse.reshape(-1,3))]:
                arrays[key+'_'+suffix]=array
            materials={p.material_index for p in mesh.polygons}
            if len(materials)!=1:raise ValueError('Separate parts by material before exporting: '+obj.name)
            material=obj.data.materials[next(iter(materials))].name
            if material not in manifest['materials']:raise ValueError('Material not in catalog: '+material)
            parts.append(dict(key=key,name='blender_'+obj.name.split('__',1)[1],material=material,
                              vertices=len(unique),triangles=len(inverse)//3))
            all_points.append(points)
        finally:evaluated.to_mesh_clear()
    points=np.concatenate(all_points);bounds=np.array([points.min(0),points.max(0)])
    original=np.array(manifest['geometry_checks'][prototype]['source_bounds_m'])
    error=float(np.abs(bounds-original).max())
    if error>2e-7:raise ValueError(f'{prototype}: envelope changed by {error} m; edit within the source dimensions')
    checks[prototype]={'source_bounds_m':original.tolist(),'visual_bounds_m':bounds.tolist(),'max_bound_error_m':error}
    descriptors[prototype]=parts
# Validation above finishes before either runtime file is replaced.
temporary=OUT/'visual_meshes.tmp.npz'
np.savez_compressed(temporary,**arrays)
manifest.update(prototypes=descriptors,geometry_checks=checks,
                edited_blend_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest())
temporary.replace(OUT/'visual_meshes.npz')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if obj.type=='MESH' and any(obj.name.startswith(p+'__') for p in descriptors):
        obj.hide_set(False);obj.select_set(True)
bpy.ops.wm.usd_export(filepath=str(OUT/'props.usdc'),selected_objects_only=True,
                     export_textures=True,relative_paths=True,export_cameras=False,export_lights=False)
print('EXPORTED',list(descriptors),flush=True)
