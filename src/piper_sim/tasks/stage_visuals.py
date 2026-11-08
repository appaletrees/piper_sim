"""Optional Blender room and link-attached details, separate from task physics.

Vertices are expressed in the named existing body's coordinates. There are no
new rigid bodies, collision shapes, sensors, joints or inertial contributions.
"""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from ..config import ROOT


def load_stage_visuals(task):
    task.stage_visuals = None
    task.stage_visual_metadata = None
    reference = task.config.get('appearance', {}).get('stage_library')
    if not reference:
        return
    path = (ROOT/reference).resolve()
    catalog = json.loads(path.read_text())
    if catalog.get('schema_version') != 1 or catalog.get('units') != 'metres':
        raise ValueError('Unsupported stage visual library')
    for relative, expected in catalog.get('source_files', {}).items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Stage visual source changed: {relative}; rebuild the Blender library')
    archive = path.parent/catalog['mesh_archive']
    parts = []
    with np.load(archive, allow_pickle=False) as arrays:
        for descriptor in catalog['parts']:
            part = copy.deepcopy(descriptor)
            for field in ('points', 'faces', 'normals', 'uv'):
                part[field] = arrays[descriptor['key']+'_'+field].tolist()
            parts.append(part)
    for name, spec in catalog['materials'].items():
        spec = copy.deepcopy(spec)
        for field in ('texture', 'detail_texture', 'roughness_texture', 'normal_texture'):
            if field in spec:
                spec[field] = str((path.parent/spec[field]).resolve())
                if not Path(spec[field]).is_file():
                    raise FileNotFoundError(spec[field])
        task.materials[name] = spec
    task.stage_visuals = dict(parts=parts, hide_environment=catalog.get('hide_environment', []),
        mujoco_material_bindings=catalog.get('mujoco_material_bindings', {}),
        usd_material_bindings=catalog.get('usd_material_bindings', {}))
    task.stage_visual_metadata = dict(library=str(path.relative_to(ROOT)),
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mesh_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        generator=catalog['generator'], status=catalog['status'], collision=False,
        parts=len(parts), source_files=catalog.get('source_files', {}))


def compose_mujoco(root, task, prefix):
    from .rigid_scene_mujoco import nums
    library = getattr(task, 'stage_visuals', None)
    if not library:
        return
    world, asset = root.find('worldbody'), root.find('asset')
    for name, material in library['mujoco_material_bindings'].items():
        geom = world.find(f".//geom[@name='{name}']")
        if geom is None:
            raise ValueError('Missing visual material target: '+name)
        geom.set('material', prefix+'/'+material)
        # An explicitly set geom RGBA overrides (rather than tints) the material
        # color in MuJoCo. Remove the old per-geom color when assigning a material.
        geom.attrib.pop('rgba', None)
    for name in library['hide_environment']:
        geom = world.find(f"geom[@name='{prefix}_environment/{name}']")
        if geom is not None:
            geom.set('rgba', '0 0 0 0')
            geom.set('group', '3')
    for part in library['parts']:
        name = 'stage_visual/'+part['name']
        parent = world if part['parent'] == 'world' else world.find(f".//body[@name='{part['parent']}']")
        if parent is None:
            raise ValueError('Missing visual attachment parent: '+part['parent'])
        ET.SubElement(asset, 'mesh', name=name, vertex=nums(part['points']), face=nums(part['faces']),
                      normal=nums(part['normals']), texcoord=nums(part['uv']), inertia='shell')
        ET.SubElement(parent, 'geom', name=name, type='mesh', mesh=name,
                      material=prefix+'/'+part['material'], contype='0', conaffinity='0',
                      density='0', group='1')


def spawn_isaac(stage, root, task):
    from pxr import Sdf, UsdGeom, UsdShade
    library = getattr(task, 'stage_visuals', None)
    if not library:
        return
    for path, name in library['usd_material_bindings'].items():
        prim = stage.GetPrimAtPath(path)
        if prim:
            material = UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/'+name))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    for name in library['hide_environment']:
        prim = stage.GetPrimAtPath(root+'/Environment/'+name)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    for part in library['parts']:
        parent = root if part['parent'] == 'world' else '/World/'+part['parent']
        # Called once before arms exist and again after spawning. Never create
        # phantom articulation ancestors during the first call.
        if not stage.GetPrimAtPath(parent):
            continue
        mesh = UsdGeom.Mesh.Define(stage, parent+'/StageVisuals/'+part['name'])
        mesh.CreatePointsAttr(part['points'])
        mesh.CreateFaceVertexCountsAttr([3]*len(part['faces']))
        mesh.CreateFaceVertexIndicesAttr(np.asarray(part['faces']).reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateNormalsAttr(part['normals']); mesh.SetNormalsInterpolation('vertex')
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                               UsdGeom.Tokens.vertex).Set(part['uv'])
        material = UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/'+part['material']))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
