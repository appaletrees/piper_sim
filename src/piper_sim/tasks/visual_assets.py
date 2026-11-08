"""Optional Blender visual library; never changes contacts or inertial geometry."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import ROOT


GEOMETRY_FIELDS = ('kind', 'size_m', 'diameter_m', 'height_m', 'top_diameter_m',
                   'bottom_diameter_m', 'wall_m', 'rim_radius_m', 'floor_height_m')


def apply_visual_library(task):
    reference = task.config.get('appearance', {}).get('asset_library')
    task.visual_asset_metadata = None
    if not reference:
        return
    path = (ROOT/reference).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('units') != 'metres':
        raise ValueError('Unsupported visual library; expected schema 1 in metres')
    archive = path.parent/manifest['mesh_archive']
    geometries = {}
    with np.load(archive, allow_pickle=False) as arrays:
        for name, descriptors in manifest['prototypes'].items():
            parts = []
            for desc in descriptors:
                key = desc['key']
                part = {field: arrays[key+'_'+field].tolist()
                        for field in ('points', 'faces', 'normals', 'uv')}
                part.update(name=desc['name'], material=desc['material'],
                            collision=False, hidden=False, smooth=True)
                parts.append(part)
            geometries[name] = parts
    for name, spec in task.specs.items():
        entry = manifest['objects'][name]
        for key in GEOMETRY_FIELDS:
            if spec.get(key) != entry['spec'].get(key):
                raise ValueError(f'Blender asset dimensions changed for {name}.{key}; rebuild the library')
        visuals = copy.deepcopy(geometries[entry['prototype']])
        points = np.concatenate([p['points'] for p in visuals])
        if not np.allclose([points.min(0), points.max(0)], task.bounds[name], atol=2e-7, rtol=0):
            raise ValueError(f'Blender visual envelope differs from task geometry: {name}')
        # Preserve the original part names and all analytic contact annotations.
        # Some old visual parts also carry collision; keep them as hidden contacts.
        contacts = [copy.deepcopy(p) for p in task.parts[name] if p['collision']]
        for p in contacts:
            p['hidden'] = True
        task.parts[name] = [*visuals, *contacts]
    for name, spec in manifest['materials'].items():
        spec = copy.deepcopy(spec)
        for field in ('texture', 'detail_texture', 'roughness_texture', 'normal_texture'):
            if field in spec:
                filename = (path.parent/spec[field]).resolve()
                if not filename.is_file():
                    raise FileNotFoundError(filename)
                spec[field] = str(filename)
        task.materials[name] = spec
    task.visual_asset_metadata = {
        'library': str(path.relative_to(ROOT)), 'generator': manifest['generator'],
        'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'mesh_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'collision_policy': manifest['collision_policy'],
        'geometry_checks': copy.deepcopy(manifest['geometry_checks']),
        'material_status': manifest['material_status'],
    }
