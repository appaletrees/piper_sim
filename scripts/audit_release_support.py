#!/usr/bin/env python3
"""Read-only geometric support estimates at the start of recorded release phases.

Rebuilds task collision meshes from recorded object specifications, without
loading or stepping a live simulator. Sampled vertical clearance is not an
exact collision distance, a support-force measurement or a stability proof.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.placement_search import VerticalSupport, collision_surface_points
from piper_sim.tasks.clean_table_meshes import build_parts
from piper_sim.transforms import pose_matrix, transform_points


def geometry(name, poses, specs, parts, active, spacing=.005):
    target = specs[name]['target']
    transforms = {n: pose_matrix(p['position'], p['quaternion_wxyz']) for n, p in poses.items()}
    points = transform_points(collision_surface_points(parts[name], spacing), transforms[name])
    local = (points-transforms[target][:3, 3])@transforms[target][:3, :3]
    floor_gap = float(local[:, 2].min()-specs[target]['wall_m'])
    half = np.asarray(specs[target]['size_m'][:2])/2-specs[target]['wall_m']
    footprint_margin = float(np.min(half-abs(local[:, :2]).max(axis=0)))
    supports = []
    for other in active:
        if other == name:
            continue
        triangles = []
        for part in parts[other]:
            if part.get('collision'):
                cloud = transform_points(part['points'], transforms[other])
                triangles.append(cloud[np.asarray(part['faces'], int)])
        if not triangles:
            continue
        surface = VerticalSupport(np.concatenate(triangles)).height(points[:, :2])
        finite = np.isfinite(surface)
        if finite.any():
            ids = np.flatnonzero(finite)
            gaps = points[ids, 2]-surface[ids]
            index = int(ids[np.argmin(gaps)])
            supports.append({'object': other, 'sampled_min_vertical_gap_m': float(gaps.min()),
                             'payload_point_world_m': points[index].tolist(),
                             'envelope_world_z_m': float(surface[index])})
    return {'target': target, 'collision_sample_count': len(points),
            'sample_spacing_m': spacing, 'basket_local_min_height_above_floor_m': floor_gap,
            'basket_local_xy_margin_m': footprint_margin,
            'support_envelopes': sorted(supports, key=lambda s: s['sampled_min_vertical_gap_m'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    seen, snapshots = set(), []
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'):
            raise ValueError('Closed source required')
        meta = json.loads(f.attrs['metadata'])['task_scene']
        specs = meta['config']['objects']
        parts = {n: build_parts(s) for n, s in specs.items()}
        active = meta['layout']['active_objects']
        for i, raw in enumerate(f['next_observation/extras_json']):
            e = json.loads(raw)
            teacher = e['teacher']
            name, phase = teacher['object'], teacher['phase']
            key = (name, phase)
            if name not in ('bowl', 'plate', 'side_dish') or phase not in ('unload_gripper', 'release') or key in seen:
                continue
            seen.add(key)
            # The observation is the state BEFORE the first action assigned to
            # this phase. next_observation has already executed 32 native steps.
            before = json.loads(f['observation/extras_json'][i])
            snapshots.append({'object': name, 'phase': phase, 'transition_index': i,
                              'sim_time': float(f['observation/sim_time'][i]),
                              'object_pose': before['objects'][name],
                              **geometry(name, before['objects'], specs, parts, active)})
    unchanged = sha256(args.source) == digest
    if not unchanged or not snapshots:
        raise ValueError('Changed source or no release snapshots')
    from piper_sim.tasks import clean_table_meshes
    from piper_sim.experts import placement_search
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': digest,
              'source_unchanged': unchanged, 'script_sha256': sha256(__file__),
              'geometry_code_sha256': sha256(clean_table_meshes.__file__),
              'support_code_sha256': sha256(placement_search.__file__),
              'snapshots': snapshots, 'training_ready': False,
              'limitations': '5 mm surface subdivision; only recorded task objects are included. '
              'A negative envelope gap can mean an obstacle above the sample, not penetration. '
              'Positive sampled gaps do not rule out unsampled edge contact. '
              'Floor clearance is in basket coordinates and does not imply absence of support by other cargo. '
              'No support forces were recorded by this diagnostic. Not a causal intervention.'}
    atomic_json(args.output, result)
    for s in snapshots:
        print(json.dumps({k: s[k] for k in ('object', 'phase', 'sim_time', 'basket_local_min_height_above_floor_m', 'support_envelopes')}))


if __name__ == '__main__':
    main()
