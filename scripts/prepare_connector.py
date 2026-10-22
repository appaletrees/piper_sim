#!/usr/bin/env python3
"""Preserve the supplied board STL and decompose its extruded layers for contact.

Only this offline preparation tool requires shapely. Runtime uses the
exported OBJ/USD/JSON and the shared engine-neutral task description.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import constrained_delaunay_triangles

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84+50*count:
        raise ValueError('Expected the supplied binary STL')
    return np.array([struct.unpack_from('<9f', data, 96+50*i) for i in range(count)]).reshape(-1, 3, 3)


def local(points):
    # Right-handed: source +Y is vertical; reverse long axis to match photo front.
    p = np.asarray(points)
    return np.stack((85-p[..., 0], p[..., 2]-30, p[..., 1]), axis=-1)*.001


def prism(poly, lo, hi):
    xy = np.array(poly.exterior.coords[:-1])
    n = len(xy)
    points = np.concatenate([np.column_stack([xy, np.full(n, z)]) for z in (lo, hi)])
    faces = [(0, i+1, i) for i in range(1, n-1)]
    faces += [(n, n+i, n+i+1) for i in range(1, n-1)]
    for i in range(n):
        j = (i+1) % n
        faces += [(i, j, n+j), (i, n+j, n+i)]
    return points, np.asarray(faces)


def save_obj(path, points, faces):
    with path.open('w') as f:
        for p in points:
            f.write('v '+' '.join(f'{x:.12g}' for x in p)+'\n')
        for face in faces:
            f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')


def convex_parts(poly):
    parts = list(constrained_delaunay_triangles(poly).geoms)
    # Merge adjacent triangles when their union remains convex. No hole is filled.
    changed = True
    while changed:
        changed = False
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                u = parts[i].union(parts[j])
                if u.geom_type == 'Polygon' and not u.interiors and abs(u.area-u.convex_hull.area) < 1e-7:
                    parts[i] = u.convex_hull
                    parts.pop(j)
                    changed = True
                    break
            if changed:
                break
    assert unary_union(parts).symmetric_difference(poly).area < 1e-6
    return parts


def prepare(output):
    source = ROOT/'scene/scene1_ peg in hole/接插件原型板1.STL'
    triangles = read_stl(source)
    np.testing.assert_allclose([triangles.reshape(-1, 3).min(0), triangles.reshape(-1, 3).max(0)],
                               [[0, 0, 0], [170, 33, 60]], atol=1e-5)
    def plane(y):
        t = triangles[np.all(np.isclose(triangles[:, :, 1], y, atol=1e-5), axis=1)]
        return unary_union([Polygon(a[:, [0, 2]]) for a in t]).simplify(1e-5, preserve_topology=True)
    top, base_top = plane(33), plane(5)
    holes = sorted([Polygon(r) for r in top.interiors], key=lambda p: p.centroid.x)
    assert len(holes) == 6 and len(base_top.interiors) == 6
    # The supplied part is an exact extrusion on each of the 0/5/30/33 mm layers.
    # Verify that there are no intermediate vertices or slanted side walls.
    assert np.all(np.min(abs(triangles[:, :, 1, None]-np.array([0, 5, 30, 33])), axis=-1) < 1e-5)
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    vertical=np.ptp(triangles[:,:,1],axis=1)>1e-5
    assert np.all(abs(normals[vertical,1])<1e-3), 'Unexpected slanted wall'
    output.mkdir(parents=True, exist_ok=True)
    meshes = output/'meshes'; meshes.mkdir(exist_ok=True)
    points = local(triangles).reshape(-1, 3)
    save_obj(meshes/'board_visual.obj', points, np.arange(len(points)).reshape(-1, 3))
    entries = []
    layers = [(Polygon([(0,0), (170,0), (170,60), (0,60)]), 0, 5)]
    layers += [(Polygon(r), 5, 30) for r in base_top.interiors]
    layers += [(p, 30, 33) for p in convex_parts(top)]
    for index, (p, lo, hi) in enumerate(layers):
        canonical = Polygon(np.column_stack([85-np.array(p.exterior.coords)[:,0],
                                             np.array(p.exterior.coords)[:,1]-30])*.001)
        pts, faces = prism(canonical, lo*.001, hi*.001)
        filename = f'board_collision_{index:02d}.obj'
        save_obj(meshes/filename, pts, faces)
        entries.append({'mesh': 'meshes/'+filename, 'points': pts.tolist(), 'faces': faces.tolist(),
                        'z_range_m': [lo*.001, hi*.001]})
    hole_records = []
    for index, hole in enumerate(holes):
        coords = np.array(hole.exterior.coords)[:-1]
        xy = np.column_stack([85-coords[:,0], coords[:,1]-30])*.001
        hole_records.append({'id': index, 'center_xy_m': xy.mean(0).tolist(),
                             'bounds_xy_m': [xy.min(0).tolist(), xy.max(0).tolist()],
                             'polygon_xy_m': xy.tolist()})
    # Source left-to-right in x: two-pin pair, three-pin group, rectangular spare.
    pair = hole_records[:2]
    three = hole_records[2:5]
    target_two = np.mean([h['center_xy_m'] for h in pair], axis=0)
    lower = sorted(three, key=lambda h: h['center_xy_m'][1])[:2]
    target_three = np.mean([h['center_xy_m'] for h in lower], axis=0)
    source_volume=abs(np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum())/6*1e-9
    collision_volume=sum(p.area*(hi-lo) for p,lo,hi in layers)*1e-9
    assert abs(source_volume-collision_volume)<1e-10
    manifest = {
        'schema_version': 1, 'units': 'm', 'source': str(source.relative_to(ROOT)),
        'source_sha256': digest(source), 'source_units': 'mm (interpreted from CAD scale and photographs; STL has no unit tag)',
        'source_bounds': [[0,0,0],[170,33,60]], 'bounds_m': [[-.085,-.03,0],[.085,.03,.033]],
        'source_to_local': [[-.001,0,0,.085],[0,0,.001,-.03],[0,.001,0,0],[0,0,0,1]],
        'top_z_m': .033, 'face_thickness_m': .003, 'visual_mesh': 'meshes/board_visual.obj',
        'collision_parts': entries, 'holes': hole_records,
        'targets': {'black_charger': {'center_xy_m': target_two.tolist(), 'hole_ids': [h['id'] for h in pair]},
                    'yellow_plug': {'center_xy_m': target_three.tolist(), 'hole_ids': [h['id'] for h in three]}},
        'geometry_policy': 'source STL exact; convex layer partitions preserve six open holes',
        'collision_volume_m3': float(collision_volume), 'source_volume_m3': float(source_volume),
        'photo_references': sorted(str(p.relative_to(ROOT)) for p in source.parent.glob('*.jpg')),
        'mesh_sha256': {p.name: digest(p) for p in sorted(meshes.glob('*.obj'))}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('bounds_m','targets','collision_volume_m3')}, indent=2))
    print(f'{len(entries)} convex pieces; {len(holes)} through holes; {output}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/tasks/connector')
    args=parser.parse_args()
    prepare(args.output)
