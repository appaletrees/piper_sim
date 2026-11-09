"""Inspect a zero distance report against an independent separating plane.

Only a reconstructed diagnostic backend is used. No solver option is changed.
"""
import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256

spec = importlib.util.spec_from_file_location('candidate_probe', 'scripts/diagnose_grasp_clearance.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
original = probe.robot_obstacle_clearance
output = Path('outputs/platform_audit_v1/tcp_grasp_distance_query_diagnostic_v1.json')
if output.exists(): raise FileExistsError(output)


class Captured(Exception): pass


def capture(model, data, *args):
    distance, pair = original(model, data, *args)
    if distance != 0. or not pair: return distance, pair
    for a in np.flatnonzero(model.geom_bodyid == model.body(pair[0]).id):
        for b in np.flatnonzero(model.geom_bodyid == model.body(pair[1]).id):
            if (model.geom_type[a] != mujoco.mjtGeom.mjGEOM_MESH
                    or model.geom_type[b] != mujoco.mjtGeom.mjGEOM_BOX): continue
            if mujoco.mj_geomDistance(model, data, int(a), int(b), .01, None) != 0.: continue
            mesh = model.geom_dataid[a]
            adr, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            av = model.mesh_vert[adr:adr+count]@data.geom_xmat[a].reshape(3, 3).T+data.geom_xpos[a]
            bv = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*model.geom_size[b]
            bv = bv@data.geom_xmat[b].reshape(3, 3).T+data.geom_xpos[b]
            adr, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            faces = model.mesh_face[adr:adr+count]
            normals = np.cross(av[faces[:, 1]]-av[faces[:, 0]], av[faces[:, 2]]-av[faces[:, 0]])
            axes = np.vstack([normals, data.geom_xmat[b].reshape(3, 3).T])
            lengths = np.linalg.norm(axes, axis=1)
            axes = axes[lengths > 1e-10]/lengths[lengths > 1e-10, None]
            ap, bp = av@axes.T, bv@axes.T
            gaps = np.maximum(ap.min(0)-bp.max(0), bp.min(0)-ap.max(0))
            caps = {str(cap): mujoco.mj_geomDistance(model, data, int(a), int(b), cap, None)
                    for cap in (.0005, .001, .0015, .002, .005, .01, .02, .1)}
            result = {'training_ready': False, 'bodies': pair,
                'geom_names': [model.geom(int(g)).name for g in (a, b)],
                'mujoco_version': mujoco.__version__, 'caps': caps,
                'ccd_iterations': model.opt.ccd_iterations, 'ccd_tolerance': model.opt.ccd_tolerance,
                'separating_axis_gap_m': float(gaps.max()),
                'separating_axis': axes[np.argmax(gaps)].tolist(),
                'world_convex_vertices': [av.tolist(), bv.tolist()], 'script_sha256': sha256(__file__),
                'scope': 'Static scratch geometry only. Positive projection gap along a unit axis '
                         'is a conservative distance lower bound for both convex vertex hulls.',
                'limitations': ['This single pose does not certify the whole path.',
                    'A prior exploratory solver-iteration sweep crashed before producing evidence; '
                    'no production solver setting or recorded source was changed.']}
            atomic_json(output, result)
            print({k: v for k, v in result.items() if k != 'world_convex_vertices'}, flush=True)
            raise Captured
    return distance, pair


probe.robot_obstacle_clearance = capture
sys.argv = ['probe', '/dev/shm/piper_tcp_holdout_v1/worker_02/episode_000001_seed_1340830695/episode.h5',
    '--object', 'chopstick_a', '--collection-profile', 'configs/collection/clean_table_tcp_clearance.yaml',
    '--output', '/dev/shm/unwritten_grasp_diagnostic.json']
try: probe.main()
except Captured: pass
