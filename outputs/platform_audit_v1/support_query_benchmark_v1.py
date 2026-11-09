"""Read-only exact-result broad-phase prototype; not collection throughput."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.experts.placement_search import support_surface


def filtered(surface, xy):
    # Conservative triangle XY bounds; leave the original barycentric tolerance
    # and exact arithmetic/reduction unchanged on all potentially relevant faces.
    vertices = np.stack((surface.a, surface.a+surface.u, surface.a+surface.v), axis=1)
    pad = np.max(np.abs(vertices), axis=(0, 1))*1e-9+1e-9
    selected = np.all(vertices.max(1)[:, :2] >= xy.min(0)-pad[:2], axis=1)
    selected &= np.all(vertices.min(1)[:, :2] <= xy.max(0)+pad[:2], axis=1)
    a, u, v, det = [getattr(surface, name)[selected] for name in ('a', 'u', 'v', 'det')]
    result = np.full(len(xy), -np.inf)
    for first in range(0, len(xy), 64):
        delta = xy[first:first+64, None]-a[None, :, :2]
        s = (delta[:, :, 0]*v[:, 1]-delta[:, :, 1]*v[:, 0])/det
        t = (u[:, 0]*delta[:, :, 1]-u[:, 1]*delta[:, :, 0])/det
        inside = (s >= -1e-10) & (t >= -1e-10) & (s+t <= 1+1e-10)
        height = a[:, 2]+s*u[:, 2]+t*v[:, 2]
        if height.shape[1]: result[first:first+64] = np.max(np.where(inside, height, -np.inf), axis=1)
    return result


def main():
    out = ROOT/'outputs/platform_audit_v1'
    destination = out/'support_query_benchmark_v1.json'
    if destination.exists(): raise FileExistsError(destination)
    sources = [(13, 'chopstick_a', out/'servo_validation_states_v1/worker_00/episode_000000_seed_13/episode.h5'),
               (15619, 'side_dish', out/'servo_validation_states_v1/worker_01/episode_000001_seed_15619/episode.h5')]
    rows = []
    for seed, name, source in sources:
        with h5py.File(source, 'r') as f:
            meta = json.loads(f.attrs['metadata'])
            event = next(e for e in meta['collection_outcome']['teacher_events'] if e['phase']=='transfer' and e['object']==name)
            index = int(np.argmin(abs(f['observation/sim_time'][:]-event['sim_time'])))
            extras = json.loads(f['observation/extras_json'][index])
        cfg = copy.deepcopy(meta['config']);cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
        b = MujocoBackend(cfg, cameras=False)
        try:
            layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
            for n, pose in extras['objects'].items():
                adr = b.model.joint('task/'+n+'/free').qposadr[0];b.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
            mujoco.mj_forward(b.model,b.data)
            surface, _ = support_surface(SimpleNamespace(b=b), 'basket', [name])
            basket = b.data.body('task/basket');rotation=basket.xmat.reshape(3,3)
            nominal = np.asarray(b.task.config['drop_slots_m'][name])
            queries = [(b.task.surface_points[name]@rotation.T+basket.xpos+rotation@np.r_[nominal+[x,y],0.])[:, :2]
                       for x in (-.05,0.,.05) for y in (-.05,0.,.05)]
            # Add exact mesh-vertex boundary queries and a no-hit query.
            queries += [surface.a[:64, :2].copy(), np.array([[100.,100.]])]
            equality = all(np.array_equal(surface.height(q), filtered(surface,q)) for q in queries)
            assert equality
            timing = {'original': [], 'broad_phase_prototype': []}
            methods = {'original': surface.height, 'broad_phase_prototype': lambda q:filtered(surface,q)}
            for repeat in range(3):
                for label in (list(methods) if repeat%2==0 else list(reversed(methods))):
                    start=time.perf_counter()
                    for q in queries: methods[label](q)
                    timing[label].append(time.perf_counter()-start)
            rows.append({'seed': seed, 'source_sha256': sha256(source), 'source_frame': index,
                'object': name, 'queries': len(queries), 'points': sum(len(q) for q in queries),
                'projectable_triangles': len(surface.a), 'exact_heights_equal': equality,
                'seconds': timing, 'median_query_speedup': float(np.median(timing['original'])/np.median(timing['broad_phase_prototype']))})
        finally: b.close()
    atomic_json(destination, {'status': 'microbenchmark_only', 'script_sha256': sha256(__file__),
        'production_source_sha256': sha256(ROOT/'src/piper_sim/experts/placement_search.py'), 'rows': rows,
        'limitations': ['Prototype only; production source is unchanged during full-task validation.',
            'Fixed geometry queries, not independent episodes or end-to-end qualified collection throughput.',
            'Shared host load; IK, collision planning and physics time are excluded.']})
    print(json.dumps(rows,indent=2))


if __name__ == '__main__': main()
