"""Exact-result support-query benchmark; deliberately not episode throughput."""
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
from piper_sim.experts.placement_search import support_surface, collision_surface_points


def unfiltered(surface, xy):
    """Original all-triangle arithmetic, kept independent of broad-phase bounds."""
    result = np.full(len(xy), -np.inf)
    a, u, v, det = (getattr(surface, key) for key in ('a', 'u', 'v', 'det'))
    for start in range(0, len(xy), 64):
        delta = xy[start:start+64, None]-a[None, :, :2]
        s = (delta[:, :, 0]*v[:, 1]-delta[:, :, 1]*v[:, 0])/det
        t = (u[:, 0]*delta[:, :, 1]-u[:, 1]*delta[:, :, 0])/det
        inside = (s >= -1e-10) & (t >= -1e-10) & (s+t <= 1+1e-10)
        height = a[:, 2]+s*u[:, 2]+t*v[:, 2]
        if height.shape[1]: result[start:start+64] = np.max(np.where(inside, height, -np.inf), axis=1)
    return result


def main():
    out = ROOT/'outputs/platform_audit_v1'
    destination = out/'support_query_benchmark_v2.json'
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
            clouds = {'visible_vertices': b.task.surface_points[name]}
            if name.startswith('chopstick'): clouds['collision_surface_5mm'] = collision_surface_points(b.task.parts[name], .005)
            for label, points in clouds.items():
                queries = [(points@rotation.T+basket.xpos+rotation@np.r_[nominal+[x,y],0.])[:, :2]
                           for x,y in ((0.,0.),(-.05,0.),(0.,.05))]
                queries += [surface.a[:64, :2].copy(), np.array([[100.,100.]])]
                assert all(np.array_equal(unfiltered(surface, q), surface.height(q)) for q in queries)
                timing = {'unfiltered': [], 'production_broad_phase': []}
                methods = {'unfiltered': lambda q:unfiltered(surface,q), 'production_broad_phase': surface.height}
                for repeat in range(3):
                    for method in (list(methods) if repeat%2==0 else list(reversed(methods))):
                        start = time.perf_counter()
                        for q in queries: methods[method](q)
                        timing[method].append(time.perf_counter()-start)
                rows.append({'seed': seed, 'source_sha256':sha256(source), 'source_frame':index,
                    'object':name, 'payload_points':label, 'queries':len(queries), 'points':sum(map(len,queries)),
                    'projectable_triangles':len(surface.a), 'exact_heights_equal':True, 'seconds':timing,
                    'median_query_speedup':float(np.median(timing['unfiltered'])/np.median(timing['production_broad_phase']))})
        finally:b.close()
    atomic_json(destination, {'status':'microbenchmark_only', 'script_sha256':sha256(__file__),
        'production_source_sha256':sha256(ROOT/'src/piper_sim/experts/placement_search.py'), 'rows':rows,
        'limitations':['Shared host load; no claim about full-task planning or qualified throughput.',
            'Exact equality is checked on these query inputs; independent tests also cover tolerance-expanded edges.',
            'Collision sampling is still discrete and does not prove dynamic landing stability.']})
    print(json.dumps([{k:r[k] for k in ('object','payload_points','exact_heights_equal','median_query_speedup')} for r in rows]))


if __name__=='__main__':main()
