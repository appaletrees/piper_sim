#!/usr/bin/env python3
"""Read-only release geometry from recorded native joints and object poses.

Uses scratch kinematics, not a dynamics replay. Recomputed overlaps do not prove
contact forces or a cause of failure. No source recording or acceptance changes.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.transforms import pose_matrix


def inspect(source, name):
    with h5py.File(source, 'r') as f:
        if not f.attrs.get('complete'):
            raise ValueError('Requires a closed complete recording')
        meta = json.loads(f.attrs['metadata'])
        changed_sources = []
        for relative, expected in meta.get('provenance', {}).items():
            path = ROOT/relative
            actual = sha256(path) if path.is_file() else None
            if actual != expected:
                changed_sources.append({'path': relative, 'recorded_sha256': expected, 'current_sha256': actual})
        event = next(e for e in meta['collection_outcome']['teacher_events']
                     if e['phase'] == 'open_gripper' and e['object'] == name)
        times = f['observation/sim_time'][:]
        first = int(np.searchsorted(times, event['sim_time']-1.))
        cfg = copy.deepcopy(meta['config'])
        cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
        backend = MujocoBackend(cfg, cameras=False)
        rows = []
        try:
            backend.reset(meta['task_scene']['layout']['seed'], meta['task_scene']['layout']['randomized'])
            model, data, task = backend.model, backend.data, backend.task
            if meta['teacher'].get('multiccd'):
                model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
            target = task.specs[name]['target']
            object_id = model.body('task/'+name).id
            inner_half = np.asarray(task.specs[target]['size_m'][:2])/2-task.specs[target]['wall_m']
            for frame in range(first, len(times)+1):
                prefix, index = ('observation', frame) if frame < len(times) else ('next_observation', frame-1)
                extra = json.loads(f[prefix+'/extras_json'][index])
                for side, native in extra['teacher']['physics']['arms'].items():
                    data.qpos[backend.indices[side]['qpos']] = native['joint_position']
                for item, pose in extra['objects'].items():
                    address = model.joint('task/'+item+'/free').qposadr[0]
                    data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
                mujoco.mj_fwdPosition(model, data)
                obj, container = (pose_matrix(extra['objects'][n]['position'], extra['objects'][n]['quaternion_wxyz'])
                                  for n in (name, target))
                local = (task.surface_points[name]@obj[:3, :3].T+obj[:3, 3]-container[:3, 3])@container[:3, :3]
                pairs = {}
                for contact in data.contact:
                    ids = model.geom_bodyid[contact.geom]
                    if object_id not in ids or contact.dist > 0.:
                        continue
                    other = model.body(int(ids[1] if ids[0] == object_id else ids[0])).name
                    pairs[other] = max(pairs.get(other, 0.), float(-contact.dist))
                rows.append({'frame': frame, 'sim_time': float(f[prefix+'/sim_time'][index]),
                    'phase': extra['teacher']['phase'], 'object_pose': extra['objects'][name],
                    'bbox_basket_m': [local.min(0).tolist(), local.max(0).tolist()],
                    'xy_inside_margin_m': float(np.min(inner_half-abs(local[:, :2]).max(0))),
                    'long_axis_tilt_deg': float(np.rad2deg(np.arcsin(np.clip(abs(obj[2, 0]), 0., 1.)))),
                    'overlap_depth_by_body_m': pairs})
        finally:
            backend.close()
    sampled = []
    for delta in (-1., 0., .5, 1., 2., 4.):
        row = min(rows, key=lambda r: abs(r['sim_time']-event['sim_time']-delta))
        if not sampled or sampled[-1]['frame'] != row['frame']:
            sampled.append(row)
    if sampled[-1]['frame'] != rows[-1]['frame']:
        sampled.append(rows[-1])
    return {'source': str(source.resolve()), 'source_sha256': sha256(source), 'object': name,
            'open_event': event, 'reconstructed_frames': len(rows), 'selected_frames': sampled,
            'source_provenance_changes': changed_sources,
            'all_observed_overlap_bodies': sorted({k for r in rows for k in r['overlap_depth_by_body_m']})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--object', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Preserve existing reports; select a new output')
    result = inspect(args.source, args.object)
    atomic_json(args.output, {'status': 'recorded_kinematic_diagnostic', 'training_ready': False,
        'scope': __doc__, 'script_sha256': sha256(__file__), **result})
    print(json.dumps({'frames': result['reconstructed_frames'], 'bodies': result['all_observed_overlap_bodies'],
                     'report': str(args.output)}))


if __name__ == '__main__': main()
