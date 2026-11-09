#!/usr/bin/env python3
"""Read-only, all-frame kinematic contact/TCP diagnostic from a recorded episode.

Uses a separate MuJoCo data instance; never drives hardware or changes the source.
This is not a dynamics replay, a contact-force estimate, or a training acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np

from piper_sim.config import ROOT
from piper_sim.backends.mujoco import MujocoBackend


def arm_of(name):
    prefix, _, _ = name.partition('/')
    return prefix if prefix in ('left', 'right') else None


def contact_category(names, target):
    arms = [arm_of(name) for name in names]
    if not any(arms):
        return None
    if all(arms):
        return 'same_arm' if arms[0] == arms[1] else 'inter_arm'
    if target and 'task/'+target in names and any(n.endswith(('/link7', '/link8')) for n in names):
        return 'intended_finger_target'
    return 'robot_environment'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    started = time.perf_counter()
    with h5py.File(args.source, 'r') as source:
        if not source.attrs.get('complete', False):
            raise ValueError('A closed, complete episode is required')
        meta = json.loads(source.attrs['metadata'])
        cfg = meta['config']
        if meta['task_scene'].get('domain_profile'):
            cfg['task_scene']['domain_randomization'] = meta['task_scene']['domain_profile']
        if meta.get('backend') != 'MujocoBackend':
            raise ValueError('This diagnostic requires a MuJoCo episode')
        q = np.concatenate((source['observation/state'][:], source['next_observation/state'][-1:]))
        times = np.r_[source['observation/sim_time'][:], source['next_observation/sim_time'][-1]]
        if not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid source states/times')
        backend = MujocoBackend(cfg, cameras=False)
        pairs, peaks = {}, {}
        tcp = {side: [] for side in backend.indices}
        changed = []
        for name, expected in meta.get('provenance', {}).items():
            path = ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(name)
        try:
            model, data = backend.model, backend.data
            names = [model.body(i).name for i in range(model.nbody)]
            geom_names = [model.geom(i).name for i in range(model.ngeom)]
            task_addresses = {name: model.joint('task/'+name+'/free').qposadr[0] for name in backend.task.specs}
            for name, pose in meta['task_scene']['layout']['poses'].items():
                address = task_addresses[name]
                data.qpos[address:address+7] = [*pose['position'], *pose['quaternion_wxyz']]
            if meta.get('teacher', {}).get('multiccd'):
                model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
            for frame, position in enumerate(q):
                prefix, index = ('observation', frame) if frame < len(q)-1 else ('next_observation', frame-1)
                extras = json.loads(source[prefix+'/extras_json'][index])
                teacher = extras.get('teacher', {})
                # No clipping: retain measured solver overshoot. The legacy
                # recording has total width only, so individual jaw asymmetry
                # cannot be recovered and is explicitly reported below.
                for side, offset in (('left', 0), ('right', 7)):
                    data.qpos[backend.indices[side]['qpos']] = [*position[offset:offset+6],
                                                               position[offset+6]/2, position[offset+6]/2]
                for name, pose in extras['objects'].items():
                    address = task_addresses[name]
                    data.qpos[address:address+7] = [*pose['position'], *pose['quaternion_wxyz']]
                data.time = times[frame]
                mujoco.mj_fwdPosition(model, data)
                for side in tcp:
                    hand = data.body(side+'/link6')
                    tcp[side].append(hand.xpos+hand.xmat.reshape(3, 3) @ np.array([0., 0., .1375]))
                frame_pairs = set()
                for contact in data.contact:
                    if contact.dist >= 0:
                        continue
                    bodies = [names[int(model.geom_bodyid[g])] for g in contact.geom]
                    category = contact_category(bodies, teacher.get('object'))
                    if category is None:
                        continue
                    key = category+':'+','.join(sorted(bodies))
                    record = pairs.setdefault(key, {'category': category, 'bodies': sorted(bodies),
                        'frames_with_contact': 0, 'max_depth_m': 0.})
                    frame_pairs.add(key)
                    if -contact.dist > record['max_depth_m']:
                        record.update(max_depth_m=float(-contact.dist), frame=frame,
                            sim_time=float(times[frame]), phase=teacher.get('phase'), object=teacher.get('object'),
                            geoms=[geom_names[g] for g in contact.geom])
                    if category not in peaks or record['max_depth_m'] > peaks[category]['max_depth_m']:
                        peaks[category] = record
                for key in frame_pairs:
                    pairs[key]['frames_with_contact'] += 1
            tcp_report = {}
            for side, positions in tcp.items():
                velocity = np.diff(positions, axis=0)/np.diff(times)[:, None]
                speed = np.linalg.norm(velocity, axis=1)
                peak = int(np.argmax(speed))
                tcp_report[side] = {'max_sampled_speed_m_s': float(speed[peak]),
                                    'p95_sampled_speed_m_s': float(np.percentile(speed, 95)),
                                    'peak_interval': [float(times[peak]), float(times[peak+1])]}
            report = {'schema_version': 'piper.geometry_diagnostic.v1', 'status': 'diagnostic_only',
                'source': str(args.source.resolve()), 'samples': len(q), 'stride': 1,
                'mujoco_version': mujoco.__version__, 'recorded_software': meta.get('software'),
                'changed_or_missing_provenance_files': changed,
                'contact_pairs': sorted(pairs.values(), key=lambda x: -x['max_depth_m']),
                'category_peaks': peaks, 'tcp': tcp_report,
                'excluded_body_pairs': [int(v) for v in model.exclude_signature],
                'limitations': [
                    'Position-only reconstruction, not dynamic replay; no contact-force, slip or torque inference.',
                    'All recorded samples including terminal state, but contacts between 30 Hz samples are unobserved.',
                    'Only total gripper width was recorded; each jaw is reconstructed as half the width.',
                    'Inactive task objects retain their recorded initial layout poses; active poses are reconstructed each frame.',
                    'Current compiled geometry retains adjacent body exclusions and convex-hull approximations.',
                    'Provenance differences are listed; this diagnostic does not certify identical source assets.',
                    'TCP is the configured link6 offset [0,0,0.1375] m; not a measured tool calibration.',
                    'No change to episode acceptance or source data.',
                ]}
        finally:
            backend.close()
    report['wall_seconds'] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'samples', 'category_peaks', 'tcp', 'wall_seconds')}))


if __name__ == '__main__':
    main()
