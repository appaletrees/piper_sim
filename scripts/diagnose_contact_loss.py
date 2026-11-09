#!/usr/bin/env python3
"""Inspect recorded grasp-loss geometry without advancing or editing a recording.

Contact distances/positions are reconstructed from recorded native joint and
object poses. Forces are copied from the original 30 Hz telemetry, never inferred
from a fresh solver run. This does not recover unrecorded substep extrema.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', required=True)
    parser.add_argument('--last-frames', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.last_frames < 1 or args.output.exists(): parser.error('Use positive frame count and a new output')
    with h5py.File(args.source, 'r') as source:
        if not source.attrs.get('complete'): raise ValueError('A closed recording is required')
        metadata = json.loads(source.attrs['metadata'])
        count = int(source.attrs['num_transitions'])
        snapshots = [(i, json.loads(source['next_observation/extras_json'][i]))
                     for i in range(max(0, count-args.last_frames), count)]
    cfg = copy.deepcopy(metadata['config'])
    cfg['task_scene']['domain_randomization'] = metadata['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    records = []
    try:
        layout = metadata['task_scene']['layout']
        backend.reset(layout['seed'], layout['randomized'])
        if metadata['teacher'].get('multiccd'):
            backend.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        for index, extras in snapshots:
            evidence = extras['teacher']['physics']
            if evidence['target'] != args.object: continue
            for side, arm in evidence['arms'].items():
                backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
            for name, pose in extras['objects'].items():
                address = backend.model.joint('task/'+name+'/free').qposadr[0]
                backend.data.qpos[address:address+7] = [*pose['position'], *pose['quaternion_wxyz']]
            # Only geometric collision queries; no force solve or integration.
            mujoco.mj_kinematics(backend.model, backend.data)
            mujoco.mj_collision(backend.model, backend.data)
            contacts = []
            for contact in backend.data.contact:
                names = [backend.model.body(int(backend.model.geom_bodyid[g])).name for g in contact.geom]
                if 'task/'+args.object not in names: continue
                contacts.append({'bodies': names,
                    'geoms': [backend.model.geom(int(g)).name for g in contact.geom],
                    'signed_distance_m': float(contact.dist), 'position_world_m': contact.pos.tolist()})
            right = evidence['arms']['right']
            records.append({'next_observation_index': index, 'sim_time': evidence['sim_time'],
                'phase': extras['teacher']['phase'], 'grasp': extras['teacher']['grasp'],
                'recorded_right_aperture_m': sum(right['joint_position'][6:]),
                'recorded_right_finger_forces': right['target_finger_contacts'],
                'recorded_object_pose': extras['objects'][args.object],
                'reconstructed_contacts': contacts})
    finally:
        backend.close()
    if not records:
        raise ValueError('No requested target in the selected recorded interval')
    report = {'status': 'geometry_diagnostic_only', 'scope': __doc__, 'training_ready': False,
        'source': str(args.source.resolve()), 'source_sha256': sha256(args.source),
        'source_contact_profile': metadata['teacher'].get('experimental_contact_profile'),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'object': args.object, 'frames': records,
        'limitations': 'Geometric contacts use reset collision masks and recorded poses, without solver history. '
            'The recorded multi-contact flag is restored. '
            'Positive-margin pairs outside their collision margin are not reported. Recorded forces are sampled.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'frames': len(records)}))


if __name__ == '__main__':
    main()
