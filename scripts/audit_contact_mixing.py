#!/usr/bin/env python3
"""Inspect effective contact friction on a recorded grasp in a scratch model.

Only object geom friction is varied. No integration, live controls, contact-force
inference, source mutation or training acceptance. This tests whether configured
object friction randomization reaches the actual finger contact constraints.
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
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', default='pen')
    parser.add_argument('--scales', nargs='+', type=float, default=[.8, 1., 1.2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output')
    if not np.isfinite(args.scales).all() or min(args.scales) <= 0:
        parser.error('Friction scales must be finite and positive')
    with h5py.File(args.source, 'r') as source:
        if not source.attrs.get('complete'): raise ValueError('A closed episode is required')
        metadata = json.loads(source.attrs['metadata'])
        selected = None
        for index, raw in enumerate(source['next_observation/extras_json']):
            extras = json.loads(raw)
            teacher = extras.get('teacher', {})
            if (teacher.get('object') == args.object and teacher.get('phase') == 'lift'
                    and len((teacher.get('grasp') or {}).get('contact_fingers', [])) == 2):
                selected = (index, extras)
                break
    if selected is None: raise ValueError('No recorded bilateral lift found')
    cfg = copy.deepcopy(metadata['config'])
    cfg['task_scene']['domain_randomization'] = metadata['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    cases = []
    try:
        layout = metadata['task_scene']['layout']
        backend.reset(layout['seed'], layout['randomized'])
        if backend.task.specs[args.object]['kind'] not in ('pen', 'chopstick'):
            raise ValueError('Only pen/chopstick baseline finger material is supported; no rim-specific overrides')
        RigidTableTeacher(SimulationSession(backend), settings=metadata['teacher']['motion_and_quality'])
        index, extras = selected
        for side, arm in extras['teacher']['physics']['arms'].items():
            backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extras['objects'].items():
            address = backend.model.joint('task/'+name+'/free').qposadr[0]
            backend.data.qpos[address:address+7] = [*pose['position'], *pose['quaternion_wxyz']]
        body = backend.model.body('task/'+args.object).id
        gids = np.flatnonzero(backend.model.geom_bodyid == body)
        nominal = np.array([cfg['task_scene']['contact']['friction'], .003, .0001])
        for scale in args.scales:
            backend.model.geom_friction[gids] = nominal*scale
            mujoco.mj_fwdPosition(backend.model, backend.data)
            contacts = []
            for contact in backend.data.contact:
                names = [backend.model.body(int(backend.model.geom_bodyid[g])).name for g in contact.geom]
                if ('task/'+args.object in names and any(n in ('right/link7', 'right/link8') for n in names)):
                    contacts.append({'bodies': names,
                        'geom_priorities': backend.model.geom_priority[contact.geom].tolist(),
                        'geom_friction': backend.model.geom_friction[contact.geom].tolist(),
                        'effective_contact_friction': contact.friction.tolist(), 'contact_dim': int(contact.dim)})
            if not contacts: raise ValueError('Restored pose did not produce finger contacts')
            cases.append({'object_friction_scale': scale, 'contacts': contacts})
    finally:
        backend.close()
    vectors = [np.array([c['effective_contact_friction'] for c in case['contacts']]) for case in cases]
    report = {'status': 'contact_parameter_diagnostic_only', 'scope': __doc__,
        'source': str(args.source.resolve()), 'object': args.object, 'frame_next_observation': index,
        'cases': cases, 'effective_friction_changes': any(
            vector.shape != vectors[0].shape or not np.array_equal(vector, vectors[0]) for vector in vectors[1:]),
        'mujoco_version': mujoco.__version__, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitation': 'Current source controller on recorded geometry. Not dynamic replay, real material calibration, '
                      'or coverage of other contact pairs. No changes to source data or acceptance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'effective_friction_changes': report['effective_friction_changes']}))


if __name__ == '__main__':
    main()
