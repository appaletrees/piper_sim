#!/usr/bin/env python3
"""Resolve recorded native contact forces into payload load contributions.

Reads original solver buffers from trace_contact_substeps, without replaying
contacts or velocities. An isolated model supplies only body membership, mass
and gravity, verified against trace geometry names and frozen input hashes.
Finger forces, other robot forces and external support are kept separate.
This is a diagnostic, not a stable-grasp or training-acceptance certificate.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256


def contact_forces_on_payload(frames, local_forces, payload_is_geom2):
    """MuJoCo contact-frame force points from geom1 to geom2."""
    frames = np.asarray(frames, float)
    local_forces = np.asarray(local_forces, float)
    signs = np.where(np.asarray(payload_is_geom2, bool), 1., -1.)
    if (frames.shape != (len(signs), 3, 3) or local_forces.shape != (len(signs), 6)
            or not np.isfinite(frames).all() or not np.isfinite(local_forces).all()):
        raise ValueError('Finite contact frame and six-dimensional local force required')
    return signs[:, None] * np.einsum('nij,ni->nj', frames, local_forces[:, :3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--object', default='plate')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.trace/'trace.json').read_text())
    if not meta.get('includes_task_contacts') or args.object not in meta['selected_objects']:
        raise ValueError('Trace must include all contacts of the selected payload')
    trace_path = args.trace/'trace.npz'
    if sha256(trace_path) != meta['trace_sha256']:
        raise ValueError('Native trace hash mismatch')
    frozen = json.loads(args.identity.read_text())['files']
    changed = [p for p, h in frozen.items() if sha256(p) != h]
    if changed:
        raise ValueError('Frozen inputs changed: '+repr(changed))
    source_hash = sha256(args.recording)
    with h5py.File(args.recording, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed recording required')
        recording_meta = json.loads(f.attrs['metadata'])
    cfg = copy.deepcopy(recording_meta['config'])
    cfg['task_scene']['domain_randomization'] = recording_meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg, cameras=False)
    try:
        layout = recording_meta['task_scene']['layout']
        backend.reset(layout['seed'], layout['randomized'])
        model = backend.model
        if [model.geom(i).name for i in range(model.ngeom)] != meta['geom_names']:
            raise ValueError('Trace geometry order differs from isolated model')
        bodies = np.asarray([model.body(int(b)).name for b in model.geom_bodyid])
        weight = float(model.body('task/'+args.object).mass[0] * np.linalg.norm(model.opt.gravity))
        up = -model.opt.gravity.copy()/np.linalg.norm(model.opt.gravity)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Positive payload weight required')
    finally:
        backend.close()
    with np.load(trace_path, allow_pickle=False) as archive:
        z = {key: archive[key] for key in ('time_before', 'phase', 'target', 'control',
            'contact_row', 'contact_geom1', 'contact_geom2', 'contact_frame', 'contact_force_local')}
    body1, body2 = bodies[z['contact_geom1']], bodies[z['contact_geom2']]
    target = 'task/'+args.object
    selected = (body1 == target) ^ (body2 == target)
    other = np.where(body2[selected] == target, body1[selected], body2[selected])
    rows = z['contact_row'][selected].astype(int)
    force = contact_forces_on_payload(z['contact_frame'][selected].reshape(-1, 3, 3),
        z['contact_force_local'][selected], body2[selected] == target)
    finger = np.isin(other, ['left/link7', 'left/link8', 'right/link7', 'right/link8'])
    robot = np.char.startswith(other, 'left/') | np.char.startswith(other, 'right/')
    masks = {'fingers': finger, 'other_robot': robot & ~finger, 'external': ~robot}
    n = len(z['time_before'])
    loads = {}
    for label, mask in masks.items():
        total = np.zeros((n, 3))
        np.add.at(total, rows[mask], force[mask])
        loads[label] = total@up
    snapshots = []
    phases = {}
    for phase in sorted(set(z['phase'][z['target'] == args.object])):
        ids = np.flatnonzero((z['target'] == args.object) & (z['phase'] == phase))
        stats = {'native_steps': len(ids), 'first_last_solver_time': z['time_before'][ids[[0, -1]]].tolist()}
        for label, values in loads.items():
            ratios = values[ids]/weight
            stats[label] = {'min_max_upward_N': [float(values[ids].min()), float(values[ids].max())],
                            'min_max_weight_fraction': [float(ratios.min()), float(ratios.max())],
                            'steps_downward': int(np.sum(values[ids] < -1e-8)),
                            'steps_upward': int(np.sum(values[ids] > 1e-8))}
        phases[phase] = stats
        # Keep evenly spaced points and extrema, not a resampled training stream.
        keep = set(ids[np.linspace(0, len(ids)-1, min(16, len(ids)), dtype=int)])
        for values in loads.values():
            keep.update((ids[np.argmin(values[ids])], ids[np.argmax(values[ids])]))
        for i in sorted(keep):
            snapshots.append({'native_row': int(i), 'phase': phase, 'solver_time': float(z['time_before'][i]),
                              'commanded_right_opening_m': float(z['control'][i, 6]),
                              'upward_force_N': {k: float(v[i]) for k, v in loads.items()}})
    if sha256(args.recording) != source_hash or sha256(trace_path) != meta['trace_sha256']:
        raise ValueError('Source changed during audit')
    out = {'scope': __doc__, 'script_sha256': sha256(__file__), 'source': str(args.recording),
           'source_sha256': source_hash, 'trace_sha256': meta['trace_sha256'], 'source_unchanged': True,
           'frozen_input_count': len(frozen), 'frozen_inputs_changed': changed,
           'object': args.object, 'weight_N': weight, 'up_axis_world': up.tolist(),
           'payload_contacts': int(selected.sum()), 'native_rows': n, 'phases': phases, 'snapshots': snapshots,
           'training_ready': False, 'timing': meta['timing'],
           'limitations': ['These are original solver force-evaluation samples, not forces recomputed from rounded observations.',
                          'No mass-acceleration balance is claimed; object velocities and accelerations were not included in this trace.',
                          'Upward load alone does not establish grasp closure, torque balance, stability or physical hardware calibration.']}
    atomic_json(args.output, out)
    print(json.dumps({'weight_N': weight, 'phases': phases}, indent=2))


if __name__ == '__main__':
    main()
