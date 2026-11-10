#!/usr/bin/env python3
"""Read original native payload contact wrenches about its recorded COM.

No contact re-solving or dynamics replay. Net contact torque about COM and the
gravity moment about a moving finger-contact centroid are diagnostics, not a
static/dynamic equilibrium certificate or a calibrated hardware force limit.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from audit_native_payload_load import contact_forces_on_payload
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256


def wrenches_about_com(frames, local_forces, payload_is_geom2, positions, centers):
    force = contact_forces_on_payload(frames, local_forces, payload_is_geom2)
    positions, centers = np.asarray(positions), np.asarray(centers)
    if positions.shape != force.shape or centers.shape != force.shape:
        raise ValueError('One contact position and payload COM per force required')
    if not np.isfinite(positions).all() or not np.isfinite(centers).all():
        raise ValueError('Finite contact position and COM required')
    # Contact force arrays are force:torque; the queried velocity is rot:lin.
    swapped = np.concatenate((local_forces[:, 3:], local_forces[:, :3]), axis=1)
    direct_torque = contact_forces_on_payload(frames, swapped, payload_is_geom2)
    return force, np.cross(positions-centers, force)+direct_torque


def summary(values):
    return {'min': float(np.min(values)), 'max': float(np.max(values)), 'mean': float(np.mean(values))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--object', default='plate')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    tm = json.loads((args.trace/'trace.json').read_text())
    source = args.trace/'trace.npz';digest = sha256(source)
    if digest != tm['trace_sha256'] or not tm['includes_task_contacts']:
        raise ValueError('Original trace with all payload contacts required')
    payload_index = next(i for i, b in enumerate(tm['payloads']) if b['name'] == 'task/'+args.object)
    frozen = json.loads(args.identity.read_text())['files']
    changed = [path for path, h in frozen.items() if sha256(path) != h]
    if changed: raise ValueError('Frozen inputs changed: '+repr(changed))
    recording_hash = sha256(args.recording)
    with h5py.File(args.recording, 'r') as f:
        if not f.attrs['complete']: raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    b = MujocoBackend(cfg, cameras=False)
    try:
        layout = meta['task_scene']['layout'];b.reset(layout['seed'], layout['randomized'])
        if [b.model.geom(i).name for i in range(b.model.ngeom)] != tm['geom_names']:
            raise ValueError('Trace geometry mapping mismatch')
        bodies = np.asarray([b.model.body(int(i)).name for i in b.model.geom_bodyid])
        mass = float(b.model.body('task/'+args.object).mass[0])
        gravity = b.model.opt.gravity.copy()
        if mass != tm['payloads'][payload_index]['mass_kg'] or not np.array_equal(gravity, tm['gravity_world_m_s2']):
            raise ValueError('Recorded payload mass/gravity mismatch')
    finally:
        b.close()
    with np.load(source, allow_pickle=False) as archive:
        z = {key: archive[key] for key in ('time_before', 'phase', 'target', 'payload_com_world',
            'payload_origin_world', 'payload_rotation_world', 'payload_velocity_world', 'payload_applied_wrench',
            'contact_row', 'contact_geom1', 'contact_geom2', 'contact_position_world', 'contact_frame', 'contact_force_local')}
    if np.any(z['payload_applied_wrench'][:, payload_index] != 0):
        raise ValueError('Unexpected external applied payload wrench')
    body1, body2 = bodies[z['contact_geom1']], bodies[z['contact_geom2']]
    target = 'task/'+args.object
    selected = (body1 == target) ^ (body2 == target)
    rows = z['contact_row'][selected].astype(int)
    other = np.where(body2[selected] == target, body1[selected], body2[selected])
    centers = z['payload_com_world'][:, payload_index]
    positions = z['contact_position_world'][selected]
    local = z['contact_force_local'][selected]
    force, torque = wrenches_about_com(z['contact_frame'][selected].reshape(-1, 3, 3),
        local, body2[selected] == target, positions, centers[rows])
    finger = np.isin(other, ['left/link7', 'left/link8', 'right/link7', 'right/link8'])
    robot = np.char.startswith(other, 'left/') | np.char.startswith(other, 'right/')
    n = len(centers);weight = mass*np.linalg.norm(gravity);up = -gravity/np.linalg.norm(gravity)
    if not weight > 0: raise ValueError('Positive weight required')
    arrays = {'time_before': z['time_before'], 'phase': z['phase'], 'target': z['target'], 'com_world': centers,
              'angular_velocity_world_rad_s': z['payload_velocity_world'][:, payload_index, :3]}
    rotation = z['payload_rotation_world'][:, payload_index].reshape(-1, 3, 3)
    arrays['tilt_from_up_rad'] = np.arccos(np.clip(rotation[:, :, 2]@up, -1., 1.))
    for label, mask in {'fingers': finger, 'other_robot': robot & ~finger, 'external_support': ~robot}.items():
        sums = np.zeros((n, 6))
        np.add.at(sums, rows[mask], np.c_[force[mask], torque[mask]])
        arrays[label+'_force_N'] = sums[:, :3]
        arrays[label+'_torque_COM_Nm'] = sums[:, 3:]
    normals = np.maximum(0., local[finger, 0])
    totals = np.zeros(n);weighted = np.zeros((n, 3))
    np.add.at(totals, rows[finger], normals)
    np.add.at(weighted, rows[finger], positions[finger]*normals[:, None])
    valid = totals > 1e-9
    centroids = np.full((n, 3), np.nan);centroids[valid] = weighted[valid]/totals[valid, None]
    arrays['finger_contact_normal_sum_N'] = totals
    arrays['finger_centroid_world'] = centroids
    arrays['gravity_moment_about_finger_centroid_Nm'] = np.cross(centers-centroids, mass*gravity)
    phases = {}
    for phase in sorted(set(z['phase'][z['target'] == args.object])):
        ids = (z['target'] == args.object) & (z['phase'] == phase)
        group = {'steps': int(ids.sum()), 'normal_sum_N': summary(totals[ids]),
                 'tilt_deg': summary(np.rad2deg(arrays['tilt_from_up_rad'][ids])),
                 'angular_speed_rad_s': summary(np.linalg.norm(arrays['angular_velocity_world_rad_s'][ids], axis=1))}
        for label in ('fingers', 'other_robot', 'external_support'):
            group[label] = {'upward_weight_fraction': summary(arrays[label+'_force_N'][ids]@up/weight),
                'torque_COM_norm_Nm': summary(np.linalg.norm(arrays[label+'_torque_COM_Nm'][ids], axis=1))}
        if np.any(ids & valid):
            group['gravity_moment_about_finger_centroid_Nm'] = summary(np.linalg.norm(
                arrays['gravity_moment_about_finger_centroid_Nm'][ids & valid], axis=1))
        phases[phase] = group
    if sha256(source) != digest or sha256(args.recording) != recording_hash:
        raise ValueError('Read-only source changed')
    args.output.mkdir(parents=True)
    with (args.output/'wrench.npz').open('xb') as f: np.savez_compressed(f, **arrays)
    atomic_json(args.output/'audit.json', {'scope': __doc__, 'training_ready': False,
        'source': str(args.recording), 'source_sha256': recording_hash, 'trace_sha256': digest,
        'script_sha256': sha256(__file__), 'series_sha256': sha256(args.output/'wrench.npz'),
        'frozen_inputs_changed': changed, 'object': args.object, 'mass_kg': mass, 'weight_N': weight,
        'phases': phases, 'timing': tm['payload_timing'],
        'limitations': ['Original force-evaluation samples only, no forces from reconstructed states.',
            'Contact centroid moves; gravity moment about it is not a fixed-pivot dynamics equation.',
            'No inertial/angular momentum balance or force-closure certification is claimed.',
            'NaN centroid/moment explicitly indicates absence of positive finger normal force.']})
    print(json.dumps(phases, indent=2))


if __name__ == '__main__': main()
