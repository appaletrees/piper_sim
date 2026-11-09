#!/usr/bin/env python3
"""Scratch-only physical sorting with constant, uncalibrated finger material.

Restore one recorded approach state per trial, then use normal dynamics/control.
Object velocities and solver warm starts were not recorded and are initialized
to zero: this is neither an exact replay nor a training/acceptance certificate.
No changes are made to the production teacher, assets, source recording or gates.
"""
import argparse
import copy
import hashlib
import json
from itertools import product
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.mujoco_ik import grasp_rotation
from piper_sim.simulation import SimulationSession
from piper_sim.transforms import pose_matrix


class FixedMaterialTeacher(RigidTableTeacher):
    """Keep material fixed; retain per-object motor force/gain control strategy."""

    def __init__(self, *args, friction_scale, rolling_m, rim_x_m=None, rim_tilt_deg=None, **kwargs):
        self.friction_scale = friction_scale
        self.rolling_m = rolling_m
        self.rim_x_m, self.rim_tilt_deg = rim_x_m, rim_tilt_deg
        super().__init__(*args, **kwargs)

    def plan_grasp(self, name):
        side, p, r, width = super().plan_grasp(name)
        if name == 'side_dish' and self.rim_x_m is not None:
            pose = self.object_pose(name)
            actual = pose_matrix(pose['position'], pose['quaternion_wxyz'])
            a = np.deg2rad(self.b.task.specs[name].get('yaw_deg', 0))
            nominal = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            rotation = actual[:3, :3]@nominal.T
            p = actual[:3, 3]+rotation@np.array([self.rim_x_m, -.049, .022])
            r = rotation@grasp_rotation(self.rim_tilt_deg, 0)
        return side, p, r, width

    def configure_contact_control(self):
        super().configure_contact_control()
        model = self.b.model
        self.pad_ids = np.array([g for g in range(model.ngeom)
            if model.body(int(model.geom_bodyid[g])).name in
            ('left/link7', 'left/link8', 'right/link7', 'right/link8')])
        model.geom_friction[self.pad_ids] = np.array([1., .015, self.rolling_m])*self.friction_scale
        # Time constant and impedance remain the same for every object.
        # The superclass initialized these from the existing controller profile.
        self.fixed_material = {key: getattr(model, key)[self.pad_ids].copy()
            for key in ('geom_friction', 'geom_solref', 'geom_solimp')}

    def sort_one(self, name):
        # Mirror only the existing motor settings. Deliberately bypass the
        # production override of pad rolling friction/softness by target kind.
        model, controller = self.b.model, self.settings['controller']
        kind = self.b.task.specs[name]['kind']
        actuator = model.actuator('right/gripper').id
        force = {'pen': .35, 'chopstick': .30, 'plate': 30., 'side_dish': 15.}.get(
            kind, controller['gripper_force_limit_N'])
        kp = 10000. if kind in ('plate', 'side_dish') else controller['gripper_kp_N_per_m']
        model.actuator_forcerange[actuator] = [-force, force]
        model.actuator_gainprm[actuator, 0] = kp
        model.actuator_biasprm[actuator, 1] = -kp
        return CleanTableTeacher.sort_one(self, name)

    def step(self):
        for key, baseline in self.fixed_material.items():
            if not np.array_equal(getattr(self.b.model, key)[self.pad_ids], baseline):
                raise RuntimeError('Fixed contact material changed inside trial')
        # Existing adaptive noslip iterations are numerical solver refinement;
        # they do not set contact material coefficients or attach an object.
        return super().step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--objects', nargs='+', default=['pen', 'plate', 'side_dish'])
    parser.add_argument('--scales', nargs='+', type=float, default=[.8, 1., 1.2])
    parser.add_argument('--rolling-m', type=float, default=.002)
    parser.add_argument('--rim-x-m', type=float, nargs='+',
                        help='Side-dish nominal-frame grasp x positions; diagnostic only')
    parser.add_argument('--rim-tilt-deg', type=float, nargs='+', default=[30.])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output')
    if not np.isfinite(args.scales).all() or min(args.scales) <= 0:
        parser.error('Invalid friction scales')
    if not np.isfinite(args.rolling_m) or args.rolling_m < 0:
        parser.error('Invalid rolling friction length')
    if (args.rim_x_m is not None and (args.objects != ['side_dish'] or
            not np.isfinite(args.rim_x_m).all() or max(abs(x) for x in args.rim_x_m) > .07)):
        parser.error('Rim variants require only side_dish and finite x positions within +/-70 mm')
    if not np.isfinite(args.rim_tilt_deg).all(): parser.error('Invalid rim tilt')
    rollout = json.loads(args.source.with_name('report.json').read_text())
    snapshots = {}
    with h5py.File(args.source, 'r') as source:
        if not source.attrs.get('complete'): raise ValueError('A closed source is required')
        metadata = json.loads(source.attrs['metadata'])
        times = source['observation/sim_time'][:]
        for name in args.objects:
            event = next(e for e in rollout['events'] if e['phase'] == 'approach' and e['object'] == name)
            index = int(np.argmin(abs(times-event['sim_time'])))
            raw = source['observation/extras_json'][index]
            command = (source['action/position'][index-1].astype(float) if index else
                       np.asarray(metadata['collection_initial_target'], float))
            snapshots[name] = (index, float(times[index]), json.loads(raw), command)
    cfg = copy.deepcopy(metadata['config'])
    cfg['task_scene']['domain_randomization'] = metadata['task_scene'].get('domain_profile')
    source_audit = json.loads(args.source.with_name('audit.json').read_text())
    backend = MujocoBackend(cfg, cameras=False)
    report = {'status': 'running', 'training_ready': False, 'scope': __doc__,
        'source': str(args.source.resolve()), 'source_trajectory_sha256':
            source_audit.get('checks', {}).get('streams', {}).get('trajectory_sha256'),
        'source_file_sha256': sha256(args.source), 'source_audit_passed': source_audit.get('passed'),
        'finger_material_nominal': {'sliding': 1., 'torsional_m': .015, 'rolling_m': args.rolling_m,
            'solref': [metadata['teacher']['motion_and_quality']['controller']['finger_contact_time_constant_s'], 1.],
            'solimp': [.95, .99, .001, .5, 2.], 'calibrated': False},
        'source_snapshot_limitations': 'Native arm position/velocity and object poses restored once; '
            'unrecorded object velocity/warm start initialized to zero. Single-object diagnostic, not exact replay.',
        'control_and_solver': 'Original target-specific motor force/kp; original loaded-rim noslip refinement.',
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'expert_source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path('src/piper_sim/experts').glob('*.py')},
        'mujoco_version': mujoco.__version__, 'trials': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in args.objects:
            index, sim_time, extras, command = snapshots[name]
            for scale, rim_x, rim_tilt in product(args.scales, args.rim_x_m or [None], args.rim_tilt_deg):
                started = time.perf_counter()
                layout = metadata['task_scene']['layout']
                backend.reset(layout['seed'], layout['randomized'])
                for side, arm in extras['teacher']['physics']['arms'].items():
                    backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
                    backend.data.qvel[backend.indices[side]['dof']] = arm['joint_velocity']
                for object_name, pose in extras['objects'].items():
                    address = backend.model.joint('task/'+object_name+'/free').qposadr[0]
                    backend.data.qpos[address:address+7] = [*pose['position'], *pose['quaternion_wxyz']]
                for side, offset in (('left', 0), ('right', 7)):
                    backend.data.ctrl[backend.indices[side]['ctrl']] = command[offset:offset+7]
                backend.data.time = sim_time
                mujoco.mj_forward(backend.model, backend.data)
                effective = set()
                def capture(_obs, teacher):
                    for contact in backend.data.contact:
                        bodies = [backend.model.body(int(backend.model.geom_bodyid[g])).name for g in contact.geom]
                        if ('task/'+name in bodies and any(b in ('right/link7', 'right/link8') for b in bodies)):
                            effective.add(tuple(contact.friction))
                teacher = FixedMaterialTeacher(SimulationSession(backend), friction_scale=scale,
                    rolling_m=args.rolling_m, rim_x_m=rim_x, rim_tilt_deg=rim_tilt,
                    settings=metadata['teacher']['motion_and_quality'])
                # frame_hook uses (observation, teacher telemetry); wrap step
                # instead to inspect the actual solver contact buffer.
                original_step = teacher.step
                def step():
                    obs = original_step(); capture(obs, teacher); return obs
                teacher.step = step
                teacher.q = command.copy()
                teacher.quality.add(backend.sim_time, command, teacher.session.read().position,
                    context={'phase': 'initial_recorded_state'})
                result = {'object': name, 'source_frame': index, 'friction_scale': scale, 'completed': False,
                          'rim_x_m': rim_x, 'rim_tilt_deg': rim_tilt if rim_x is not None else None}
                try:
                    teacher.sort_one(name)
                    result['completed'] = True
                except SortFailure as error:
                    result['error'] = str(error)
                result.update(wall_seconds=time.perf_counter()-started,
                    sim_seconds=backend.sim_time-sim_time, motion=teacher.quality.report(),
                    collision_peak=teacher.collision_peak, events=teacher.events, grasps=teacher.grasps,
                    effective_contact_friction=[list(v) for v in sorted(effective)])
                report['trials'].append(result)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(json.dumps({k: result[k] for k in ('object', 'friction_scale', 'rim_x_m',
                    'rim_tilt_deg', 'completed', 'wall_seconds')}
                    | {'error': result.get('error')}), flush=True)
        report['status'] = 'completed_diagnostic_only'
    finally:
        backend.close()
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
