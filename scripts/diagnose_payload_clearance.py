#!/usr/bin/env python3
"""Test conservative payload clearance in isolated recorded-state transfer trials.

Restores a settled transfer-start snapshot once per trial, with unavailable object
velocities/warm-start reset. Subsequent movement is physical control. No recording
is produced and no trial is training-qualified or a bitwise historical replay.
"""
import argparse
import copy
import hashlib
from itertools import product
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.simulation import SimulationSession
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.collision import carried_obstacle_clearance as payload_clearance
from piper_sim.experts.mujoco_ik import grasp_rotation
import piper_sim.experts.clean_table as teacher_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--object', default='chopstick_a')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clearances', type=float, nargs='+', default=[0., .001, .003, .006])
    parser.add_argument('--via-fallback', action='store_true', help='Use the production intermediate pose if direct placement fails')
    parser.add_argument('--slot-x-offsets', type=float, nargs='+', default=[0.], help='Basket-local placement-goal offsets only; does not move objects')
    parser.add_argument('--release', action='store_true', help='Continue with physical unload, release, retreat and containment check')
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < 0 for value in args.clearances):
        parser.error('Clearances must be finite and nonnegative')
    if not np.isfinite(args.slot_x_offsets).all(): parser.error('Slot offsets must be finite')
    if args.release and not args.object.startswith('chopstick'):
        parser.error('Release diagnostic currently implements chopstick release width only')
    if args.output.exists(): raise FileExistsError(args.output)
    prior = json.loads(args.source.with_name('report.json').read_text())
    event = next(e for e in prior['events'] if e['phase'] == 'transfer' and e['object'] == args.object)
    with h5py.File(args.source, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        i = int(np.argmin(abs(f['observation/sim_time'][:]-event['sim_time'])))
        extras = json.loads(f['observation/extras_json'][i])
        command = f['action/position'][i-1].astype(float)
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    backend = MujocoBackend(cfg)
    results = []
    original = teacher_module.carried_obstacle_penetration
    base_slot = copy.deepcopy(backend.task.config['drop_slots_m'][args.object])
    try:
        for margin, slot_x in product(args.clearances, args.slot_x_offsets):
            backend.reset(meta['task_scene']['layout']['seed'], meta['task_scene']['layout']['randomized'])
            backend.task.config['drop_slots_m'][args.object] = [base_slot[0]+slot_x, base_slot[1]]
            for side, arm in extras['teacher']['physics']['arms'].items():
                backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
                backend.data.qvel[backend.indices[side]['dof']] = arm['joint_velocity']
            for name, pose in extras['objects'].items():
                address = backend.model.joint('task/'+name+'/free').qposadr[0]
                backend.data.qpos[address:address+7] = pose['position']+pose['quaternion_wxyz']
            for side, offset in (('left', 0), ('right', 7)):
                backend.data.ctrl[backend.indices[side]['ctrl']] = command[offset:offset+7]
            mujoco.mj_forward(backend.model, backend.data)
            teacher = RigidTableTeacher(SimulationSession(backend), settings=meta['teacher']['motion_and_quality'])
            # This diagnostic supplies its own explicit margin below. Avoid
            # silently retaining a newer production margin in the zero case.
            teacher.pacing['payload_transport_clearance_m_by_kind'] = {}
            teacher.current, teacher.phase = args.object, 'transfer'
            backend.model.actuator_forcerange[backend.model.actuator('right/gripper').id] = [-.30, .30]
            teacher.track_grasp('right', args.object)
            def guarded(model, data, carried):
                if margin:
                    distance, pair = payload_clearance(model, data, carried, margin)
                    if distance < margin:
                        raise SortFailure(f'Payload clearance {distance:.6f} m < {margin} m: {pair}')
                return original(model, data, carried)
            teacher_module.carried_obstacle_penetration = guarded
            result = {'clearance_m': margin, 'slot_x_offset_m': slot_x}
            try:
                try:
                    score, high, low, r, q_high, q_low = teacher.placement('right', args.object, 'basket')
                except SortFailure:
                    if not args.via_fallback: raise
                    result['used_intermediate_pose'] = True
                    teacher.pose('right', [.38, -.20, .32], grasp_rotation(45, 15))
                    score, high, low, r, q_high, q_low = teacher.placement('right', args.object, 'basket')
                result.update(placement_score=score, high=high.tolist(), low=low.tolist())
                teacher.move_joints('right', q_high)
                result['transferred'] = True
                teacher.move_joints('right', q_low)
                result['lowered'] = True
                if args.release:
                    teacher.carried = teacher.armed = None
                    measured = float(backend.data.qpos[backend.indices['right']['qpos'][6:]].sum())
                    if measured-teacher.q[13] > .003:
                        teacher.event('unload_gripper')
                        teacher.grip('right', min(.04, measured+.002))
                    teacher.event('release')
                    teacher.grip('right', .04)
                    teacher.wait(teacher.pacing.get('release_hold_s', .5))
                    teacher.pose_path('right', [(high, r)])
                    teacher.wait(teacher.pacing.get('placement_hold_s', 2.))
                    result['placement_metric'] = teacher.session.read().extras['task']['metrics'][args.object]
                    result['final_pose'] = teacher.object_pose(args.object)
            except SortFailure as error:
                result['error'] = str(error)
            result.update(sim_seconds=backend.sim_time, grasp=teacher.grasps[-1], motion=teacher.quality.report())
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != 'motion'}), flush=True)
    finally:
        teacher_module.carried_obstacle_penetration = original
        backend.close()
    report = {'status': 'isolated_transfer_diagnostic', 'training_ready': False, 'scope': __doc__,
        'source': str(args.source.resolve()), 'source_frame': i, 'recorded_event': event, 'trials': results,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
