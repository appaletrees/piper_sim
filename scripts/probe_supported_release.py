#!/usr/bin/env python3
"""Fresh full-task experiment: lower a held plate onto measured physical support.

Only position commands pass through the existing trajectory/collision gates.
No solver, gain, force-limit, geometry or object-state modifications. Original
grasp/slip checks stay active through the supported descent. This is an opt-in
state-only experiment, not a production policy or hardware calibration.
"""
import argparse
import copy
import sys

import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.clean_table import CleanTableTeacher, SortFailure
from piper_sim.experts.support_evidence import payload_support
from probe_supported_lowering import clearance
import trace_contact_substeps as tracing


PROFILE = {'mode': 'plate_supported_release_v1', 'objects': ['plate'],
           'initial_gap_m': .003, 'max_descent_m': .16, 'seek_increment_m': .0005,
           'max_seek_m': .006, 'speed_m_s': .06, 'required_support_fraction': .5,
           'max_support_fraction': 3., 'support_hold_s': .1,
           'hardware_calibration': 'unidentified',
           'inputs': ['privileged task collision geometry and poses', 'native joint feedback',
                      'non-robot upward-facing solver contact force'],
           'not_modified': ['object states', 'motor gains or force limits', 'solver settings',
                            'collision geometry', 'original motion and quality thresholds']}


def opening_check(teacher, joints, width):
    aperture = float(teacher.b.data.qpos[teacher.b.indices['right']['qpos'][6:]].sum())
    widths = np.linspace(aperture, width, max(2, int(np.ceil(abs(width-aperture)/.001))+1))
    for a, b in zip(widths[:-1], widths[1:]):
        teacher.check_joint_path('right', joints, joints, gripper=(a, b))


def supported_lower(teacher, profile):
    name = teacher.current
    if teacher.mode != 'physical' or teacher.carried is None or teacher.carried[:2] != ('right', name):
        raise SortFailure('Supported lowering requires an active physical right-hand grasp')
    gap, obstacles = clearance(teacher, name)
    descent = max(0., gap-profile['initial_gap_m'])
    if descent > profile['max_descent_m'] or gap < -.001:
        raise SortFailure('Unsupported lowering geometry: gap='+str(gap))
    tcp, rotation = teacher.tcp('right')
    width = min(.095, teacher.plan_grasp(name)[3]+.020)
    target = tcp-[0, 0, descent]
    trajectory = teacher.plan_pose_path('right', [(target, rotation)], speed=profile['speed_m_s'])
    opening_check(teacher, trajectory[-1, 7:13], width)
    teacher.event('support_lower', sampled_gap_m=gap, descent_m=descent,
                  support_obstacles=obstacles, release_width_m=width)
    for command in trajectory:
        teacher.q = command
        teacher.step()
    teacher.settle()
    stable = 0
    samples = int(np.ceil(profile['support_hold_s']*teacher.fps))
    for attempt in range(int(round(profile['max_seek_m']/profile['seek_increment_m']))+1):
        force = payload_support(teacher.b.model, teacher.b.data, name)
        if force['support_fraction_of_weight'] >= profile['required_support_fraction']:
            for _ in range(samples):
                teacher.wait(1/teacher.fps)
                force = payload_support(teacher.b.model, teacher.b.data, name)
                stable = stable+1 if force['support_fraction_of_weight'] >= profile['required_support_fraction'] else 0
            if stable >= samples:
                teacher.event('support_verified', support=force, stable_samples=stable,
                              total_commanded_descent_m=descent+attempt*profile['seek_increment_m'])
                return
        if attempt == int(round(profile['max_seek_m']/profile['seek_increment_m'])):
            break
        stable = 0
        target = target-[0, 0, profile['seek_increment_m']]
        path = teacher.plan_pose_path('right', [(target, rotation)], speed=.01)
        opening_check(teacher, path[-1, 7:13], width)
        teacher.event('support_seek', seek_index=attempt+1, support_fraction=force['support_fraction_of_weight'])
        for command in path:
            teacher.q = command
            teacher.step()
        teacher.settle()
    raise SortFailure('Support not verified within bounded lowering; refusing high release')


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    _, forwarded = parser.parse_known_args()
    if '--rgb' in forwarded or '--viewer' in forwarded or '--objects' in forwarded:
        raise ValueError('This probe requires a full state-only rollout')
    profile = copy.deepcopy(PROFILE)
    profile['script_sha256'] = sha256(__file__)
    original_event = CleanTableTeacher.event
    original_step = CleanTableTeacher.step
    original_metadata = CleanTableTeacher.metadata
    original_telemetry = CleanTableTeacher.telemetry

    def event(teacher, phase, **fields):
        if phase == 'open_gripper' and teacher.current in profile['objects']:
            supported_lower(teacher, profile)
            # Caller supplied this pose before the additional physical motion.
            fields['object_pose'] = teacher.object_pose(teacher.current)
        return original_event(teacher, phase, **fields)

    def step(teacher):
        result = original_step(teacher)
        if teacher.current in profile['objects'] and teacher.phase in ('support_lower', 'support_seek'):
            evidence = payload_support(teacher.b.model, teacher.b.data, teacher.current)
            if evidence['support_fraction_of_weight'] > profile['max_support_fraction']:
                raise SortFailure('Supported lowering exceeds experimental load guard: '+str(evidence['support_fraction_of_weight']))
        return result

    def metadata(teacher):
        return {**original_metadata(teacher), 'experimental_supported_release': profile}

    def telemetry(teacher):
        value = original_telemetry(teacher)
        if teacher.current in profile['objects']:
            value['payload_support'] = payload_support(teacher.b.model, teacher.b.data, teacher.current)
        return value

    try:
        CleanTableTeacher.event, CleanTableTeacher.step = event, step
        CleanTableTeacher.metadata, CleanTableTeacher.telemetry = metadata, telemetry
        sys.argv = [tracing.__file__, *forwarded]
        return tracing.main(extra_metadata=profile)
    finally:
        CleanTableTeacher.event, CleanTableTeacher.step = original_event, original_step
        CleanTableTeacher.metadata, CleanTableTeacher.telemetry = original_metadata, original_telemetry


if __name__ == '__main__':
    raise SystemExit(main())
