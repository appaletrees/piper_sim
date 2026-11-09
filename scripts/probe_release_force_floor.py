#!/usr/bin/env python3
"""Experimental full-rollout release force limit; never enabled by default.

During bowl/dish release, smoothly reduce the available closing force to zero.
Keep the original opening-force ceiling, references, physics and acceptance
policy. Restore the original closing-force bound smoothly during retreat. This
is an uncalibrated controller experiment, not a change to material/contact laws.
Requires trace_contact_substeps' normal object/phase selection and state-only
recording. Every affected native step records the actual force bounds.
"""
import argparse
import sys
import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.experts.rigid_table import RigidTableTeacher
import trace_contact_substeps as tracing


def ease(value):
    t = np.clip(value, 0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--force-ramp-seconds', type=float, default=.4)
    args, forwarded = parser.parse_known_args()
    if not np.isfinite(args.force_ramp_seconds) or args.force_ramp_seconds <= 0:
        raise ValueError('Positive finite ramp duration required')
    if any(flag in forwarded for flag in ('--rgb', '--viewer', '--trace-objects', '--trace-phases')):
        raise ValueError('Probe requires default trace coverage and a state-only rollout')
    original_step, original_metadata = RigidTableTeacher.step, RigidTableTeacher.metadata
    original_before = tracing.ContactTrace.before
    active = None
    transitions = []
    profile = {'mode': 'release_closing_force_floor_quintic_v1',
               'ramp_seconds': args.force_ramp_seconds,
               'targets': ['bowl', 'plate', 'side_dish'], 'hardware_calibration': 'unidentified',
               'script_sha256': sha256(__file__),
               'scope': 'Experimental actuator force bound only; no added forces, object assignments or contact changes.'}

    def metadata(teacher):
        return {**original_metadata(teacher), 'experimental_release_force_floor': profile,
                'release_force_floor_transitions': list(transitions)}

    def step(teacher, *a, **kw):
        nonlocal active
        releasing = teacher.releasing and teacher.releasing[1] in profile['targets']
        if releasing and active is None:
            side, target = teacher.releasing
            aid = teacher.b.model.actuator(side+'/gripper').id
            bounds = teacher.b.model.actuator_forcerange[aid].copy()
            if not bounds[0] < 0 < bounds[1]:
                raise ValueError('Expected original two-sided gripper force limits')
            active = {'teacher': teacher, 'aid': aid, 'bounds': bounds,
                      'start': float(teacher.b.data.time), 'restore': None}
            transitions.append({'event': 'begin_unload', 'target': target,
                                'time': active['start'], 'original_range_N': bounds.tolist()})
        elif active is not None and not releasing and active['restore'] is None:
            active['restore'] = float(teacher.b.data.time)
            active['restore_from'] = float(teacher.b.model.actuator_forcerange[active['aid'], 0])
            transitions.append({'event': 'begin_restore', 'time': active['restore']})
        return original_step(teacher, *a, **kw)

    def before(tracer, data):
        nonlocal active
        if active is not None:
            if tracer.model is not active['teacher'].b.model or data is not active['teacher'].b.data:
                raise ValueError('Unexpected force-probe model/data')
            original = active['bounds'][0]
            if active['restore'] is None:
                lower = original*(1-ease((data.time-active['start'])/args.force_ramp_seconds))
            else:
                t = (data.time-active['restore'])/args.force_ramp_seconds
                lower = active['restore_from']+(original-active['restore_from'])*ease(t)
            tracer.model.actuator_forcerange[active['aid'], 0] = lower
            if active['restore'] is not None and t >= 1:
                tracer.model.actuator_forcerange[active['aid']] = active['bounds']
                transitions.append({'event': 'restored', 'time': float(data.time)})
                active = None
        return original_before(tracer, data)

    try:
        RigidTableTeacher.step, RigidTableTeacher.metadata = step, metadata
        tracing.ContactTrace.before = before
        sys.argv = [tracing.__file__, *forwarded]
        return tracing.main()
    finally:
        RigidTableTeacher.step, RigidTableTeacher.metadata = original_step, original_metadata
        tracing.ContactTrace.before = original_before
        if active is not None:
            active['teacher'].b.model.actuator_forcerange[active['aid']] = active['bounds']


if __name__ == '__main__':
    raise SystemExit(main())
