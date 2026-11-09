#!/usr/bin/env python3
"""Experimental directional gripper compliance from public control/state only.

While opening, reduce only the negative position-error spring term. Preserve
the velocity-feedback term and both original motor force bounds. Direction is
latched from reference increments, and two cascaded low-pass states make the
compliance coefficient continuous with continuous first derivative in time.
No object name, pose, task phase, contact or teacher grasp state is an input.
This uncalibrated state-only probe is not a production controller.
"""
import argparse
import copy
import sys

import mujoco
import numpy as np

from piper_sim.collection_jobs import sha256
from piper_sim.mujoco_controller import MujocoReferenceController
import trace_contact_substeps as tracing


class OpeningCompliance:
    def __init__(self, model, *, stiffness_floor=.1, time_constant=.06):
        if (not np.isfinite(stiffness_floor) or not 0 < stiffness_floor <= 1
                or not np.isfinite(time_constant) or time_constant <= 0):
            raise ValueError('Finite positive stiffness floor <= 1 and time constant required')
        self.model = model
        self.ids = np.array([model.actuator(side+'/gripper').id for side in ('left', 'right')])
        self.qids = np.array([[int(model.joint(side+'/'+joint).qposadr[0])
                              for joint in ('joint7', 'joint8')] for side in ('left', 'right')])
        self.floor, self.tau = stiffness_floor, time_constant
        self.reset()

    def reset(self):
        self.previous = None
        self.direction = np.zeros(2)
        self.first = np.zeros(2)
        self.second = np.zeros(2)
        self.last = None
        self.last_time = None

    def apply(self, data):
        model, ids = self.model, self.ids
        # The reference controller must write its unmodified velocity bias
        # before this hook, on every native step. Never accumulate this offset.
        reference = data.ctrl[ids].copy()
        measured = data.qpos[self.qids].sum(axis=1)
        kp = model.actuator_gainprm[ids, 0]
        if (not np.isfinite([reference, measured, kp]).all() or np.any(kp <= 0)
                or not np.array_equal(model.actuator_biasprm[ids, 1], -kp)
                or np.any(model.actuator_biasprm[ids, 2] > 0)):
            raise ValueError('Finite affine position servo with nonnegative damping required')
        dt = float(model.opt.timestep)
        if self.last_time is not None and abs(data.time-self.last_time-dt) > 1e-8:
            raise ValueError('Compliance state requires contiguous native steps or an explicit reset')
        if self.previous is not None:
            delta = reference-self.previous
            self.direction[delta > 1e-12] = 1.
            self.direction[delta < -1e-12] = 0.
        decay = np.exp(-dt/self.tau)
        first = self.first.copy()
        self.first = self.direction+(first-self.direction)*decay
        self.second = self.direction+(self.second-self.direction+(first-self.direction)*dt/self.tau)*decay
        error = measured-reference
        offset = (1-self.floor)*self.second*kp*np.maximum(error, 0.)
        model.actuator_biasprm[ids, 0] += offset
        self.last = {'opening_direction': self.direction.tolist(), 'filter_first': self.first.tolist(),
                     'filter_second': self.second.tolist(), 'position_error_m': error.tolist(),
                     'force_bias_N': offset.tolist(), 'sim_time': float(data.time)}
        self.previous, self.last_time = reference, float(data.time)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--stiffness-floor', type=float, default=.1)
    parser.add_argument('--compliance-time-constant', type=float, default=.06)
    args, forwarded = parser.parse_known_args()
    if '--rgb' in forwarded or '--viewer' in forwarded:
        raise ValueError('Experimental controller supports state-only diagnostics')
    profile = {'mode': 'opening_negative_error_compliance_v1', 'stiffness_floor': args.stiffness_floor,
               'time_constant_s': args.compliance_time_constant, 'script_sha256': sha256(__file__),
               'hardware_calibration': 'unidentified',
               'inputs': ['native actuator position reference and its history', 'measured full gripper opening',
                          'existing kp/kv and motor force bounds'],
               'force_semantics': 'For negative position error only, kp is scaled by 1-(1-floor)*alpha. '
                                  'kv*(vref-v) and original two-sided force limits remain unchanged.',
               'filter': 'Exact two cascaded first-order filters; direction latch holds through zero command increments.'}
    original_advance = MujocoReferenceController.advance
    original_reset = MujocoReferenceController.reset
    original_metadata = MujocoReferenceController.metadata
    original_telemetry = MujocoReferenceController.telemetry
    instances = {}
    active = None

    def get(controller):
        if controller.mode != 'monotone_cubic_velocity':
            raise ValueError('Probe requires the native monotone reference controller')
        if id(controller) not in instances:
            instances[id(controller)] = OpeningCompliance(controller.model, stiffness_floor=args.stiffness_floor,
                                                          time_constant=args.compliance_time_constant)
        return instances[id(controller)]

    def reset(controller, target):
        original_reset(controller, target)
        get(controller).reset()

    def advance(controller, data, **kwargs):
        nonlocal active
        if active is not None:
            raise RuntimeError('Probe supports one live controller at a time')
        active = get(controller)
        try:
            return original_advance(controller, data, **kwargs)
        finally:
            active = None

    def before_step(model, data):
        if active is None or active.model is not model:
            raise RuntimeError('Compliance hook must run inside the reference controller')
        active.apply(data)

    def metadata(controller):
        value = original_metadata(controller)
        value['experimental_opening_compliance'] = profile
        value['force_semantics'] += ' '+profile['force_semantics']
        return value

    def telemetry(controller):
        value = original_telemetry(controller)
        if value is not None:
            value['opening_compliance'] = copy.deepcopy(get(controller).last)
        return value

    try:
        MujocoReferenceController.advance, MujocoReferenceController.reset = advance, reset
        MujocoReferenceController.metadata, MujocoReferenceController.telemetry = metadata, telemetry
        sys.argv = [tracing.__file__, *forwarded]
        return tracing.main(before_step=before_step, extra_metadata=profile)
    finally:
        MujocoReferenceController.advance, MujocoReferenceController.reset = original_advance, original_reset
        MujocoReferenceController.metadata, MujocoReferenceController.telemetry = original_metadata, original_telemetry


if __name__ == '__main__':
    raise SystemExit(main())
