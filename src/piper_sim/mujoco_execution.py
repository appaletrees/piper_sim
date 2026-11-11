"""Explicit position-only execution shared by teachers and external callers.

Fixed, uncalibrated servo and solver settings, plus a declared robot-only bias
compensation rule. No target identity, teacher phase, object pose or contact
label is used to select controller parameters. This does not certify grasping.
"""
import copy
import hashlib
from pathlib import Path

import mujoco
import numpy as np


from .position_execution import SCHEMA, INTERVAL_SCHEMA, profile_digest, validate_profile


class MujocoPositionExecution:
    def __init__(self, model, indices, profile):
        self.profile = validate_profile(profile)
        self.model = model
        self.actuators = np.concatenate([indices[s]['ctrl'] for s in ('left', 'right')])
        self.dofs = np.concatenate([indices[s]['dof'] for s in ('left', 'right')])
        self.other_dofs = np.setdiff1d(np.arange(model.nv), self.dofs)
        p = self.profile
        for side in ('left', 'right'):
            a = model.actuator(side+'/gripper').id
            model.actuator_gainprm[a, 0] = p['gripper_kp_N_per_m']
            model.actuator_biasprm[a, 1:3] = [-p['gripper_kp_N_per_m'], -p['gripper_kv_Ns_per_m']]
            model.actuator_forcerange[a] = [-p['gripper_force_limit_N'], p['gripper_force_limit_N']]
        model.opt.impratio = p.get('impratio',10.)
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        model.opt.noslip_iterations = p['noslip_iterations']
        self.expected = {key: getattr(model, key)[self.actuators].copy() for key in (
            'actuator_gainprm', 'actuator_forcerange', 'actuator_ctrlrange', 'actuator_ctrllimited',
            'actuator_forcelimited', 'actuator_gaintype', 'actuator_biastype', 'actuator_dyntype')}
        self.bias_coefficients = model.actuator_biasprm[self.actuators, 1:].copy()
        self.expected_applied = np.zeros(model.nv)
        self.last = None

    def assert_unchanged(self):
        for key, expected in self.expected.items():
            if not np.array_equal(getattr(self.model, key)[self.actuators], expected):
                raise RuntimeError('Position execution parameters changed: '+key)
        if not np.array_equal(self.model.actuator_biasprm[self.actuators, 1:], self.bias_coefficients):
            raise RuntimeError('Position execution servo coefficients changed')
        opt = self.model.opt
        if (opt.noslip_iterations != self.profile['noslip_iterations'] or opt.impratio != self.profile.get('impratio',10.)
                or not opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)):
            raise RuntimeError('Position execution solver settings changed')

    def reset(self):
        # mj_resetData clears applied forces; reset never hides model changes.
        self.assert_unchanged()
        self.expected_applied[:] = 0.
        self.last = None

    def begin(self, data):
        self.assert_unchanged()
        if (not np.array_equal(data.qfrc_applied, self.expected_applied)
                or np.any(data.xfrc_applied != 0)):
            raise RuntimeError('Position execution does not accept undeclared external forces')
        self.expected_applied[:] = 0.
        if self.profile['bias_compensation'] == 'robot_bias':
            self.expected_applied[self.dofs] = data.qfrc_bias[self.dofs]
        data.qfrc_applied[:] = self.expected_applied
        self.last = {'schema_version': INTERVAL_SCHEMA, 'profile_sha256': profile_digest(self.profile),
                     'start_sim_time': float(data.time), 'end_sim_time': None,
                     'applied_robot_generalized_force': self.expected_applied[self.dofs].tolist(),
                     'nonrobot_applied_force_abs_max': 0., 'external_body_force_abs_max': 0.,
                     'impratio':float(self.model.opt.impratio),
                     'noslip_iterations': int(self.model.opt.noslip_iterations)}

    def finish(self, data):
        self.assert_unchanged()
        if not np.array_equal(data.qfrc_applied, self.expected_applied) or np.any(data.xfrc_applied != 0):
            raise RuntimeError('Applied forces changed inside the execution interval')
        self.last['end_sim_time'] = float(data.time)

    def metadata(self):
        self.assert_unchanged()
        ids = self.actuators
        return {'schema_version': SCHEMA, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'profile': copy.deepcopy(self.profile), 'profile_sha256': profile_digest(self.profile),
                'position_gain': self.model.actuator_gainprm[ids, 0].tolist(),
                'velocity_gain': (-self.model.actuator_biasprm[ids, 2]).tolist(),
                'actuator_force_range': self.model.actuator_forcerange[ids].tolist(),
                'parameter_origin': 'uncalibrated_configuration', 'constant_within_episode': True,
                'public_action': '14 absolute joint/full-opening targets, rad/m',
                'bias_timing': 'Recompute from current robot qfrc_bias once before each control interval; hold through its substeps.',
                'solver': {'impratio': self.profile.get('impratio',10.), 'multiccd': True, 'noslip_iterations': self.profile['noslip_iterations']},
                'scope': __doc__}

    def telemetry(self):
        return copy.deepcopy(self.last)
