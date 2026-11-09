"""Synthetic fixed-executor evidence for file/reader tests, not physics data."""
import hashlib
from pathlib import Path

import numpy as np

from piper_sim.position_execution import SCHEMA, INTERVAL_SCHEMA, profile_digest


def declaration():
    profile = {'schema_version': 1, 'bias_compensation': 'none', 'gripper_kp_N_per_m': 1000.,
               'gripper_kv_Ns_per_m': 120., 'gripper_force_limit_N': 5., 'noslip_iterations': 0}
    kp = np.full(14, 200.);kp[[6, 13]] = 1000.
    kv = np.full(14, 20.);kv[[6, 13]] = 120.
    limits = np.tile([-50., 50.], (14, 1));limits[[6, 13]] = [-5., 5.]
    implementation = Path(__file__).resolve().parents[1]/'src/piper_sim/mujoco_execution.py'
    return {'schema_version': SCHEMA, 'constant_within_episode': True, 'profile': profile,
            'profile_sha256': profile_digest(profile), 'implementation_sha256': hashlib.sha256(implementation.read_bytes()).hexdigest(),
            'position_gain': kp.tolist(), 'velocity_gain': kv.tolist(), 'actuator_force_range': limits.tolist()}


def interval(executor, command, start, end):
    return {'position_execution_interval': {'schema_version': INTERVAL_SCHEMA,
        'profile_sha256': executor['profile_sha256'], 'start_sim_time': start, 'end_sim_time': end,
        'applied_robot_generalized_force': [0.]*16, 'nonrobot_applied_force_abs_max': 0.,
        'external_body_force_abs_max': 0., 'noslip_iterations': 0},
        'controller_interval': {**{k: executor[k] for k in ('position_gain', 'velocity_gain', 'actuator_force_range')},
                                'command': np.asarray(command).tolist(), 'mode': 'zero_order_observed'}}
