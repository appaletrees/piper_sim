import copy
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from piper_sim.contracts import Command, Observation
from piper_sim.mujoco_execution import MujocoPositionExecution
from piper_sim.position_execution import SCHEMA, INTERVAL_SCHEMA, audit_position_execution, profile_digest, validate_profile
from piper_sim.recording import EpisodeWriter


PROFILE = {'schema_version': 1, 'bias_compensation': 'robot_bias', 'gripper_kp_N_per_m': 1000.,
           'gripper_kv_Ns_per_m': 120., 'gripper_force_limit_N': 5., 'noslip_iterations': 0}


def model_and_executor():
    bodies, actuators = [], []
    for side in ('left', 'right'):
        for i in range(1, 9):
            name = f'{side}/joint{i}'
            bodies.append(f'<body><joint name="{name}" type="slide" axis="0 0 1"/><geom type="sphere" size=".01" mass=".1"/></body>')
            if i < 8:
                actuator = side+'/gripper' if i == 7 else name
                actuators.append(f'<position name="{actuator}" joint="{name}" kp="200" kv="20" ctrlrange="-2 2" forcerange="-50 50"/>')
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(bodies)+'</worldbody><actuator>'+''.join(actuators)+'</actuator></mujoco>')
    indices = {s: {'ctrl': list(range(7*k, 7*k+7)), 'dof': list(range(8*k, 8*k+8))} for k, s in enumerate(('left', 'right'))}
    executor = MujocoPositionExecution(model, indices, PROFILE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, executor


def test_bias_is_owned_recomputed_and_reset_without_target_labels():
    model, data, executor = model_and_executor()
    executor.reset();executor.begin(data)
    np.testing.assert_array_equal(data.qfrc_applied, data.qfrc_bias)
    data.time = .033;executor.finish(data)
    assert executor.telemetry()['end_sim_time'] == .033
    info = executor.telemetry();info['applied_robot_generalized_force'][0] = 999
    assert executor.telemetry()['applied_robot_generalized_force'][0] != 999
    mujoco.mj_resetData(model, data);executor.reset()
    assert executor.telemetry() is None


@pytest.mark.parametrize('change', ['gain', 'force_limit', 'solver', 'external_generalized', 'external_body'])
def test_undeclared_changes_fail_before_integration(change):
    model, data, executor = model_and_executor()
    if change == 'gain': model.actuator_gainprm[6, 0] += 1
    if change == 'force_limit': model.actuator_forcerange[6, 1] += 1
    if change == 'solver': model.opt.noslip_iterations = 10
    if change == 'external_generalized': data.qfrc_applied[0] = 1.
    if change == 'external_body': data.xfrc_applied[1, 0] = 1.
    with pytest.raises(RuntimeError): executor.begin(data)
    assert data.time == 0.


@pytest.mark.parametrize('key,value', [('gripper_force_limit_N', 0), ('gripper_kp_N_per_m', np.nan),
    ('schema_version', True), ('noslip_iterations', True), ('bias_compensation', 'teacher_target')])
def test_invalid_profile_rejected(key, value):
    profile = {**PROFILE, key: value}
    with pytest.raises(ValueError): validate_profile(profile)


def fixture_episode(tmp_path):
    profile = {**PROFILE, 'bias_compensation': 'none'}
    kp = np.full(14, 200.);kp[[6, 13]] = 1000.
    kv = np.full(14, 20.);kv[[6, 13]] = 120.
    limits = np.tile([-50., 50.], (14, 1));limits[[6, 13]] = [-5., 5.]
    implementation = Path(__file__).resolve().parents[1]/'src/piper_sim/mujoco_execution.py'
    declaration = {'schema_version': SCHEMA, 'constant_within_episode': True, 'profile': profile,
        'profile_sha256': profile_digest(profile), 'implementation_sha256': hashlib.sha256(implementation.read_bytes()).hexdigest(),
        'position_gain': kp.tolist(), 'velocity_gain': kv.tolist(), 'actuator_force_range': limits.tolist()}
    metadata = {'config': {'mujoco': {'position_execution': profile, 'controller': {'mode': 'zero_order_observed'}}},
                'physics_model': {'position_execution': declaration, 'controller': {'mode': 'zero_order_observed'}}}
    path = tmp_path/'execution.h5';q = np.array([0, .2, -.2, 0, 0, 0, .04]*2)
    with EpisodeWriter(path, metadata) as writer:
        for i in range(3):
            execution = {'schema_version': INTERVAL_SCHEMA, 'profile_sha256': profile_digest(profile),
                'start_sim_time': i/30, 'end_sim_time': (i+1)/30, 'applied_robot_generalized_force': [0.]*16,
                'nonrobot_applied_force_abs_max': 0., 'external_body_force_abs_max': 0., 'noslip_iterations': 0}
            controller = {k: copy.deepcopy(declaration[k]) for k in ['position_gain', 'velocity_gain', 'actuator_force_range']}
            controller['command'] = q.tolist()
            controller['mode'] = 'zero_order_observed'
            before = Observation(q, 10+i, i/30)
            after = Observation(q, 10+i+.5, (i+1)/30, extras={'position_execution_interval': execution, 'controller_interval': controller})
            writer.append(before, Command(q, 10+i+.2, i), after)
    return path, metadata


def test_complete_fixed_execution_is_accepted_independently_of_task_success(tmp_path):
    path, metadata = fixture_episode(tmp_path)
    with h5py.File(path) as f:
        assert not f.attrs['success']
        result = audit_position_execution(f, metadata)
    assert result['passed'] and result['checked_intervals'] == 3
    assert len(result['gripper_regime_segments']) == 1


@pytest.mark.parametrize('fault,expected', [
    ('hidden_force_limit', 'interval_parameter_mismatch/actuator_force_range'),
    ('wrong_time', 'interval_clock_mismatch'), ('wrong_action', 'interval_action_mismatch'),
    ('missing_interval', 'missing_execution_intervals'), ('extra_force', 'undeclared_solver_or_external_force'),
    ('undeclared_bias', 'unexpected_bias_force'), ('reference_mode', 'interval_reference_mode_mismatch')])
def test_middle_interval_corruption_cannot_be_hidden(tmp_path, fault, expected):
    path, metadata = fixture_episode(tmp_path)
    with h5py.File(path, 'r+') as f:
        e = json.loads(f['next_observation/extras_json'][1]);x = e['position_execution_interval']
        if fault == 'hidden_force_limit': e['controller_interval']['actuator_force_range'][13] = [-30, 30]
        if fault == 'wrong_time': x['end_sim_time'] += 1.
        if fault == 'wrong_action': e['controller_interval']['command'][0] += .1
        if fault == 'missing_interval': e['position_execution_interval'] = None
        if fault == 'extra_force': x['nonrobot_applied_force_abs_max'] = .1
        if fault == 'undeclared_bias': x['applied_robot_generalized_force'][0] = .1
        if fault == 'reference_mode': e['controller_interval']['mode'] = 'monotone_cubic_velocity'
        f['next_observation/extras_json'][1] = json.dumps(e)
    with h5py.File(path) as f: result = audit_position_execution(f, metadata)
    assert not result['passed'] and expected in result['violations']


def test_legacy_hidden_controller_is_reported_without_modifying_source(tmp_path):
    path, metadata = fixture_episode(tmp_path)
    metadata.pop('physics_model');metadata['teacher'] = {'per_kind_gripper_force_limits_N': {'plate': 30}}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path) as f: result = audit_position_execution(f, metadata)
    assert 'missing_backend_position_execution' in result['violations']
    assert 'teacher_selects_controller_by_target' in result['violations']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
