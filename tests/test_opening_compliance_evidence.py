import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from piper_sim.mujoco_controller import MujocoReferenceController


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def evidence(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    fixture = load('compliance_fixture', root/'tests/test_opening_compliance.py')
    probe = load('compliance_probe_evidence', root/'scripts/probe_opening_compliance.py')
    audit = load('compliance_audit_evidence', root/'scripts/audit_opening_compliance.py')
    m, d = fixture.setup()
    m.actuator_ctrllimited[:] = True
    m.actuator_ctrlrange[:] = [0., .095]
    d.qpos[:] = .003
    projection = np.zeros((2, 4))
    projection[0, :2] = 1
    projection[1, 2:] = 1
    control = MujocoReferenceController(m, [0, 1], projection, 32,
        {'schema_version': 1, 'mode': 'monotone_cubic_velocity'})
    control.reset([.006, .006])
    compensation = probe.OpeningCompliance(m)
    native = mujoco.mj_step

    def step(model, data):
        compensation.apply(data)
        native(model, data)

    monkeypatch.setattr(mujoco, 'mj_step', step)

    def record():
        value = control.telemetry()
        for key in ('command', 'previous_command', 'last_velocity_reference', 'position_gain', 'velocity_gain'):
            expanded = np.zeros(14)
            expanded[[6, 13]] = value[key]
            value[key] = expanded.tolist()
        bounds = np.zeros((14, 2))
        bounds[[6, 13]] = value['actuator_force_range']
        value['actuator_force_range'] = bounds.tolist()
        value['opening_compliance'] = copy.deepcopy(compensation.last)
        return value

    d.ctrl[:] = .006
    control.advance(d)
    mujoco.mj_forward(m, d)
    before = record()
    records = []
    for target in (.0061, .0063, .0066, .0066, .0062, .0058):
        d.ctrl[:] = target
        control.advance(d)
        mujoco.mj_forward(m, d)
        after = record()
        physics = {'arms': {}}
        for j, side in enumerate(('left', 'right')):
            ids = slice(j*2, j*2+2)
            physics['arms'][side] = {
                key: [0.]*6+value[ids].tolist()
                for key, value in (('joint_position', d.qpos), ('joint_velocity', d.qvel),
                                   ('actuator_generalized_force', d.qfrc_actuator))}
        records.append((before, after, physics))
        before = after
    profile = {'mode': 'opening_negative_error_compliance_v1', 'stiffness_floor': .1, 'time_constant_s': .06}
    return audit, profile, records


def test_reconstruction_matches_actual_tendon_dynamics_through_open_hold_and_reverse(evidence):
    audit, profile, records = evidence
    for before, after, physics in records:
        errors = audit.reconstruct(before, after, physics, profile)
        assert max(errors.values()) < 1e-10


@pytest.mark.parametrize('corruption', ['filter', 'bias', 'clock', 'force'])
def test_damaged_compliance_evidence_is_rejected(evidence, corruption):
    audit, profile, records = evidence
    before, after, physics = copy.deepcopy(records[2])
    if corruption == 'filter':
        after['opening_compliance']['filter_second'][0] += .01
    elif corruption == 'bias':
        after['opening_compliance']['force_bias_N'][0] += .1
    elif corruption == 'clock':
        after['opening_compliance']['sim_time'] += .01
    else:
        physics['arms']['right']['actuator_generalized_force'][6] += 1
    with pytest.raises(ValueError):
        audit.reconstruct(before, after, physics, profile)
