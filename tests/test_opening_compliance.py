import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('opening_compliance_test', directory/'probe_opening_compliance.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def setup():
    bodies, tendons, actuators = '', '', ''
    for index, side in enumerate(('left', 'right')):
        for jaw in (7, 8):
            bodies += f'<body pos="{index} {jaw} 0"><joint name="{side}/joint{jaw}" type="slide"/><geom size=".02" mass=".01"/></body>'
        tendons += f'<fixed name="{side}/width"><joint joint="{side}/joint7" coef="1"/><joint joint="{side}/joint8" coef="1"/></fixed>'
        actuators += f'<position name="{side}/gripper" tendon="{side}/width" kp="1000" kv="120" forcerange="-5 5"/>'
    m = mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".0010416666666666667" integrator="implicitfast" gravity="0 0 0"/><worldbody>{bodies}</worldbody><tendon>{tendons}</tendon><actuator>{actuators}</actuator></mujoco>')
    return m, mujoco.MjData(m)


def full_state(m, d):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(m, kind))
    mujoco.mj_getState(m, d, result, kind)
    return result


def test_disabled_compliance_preserves_full_native_integration(module):
    m, d = setup()
    other_m, other = copy.copy(m), copy.copy(d)
    observer = module.OpeningCompliance(other_m, stiffness_floor=1.)
    for i in range(800):
        reference = .002+.001*np.sin(i/100)
        velocity = .01*np.cos(i/100)
        d.ctrl[:] = other.ctrl[:] = reference
        m.actuator_biasprm[:, 0] = other_m.actuator_biasprm[:, 0] = 120*velocity
        observer.apply(other)
        mujoco.mj_step(m, d)
        mujoco.mj_step(other_m, other)
        np.testing.assert_array_equal(full_state(m, d), full_state(other_m, other))
    assert np.all(observer.second >= 0) and np.all(observer.second <= 1)


def test_formula_reduces_only_negative_position_error_and_keeps_velocity_damping(module):
    m, d = setup()
    c = module.OpeningCompliance(m)
    d.qpos[:] = .003
    d.qvel[:] = [.01, .02, -.01, -.02]
    for i in range(800):
        # Opening followed by a hold. The latch must persist through a hold.
        d.ctrl[:] = min(i, 400)*1e-6
        d.time = i*m.opt.timestep
        m.actuator_biasprm[:, 0] = 120*.01
        c.apply(d)
        mujoco.mj_forward(m, d)
        error = d.ctrl-d.qpos[c.qids].sum(axis=1)
        expected = 1000*(1-.9*c.second)*error+120*(.01-d.qvel[c.qids].sum(axis=1))
        np.testing.assert_allclose(d.actuator_force, np.clip(expected, -5, 5), atol=1e-12)
        np.testing.assert_array_equal(m.actuator_forcerange, [[-5, 5], [-5, 5]])
        np.testing.assert_array_equal(m.actuator_biasprm[:, 2], -120)
        np.testing.assert_array_equal(d.qfrc_applied, 0)
    assert np.all(c.direction == 1) and np.all(c.second > .999)
    # With the jaw behind an opening target, positive tracking force is intact.
    d.time += m.opt.timestep
    d.ctrl[:] = .007
    m.actuator_biasprm[:, 0] = 0
    c.apply(d)
    np.testing.assert_array_equal(c.last['force_bias_N'], [0., 0.])


def test_filter_has_bounded_history_across_reversals_and_explicit_reset(module):
    m, d = setup()
    c = module.OpeningCompliance(m)
    for i in range(900):
        d.time = i*m.opt.timestep
        d.ctrl[:] = .01+.003*np.sin(i/100)
        m.actuator_biasprm[:, 0] = 0
        c.apply(d)
        assert np.all(c.first >= 0) and np.all(c.first <= 1)
        assert np.all(c.second >= 0) and np.all(c.second <= 1)
    before = m.actuator_biasprm.copy()
    with pytest.raises(ValueError, match='contiguous'):
        c.apply(d)
    np.testing.assert_array_equal(before, m.actuator_biasprm)
    c.reset()
    d.time = 0
    c.apply(d)
    assert c.last_time == 0 and np.all(c.second == 0)


@pytest.mark.parametrize('kwargs', [{'stiffness_floor': 0}, {'stiffness_floor': 1.01}, {'time_constant': 0}, {'time_constant': np.nan}])
def test_invalid_controller_parameters_rejected(module, kwargs):
    m, _ = setup()
    with pytest.raises(ValueError):
        module.OpeningCompliance(m, **kwargs)


def test_mutating_trace_hook_cannot_be_mislabeled_as_read_only(module):
    with pytest.raises(ValueError, match='metadata'):
        module.tracing.main(before_step=lambda m, d: None)
