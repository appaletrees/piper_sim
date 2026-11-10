import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('contact_trace',
    Path(__file__).resolve().parents[1]/'scripts/trace_contact_substeps.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, value, kind)
    return value


@pytest.mark.parametrize('integrator', ['Euler', 'implicitfast'])
@pytest.mark.parametrize('jacobian', ['dense', 'sparse'])
def test_force_trace_preserves_state_and_resolves_constraints(integrator, jacobian, tmp_path):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".001" integrator="{integrator}" jacobian="{jacobian}"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .07"><joint name="a" type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size=".05" mass="1"/></body></worldbody>
      <actuator><position joint="a" kp="400" kv="40" forcerange="-10 10"/></actuator></mujoco>''')
    original_model = copy.copy(model)
    plain, observed = mujoco.MjData(model), mujoco.MjData(model)
    tracer = module.ContactTrace(model, [0], [0], [0], [1], payload_bodies=[1])
    tracer.projected.fill(123.)  # prove no-contact rows cannot reuse scratch memory
    for i in range(250):
        plain.ctrl[:] = observed.ctrl[:] = -.07+.005*np.sin(i/30)
        mujoco.mj_step(model, plain)
        before = tracer.before(observed)
        mujoco.mj_step(model, observed)
        tracer.after(observed, before, 'release', 'ball')
        np.testing.assert_array_equal(state(model, plain), state(model, observed))
        np.testing.assert_array_equal(plain.qacc, observed.qacc)
        np.testing.assert_array_equal(before['payload_com_world'][0], plain.xipos[1])
        np.testing.assert_array_equal(before['payload_origin_world'][0], plain.xpos[1])
        expected_velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, plain, mujoco.mjtObj.mjOBJ_BODY, 1, expected_velocity, 0)
        np.testing.assert_array_equal(before['payload_velocity_world'][0], expected_velocity)
        np.testing.assert_allclose(before['constraint_force_by_type'].sum(axis=0),
                                   before['qfrc_constraint'], atol=1e-12, rtol=1e-12)
        expected = np.clip(400*(before['control']-before['actuator_length'])
                           -40*before['actuator_velocity'], -10, 10)
        np.testing.assert_allclose(before['actuator_force'], expected, atol=1e-12)
    assert len(tracer.contacts) > 0
    for field in ('actuator_gainprm', 'actuator_biasprm', 'geom_friction', 'geom_solref', 'geom_solimp'):
        np.testing.assert_array_equal(getattr(model, field), getattr(original_model, field))
    tracer.save(tmp_path, {})
    with np.load(tmp_path/'trace.npz', allow_pickle=False) as arrays:
        assert arrays['qpos_after'].shape == (250, 1)
        assert arrays['constraint_force_by_type'].shape == (250, 8, 1)
        assert arrays['contact_force_local'].shape[1] == 6
        assert arrays['payload_com_world'].shape == (250, 1, 3)
        np.testing.assert_allclose(arrays['time_after']-arrays['time_before'], .001)


def test_trace_rejects_ambiguous_timing_and_capacity_exhaustion():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    with pytest.raises(ValueError, match='timing'):
        module.ContactTrace(model, [0], [0], [], [])
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    data = mujoco.MjData(model)
    tracer = module.ContactTrace(model, [0], [0], [], [], max_steps=1)
    before = tracer.before(data)
    mujoco.mj_step(model, data, nstep=2)
    with pytest.raises(ValueError, match='exactly one'):
        tracer.after(data, before, 'release', 'ball')
    before = tracer.before(data)
    mujoco.mj_step(model, data)
    tracer.after(data, before, 'release', 'ball')
    with pytest.raises(RuntimeError, match='limit reached'):
        tracer.before(data)
