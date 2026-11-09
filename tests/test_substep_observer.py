import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('substep_diagnostic',
    Path(__file__).resolve().parents[1]/'scripts/diagnose_substep_motion.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.mark.parametrize('integrator', ['Euler','implicitfast'])
@pytest.mark.parametrize('contact', [False,True])
def test_observing_substeps_preserves_entire_integration_state(integrator, contact):
    floor = '<geom type="plane" size="1 1 .1"/>' if contact else ''
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".0010416666666666667" integrator="{integrator}"/>
      <worldbody>{floor}<body pos="0 0 .15"><joint name="joint" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody>
      <actuator><position joint="joint" kp="400" kv="40"/></actuator></mujoco>''')
    a = mujoco.MjData(model); mujoco.mj_forward(model,a); b=copy.copy(a)
    observed = []
    for frame in range(25):
        a.ctrl[:] = b.ctrl[:] = -.15+.02*np.sin(frame/8)
        mujoco.mj_step(model,a,nstep=32)
        t,q,v=module.observed_steps(model,b,32,[0],[0]); observed.append(t)
        np.testing.assert_array_equal(module.integration_state(model,a),module.integration_state(model,b))
        np.testing.assert_array_equal(a.qacc,b.qacc)
        assert t.shape == (32,) and q.shape == v.shape == (32,1)
        if contact and frame == 24: assert a.ncon > 0
    times = np.concatenate(observed)
    assert np.allclose(np.diff(times),model.opt.timestep,rtol=0,atol=1e-12)


@pytest.mark.parametrize('steps,positions,velocities', [(0,[0],[0]),(1.5,[0],[0]),
    (1,[-1],[0]),(1,[0],[10]),(1,[0.],[0]),(1,[],[0])])
def test_invalid_observer_request_does_not_advance_state(steps, positions, velocities):
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="slide"/><geom size=".05"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    with pytest.raises(ValueError): module.observed_steps(model,data,steps,positions,velocities)
    assert data.time == 0


def test_interval_motion_and_native_velocity_are_reported_separately():
    # A velocity pulse hidden between observation times must remain visible.
    times=np.arange(65)/960
    q=np.zeros((65,16));v=np.zeros_like(q)
    v[10,2]=2.;q[20:,2]=.01
    result=module.summarize(times,q,v,32)
    assert result['left/arm']['native_substep_velocity']['absolute_max'] == 2.
    assert result['left/arm']['native_control_boundary_velocity']['absolute_max'] == 0.
    assert result['left/arm']['control_position_difference_velocity']['absolute_max'] == pytest.approx(.3)


def test_linear_candidate_preserves_endpoints_and_joint_limits_without_clipping():
    start=np.array([-.5,.3,.095]);end=np.array([.5,-.3,0.])
    targets=module.linear_targets(start,end,32)
    np.testing.assert_array_equal(targets[-1],end)
    np.testing.assert_allclose(np.diff(np.vstack([start,targets]),axis=0),np.tile((end-start)/32,(32,1)),atol=1e-16)
    assert np.all(targets >= np.minimum(start,end)) and np.all(targets <= np.maximum(start,end))
    assert targets.shape == (32,3)
