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


@pytest.mark.parametrize('tendon', [False, True])
@pytest.mark.parametrize('reference', [0., .4, 100., -100.])
def test_velocity_reference_has_analytic_pd_force_inside_original_actuator_limit(tendon, reference):
    bodies = '<body><joint name="a" type="slide"/><geom size=".05" mass="1"/></body>'
    transmission = 'joint="a"'
    extra = ''
    if tendon:
        bodies += '<body pos=".2 0 0"><joint name="b" type="slide"/><geom size=".05" mass="1"/></body>'
        extra = '<tendon><fixed name="width"><joint joint="a" coef="1"/><joint joint="b" coef="1"/></fixed></tendon>'
        transmission = 'tendon="width"'
    original = mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
        <worldbody>{bodies}</worldbody>{extra}<actuator>
        <position {transmission} kp="40" kv="4" forcerange="-10 10"/>
        </actuator></mujoco>''')
    model = copy.copy(original); data = mujoco.MjData(model)
    data.qpos[:] = .05/model.nq; data.qvel[:] = .3/model.nv; data.ctrl[:] = .2
    module.set_velocity_reference(model, [0], [reference])
    mujoco.mj_forward(model, data)
    assert data.actuator_force[0] == pytest.approx(np.clip(40*(.2-.05)+4*(reference-.3),-10,10))
    np.testing.assert_array_equal(data.qfrc_applied, np.zeros(model.nv))
    assert original.actuator_biasprm[0,0] == 0  # no mutation of the baseline model


@pytest.mark.parametrize('actuator,ids,velocity', [('motor',[0],[.1]),('position',[0],[np.nan]),
    ('position',[-1],[0.]),('position',[0,0],[0.,0.])])
def test_invalid_velocity_reference_preserves_model(actuator,ids,velocity):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body><joint name="a"/>
        <geom size=".05"/></body></worldbody><actuator><{actuator} joint="a"/></actuator></mujoco>''')
    original = model.actuator_biasprm.copy()
    with pytest.raises(ValueError): module.set_velocity_reference(model, ids, velocity)
    np.testing.assert_array_equal(model.actuator_biasprm, original)


def test_hermite_linear_motion_matches_exact_position_and_velocity():
    q=np.array([.2,.03]);v=np.array([.5,-.1]);dt=1/30
    targets,speeds=module.hermite_targets(q,q+dt*v,v,v,dt,32)
    np.testing.assert_allclose(targets,q+np.arange(1,33)[:,None]*dt/32*v,atol=1e-15)
    np.testing.assert_allclose(speeds,np.tile(v,(32,1)),atol=1e-14)
    np.testing.assert_array_equal(targets[-1],q+dt*v)
    np.testing.assert_array_equal(speeds[-1],v)


def test_hermite_stop_keeps_overshoot_visible_instead_of_clipping():
    q,v=module.hermite_targets([0.],[0.],[1.],[0.],1/30,32)
    assert q.max()>0 and v.min()<0  # constant endpoints do not imply a constant path
    assert q[-1,0] == v[-1,0] == 0


def test_bounded_hermite_explicitly_limits_derivatives_at_stop():
    q,v=module.bounded_hermite_targets([0.],[0.],[1.],[0.],1/30,32)
    np.testing.assert_array_equal(q,np.zeros((32,1)))
    np.testing.assert_array_equal(v,np.zeros((32,1)))


@pytest.mark.parametrize('start_speed,end_speed', [(-100,100),(100,100),(100,-100),(-100,-100),(1,0)])
def test_bounded_hermite_keeps_all_applied_targets_monotone(start_speed,end_speed):
    a=np.array([-.4,.095]);b=np.array([.3,0.])
    q,v=module.bounded_hermite_targets(a,b,[start_speed]*2,[end_speed]*2,1/30,32)
    assert np.all(q >= np.minimum(a,b)) and np.all(q <= np.maximum(a,b))
    assert np.all(np.diff(np.vstack([a,q]),axis=0)[:,0]>=0)
    assert np.all(np.diff(np.vstack([a,q]),axis=0)[:,1]<=0)
    np.testing.assert_array_equal(q[-1],b)
