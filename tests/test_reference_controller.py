import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from piper_sim import mujoco_controller as module
spec2=importlib.util.spec_from_file_location('original_diagnostic',
    Path(__file__).resolve().parents[1]/'scripts/diagnose_substep_motion.py')
diagnostic=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(diagnostic)
CONFIG={'schema_version':1,'mode':'monotone_cubic_velocity'}


def setup():
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".0010416666666666667" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody><body><joint name="slide" type="slide"/>
      <geom type="sphere" size=".05" mass=".1"/></body></worldbody>
      <actuator><position joint="slide" kp="400" kv="40" ctrlrange="-1 1" forcerange="-10 10"/></actuator>
      </mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    controller=module.MujocoReferenceController(model,[0],[[1]],32,CONFIG)
    controller.reset([0])
    return model,data,controller


def test_complete_integration_and_native_extrema_match_the_independent_diagnostic():
    model,data,controller=setup()
    expected_model=copy.copy(model);expected=copy.copy(data)
    previous=np.zeros(1);previous_velocity=np.zeros(1)
    for target in [0., .001, .003, .004, .004, .001, 0.,0.]:
        data.ctrl[:]=target
        controller.advance(data);mujoco.mj_forward(model,data)
        speed=(np.array([target])-previous)/(32*model.opt.timestep)
        q,v=diagnostic.bounded_hermite_targets(previous,[target],previous_velocity,speed,32*model.opt.timestep,32)
        trace=[expected.qvel.copy()]
        for position,velocity in zip(q,v):
            expected.ctrl[:]=position
            diagnostic.set_velocity_reference(expected_model,[0],velocity)
            mujoco.mj_step(expected_model,expected);trace.append(expected.qvel.copy())
        mujoco.mj_forward(expected_model,expected)
        np.testing.assert_array_equal(diagnostic.integration_state(model,data),diagnostic.integration_state(expected_model,expected))
        np.testing.assert_array_equal(data.qacc,expected.qacc)
        info=controller.telemetry()
        np.testing.assert_array_equal(info['native_velocity_min'],np.min(trace,axis=0))
        np.testing.assert_array_equal(info['native_velocity_max'],np.max(trace,axis=0))
        assert info['integration_steps']==32
        assert info['end_sim_time']-info['start_sim_time']==pytest.approx(1/30)
        assert abs(data.actuator_force[0])<=10
        previous=np.array([target]);previous_velocity=v[-1]
    assert controller.telemetry()['native_jerk_samples']==32


def test_reset_removes_stale_reference_and_evidence():
    model,data,c=setup()
    data.ctrl[0]=.005;c.advance(data)
    assert model.actuator_biasprm[0,0]!=0
    c.reset([.1])
    assert model.actuator_biasprm[0,0]==0 and c.telemetry() is None
    np.testing.assert_array_equal(c.previous,[.1])
    np.testing.assert_array_equal(c.previous_velocity,[0])
    assert c.previous_acceleration is None


@pytest.mark.parametrize('target',[np.nan,np.inf,-1.1,1.1])
def test_invalid_command_does_not_advance_or_mutate_controller(target):
    model,data,c=setup();data.ctrl[0]=target
    state=diagnostic.integration_state(model,data);bias=model.actuator_biasprm.copy()
    with pytest.raises(ValueError):c.advance(data)
    np.testing.assert_array_equal(state,diagnostic.integration_state(model,data))
    np.testing.assert_array_equal(bias,model.actuator_biasprm)
    np.testing.assert_array_equal(c.previous,[0])


@pytest.mark.parametrize('config',[{}, {'schema_version':True,'mode':'monotone_cubic_velocity'},
    {'schema_version':1,'mode':'unknown'}, {**CONFIG,'relax_limits':True}])
def test_invalid_configuration_fails_before_model_mutation(config):
    model,_,_=setup();old=model.actuator_biasprm.copy()
    with pytest.raises(ValueError):module.MujocoReferenceController(model,[0],[[1]],32,config)
    np.testing.assert_array_equal(old,model.actuator_biasprm)


def test_runtime_servo_or_clock_change_is_not_silently_accepted():
    m,d,c=setup();m.actuator_biastype[0]=mujoco.mjtBias.mjBIAS_NONE
    with pytest.raises(ValueError,match='servos'):c.advance(d)
    assert d.time==0
    m,d,c=setup();m.opt.timestep*=2
    with pytest.raises(ValueError,match='timestep'):c.advance(d)
    assert d.time==0


def test_interval_telemetry_is_detached_from_controller_state_and_forces_are_separate():
    m,d,c=setup();d.qfrc_applied[0]=.2;d.ctrl[0]=.001;c.advance(d)
    record=c.telemetry();record['command'][0]=999
    assert c.telemetry()['command']==[.001]
    assert d.qfrc_applied[0]==.2
    assert c.telemetry()['native_jerk_samples']==31
    assert c.metadata()['experimental'] and c.metadata()['hardware_calibration']=='unidentified'


def test_derivative_limiter_jump_is_recorded_on_a_stop():
    _,data,c=setup();data.ctrl[0]=.01;c.advance(data);c.advance(data)
    assert c.telemetry()['limiter_start_velocity_jump'][0]==pytest.approx(-.3)
    assert c.telemetry()['position_reference_min']==c.telemetry()['position_reference_max']==[.01]


def test_autoreset_on_the_first_substep_is_detected_even_when_time_still_matches():
    model,data,c=setup();data.qvel[0]=1e20
    with pytest.raises(RuntimeError,match='physics warning'):c.advance(data)
    assert data.time==model.opt.timestep  # a timestamp-only check would miss this reset
    assert c.telemetry() is None
    np.testing.assert_array_equal(c.previous,[0])


def test_holding_exact_decimal_limit_does_not_fail_on_polynomial_roundoff():
    m,d,c=setup();m.actuator_ctrlrange[0]=[-.095,.095]
    d.qpos[0]=d.ctrl[0]=.095;c.reset([.095]);mujoco.mj_forward(m,d)
    c.advance(d)
    assert c.telemetry()['position_reference_min']==c.telemetry()['position_reference_max']==[.095]


def test_gripper_native_velocity_uses_the_full_tendon_opening():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0" timestep=".001"/>
        <worldbody><body><joint name="a" type="slide"/><geom size=".01"/></body>
        <body pos=".1 0 0"><joint name="b" type="slide"/><geom size=".01"/></body></worldbody>
        <tendon><fixed name="width"><joint joint="a" coef="1"/><joint joint="b" coef="1"/></fixed></tendon>
        <actuator><position tendon="width" kp="1000" kv="25" ctrlrange="0 .095" forcerange="-10 10"/></actuator>
        </mujoco>''')
    d=mujoco.MjData(m);d.qvel[:]=[.01,.02]
    c=module.MujocoReferenceController(m,[0],[[1,1]],1,CONFIG);c.reset([0]);c.advance(d)
    observed=c.telemetry()
    assert observed['native_velocity_max'][0]>=.03
    expected=(d.qvel.sum()-.03)/m.opt.timestep
    assert observed['native_acceleration_abs_max'][0]==pytest.approx(abs(expected))


def test_backend_records_endpoint_semantics_and_full_interval_then_resets(tmp_path):
    import h5py
    import json
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    from piper_sim.simulation import SimulationSession
    cfg=load('configs/clean_table_servo.yaml')
    backend=MujocoBackend(cfg,cameras=False)
    session=SimulationSession(backend,record_dir=tmp_path)
    try:
        target=np.array(cfg['robot']['home']*2);target[[0,7]]+=.002;target[[6,13]]+=.001
        session.start_recording('servo')
        session.step(target);session.step(target)
        session.stop_recording(success=False)
        with h5py.File(tmp_path/'servo.h5','r') as f:
            meta=json.loads(f.attrs['metadata'])
            assert meta['physics_model']['controller']['mode']=='monotone_cubic_velocity'
            assert 'following observation-to-next_observation interval' in meta['action_semantics']
            np.testing.assert_allclose(f['action/position'][:],np.tile(target,(2,1)),atol=1e-8)
            for i,raw in enumerate(f['next_observation/extras_json'][:]):
                record=json.loads(raw)['controller_interval']
                assert record['start_sim_time']==pytest.approx(f['observation/sim_time'][i])
                assert record['end_sim_time']==pytest.approx(f['next_observation/sim_time'][i])
                assert len(record['native_velocity_min'])==len(record['command'])==14
                np.testing.assert_array_equal(record['command'],target)
        session.reset()
        assert session.read().extras['controller_interval'] is None
        np.testing.assert_array_equal(backend.model.actuator_biasprm[:,0],0)
        np.testing.assert_array_equal(backend.control_targets(),cfg['robot']['home']*2)
    finally:session.close()
