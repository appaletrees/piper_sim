import copy
import json

import h5py
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from piper_sim.experts.trajectory import smooth_trajectory, path_trajectory
from piper_sim.experts.tcp_limits import speed_constraint, pose_speed_peaks, validate_limits, measured_violations
from piper_sim.mujoco_tcp import MujocoTcpObserver
from piper_sim.tcp_audit import audit_tcp_motion

LIMITS = {'linear_speed_m_s': .25, 'angular_speed_rad_s': 1.5, 'planning_fraction': .95}


def arc(q):
    c, s = np.cos(q[0]), np.sin(q[0])
    return np.array([c, s, 0]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@pytest.mark.parametrize('kind', ['smooth','natural','convex'])
def test_retiming_bounds_real_fk_arc_and_preserves_endpoints_and_joint_limits(kind):
    start, end = np.array([0.]), np.array([1.5])
    limits = dict(velocity=1., acceleration=2.5, jerk=20.)
    def generate(check):
        if kind == 'smooth':
            return smooth_trajectory(start,end,1/30,speed_ratio=check,**limits)
        return path_trajectory([[0.],[.4],[1.],[1.5]],1/30,interpolation=kind,speed_ratio=check,**limits)
    original = generate(None)
    assert pose_speed_peaks(start,original,arc,1/30)[0] > .5
    path = generate(speed_constraint(start,arc,1/30,LIMITS))
    assert len(path) > len(original)
    assert pose_speed_peaks(start,path,arc,1/30)[0] <= .25*.95*(1+1e-9)
    np.testing.assert_array_equal(path[-1], end)
    assert np.all(np.diff(path[:,0]) >= -1e-9)
    padded = np.vstack([np.tile(start,(4,1)),path,np.tile(end,(4,1))])
    for n, bound in enumerate(limits.values(),1):
        assert np.max(abs(np.diff(padded,n=n,axis=0))*30**n) <= bound+1e-6


def test_rotation_only_is_limited_and_invalid_budgets_cannot_relax_constraints():
    forward = lambda q: (np.zeros(3),arc(q)[1])
    config = {**LIMITS,'angular_speed_rad_s':.2}
    path = smooth_trajectory([0.],[1.],1/30,velocity=1.,acceleration=2.,jerk=10.,
                             speed_ratio=speed_constraint([0.],forward,1/30,config))
    assert pose_speed_peaks([0.],path,forward,1/30)[1] <= .2*.95*(1+1e-9)
    for value in (0.,-1.,float('nan'),True,'0.25'):
        with pytest.raises(ValueError):validate_limits({**LIMITS,'linear_speed_m_s':value})
    with pytest.raises(ValueError):validate_limits({**LIMITS,'planning_fraction':1.1})
    with pytest.raises(ValueError):validate_limits({**LIMITS,'typo':1})
    with pytest.raises(ValueError):
        smooth_trajectory([0.],[1.],1/30,velocity=1.,acceleration=2.,jerk=10.,speed_ratio=lambda _:np.nan)


def synthetic_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".01" gravity="0 0 0"/>
      <worldbody><body name="left/link6"><joint type="hinge" axis="0 0 1"/>
      <geom type="sphere" size=".02" pos="1 0 0"/></body>
      <body name="right/link6" pos="0 2 0"><joint type="hinge" axis="0 0 1"/>
      <geom type="sphere" size=".02" pos="1 0 0"/></body></worldbody></mujoco>''')


def test_observer_catches_interior_peak_with_zero_boundary_speed_without_mutating_solver():
    model=synthetic_model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    observer=MujocoTcpObserver(model,4,offset=(1.,0.,0.))
    observer.begin(data)
    for i, speed in enumerate((0.,.8,0.,0.),1):
        data.time=i*.01;data.qpos[:]=.3;data.qvel[:]=speed
        before={k:getattr(data,k).copy() for k in ('qpos','qvel','qacc','qacc_warmstart','qfrc_applied','xpos','xmat')}
        observer.sample(data)
        for k,v in before.items():np.testing.assert_array_equal(getattr(data,k),v)
    observer.finish();evidence=observer.telemetry()
    assert evidence['samples']==5
    for arm in evidence['arms'].values():
        assert arm['linear_speed_max']==pytest.approx(.8)
        assert arm['angular_speed_max']==pytest.approx(.8)
        assert arm['linear_peak_step']==2
        assert arm['start_linear_velocity']==arm['end_linear_velocity']==[0.,0.,0.]
    assert measured_violations(evidence,LIMITS)
    observer.reset();assert observer.telemetry() is None
    observer.begin(data)
    with pytest.raises(ValueError,match='Incomplete'):observer.finish()


@pytest.mark.parametrize('controlled',[False,True])
def test_native_observer_is_dynamically_inert_and_clears_on_reset(controlled):
    from piper_sim.config import load
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.contracts import Command
    cfg=load('configs/clean_table_servo.yaml')
    if not controlled:cfg.pop('mujoco')
    observed=copy.deepcopy(cfg);observed.setdefault('mujoco',{})['tcp_motion_evidence']=True
    a,b=MujocoBackend(cfg),MujocoBackend(observed)
    try:
        for i in range(12):
            q=np.array(cfg['robot']['home']*2);q[0]=q[7]=.02*i/12
            for backend in (a,b):backend.apply(Command(q,i,i));backend.step()
            for field in ('qpos','qvel','qacc','qacc_warmstart','qfrc_applied','ctrl'):
                np.testing.assert_array_equal(getattr(a.data,field),getattr(b.data,field))
            evidence=b.read().extras['tcp_motion_interval']
            assert evidence['samples']==b.substeps*b.decimation+1
            assert evidence['end_sim_time']==b.sim_time
        b.reset();assert b.read().extras['tcp_motion_interval'] is None
    finally:a.close();b.close()


def write_evidence(path, speed=.8):
    steps=4;dt=.01;rows=[]
    for i in range(2):
        arm={'start_linear_velocity':[0.,0.,0.],'end_linear_velocity':[0.,0.,0.],
             'start_angular_velocity':[0.,0.,0.],'end_angular_velocity':[0.,0.,0.],
             'linear_speed_max':speed,'angular_speed_max':0.,'linear_peak_step':2,'angular_peak_step':0}
        rows.append({'schema_version':1,'start_sim_time':i*.04,'end_sim_time':(i+1)*.04,
                     'integration_dt':dt,'integration_steps':steps,'samples':5,
                     'arms':{'left':copy.deepcopy(arm),'right':copy.deepcopy(arm)}})
    physics={'arms':{s:{'tcp_linear_velocity_world_m_s':[0,0,0],
                       'tcp_angular_velocity_world_rad_s':[0,0,0]} for s in ('left','right')}}
    with h5py.File(path,'w') as f:
        f.create_dataset('action/position',data=np.zeros((2,14)))
        for prefix, times, intervals in (
                ('observation',[0.,.04],[None,rows[0]]),('next_observation',[.04,.08],rows)):
            f.create_dataset(prefix+'/sim_time',data=times)
            f.create_dataset(prefix+'/extras_json',data=[json.dumps({'tcp_motion_interval':v,
                'teacher':{'physics':physics,'phase':'transfer','tcp_limits':{'linear_speed_m_s':999}}}) for v in intervals])
    meta={'fps':25,'physics_model':{'tcp_motion':{'schema_version':1,'integration_steps':steps,
            'integration_dt':dt,'tcp_in_link6_m':[0.,0.,.1375]}}}
    return meta


def test_independent_tcp_gate_rejects_between_frame_peak_and_ignores_producer_limits(tmp_path):
    path=tmp_path/'episode.h5';meta=write_evidence(path)
    policy={k:LIMITS[k] for k in ('linear_speed_m_s','angular_speed_rad_s')}
    with h5py.File(path) as f:
        report=audit_tcp_motion(f,meta,policy)
    assert not report['passed'] and report['violations']==['left/linear_speed_limit','right/linear_speed_limit']
    assert report['peaks']['left/linear']['substep']==2
    assert report['integration_steps']==8
    assert report['peaks']['left/linear']['sim_time']==pytest.approx(.02)
    meta=write_evidence(path,.2)
    with h5py.File(path,'r+') as f:
        assert audit_tcp_motion(f,meta,policy)['passed']
        row=json.loads(f['next_observation/extras_json'][0]);row['tcp_motion_interval']['samples']=4
        f['next_observation/extras_json'][0]=json.dumps(row)
        with pytest.raises(ValueError,match='samples or clocks'):audit_tcp_motion(f,meta,policy)


def test_strict_policy_keeps_every_original_threshold():
    from piper_sim.collection_audit import load_acceptance_policy
    original,_=load_acceptance_policy();strict,_=load_acceptance_policy('configs/collection/quality_policy_tcp.yaml')
    assert strict.pop('tcp_motion')=={k:LIMITS[k] for k in ('linear_speed_m_s','angular_speed_rad_s')}
    assert strict==original


def test_journal_resume_retains_operator_policy_and_does_not_trust_result_path(tmp_path, monkeypatch):
    from piper_sim import collection_jobs as jobs
    seen=[]
    monkeypatch.setattr(jobs,'validate_completed',lambda result,**kwargs:seen.append(kwargs))
    identity={'options':{'acceptance_policy':'/operator/strict.yaml'}}
    with jobs.BatchJournal(tmp_path/'batch',identity) as journal:
        journal.save(0,{'accepted':False,'diagnostic_passed':False,'acceptance_policy':'/untrusted/loose.yaml'})
        journal.completed(0)
    assert seen==[{'policy':'/operator/strict.yaml'}]


def test_strict_audit_cannot_accept_missing_native_evidence(tmp_path):
    path=tmp_path/'episode.h5';meta=write_evidence(path,.2)
    policy={k:LIMITS[k] for k in ('linear_speed_m_s','angular_speed_rad_s')}
    with h5py.File(path,'r+') as f:
        row=json.loads(f['next_observation/extras_json'][0]);del row['tcp_motion_interval']
        f['next_observation/extras_json'][0]=json.dumps(row)
        with pytest.raises(KeyError):audit_tcp_motion(f,meta,policy)
