import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from piper_sim.collection_jobs import sha256

spec = importlib.util.spec_from_file_location('dynamics_diagnostic',
    Path(__file__).resolve().parents[1]/'scripts/audit_recorded_dynamics.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def recorded_source(tmp_path):
    path = tmp_path/'episode.h5'
    q = [0.1]*6+[.01, .01]
    v = [0.02]*6+[.003, .004]
    arm = {'joint_position': q, 'joint_velocity': v,
        'actuator_generalized_force': [1.]*8, 'applied_generalized_force': [2.]*8,
        'combined_drive_generalized_force': [3.]*8, 'constraint_generalized_force': [4.]*8,
        'tcp_position_world_m': [.1, .2, .3], 'tcp_linear_velocity_world_m_s': [.03, .04, 0.],
        'tcp_angular_velocity_world_rad_s': [0., 0., .1]}
    with h5py.File(path, 'w') as f:
        f.attrs['complete'] = True
        f.attrs['metadata'] = json.dumps({'backend': 'MujocoBackend', 'fps': 30})
        for prefix, offset in (('observation', 0), ('next_observation', 1)):
            times = np.arange(offset, offset+4)/30
            f.create_dataset(prefix+'/sim_time', data=times)
            f.create_dataset(prefix+'/state', data=np.tile([*q[:6], .02]*2, (4, 1)).astype(np.float32))
            extras = []
            for t in times:
                extras.append(json.dumps({'teacher': {'phase': 'diagnostic', 'object': 'pen', 'physics': {
                    'schema_version': 'piper.physics_evidence.v1', 'sim_time': float(t),
                    'arms': {side: copy.deepcopy(arm) for side in ('left', 'right')}}}}))
            f.create_dataset(prefix+'/extras_json', data=extras, dtype=h5py.string_dtype())
    return path


def mutate(path, change):
    with h5py.File(path, 'r+') as f:
        raw = json.loads(f['next_observation/extras_json'][1])
        change(raw['teacher']['physics'])
        f['next_observation/extras_json'][1] = json.dumps(raw)


def test_read_only_native_evidence_keeps_measurements_distinct(tmp_path):
    path = recorded_source(tmp_path)
    checksum, modified = sha256(path), path.stat().st_mtime_ns
    result = diagnostic.analyze(path)
    assert result['integrity_status'] == 'consistent_recorded_evidence'
    assert result['checked_snapshots'] == 8 and result['unique_state_samples'] == 5
    assert not result['training_ready'] and not result['violations']
    arm = result['arms']['right']
    assert arm['unique_state_samples'] == 5
    assert arm['peaks_all_observation_snapshots']['gripper_opening_velocity_m_s']['absolute_max'] == pytest.approx(.007)
    assert arm['peaks_all_observation_snapshots']['tcp_linear_speed_m_s']['absolute_max'] == pytest.approx(.05)
    assert arm['post_step_force_abs_max_per_joint']['combined_drive_generalized_force'] == [3.]*8
    assert arm['interval_position_difference_velocity_abs_max_per_channel'] == [0.]*7
    assert arm['interval_tcp_speed_peak']['m_s'] == 0.
    assert path.stat().st_mtime_ns == modified and sha256(path) == checksum


@pytest.mark.parametrize('change,reason', [
    (lambda p: p.update(sim_time=1.), 'clock mismatch'),
    (lambda p: p.update(sim_time=float('nan')), 'clock mismatch'),
    (lambda p: p.update(schema_version='unknown'), 'schema'),
    (lambda p: p['arms'].pop('right'), 'inventory'),
    (lambda p: p['arms']['left']['joint_position'].__setitem__(6, .03), 'joint state mismatch'),
    (lambda p: p['arms']['left'].update(joint_velocity=[float('nan')]*8), 'joint_velocity'),
    (lambda p: p['arms']['left'].update(joint_velocity=[0.]*7), 'joint_velocity'),
    (lambda p: p['arms']['left'].update(tcp_linear_velocity_world_m_s=[0., 0.]), 'tcp_linear_velocity'),
    (lambda p: p['arms']['left'].update(combined_drive_generalized_force=[1.]*8), 'drive identity'),
    (lambda p: p['arms']['left']['joint_velocity'].__setitem__(0, .3), 'Native transition discontinuity'),
])
def test_corrupt_native_evidence_cannot_report_consistency(tmp_path, change, reason):
    path = recorded_source(tmp_path)
    mutate(path, change)
    result = diagnostic.analyze(path)
    assert result['integrity_status'] == 'invalid'
    assert any(reason in v for v in result['violations'])
    assert not result['training_ready']


def test_feedforward_can_change_at_boundary_without_changing_state(tmp_path):
    path = recorded_source(tmp_path)
    with h5py.File(path, 'r+') as f:
        extra = json.loads(f['observation/extras_json'][2])
        arm = extra['teacher']['physics']['arms']['left']
        arm['applied_generalized_force'] = [4.]*8
        arm['combined_drive_generalized_force'] = [5.]*8
        f['observation/extras_json'][2] = json.dumps(extra)
    result = diagnostic.analyze(path)
    assert not result['violations']
    peaks = result['arms']['left']['peaks_all_observation_snapshots']
    assert peaks['combined_drive_generalized_force_arm']['absolute_max'] == 5.


@pytest.mark.parametrize('fault', ['partial', 'missing_native', 'time_gap', 'extra_frame', 'non_mujoco'])
def test_incomplete_or_incompatible_sources_are_not_supported(tmp_path, fault):
    path = recorded_source(tmp_path)
    with h5py.File(path, 'r+') as f:
        if fault == 'partial':
            f.attrs['complete'] = False
        elif fault == 'missing_native':
            f['observation/extras_json'][0] = json.dumps({'teacher': {}})
        elif fault == 'time_gap':
            f['observation/sim_time'][1] = 4.
        elif fault == 'extra_frame':
            del f['observation/extras_json']
            f.create_dataset('observation/extras_json', data=['{}']*5, dtype=h5py.string_dtype())
        else:
            f.attrs['metadata'] = json.dumps({'backend': 'IsaacBackend', 'fps': 30})
    result = diagnostic.analyze(path)
    assert result['integrity_status'] == 'invalid' and result['violations']
    assert not result['training_ready']


def controller_source(tmp_path):
    path=recorded_source(tmp_path)
    with h5py.File(path,'r+') as f:
        meta=json.loads(f.attrs['metadata'])
        meta.update(physics_model={'controller':{'schema_version':1,'mode':'monotone_cubic_velocity',
            'integration_steps':32,'integration_dt':1/960}},collection_initial_target=[0.]*14)
        f.attrs['metadata']=json.dumps(meta)
        f.create_dataset('action/position',data=np.zeros((4,14),np.float32))
        previous=None
        for i in range(4):
            record={'mode':'monotone_cubic_velocity','integration_steps':32,'integration_dt':1/960,
                'start_sim_time':i/30,'end_sim_time':(i+1)/30,'native_jerk_samples':32,
                'actuator_force_range':[[-10,10]]*14}
            for key in ('previous_command','command','position_reference_min','position_reference_max',
                        'velocity_reference_min','velocity_reference_max','last_velocity_reference',
                        'limiter_start_velocity_jump','native_velocity_min'):
                record[key]=[0.]*14
            record['native_velocity_max']=[.1]*14
            for key in ('native_acceleration_abs_max','native_jerk_abs_max','position_gain','velocity_gain'):
                record[key]=[1.]*14
            for prefix,value in [('observation',previous),('next_observation',record)]:
                extra=json.loads(f[prefix+'/extras_json'][i]);extra['controller_interval']=value
                f[prefix+'/extras_json'][i]=json.dumps(extra)
            previous=record
    return path


def test_controller_summary_checks_all_intervals_without_granting_training(tmp_path):
    path=controller_source(tmp_path);checksum=sha256(path)
    result=diagnostic.analyze(path)
    assert result['integrity_status']=='consistent_recorded_evidence' and not result['training_ready']
    report=result['controller_intervals']
    assert report['integration_steps']==128 and report['intervals']==4
    assert report['peaks']['native_velocity_abs_max'][6]['absolute_max']==.1
    assert sha256(path)==checksum


@pytest.mark.parametrize('change',[
    lambda x:x.update(end_sim_time=99),
    lambda x:x.update(integration_steps=31),
    lambda x:x['command'].__setitem__(0,.2),
    lambda x:x['previous_command'].__setitem__(0,.2),
    lambda x:x['position_reference_min'].__setitem__(6,-.1),
    lambda x:x['native_velocity_min'].__setitem__(0,.03),
    lambda x:x['native_acceleration_abs_max'].__setitem__(0,-1e-12),
    lambda x:x.update(native_jerk_samples=0),
    lambda x:x.update(actuator_force_range=[[1,-1]]*14),
    lambda x:x['last_velocity_reference'].__setitem__(0,float('nan')),
    lambda x:x.pop('command'),
])
def test_corrupt_controller_summary_is_rejected_even_after_native_snapshot_checks_pass(tmp_path,change):
    path=controller_source(tmp_path)
    with h5py.File(path,'r+') as f:
        extra=json.loads(f['next_observation/extras_json'][1]);change(extra['controller_interval'])
        f['next_observation/extras_json'][1]=json.dumps(extra)
    result=diagnostic.analyze(path)
    assert result['integrity_status']=='invalid' and result['violations'] and not result['training_ready']
