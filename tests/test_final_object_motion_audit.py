import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('object_motion_audit',
    Path(__file__).resolve().parents[1]/'scripts/audit_final_object_motion.py')
audit = importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def recording(path, *, speed=0., angular=0., contained=True, sign_flips=False):
    times = np.arange(61)/30
    rows = []
    for i,t in enumerate(times):
        quaternion = np.array([np.cos(t*angular/2),np.sin(t*angular/2),0.,0.])
        if sign_flips and i%2: quaternion *= -1
        rows.append(json.dumps({'objects':{'stick':{'position':[t*speed,0.,0.],
            'quaternion_wxyz':quaternion.tolist()}},
            'task':{'metrics':{'summary':{'success':contained,'required_count':1}}}}))
    with h5py.File(path,'w') as f:
        f.attrs['complete'] = True
        f.attrs['metadata'] = json.dumps({'task_scene':{'config':{'success':{'required':['stick']}}}})
        f['observation/sim_time'] = times[:-1];f['next_observation/sim_time'] = times[1:]
        f.create_dataset('observation/extras_json',data=rows[:-1],dtype=h5py.string_dtype())
        f.create_dataset('next_observation/extras_json',data=rows[1:],dtype=h5py.string_dtype())


def test_quaternion_sign_flips_are_not_spurious_motion(tmp_path):
    path=tmp_path/'episode.h5';recording(path,sign_flips=True)
    result=audit.analyze(path,2.)
    assert result['ending_all_contained_stable_s'] == pytest.approx(2.)
    assert result['objects']['stick']['angular_speed_max_rad_s'] == 0.
    assert not result['training_ready']


def test_known_axial_rotation_and_translation_exceed_independent_policy(tmp_path):
    path=tmp_path/'episode.h5';recording(path,speed=.02,angular=.2)
    result=audit.analyze(path,2.);row=result['objects']['stick']
    assert row['linear_speed_max_m_s'] == pytest.approx(.02)
    assert row['angular_speed_max_rad_s'] == pytest.approx(.2,abs=1e-9)
    assert row['local_x_angular_component_max_rad_s'] == pytest.approx(.2,abs=1e-9)
    assert result['ending_all_contained_stable_s'] == 0.
    assert not result['training_ready']


def test_static_but_uncontained_objects_do_not_satisfy_stability(tmp_path):
    path=tmp_path/'episode.h5';recording(path,contained=False)
    assert audit.analyze(path,2.)['ending_all_contained_stable_s'] == 0.


def test_incomplete_source_rejected(tmp_path):
    path=tmp_path/'episode.h5';recording(path)
    with h5py.File(path,'r+') as f: f.attrs['complete'] = False
    with pytest.raises(ValueError,match='closed'): audit.analyze(path,2.)
