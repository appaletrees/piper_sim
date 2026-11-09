import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from piper_sim.collection_jobs import sha256

spec = importlib.util.spec_from_file_location('supported_audit',
    Path(__file__).resolve().parents[1]/'scripts/audit_supported_release.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('contact_gap', [False, True])
def test_native_hold_checks_every_step_and_excludes_robot_by_body_name(tmp_path, contact_gap):
    n, dt = 100, .001
    force = np.tile([.75, 0, 0, 0, 0, 0], (2*n, 1))
    force[1::2, 0] = 100.  # A robot force must never count as basket support.
    if contact_gap:
        force[2*19] = 0  # A native interruption between coarse observations.
    np.savez_compressed(tmp_path/'trace.npz', time_before=np.arange(n)*dt,
        time_after=np.arange(1,n+1)*dt, target=np.full(n,'plate'),
        contact_offset=2*np.arange(n), contact_count=np.full(n,2),
        contact_geom1=np.tile([0,2],n), contact_geom2=np.ones(2*n,dtype=int),
        contact_frame=np.tile([0,0,1,1,0,0,0,1,0],(2*n,1)), contact_force_local=force)
    metadata={'includes_task_contacts':True,'trace_sha256':sha256(tmp_path/'trace.npz'),
              'integration_dt':dt,'geom_names':['task/basket/floor','task/plate/mesh','gripper_right_46']}
    (tmp_path/'trace.json').write_text(json.dumps(metadata))
    result=audit.native_support(tmp_path,[{'phase':'support_verified','object':'plate','sim_time':.1,
        'support':{'up_axis_world':[0,0,1],'weight_N':1.}}],['task/basket','task/plate','right/link7'])[0]
    assert result['complete_window'] and result['native_samples']==n
    assert result['max_fraction']==pytest.approx(.75)
    assert result['min_fraction']==pytest.approx(0 if contact_gap else .75)
    assert result['all_native_samples_at_least_half_weight'] is not contact_gap
    assert sha256(tmp_path/'trace.npz')==metadata['trace_sha256']
