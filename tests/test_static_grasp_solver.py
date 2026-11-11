"""Analytic static force and cantilever torque checks (system Python/SciPy)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
pytest.importorskip('scipy')
from static_grasp_solver import solve


def case(*, offset=0., rolling=0.):
    contacts=[]
    for sign,x in ((1.,offset-.002),(-1.,offset+.002)):
        contacts.append({'position_world_m':[x,0.,0.],
            'basis_on_object_world':np.diag([sign,sign,1.]).tolist(),
            'friction':[1.,1.,0.,rolling,rolling],
            'motor_projection_local':[-.5,0.,0.,0.,0.,0.]})
    return {'mass_kg':.225,'gravity_world_m_s2':[0.,0.,-9.81],
            'com_world_m':[0.,0.,0.],'contacts':contacts}


def test_opposed_center_grasp_matches_weight_over_twice_friction():
    result=solve(case())
    assert result['static_elliptic_solution']
    assert result['optimistic_motor_lower_bound_N']==pytest.approx(.225*9.81/2,abs=1e-7)
    assert result['force_balance_abs_max_N']<1e-7
    assert result['torque_balance_abs_max_Nm']<1e-7


def test_cantilever_needs_more_motor_force_even_with_friction_torques():
    result=solve(case(offset=.1,rolling=.002))
    assert result['static_elliptic_solution']
    assert result['optimistic_motor_lower_bound_N']>20.
    assert result['force_balance_abs_max_N']<1e-7
    assert result['torque_balance_abs_max_Nm']<1e-7
    history=[r['lower_bound_motor_abs_N'] for r in result['history']]
    assert min(np.diff(history),default=0.)>=-1e-6
    for contact in result['contact_wrenches_local_N_Nm']:
        f=np.asarray(contact)
        normalized=np.r_[f[1:3],f[4:]/.002]
        assert np.linalg.norm(normalized)<=f[0]+1e-6


def test_no_finger_contacts_cannot_hold_a_free_payload():
    c=case();c['contacts']=[]
    result=solve(c)
    assert result['status']=='outer_problem_infeasible'
    assert not result['static_elliptic_solution']
