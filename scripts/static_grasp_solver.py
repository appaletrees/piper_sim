#!/usr/bin/env python3
"""Frozen-contact static wrench feasibility, not a physical grasp certificate.

Uses successive outer polyhedral approximations of MuJoCo's elliptic friction
cones. Every LP objective is an optimistic lower bound on required tendon force.
Convergence additionally supplies a numerically checked elliptic-cone solution.
Contact compliance, collisions, dynamic stability and control are not certified.
Run with system Python (SciPy); this is not a simulation runtime dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog


def solve(case, *, max_iterations=120, tolerance=2e-7):
    contacts=case['contacts'];n=len(contacts);dim=6*n+2;motor,limit=dim-2,dim-1
    mass=float(case['mass_kg']);gravity=np.asarray(case['gravity_world_m_s2'],float)
    com=np.asarray(case['com_world_m'],float);bias=float(case.get('motor_bias_residual_N',0.))
    if mass<=0 or gravity.shape!=(3,) or com.shape!=(3,) or not np.isfinite(np.r_[mass,gravity,com,bias]).all():
        raise ValueError('Finite positive mass, gravity, COM and bias required')
    eq=np.zeros((7,dim));rhs=np.r_[-mass*gravity,np.zeros(3),-bias]
    eq[6,motor]=-1.;objective=np.zeros(dim);objective[limit]=1.
    bounds=[];inequalities=[];matrices=[]
    for i,c in enumerate(contacts):
        basis=np.asarray(c['basis_on_object_world'],float)
        pos=np.asarray(c['position_world_m'],float)
        mu=np.asarray(c['friction'],float)
        projection=np.asarray(c['motor_projection_local'],float)
        if (basis.shape!=(3,3) or pos.shape!=(3,) or mu.shape!=(5,) or projection.shape!=(6,)
                or not np.isfinite(np.r_[basis.ravel(),pos,mu,projection]).all() or np.any(mu<0)
                or not np.allclose(basis.T@basis,np.eye(3),atol=1e-8)):
            raise ValueError('Invalid orthonormal basis, position, friction or tendon projection')
        scales=np.r_[1.,mu];b=np.zeros((6,6))
        b[:3,:3]=basis;b[3:,3:]=basis
        r=pos-com;k=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
        b[3:,:3]=k@basis;b*=scales[None,:];matrices.append(b)
        start=6*i;eq[:6,start:start+6]=b;eq[6,start:start+6]=projection*scales
        bounds += [(0.,None)]+[(None,None) if s else (0.,0.) for s in mu]
        for j in range(1,6):
            for sign in (-1.,1.):
                row=np.zeros(dim);row[start]=-1.;row[start+j]=sign;inequalities.append(row)
    bounds += [(None,None),(0.,None)]
    for sign in (-1.,1.):
        row=np.zeros(dim);row[motor]=sign;row[limit]=-1.;inequalities.append(row)
    # Numerical row scaling changes neither the feasible set nor the objective.
    row_scale=np.r_[np.ones(3),np.full(3,10.),1.]
    history=[];solution=None
    for iteration in range(max_iterations):
        result=linprog(objective,A_ub=np.array(inequalities),b_ub=np.zeros(len(inequalities)),
            A_eq=eq*row_scale[:,None],b_eq=rhs*row_scale,bounds=bounds,method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not result.success:
            return {'status':'outer_problem_infeasible' if result.status==2 else 'solver_inconclusive',
                'solver_status':int(result.status),'message':result.message,'history':history,
                'static_elliptic_solution':False,'training_ready':False}
        x=result.x;violations=[]
        for i in range(n):
            w=x[6*i:6*i+6];norm=float(np.linalg.norm(w[1:]));violation=norm-w[0]
            violations.append(violation)
            if violation>tolerance:
                row=np.zeros(dim);row[6*i]=-1.;row[6*i+1:6*i+6]=w[1:]/norm
                inequalities.append(row)
        worst=max(violations,default=0.)
        history.append({'iteration':iteration,'lower_bound_motor_abs_N':float(result.fun),
                        'max_cone_violation_N':float(worst)})
        if worst<=tolerance:
            solution=x;break
    if solution is None:
        return {'status':'iteration_limit_inconclusive','history':history,
                'optimistic_motor_lower_bound_N':float(result.fun),'static_elliptic_solution':False,'training_ready':False}
    locals_=[(solution[6*i:6*i+6]*np.r_[1.,c['friction']]).tolist() for i,c in enumerate(contacts)]
    residual=eq@solution-rhs
    return {'status':'converged_frozen_contact_static_solution','history':history,
        'optimistic_motor_lower_bound_N':float(result.fun),'motor_force_N':float(solution[motor]),
        'static_elliptic_solution':True,'force_balance_abs_max_N':float(abs(residual[:3]).max()),
        'torque_balance_abs_max_Nm':float(abs(residual[3:6]).max()),
        'tendon_balance_abs_error_N':float(abs(residual[6])),
        'contact_wrenches_local_N_Nm':locals_,
        'normal_sum_N':float(sum(v[0] for v in locals_)),
        'scope':__doc__,'training_ready':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    raw=args.input.read_bytes();document=json.loads(raw);results=[]
    for case in document['cases']:
        result=solve(case)
        results.append({'label':case['label'],'time_before':case.get('time_before'),
            'case_hypothesis':case.get('hypothesis','Frozen original native contact geometry; static query only.'),
            'recorded_motor_force_N':case.get('recorded_motor_force_N'),
            'recorded_normal_sum_N':case.get('recorded_normal_sum_N'),**result})
    out={'scope':__doc__,'training_ready':False,'input_sha256':hashlib.sha256(raw).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scipy_version':scipy.__version__,'numpy_version':np.__version__,'results':results}
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k in ('label','time_before','status','optimistic_motor_lower_bound_N','recorded_motor_force_N','recorded_normal_sum_N')} for r in results],indent=2))


if __name__=='__main__':main()
