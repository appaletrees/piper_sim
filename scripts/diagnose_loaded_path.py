#!/usr/bin/env python3
"""Isolated cup-stack trial of bounded loaded-path interpolation.

The production source is never edited. This process tries a convex spline only
when a natural spline overshoots valid IK knots while carrying a cup; production
joint limits, actual-curve collision checks and physical grasp checks still run.
No HDF5, RGB or training dataset is created. This is not a full-task acceptance.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.contracts import LOWER, UPPER
from piper_sim.experts import clean_table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Retain existing trials; choose a new output')
    original=clean_table.CleanTableTeacher.plan_pose_path
    natural=clean_table.path_trajectory
    trials=[]
    def plan(self,*a,**kw):
        pending=[]
        def bounded(points,dt,**options):
            trajectory=natural(points,dt,**options)
            knots=np.asarray(points)
            if (self.carried is not None and
                    (np.any(trajectory<LOWER) or np.any(trajectory>UPPER))
                    and np.all(knots>=LOWER) and np.all(knots<=UPPER)):
                candidate=natural(points,dt,**{**options,'interpolation':'convex'})
                pending.append({'phase':self.phase,'object':self.current,'sim_time':self.b.sim_time,
                    'carried_objects':self.carried_objects(),'ik_knots':len(knots),
                    'natural_samples':len(trajectory),'candidate_samples':len(candidate),
                    'natural_limit_excess':float(np.maximum(LOWER-trajectory,trajectory-UPPER).max()),
                    'candidate_inside_joint_bounds':bool(np.all(candidate>=LOWER) and np.all(candidate<=UPPER)),
                    'natural_path':trajectory.tolist(),'candidate_path':candidate.tolist(),
                    'knots':knots.tolist(),'derivative_limits':self.motion_limits(),
                    'training_ready':False})
                return candidate
            return trajectory
        clean_table.path_trajectory=bounded
        try:
            result=original(self,*a,**kw)
            for row in pending:row['production_geometric_path_check_passed']=True
            return result
        except Exception as error:
            for row in pending:row.update(production_geometric_path_check_passed=False,error=str(error))
            raise
        finally:
            clean_table.path_trajectory=natural
            trials.extend(pending)
    clean_table.CleanTableTeacher.plan_pose_path=plan
    spec=importlib.util.spec_from_file_location('diagnostic_demo',ROOT/'scripts/demo_clean_table.py')
    demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    sys.argv=[str(spec.origin),'--config','configs/clean_table_servo.yaml','--seed',str(args.seed),
        '--domain-randomization','configs/domain_randomization/clean_table.yaml',
        '--objects','cup_stack','--output',str(args.output)]
    try:
        code=demo.main()
    finally:
        clean_table.CleanTableTeacher.plan_pose_path=original
        clean_table.path_trajectory=natural
        atomic_json(args.output/'candidate_diagnostic.json',{'scope':__doc__,'seed':args.seed,
            'training_ready':False,'implementation_sha256':sha256(__file__),
            'production_planner_sha256':sha256(clean_table.__file__),
            'trials':trials,'limitations':[
                'Interior IK knots are approximated, not interpolated exactly.',
                'Sampled geometric checks do not prove continuous collision freedom.',
                'No independent closed-file audit, no RGB, no hardware validity, no full-task success claim.',
                'Only this diagnostic process changes path interpolation; production defaults remain unchanged.']})
    return code


if __name__=='__main__':raise SystemExit(main())
