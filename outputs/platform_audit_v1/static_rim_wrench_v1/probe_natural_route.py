#!/usr/bin/env python3
"""Offline curve proposal only; retains all collision and orientation gates."""
import json
from pathlib import Path
import sys
from piper_sim.collection_jobs import atomic_json,sha256
import probe_measured_rim_transport as base

original=base.search

def search(teacher,*args,**kwargs):
    plan=teacher.plan_pose_path
    calls=[]
    def natural(*a,**k):
        k.pop('bounded',None)
        result=plan(*a,**k)
        calls.append(len(result))
        return result
    teacher.plan_pose_path=natural
    try:
        result=original(teacher,*args,**kwargs)
        result['evidence']['curve_experiment']={'scope':__doc__,'script_sha256':sha256(__file__),
            'requested':'Natural interpolation; original plan_pose_path may still fall back to bounded when joint limits require it.',
            'all_geometry_and_orientation_gates_retained':True,'planned_path_lengths':calls}
        return result
    finally:teacher.plan_pose_path=plan

base.search=search
base.main()
