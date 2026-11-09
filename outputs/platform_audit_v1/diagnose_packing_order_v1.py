"""Unrecorded full physical packing-order trial; no dataset acceptance.

Only the order of existing sort_one calls changes. Existing physical grasp,
motion, geometry and final-stability gates are retained. No object pose writes,
welds, material edits, timeout extensions or acceptance-policy changes here.
"""
import argparse
import json
from pathlib import Path
import time

import yaml

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT, load
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.simulation import SimulationSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=15619)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Preserve previous diagnostics')
    config=ROOT/'configs/clean_table_servo.yaml'
    task=ROOT/'configs/tasks/clean_table_rigid.yaml'
    profile=ROOT/'configs/collection/clean_table_collision_support.yaml'
    domain=ROOT/'configs/domain_randomization/clean_table.yaml'
    cfg=load(config);cfg['task_scene']=yaml.safe_load(task.read_text());cfg['task_scene']['domain_randomization']=str(domain)
    settings=yaml.safe_load(profile.read_text())
    order=['chopstick_a','chopstick_b','pen','charger','spoon','bowl','plate','side_dish']
    sources=[*ROOT.glob('src/piper_sim/**/*.py'),config,task,profile,domain]
    result={'scope':__doc__,'training_ready':False,'recorded':False,'seed':args.seed,'order_after_cup_stack':order,
        'script_sha256':sha256(__file__),'source_sha256':{str(f.relative_to(ROOT)):sha256(f) for f in sources if 'rl' not in f.parts},
        'acceptance_policy_sha256':sha256(ROOT/'configs/collection/quality_policy.yaml'),'error':None}
    backend=teacher=None;started=time.monotonic()
    try:
        backend=MujocoBackend(cfg,cameras=False)
        session=SimulationSession(backend);session.reset(args.seed,True)
        teacher=RigidTableTeacher(session,settings=settings)
        teacher.prepare();teacher.stack_cups()
        for name in order:
            if name in backend.task.active_names:teacher.sort_one(name)
        teacher.wait_for_final_stability()
        summary=session.read().extras['task']['metrics']['summary']
        if not summary['success']:raise SortFailure('Full packing trial final containment failed')
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error)
    finally:
        if teacher:
            result.update(quality=teacher.quality.report(),events=teacher.events,grasps=teacher.grasps,
                sim_seconds=backend.sim_time,final_metrics=backend.read().extras['task']['metrics'],
                final_objects=backend.read().extras['objects'])
        if backend:backend.close()
        result['wall_seconds']=time.monotonic()-started
        atomic_json(args.output,result)
    print(json.dumps({'error':result['error'],'summary':result.get('final_metrics',{}).get('summary'),
        'online_quality_passed':result.get('quality',{}).get('passed'),'training_ready':False}))
    return 2 if result['error'] else 0


if __name__=='__main__':raise SystemExit(main())
