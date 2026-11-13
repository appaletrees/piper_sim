#!/usr/bin/env python3
"""Alternating same-state frame timings; excluded from training/episode quotas.

Includes physics, three RGB cameras, unchanged sensor transforms and gzip HDF5
appends. Not a full-task or successful-episodes/hour benchmark. Models/lighting,
camera optics, random parameters and physics settings are unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=30)
    p.add_argument('--modes',nargs='+',choices=('serial','pool3','pipeline3'),
        default=['serial','pool3','pipeline3','pipeline3','pool3','serial'])
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh report path')
    if args.frames<1:p.error('Need at least one frame')
    from piper_sim.config import load
    from piper_sim import randomization
    from piper_sim.simulation import SimulationSession
    from piper_sim.experts.rigid_table import RigidTableTeacher
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
    sensor=module('piper_sim.pipeline_probe_sensor',args.candidate/'sensor_candidate.py')
    backend=module('piper_sim.backends.pipeline_probe_backend',args.candidate/'backend_candidate.py')
    backend.SensorImageProcessor=sensor.SensorImageProcessor
    run=args.run.resolve();spec=json.loads((run/'run.json').read_text());options=spec['options'];snap=run/'inputs/project'
    cfg=load(snap/options['config']);cfg['task_scene']=yaml.safe_load((snap/options['task_scene']).read_text())
    cfg['task_scene']['domain_randomization']=str(snap/options['domain_randomization'])
    settings=yaml.safe_load((snap/options['collection_profile']).read_text())
    fast=Path(json.loads((run/'status.json').read_text())['storage']['fast_root'])
    rows=[];result={'scope':__doc__,'training_ready':False,'rows':rows,'status':'running'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    b=None;session=None
    try:
        b=backend.MujocoBackend(cfg,cameras=True)
        session=SimulationSession(b);teacher=RigidTableTeacher(session,settings=settings,grasp_mode='physical')
        raw={};render=b._render
        def capture(name,*size):
            value=render(name,*size);raw[name]=value;return value
        b._render=capture
        with tempfile.TemporaryDirectory(prefix='pipeline_probe_',dir=fast.parent) as temp:
            session.record_dir=Path(temp)
            for index,mode in enumerate(args.modes):
                b.sensor_processor.close()
                b.sensor_processor=sensor.SensorImageProcessor(1 if mode=='serial' else 3)
                b.sensor_capture_pipeline=mode=='pipeline3'
                session.reset(12345,True)
                for _ in range(3):session.step()
                session.start_recording('probe_'+str(index),'diagnostic only',image_compression='gzip',
                    image_storage='shared',flush_every=1,extras_storage='compressed')
                timings=[];mismatches=0;cpu=time.process_time()
                for frame in range(args.frames):
                    command=np.array(cfg['robot']['home']*2)
                    command[5]=command[12]=.02*np.sin(frame/10)
                    started=time.perf_counter();obs=session.step(command);timings.append(time.perf_counter()-started)
                    if frame in (0,args.frames//2,args.frames-1):
                        tick=round(b.sim_time/(b.dt*b.decimation))
                        expected=randomization.sensor_images(raw,b.task.domain_parameters,tick)
                        mismatches+=sum(np.count_nonzero(obs.images[n]!=expected[n]) for n in expected)
                cpu=time.process_time()-cpu
                row={'mode':mode,'frames':args.frames,'wall_s':sum(timings),'fps':args.frames/sum(timings),
                    'median_frame_s':float(np.median(timings)),'cpu_s_including_reference_checks':cpu,
                    'different_channel_values':int(mismatches),'seconds':timings,
                    'physics_substeps':b.decimation*b.substeps,'shadow_size':b.model.vis.quality.shadowsize,
                    'msaa_samples':b.model.vis.quality.offsamples}
                session.writer.close(complete=False);session.writer=None
                rows.append(row);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
                if mismatches:raise ValueError('Sensor pixels differ from frozen production arithmetic')
        result['status']='completed'
        result['median_frame_s']={mode:float(np.median([t for r in rows if r['mode']==mode for t in r['seconds']]))
            for mode in dict.fromkeys(args.modes)}
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        if session:session.close(complete=False)
        elif b:b.close()
        args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
