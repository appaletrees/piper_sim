"""One rendering implementation per process, unchanged recorded poses and visual settings."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from concurrent.futures import wait
import h5py
import mujoco
import numpy as np
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.profiling import PipelineProfiler
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_render_workers import CameraOwner

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['baseline','fit_buffer','workers'],required=True)
p.add_argument('--repeats',type=int,default=20)
args=p.parse_args()
assert not args.output.exists() and args.repeats>0
args.output.mkdir(parents=True)
report=dict(status='running',training_ready=False,scope=__doc__,mode=args.mode,script_sha256=sha256(__file__))
b=profiler=None; owners={}; samples=[]; snapshots={}; differences=[]
try:
    before=sha256(args.source)
    with h5py.File(args.source,'r') as f:
        meta=json.loads(f.attrs['metadata']); cfg=copy.deepcopy(meta['config'])
        cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
        b=MujocoBackend(cfg,cameras=False)
        layout=meta['task_scene']['layout']; b.reset(layout['seed'],layout['randomized'])
        names=list(b.rig['cameras']); assert len(names)==3
        assert all(v['rgb']['resolution']==[640,480] for v in b.rig['cameras'].values())
        profiler=PipelineProfiler(sample_interval_s=1.)
        if args.mode=='workers':
            for name in names: owners[name]=CameraOwner(b.model,name,640,480)
            context=next(iter(owners.values())).renderer._mjr_context
        else:
            b._render(names[0],640,480)
            renderer=b.renderers[(640,480)];context=renderer._mjr_context
            if args.mode=='fit_buffer': mujoco.mjr_resizeOffscreen(640,480,context)
        report.update(framebuffer=[context.offWidth,context.offHeight],msaa_samples=context.offSamples,shadow_size=context.shadowSize)
        def render():
            if args.mode!='workers':return {name:b._render(name,640,480) for name in names}
            pending={}
            try:
                for name,owner in owners.items():pending[name]=owner.submit(b.data)
                return {name:future.result()[0] for name,future in pending.items()}
            finally:wait(pending.values())
        frames=[0,2450,4385,8770]
        for frame in frames:
            extra=json.loads(f['observation/extras_json'][frame])
            for side,value in extra['teacher']['physics']['arms'].items():
                b.data.qpos[b.indices[side]['qpos']]=value['joint_position']
            for name,pose in extra['objects'].items():
                adr=b.model.joint('task/'+name+'/free').qposadr[0]
                b.data.qpos[adr:adr+7]=pose['position']+pose['quaternion_wxyz']
            mujoco.mj_forward(b.model,b.data)
            kind=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(b.model,kind))
            mujoco.mj_getState(b.model,b.data,state,kind)
            for _ in range(5):render()
            reference=render()
            for name,pixels in reference.items():snapshots[f'frame_{frame}_{name}']=pixels
            for repeat in range(args.repeats):
                start=time.perf_counter();pixels=render();elapsed=time.perf_counter()-start
                samples.append(dict(frame=frame,repeat=repeat,seconds=elapsed))
                differences.append(dict(frame=frame,repeat=repeat,cameras={name:dict(
                    different_channel_values=int(np.count_nonzero(pixels[name]!=reference[name])),
                    max_abs=int(np.abs(pixels[name].astype(np.int16)-reference[name]).max())) for name in names}))
            after=np.empty_like(state);mujoco.mj_getState(b.model,b.data,after,kind);np.testing.assert_array_equal(state,after)
        assert sha256(args.source)==before
        for owner in owners.values():owner.close()
        owners.clear()
        prof=profiler.close();profiler=None
        np.savez_compressed(args.output/'pixels.npz',**snapshots)
        report.update(status='isolated_render_probe_passed',source=str(args.source),source_sha256=before,source_unchanged=True,
            integration_state_unchanged=True,frames=frames,repeats=args.repeats,samples=samples,repeat_differences=differences,
            median_seconds=float(np.median([row['seconds'] for row in samples])),
            per_pose_median_seconds={str(frame):float(np.median([row['seconds'] for row in samples if row['frame']==frame])) for frame in frames},
            pixels_path=str(args.output/'pixels.npz'),pixels_sha256=sha256(args.output/'pixels.npz'),resources=prof['resources'],
            mean_cpu_percent_one_core=prof['mean_cpu_percent_one_core'],worker_threads_joined=True,
            mujoco_version=mujoco.__version__,input_hashes={str(path):sha256(path) for path in [ROOT/'scripts/benchmark_render_workers.py',
                ROOT/'src/piper_sim/backends/mujoco.py',ROOT/'src/piper_sim/mujoco_asset.py']},
            limitations=['Warmed render component at four recorded poses; no full-task or physics replay claim.',
                'One implementation per process avoids benchmark-only active context switching.',
                'Worker variant includes data snapshot copying and camera ownership checks.',
                'Source sensor postprocessing and episode writing are excluded.'])
except BaseException as error:
    report.update(status='failed_probe',error=repr(error));raise
finally:
    for owner in owners.values():owner.close()
    if profiler:profiler.close()
    if b:b.close()
    atomic_json(args.output/'report.json',report)
print(json.dumps({k:report[k] for k in ['status','mode','median_seconds']},indent=2))
