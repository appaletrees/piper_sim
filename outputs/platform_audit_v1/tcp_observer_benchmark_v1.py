"""Paired native observer cost and exact dynamics check; no collection claims."""
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import ROOT, load
from piper_sim.contracts import Command
from piper_sim.collection_jobs import atomic_json, sha256

out=ROOT/'outputs/platform_audit_v1/tcp_observer_benchmark_v1.json'
if out.exists():raise FileExistsError(out)
cfg=load('configs/clean_table_servo.yaml')
cfg['task_scene']=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
results=[]
for repeat in range(3):
    for enabled in ((False,True) if repeat%2==0 else (True,False)):
        config=copy.deepcopy(cfg);config['mujoco']['tcp_motion_evidence']=enabled
        b=MujocoBackend(config,cameras=False)
        try:
            # Nominal reset and a fixed mild controller trajectory; not a task.
            for _ in range(15):b.step()
            digest=hashlib.sha256();started=time.perf_counter()
            for i in range(150):
                q=np.array(cfg['robot']['home']*2);q[0]=q[7]=.015*np.sin(i/30)
                b.apply(Command(q,0,i));b.step()
                for field in ('qpos','qvel','qacc','qacc_warmstart','qfrc_applied','ctrl'):
                    digest.update(getattr(b.data,field).tobytes())
            results.append({'observer':enabled,'repeat':repeat,'step_seconds':time.perf_counter()-started,
                            'transitions':150,'integration_steps':150*b.substeps*b.decimation,
                            'dynamics_sha256':digest.hexdigest()})
        finally:b.close()
assert len({r['dynamics_sha256'] for r in results})==1
medians={str(mode):statistics.median(r['step_seconds'] for r in results if r['observer']==mode) for mode in (False,True)}
report={'results':results,'medians_seconds':medians,'observer_step_time_ratio':medians['True']/medians['False'],
 'all_dynamics_hashes_identical':True,'script_sha256':sha256(__file__),
 'observer_sha256':sha256(ROOT/'src/piper_sim/mujoco_tcp.py'),
 'backend_sha256':sha256(ROOT/'src/piper_sim/backends/mujoco.py'),
 'mujoco_version':mujoco.__version__,'training_ready':False,
 'scope':'Fixed 150-control-step low-motion window, full task geometry, no RGB, teacher, recording, audit or export; cost includes state hashing, excludes initialization; not qualified-throughput evidence.'}
atomic_json(out,report);print(json.dumps(medians),report['observer_step_time_ratio'])
