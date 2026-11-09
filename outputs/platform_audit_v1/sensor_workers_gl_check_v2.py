"""Compare actual backend RGB processing with the frozen serial implementation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import yaml

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT, load
from piper_sim.contracts import Command

base=Path(__file__).resolve().parent
out=base/'sensor_workers_gl_check_v2.json'; assert not out.exists()
reference=base/'sensor_workers_reference_v1.py'
spec=importlib.util.spec_from_file_location('piper_sim.randomization._serial_reference',reference)
serial=importlib.util.module_from_spec(spec); spec.loader.exec_module(serial)
cfg=load(ROOT/'configs/clean_table_tcp_sensor_workers.yaml')
cfg['task_scene']=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
cfg['task_scene']['domain_randomization']='configs/domain_randomization/clean_table.yaml'
b=None; checks=[]; results=[]
started=time.perf_counter()
try:
    b=MujocoBackend(cfg,cameras=True)
    processor=b.sensor_processor
    class CheckedProcessor:
        def __call__(self,images,parameters,tick):
            before={name:hashlib.sha256(pixels.tobytes()).hexdigest() for name,pixels in images.items()}
            actual=processor(images,parameters,tick)
            expected=serial.sensor_images(images,parameters,tick)
            record=dict(seed=parameters['seed'],tick=tick,cameras={})
            for name in images:
                assert images[name].shape == actual[name].shape == (480,640,3)
                np.testing.assert_array_equal(actual[name],expected[name])
                assert hashlib.sha256(images[name].tobytes()).hexdigest()==before[name]
                record['cameras'][name]=dict(input_sha256=before[name],output_sha256=hashlib.sha256(actual[name].tobytes()).hexdigest())
            checks.append(record)
            return actual
        def close(self): processor.close()
    b.sensor_processor=CheckedProcessor()
    for seed in (1340830695,97):
        b.reset(seed=seed,randomize=True)
        for frame in range(32):
            obs=b.read(); again=b.read()
            assert all(t == obs.sim_time for t in obs.image_timestamps.values())
            for name in obs.images: np.testing.assert_array_equal(obs.images[name],again.images[name])
            expected_calls=len(checks)
            # Returned arrays cannot corrupt the retained processed snapshot.
            obs.images['overhead'][:]=0
            np.testing.assert_array_equal(b.read().images['overhead'],again.images['overhead'])
            assert len(checks)==expected_calls
            command=np.array(cfg['robot']['home']*2); command[0]=.02*np.sin(frame*.1)
            b.apply(Command(command,time.monotonic(),frame)); b.step()
        results.append(dict(seed=seed,frames=32,sim_end=b.sim_time))
    assert len(checks)==64
    report=dict(status='passed_actual_render_equivalence',training_ready=False,scope=__doc__,
        checks=checks,resets=results,processed_triplets=len(checks),different_channel_values=0,
        reference_sha256=sha256(reference),script_sha256=sha256(__file__),
        inputs={str(ROOT/file):sha256(ROOT/file) for file in ('src/piper_sim/randomization/__init__.py',
          'src/piper_sim/sensor_processing.py','src/piper_sim/backends/mujoco.py','configs/clean_table_tcp_sensor_workers.yaml')},
        wall_seconds=time.perf_counter()-started,
        limitations=['Actual EGL frames and real backend read/cache/reset exercised; not a full task or training acceptance.',
                     'Serial comparison is additional work, so elapsed time is not an optimized collection benchmark.',
                     'No simulation steps occur between parallel and frozen-reference processing of the same raw frame.'])
    atomic_json(out,report)
    print(json.dumps({k:report[k] for k in ('status','processed_triplets','different_channel_values','wall_seconds')}))
finally:
    if b: b.close()
