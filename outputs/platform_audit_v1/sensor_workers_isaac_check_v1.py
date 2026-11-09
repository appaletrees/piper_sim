"""Actual Isaac camera-worker equivalence/reset/capture/lifecycle diagnostic.

Not a manipulation task, trajectory-quality grant or collection benchmark.
Only run after the full MuJoCo profile finishes to avoid GPU interference.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if not args.headless or not args.enable_cameras: p.error('Require --headless --enable_cameras')
if args.output.exists(): p.error('Choose a new output directory')
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith(' kB')}
if mem['MemAvailable'] < 12*1024**3: p.error('Need 12 GiB available RAM before starting Kit')
args.output.mkdir(parents=True)
base=Path(__file__).resolve().parent
report=dict(status='running',training_ready=False,scope=__doc__,checks=[],resets=[],mem_available_before=mem['MemAvailable'])
started=time.perf_counter(); app=b=None
try:
    app=AppLauncher(args).app
    import numpy as np
    import yaml
    from piper_sim.backends.isaac import IsaacBackend
    from piper_sim.collection_jobs import atomic_json,sha256
    from piper_sim.config import ROOT,load
    from piper_sim.contracts import Command
    ref=base/'sensor_workers_reference_v1.py'
    spec=importlib.util.spec_from_file_location('piper_sim.randomization._serial_reference',ref)
    serial=importlib.util.module_from_spec(spec); spec.loader.exec_module(serial)
    cfg=load(ROOT/'configs/clean_table_workcell.yaml')
    cfg['runtime']['sensor_postprocess_workers']=3
    cfg['task_scene']=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    cfg['task_scene']['domain_randomization']='configs/domain_randomization/clean_table.yaml'
    original_urdf=ROOT/cfg['robot']['urdf']; isolated=args.output/'asset_cache'/original_urdf.name
    isolated.parent.mkdir(); shutil.copy2(original_urdf,isolated)
    if (original_urdf.parent/'usd').exists(): shutil.copytree(original_urdf.parent/'usd',isolated.parent/'usd')
    cfg['robot']['urdf']=str(isolated.resolve())
    report.update(config=cfg,script_sha256=sha256(__file__),reference_sha256=sha256(ref),
        inputs={str(ROOT/file):sha256(ROOT/file) for file in ('src/piper_sim/backends/isaac.py',
                'src/piper_sim/randomization/__init__.py','src/piper_sim/sensor_processing.py')})
    b=IsaacBackend(cfg,args.device,cameras=True)
    processor=b.sensor_processor
    class CheckedProcessor:
        def __call__(self,images,parameters,tick):
            before={name:hashlib.sha256(pixels.tobytes()).hexdigest() for name,pixels in images.items()}
            actual=processor(images,parameters,tick)
            expected=serial.sensor_images(images,parameters,tick)
            item=dict(seed=parameters['seed'],tick=tick,cameras={})
            for name in images:
                assert actual[name].shape == images[name].shape == (480,640,3)
                np.testing.assert_array_equal(actual[name],expected[name])
                assert hashlib.sha256(images[name].tobytes()).hexdigest()==before[name]
                item['cameras'][name]=dict(input_sha256=before[name],output_sha256=hashlib.sha256(actual[name].tobytes()).hexdigest())
            report['checks'].append(item)
            return actual
        def close(self): processor.close()
    b.sensor_processor=CheckedProcessor()
    for seed in (1340830695,97):
        b.reset(seed=seed,randomize=True)
        for frame in range(24):
            obs=b.read(); count=len(report['checks']); again=b.read()
            assert len(report['checks'])==count
            assert all(t==obs.sim_time for t in obs.image_timestamps.values())
            for name in obs.images: np.testing.assert_array_equal(obs.images[name],again.images[name])
            obs.images['overhead'][:]=0
            np.testing.assert_array_equal(b.read().images['overhead'],again.images['overhead'])
            assert len(report['checks'])==count
            target=np.array(cfg['robot']['home']*2); target[0]=.02*np.sin(frame*.1)
            b.apply(Command(target,time.monotonic(),frame)); b.step()
        b.read(); previous=b.camera_cache.key; count=len(report['checks'])
        for camera in b.cameras.values(): camera.update(b.dt*b.decimation,force_recompute=True)
        forced=b.read()
        assert b.camera_cache.key != previous and len(report['checks'])==count+1
        assert forced.sim_time==b.sim_time
        report['resets'].append(dict(seed=seed,frames=24,sim_end=b.sim_time,forced_capture_refreshed=True))
    pool=processor._pool
    b.close(); b.close(); b=None
    assert pool is not None and not any(thread.is_alive() for thread in pool._threads)
    report.update(status='passed_actual_isaac_sensor_equivalence',processed_triplets=len(report['checks']),
        different_channel_values=0,owned_workers_joined=True,
        limitations=['Short actual-engine equivalence and cache probe; no full Isaac sorting or throughput acceptance.',
                     'The serial reference is extra work; elapsed time is not candidate collection speed.',
                     'Nominal optical parameters remain uncalibrated against hardware.'])
except BaseException as error:
    report.update(status='failed_diagnostic',error=repr(error))
    raise
finally:
    if b: b.close()
    if app: app.close()
    report.update(wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
