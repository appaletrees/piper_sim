#!/usr/bin/env python3
"""Collect a resumable quota of independently accepted, randomized RGB episodes.

Each GPU runs one ordinary position-control teacher. Failed and interrupted
episodes are retained separately and never contribute to the success quota.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

from piper_sim.collection_jobs import atomic_json, atomic_text, sha256
from piper_sim.config import ROOT
from piper_sim.contracts import FEATURE_NAMES
from piper_sim.process_lifecycle import stop_process_groups, termination_guard
from piper_sim.training_acceptance import verify_training_acceptance


DEFAULTS = dict(target_success=2000, gpus=list(range(8)), seed_start=None, workers_per_gpu=1,
    timeout_s=7200., min_free_gib=512., warmup_successes=1,
    config='configs/clean_table_position_execution_rim25_damped.yaml',
    task_scene='configs/tasks/clean_table_rigid_bowl_clearance.yaml',
    domain_randomization='configs/domain_randomization/clean_table.yaml',
    collection_profile='configs/collection/clean_table_rim_damped.yaml',
    acceptance_policy='configs/collection/quality_policy_tcp.yaml')
INPUT_OPTIONS = ('config', 'task_scene', 'domain_randomization', 'collection_profile', 'acceptance_policy')


def versions():
    return {name: importlib.metadata.version(name) for name in ('mujoco', 'numpy', 'h5py', 'Pillow', 'PyYAML')}


def fingerprints(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def initialize(output, options):
    """Freeze the actual working tree, including its uncommitted source changes."""
    for name in INPUT_OPTIONS:
        path = (ROOT/options[name]).resolve()
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise ValueError('Input must be an existing project file: '+str(path))
        options[name] = path.relative_to(ROOT).as_posix()
    cfg = yaml.safe_load((ROOT/options['config']).read_text())
    if not math.isclose(1/(cfg['runtime']['physics_dt']*cfg['runtime']['decimation']), 30., abs_tol=1e-9):
        raise ValueError('This collector requires exactly 30 Hz simulation-time sampling')
    profile = yaml.safe_load((ROOT/options['collection_profile']).read_text())
    if profile['dataset'].get('image_storage') != 'shared':
        raise ValueError('Use self-contained shared-frame storage for this large collection')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'inputs/project'
    snapshot.mkdir(parents=True)
    for name in ('src', 'scripts', 'configs', 'assets', 'requirements', 'docs', 'examples'):
        if (ROOT/name).exists():
            shutil.copytree(ROOT/name, snapshot/name,
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc'))
    for name in ('pyproject.toml', 'README.md'):
        shutil.copy2(ROOT/name, snapshot/name)
    for name in ('success', 'failed', 'running'):
        (output/name).mkdir()
    git = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    atomic_json(output/'run.json', {'schema_version': 'piper.success_quota.v1',
        'created_unix': time.time(), 'source_project': str(ROOT), 'source_git_commit': git,
        'options': options, 'files': fingerprints(snapshot), 'versions': versions(),
        'contract': {'fps': 30., 'action': 'absolute_joint_position', 'features': list(FEATURE_NAMES),
            'units': ['rad']*6+['m']+['rad']*6+['m'], 'cameras': ['overhead','left_wrist','right_wrist'],
            'task': 'stack cups on the table, move the entire stack and all rigid utensils to the basket',
            'randomized': True, 'success_rule': 'complete RGB episode passes the unchanged independent policy',
            'random_seed_mode': 'system_entropy_unique' if options['seed_start'] is None else 'deterministic_sequence',
            'failed_rule': 'all other attempts retained, including quality rejection and incomplete recordings'}})


def quota_slots(target, successes, active):
    return max(0, target-successes-active)


def choose_seed(index, options, used):
    if options['seed_start'] is not None:
        value=options['seed_start']+index
        if value in used:raise ValueError('Duplicate deterministic seed')
        return value
    # Exact integers even in JSON clients using IEEE doubles; enough entropy
    # for a new independent episode, and retained in the claim for reproduction.
    while True:
        value=secrets.randbits(53)
        if value not in used:return value


def worker_environment(gpu, snapshot):
    return {**os.environ, 'PYTHONPATH': str(snapshot/'src'), 'PYTHONDONTWRITEBYTECODE': '1',
        'MUJOCO_GL': 'egl', 'MUJOCO_EGL_DEVICE_ID': str(gpu),
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1'}


def command_for(spec, snapshot, folder, seed):
    options = spec['options']
    command = [sys.executable, '-u', str(snapshot/'scripts/demo_clean_table.py'),
               '--record', '--rgb', '--grasp-mode', 'physical', '--seed', str(seed), '--output', str(folder/'recording')]
    for name in INPUT_OPTIONS:
        command += ['--'+name.replace('_','-'), str(snapshot/options[name])]
    return command


def load_json(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def check_contract(f):
    metadata = json.loads(f.attrs['metadata'])
    if float(metadata['fps']) != 30.:
        raise ValueError('Wrong sampling frequency')
    if not metadata.get('action_semantics', '').startswith('accepted absolute joint targets'):
        raise ValueError('Wrong action semantics')
    if json.loads(f.attrs['units']) != ['rad']*6+['m']+['rad']*6+['m']:
        raise ValueError('Wrong joint/gripper units')
    if json.loads(f.attrs['features']) != list(FEATURE_NAMES):
        raise ValueError('Wrong feature order')
    if metadata['task_scene']['layout'].get('randomized') is not True:
        raise ValueError('Randomization was not applied')
    times = f['observation/sim_time'][:]
    if len(times)<2 or not np.allclose(np.diff(times), 1/30, atol=1e-6, rtol=0):
        raise ValueError('Noncontiguous 30 Hz recording')


def assess(folder, policy, returncode):
    """Read and verify closed evidence; never edit HDF5 or its audit sidecar."""
    recording = folder/'recording'
    report, audit = load_json(recording/'report.json'), load_json(recording/'audit.json')
    path = recording/'episode.h5'
    verified = None
    error = report.get('error')
    if returncode in (None, 0) and not error and report.get('training_ready') and audit.get('training_ready'):
        try:
            with h5py.File(path, 'r') as f:
                checked, digest = verify_training_acceptance(f, path, policy=policy)
                check_contract(f)
                verified = {'source_sha256': digest, 'frames': int(f.attrs['num_transitions']),
                    'duration_s': checked['checks']['streams']['duration_s'],
                    'trajectory_sha256': checked['checks']['streams']['trajectory_sha256']}
        except Exception as exc:
            error = 'FinalReadOnlyVerification: '+type(exc).__name__+': '+str(exc)
    if verified is None:
        error = error or ('episode_timeout' if returncode == -1 else 'incomplete_or_rejected_episode')
    return {'accepted': verified is not None, 'training_ready': verified is not None,
        'returncode': returncode, 'error': error, 'violations': audit.get('violations', ['missing_audit']),
        'physical_summary': report.get('summary'), 'randomized': report.get('randomized'),
        'inventory': report.get('inventory'), 'verified': verified,
        'infrastructure_failure': not report,
        'has_rgb_recording': path.exists() or path.with_suffix('.h5.partial').exists()}


def process_identity(pid):
    try:
        # /proc stat field 22, robust to spaces in the process comm field.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError: return None


def recover_owned_child(folder):
    marker = load_json(folder/'process.json')
    pid = marker.get('pid')
    if not pid or process_identity(pid) != marker.get('proc_start_ticks'): return
    try: args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except OSError: return
    if str(folder/'recording') not in args or not any(x.endswith('/demo_clean_table.py') for x in args):
        raise ValueError('Saved PID now belongs to a different process')
    # Only this batch's demonstrably owned session may be stopped on resume.
    try: os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError: return
    for _ in range(50):
        if process_identity(pid) != marker.get('proc_start_ticks'): return
        time.sleep(.2)
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass


def finalize(output, folder, policy, returncode):
    claim = load_json(folder/'claim.json')
    result = {**claim, **assess(folder, policy, returncode), 'finished_unix': time.time()}
    category = 'success' if result['accepted'] else 'failed'
    destination = output/category/folder.name
    if destination.exists(): raise FileExistsError(destination)
    result.update(category=category, directory=str(destination),
                  original_recorded_directory=str(folder/'recording'))
    for name, filename in (('path','episode.h5'),('report','report.json'),('audit','audit.json')):
        path = folder/'recording'/filename
        if name=='path' and not path.exists() and path.with_suffix('.h5.partial').exists():
            filename='episode.h5.partial';path=folder/filename
        result[name] = str(destination/'recording'/filename) if path.exists() else None
    atomic_json(folder/'result.json', result)
    folder.rename(destination)
    return result


def publish(output, spec, results, active, status, started, error=None, *, manifests=True):
    successes = sorted((r for r in results if r['accepted']), key=lambda r:r['index'])
    failures = sorted((r for r in results if not r['accepted']), key=lambda r:r['index'])
    if manifests:
        for name, rows in (('success',successes),('failed',failures)):
            atomic_text(output/(name+'.jsonl'), ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    state = {'status': status, 'manager_pid': os.getpid(), 'updated_unix': time.time(),
        'invocation_started_unix': started, 'target_success': spec['options']['target_success'],
        'success_count': len(successes), 'failed_count': len(failures),
        'completed_attempts': len(results), 'fps': 30., 'action': 'absolute_joint_position',
        'randomized': True, 'error': error, 'gpus': spec['options']['gpus'],
        'planned_workers':len(spec['options']['gpus'])*spec['options']['workers_per_gpu'],
        'active': [{'index':j['index'],'seed':j['seed'],'gpu':j['gpu'],'slot':slot,'pid':j['process'].pid,
                    'directory':str(j['folder']),'elapsed_s':time.monotonic()-j['started']}
                   for slot,j in active.items()]}
    atomic_json(output/'status.json', state)
    return state


def collect(output, spec):
    snapshot = output/'inputs/project'
    options=spec['options'];policy=snapshot/options['acceptance_policy']
    if fingerprints(snapshot) != spec['files'] or versions() != spec['versions']:
        raise ValueError('Frozen collection inputs or installed dependency versions changed')
    results=[]
    for category in ('success','failed'):
        for folder in sorted((output/category).iterdir()):
            result=load_json(folder/'result.json')
            if not result: raise ValueError('Missing terminal result: '+str(folder))
            if result['accepted']:
                with h5py.File(result['path'],'r') as f:
                    _,digest=verify_training_acceptance(f,result['path'],policy=policy);check_contract(f)
                if digest!=result['verified']['source_sha256']:raise ValueError('Changed accepted episode')
            results.append(result)
    for folder in sorted((output/'running').iterdir()):
        recover_owned_child(folder)
        results.append(finalize(output,folder,policy,None))
    if len({r['index'] for r in results})!=len(results) or len({r['seed'] for r in results})!=len(results):
        raise ValueError('Duplicate episode index or random seed')
    successes=sum(r['accepted'] for r in results)
    if successes>options['target_success']:raise ValueError('Success quota already exceeded')
    index=max((r['index'] for r in results),default=-1)+1
    used_seeds={r['seed'] for r in results}
    gpu_slots=[gpu for gpu in options['gpus'] for _ in range(options['workers_per_gpu'])]
    active={};started=time.time();last_publish=0.;infra_streak=0
    status='running';error=None
    try:
        while successes<options['target_success'] or active:
            for slot,job in list(active.items()):
                code=job['process'].poll()
                if code is None and time.monotonic()-job['started']>options['timeout_s']:
                    stop_process_groups([job['process']]);code=-1
                if code is None:continue
                job['log'].close()
                result=finalize(output,job['folder'],policy,code)
                results.append(result);successes+=int(result['accepted']);del active[slot]
                infra_streak=infra_streak+1 if result['infrastructure_failure'] else 0
                print(json.dumps({k:result[k] for k in ('index','seed','gpu','category','error','path')},ensure_ascii=False),flush=True)
                if infra_streak>=len(options['gpus']):
                    raise RuntimeError('Repeated infrastructure failures; inspect failed episode logs before resuming')
                publish(output,spec,results,active,'running',started)
            if successes==options['target_success'] and not active:break
            free=shutil.disk_usage(output).free/1024**3
            if free<options['min_free_gib']:
                if not active:raise OSError(f'Insufficient free disk: {free:.1f} GiB; resume after freeing space')
            else:
                warmup = successes<options['warmup_successes']
                launched=False
                for slot,gpu in enumerate(gpu_slots):
                    if slot in active or not quota_slots(options['target_success'],successes,len(active)):continue
                    if warmup and active:break
                    seed=choose_seed(index,options,used_seeds);used_seeds.add(seed)
                    folder=output/'running'/f'episode_{index:06d}_seed_{seed}'
                    folder.mkdir()
                    claim={'index':index,'seed':seed,'gpu':gpu,'slot':slot,'started_unix':time.time()}
                    atomic_json(folder/'claim.json',claim)
                    log=(folder/'collection.log').open('x')
                    try:
                        process=subprocess.Popen(command_for(spec,snapshot,folder,claim['seed']),
                            cwd=snapshot,env=worker_environment(gpu,snapshot),stdin=subprocess.DEVNULL,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    except BaseException:
                        log.close();raise
                    atomic_json(folder/'process.json',{'pid':process.pid,'proc_start_ticks':process_identity(process.pid)})
                    active[slot]={**claim,'folder':folder,'log':log,'process':process,'started':time.monotonic()}
                    index+=1
                    launched=True
                if launched:publish(output,spec,results,active,'warmup' if warmup else 'running',started)
            if time.monotonic()-last_publish>=15:
                publish(output,spec,results,active,'warmup' if successes<options['warmup_successes'] else 'running',started,manifests=False)
                last_publish=time.monotonic()
            time.sleep(1)
        status='completed'
        return 0
    except BaseException as exc:
        status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        stop_process_groups([j['process'] for j in active.values()])
        for job in active.values():job['log'].close()
        # Interrupted working folders are preserved for read-only recovery on resume.
        publish(output,spec,results,active,status,started,error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    for name,default in DEFAULTS.items():
        kwargs={'default':None}
        if name=='gpus':kwargs.update(type=int,nargs='+')
        elif isinstance(default,int) or name=='seed_start':kwargs['type']=int
        elif isinstance(default,float):kwargs['type']=float
        parser.add_argument('--'+name.replace('_','-'),**kwargs)
    args=parser.parse_args();output=args.output.resolve()
    if not args.resume:
        options={name:getattr(args,name) if getattr(args,name) is not None else default
                 for name,default in DEFAULTS.items()}
        if (options['target_success']<1 or (options['seed_start'] is not None and options['seed_start']<0) or not options['gpus']
                or min(options['gpus'])<0 or len(set(options['gpus']))!=len(options['gpus'])
                or options['workers_per_gpu']<1
                or options['timeout_s']<=0 or options['min_free_gib']<0
                or not 0<=options['warmup_successes']<=options['target_success']):
            parser.error('Invalid collection quota, GPU list, timeout, or reserve')
        initialize(output,options)
    else:
        spec=json.loads((output/'run.json').read_text())
        for name in DEFAULTS:
            value=getattr(args,name)
            if value is not None and value!=spec['options'][name]:
                parser.error('Resume must use the original options: '+name)
    snapshot=output/'inputs/project';script=snapshot/'scripts'/Path(__file__).name
    if Path(__file__).resolve()!=script:
        environment={**os.environ,'PYTHONPATH':str(snapshot/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
        os.execve(sys.executable,[sys.executable,'-u',str(script),'--output',str(output),'--resume'],environment)
    spec=json.loads((output/'run.json').read_text())
    with (output/'.manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with termination_guard():
            try:return collect(output,spec)
            except Exception as exc:
                state=load_json(output/'status.json')
                state.update(status='failed',manager_pid=os.getpid(),updated_unix=time.time(),
                             error=type(exc).__name__+': '+str(exc))
                atomic_json(output/'status.json',state)
                raise


if __name__=='__main__':
    raise SystemExit(main())
