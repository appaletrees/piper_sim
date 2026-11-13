#!/usr/bin/env python3
"""Bounded frame-capacity probe with independently guarded worker restoration.

Only this run's PID/start/command-verified workers are signalled. Simulation
clocks remain stationary while a process is stopped. Reports are diagnostics,
never accepted episodes or successful-task throughput measurements.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def info(marker):
    try:
        p=Path('/proc')/str(marker['pid'])
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        argv=(p/'cmdline').read_bytes().decode().split('\0')
        if (fields[19]==marker['proc_start_ticks'] and marker['script'] in argv
                and marker['recording'] in argv and fields[0]!='Z'):
            return fields[0]
    except (OSError,IndexError):pass
    return None


def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)


def restore(lease):
    while True:
        value=json.loads(lease.read_text());p=Path('/proc')/str(value['pid'])
        try:alive=(p/'stat').read_text().rsplit(')',1)[1].split()[19]==value['proc_start_ticks']
        except OSError:alive=False
        if value['done'] or not alive or time.time()-value['heartbeat']>25:break
        time.sleep(2)
    resumed=[]
    for marker in value['markers']:
        if info(marker):
            try:os.kill(marker['pid'],signal.SIGCONT);resumed.append(marker['pid'])
            except ProcessLookupError:pass
    drain=Path(value['drain'])
    if drain.exists() and drain.read_text()==value['drain_contents']:drain.unlink()
    atomic(lease.with_name('restored.json'),{'resumed':resumed,'time':time.time(),
        'owner':value['owner'],'heartbeat_expired':time.time()-value['heartbeat']>25})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-lease',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--levels',type=int,nargs='+',help='Explicit environments per GPU, in measured order')
    args=parser.parse_args()
    if args.restore_lease:return restore(args.restore_lease)
    if not all((args.run,args.candidate,args.output)):parser.error('Need run, candidate and output')
    run=args.run.resolve();drain=run/'DRAIN';state=json.loads((run/'status.json').read_text())
    if drain.exists():raise ValueError('Preserve pre-existing drain request')
    markers=[]
    for job in state['active']:
        folder=Path(job['directory']);claim=json.loads((folder/'claim.json').read_text())
        snapshot=Path(claim['worker_revision'])/'project' if claim.get('worker_revision') else run/'inputs/project'
        marker={**json.loads((folder/'process.json').read_text()),'gpu':job['gpu'],'index':job['index'],
            'script':str(snapshot/'scripts/demo_clean_table.py'),'recording':str(folder/'recording'),
            'nvme':folder.is_symlink()}
        if info(marker) not in (None,'T','t'):markers.append(marker)
    gpus=state['gpus'];maximum=min(sum(m['gpu']==g for m in markers) for g in gpus)
    if args.levels:
        if min(args.levels)<1 or max(args.levels)>maximum:raise ValueError('Requested levels exceed verified live workers')
        targets=args.levels
    else:
        maximum=min(4,maximum)
        if maximum<3:raise ValueError('Need at least three verified live workers per GPU')
        targets=[maximum,maximum-1,2,maximum-1,maximum]
    args.output.mkdir(exist_ok=False);lease=args.output/'lease.json';owner=uuid.uuid4().hex
    payload=json.dumps({'owner':owner});start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    def heartbeat(done=False):
        atomic(lease,{'owner':owner,'pid':os.getpid(),'proc_start_ticks':start,'heartbeat':time.time(),
            'done':done,'markers':markers,'drain':str(drain),'drain_contents':payload})
    heartbeat()
    guard_log=(args.output/'guard.log').open('w')
    guard=subprocess.Popen(['python3','-u',str(Path(__file__).resolve()),'--restore-lease',str(lease)],
        stdin=subprocess.DEVNULL,stdout=guard_log,stderr=subprocess.STDOUT,start_new_session=True)
    guard_log.close();atomic(args.output/'guard.json',{'pid':guard.pid})
    rows=[]
    try:
        with drain.open('x') as f:f.write(payload)
        for iteration,per_gpu in enumerate(targets):
            selected=set()
            for gpu in gpus:
                group=sorted((m for m in markers if m['gpu']==gpu),key=lambda m:(not m['nvme'],m['index']))
                selected.update(m['pid'] for m in group[:per_gpu-(gpu==3)])
            for marker in markers:
                if info(marker):os.kill(marker['pid'],signal.SIGCONT if marker['pid'] in selected else signal.SIGSTOP)
            heartbeat();report=args.output/f'stage_{iteration}_{per_gpu}.json'
            env={**os.environ,'PYTHONPATH':str(run/'inputs/project/src'),'PYTHONDONTWRITEBYTECODE':'1',
                'MUJOCO_GL':'egl','MUJOCO_EGL_DEVICE_ID':'3','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            command=['/mnt/piper_sim/.venv-mujoco/bin/python','-u',str(args.candidate/'benchmark_sensor_pipeline.py'),
                '--run',str(run),'--candidate',str(args.candidate),'--output',str(report),'--frames','30','--modes','pool3']
            log=(args.output/f'stage_{iteration}.log').open('w')
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            started=time.monotonic()
            while process.poll() is None:
                heartbeat();time.sleep(1)
                if time.monotonic()-started>180:
                    process.terminate();process.wait(timeout=10);raise RuntimeError('Probe timeout')
            log.close()
            if process.returncode:raise RuntimeError((args.output/f'stage_{iteration}.log').read_text()[-1500:])
            row=json.loads(report.read_text())['rows'][0]
            row.update(target_environments=per_gpu*len(gpus),capacity_proxy_fps=per_gpu*len(gpus)*row['fps'])
            rows.append(row);atomic(args.output/'summary.json',{'rows':rows,'scope':__doc__,'training_ready':False})
            print(json.dumps({k:row[k] for k in ('target_environments','fps','capacity_proxy_fps','different_channel_values')}),flush=True)
    finally:
        heartbeat(True)
        for _ in range(20):
            if (args.output/'restored.json').exists():break
            time.sleep(1)
        print('RESTORE_RECEIPT '+str(args.output/'restored.json'),flush=True)


if __name__=='__main__':main()
