#!/usr/bin/env python3
"""Elastic supervisor over immutable success-quota worker inputs.

A legacy supervisor with a consistent atomic claim catalog can be replaced without signalling its independent
worker sessions. All quota claims, seeds, recordings and audits stay in place.
"""
import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid

UNKNOWN_EXIT = 257
SIM_TIME = re.compile(r"['\"]sim_time['\"]\s*:\s*([0-9.eE+-]+)")


def process_info(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'start': fields[19], 'exit': int(fields[49]),
                'argv': Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')}
    except FileNotFoundError:
        return None


def owned(info, marker, folder, snapshot):
    return bool(info and info['start'] == marker['proc_start_ticks'] and
        (info['state'] == 'Z' or (str(folder/'recording') in info['argv'] and
         str(snapshot/'scripts/demo_clean_table.py') in info['argv'])))


class AttachedProcess:
    def __init__(self, marker):
        self.pid = marker['pid']; self.start = marker['proc_start_ticks']

    def poll(self):
        info = process_info(self.pid)
        if info is None or info['start'] != self.start:
            return UNKNOWN_EXIT  # closed-file evidence decides; never invent an exit code
        if info['state'] == 'Z':
            return os.waitstatus_to_exitcode(info['exit'])
        return None


def signal_owned(job, snapshot, signum):
    marker = job['marker']; info = process_info(marker['pid'])
    if not owned(info, marker, job['folder'], snapshot) or info['state'] == 'Z':
        return False
    if os.getpgid(marker['pid']) != marker['pid']:
        raise ValueError('Worker no longer owns its saved process group')
    os.killpg(marker['pid'], signum)
    return True


def stop_owned(job, snapshot):
    if not signal_owned(job, snapshot, signal.SIGTERM):
        return
    until = time.monotonic()+10
    while job['process'].poll() is None and time.monotonic() < until:
        time.sleep(.1)
    signal_owned(job, snapshot, signal.SIGKILL)
    if isinstance(job['process'], subprocess.Popen):
        job['process'].wait(timeout=10)


def acquire_manager(lock, output, controller, cli, *, takeover=False):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        if not takeover:
            raise ValueError('Another supervisor owns this run; explicit takeover required')
    state = cli.load_json(output/'status.json'); pid = state['manager_pid']
    info = process_info(pid)
    expected_script = str(output/'inputs/project/scripts/collect_success_quota.py')
    expected = bool(info and expected_script in info['argv'])
    if info and '--controller-dir' in info['argv']:
        old_controller=Path(info['argv'][info['argv'].index('--controller-dir')+1])
        old_config=cli.load_json(old_controller/'controller.json')
        expected=(old_controller.is_relative_to(output/'schedulers') and
                  str(old_controller/'controller.py') in info['argv'] and
                  cli.sha256(old_controller/'controller.py')==old_config.get('controller_sha256') and
                  old_config.get('base_run_sha256')==cli.sha256(output/'run.json'))
    if not expected or str(output) not in info['argv']:
        raise ValueError('Lock holder is not a verified quota supervisor')
    original_start = info['start']; stopped = False; replaced = False
    try:
        # /proc syscall inspection may be forbidden for a non-child process.
        # Freeze the manager, then validate the atomic on-disk claim catalog.
        # A legacy launch gap (mkdir/claim/Popen/process marker) causes a retry,
        # never an orphaned unregistered worker. Terminal rename happens only
        # after result.json is committed, so a completed result cannot be lost.
        for _ in range(250):
            current = process_info(pid)
            if not current or current['start'] != original_start:
                raise ValueError('Supervisor identity changed during handoff')
            os.kill(pid, signal.SIGSTOP); stopped = True
            for _ in range(100):
                current = process_info(pid)
                if current and current['state'] in ('T', 't'):break
                time.sleep(.005)
            consistent = bool(current and current['state'] in ('T', 't'))
            if consistent:
                for folder in (output/'running').iterdir():
                    claim=cli.load_json(folder/'claim.json');marker=cli.load_json(folder/'process.json')
                    if not ({'index','seed','gpu','started_unix'} <= claim.keys()
                            and {'pid','proc_start_ticks'} <= marker.keys()):
                        consistent=False;break
                for category in ('success','failed'):
                    for folder in (output/category).iterdir():
                        result=cli.load_json(folder/'result.json')
                        if not {'index','seed','accepted'} <= result.keys():
                            consistent=False;break
            if consistent:break
            os.kill(pid, signal.SIGCONT); stopped = False
            time.sleep(.05)
        else:
            raise RuntimeError('Could not verify atomic claim catalog; workers left untouched')
        workers = []
        snapshot = output/'inputs/project'
        for folder in (output/'running').iterdir():
            marker = cli.load_json(folder/'process.json'); p = process_info(marker['pid'])
            if p and p['start'] == marker['proc_start_ticks']:
                if not owned(p, marker, folder, snapshot):
                    raise ValueError('Unexpected worker command in saved claim')
                workers.append({**marker, 'directory': str(folder), 'state': p['state']})
        receipt = {'legacy_pid': pid, 'legacy_start': original_start, 'atomic_claim_catalog_verified': True,
                   'workers': workers, 'created_unix': time.time(),
                   'operation': 'replace supervisor PID only; preserve all worker sessions'}
        cli.atomic_json(controller/'handoff.json', receipt)
        # SIGKILL avoids the legacy finally block, which would stop its workers.
        # Workers were created as separate sessions and receive no signal here.
        os.kill(pid, signal.SIGKILL); replaced = True
        until = time.monotonic()+10
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() > until:raise RuntimeError('Supervisor lock did not release')
                time.sleep(.05)
        receipt['preserved_live_pids'] = [m['pid'] for m in workers
            if (p := process_info(m['pid'])) and p['start'] == m['proc_start_ticks'] and p['state'] != 'Z']
        cli.atomic_json(controller/'handoff.json', receipt)
    finally:
        if stopped and not replaced:
            current = process_info(pid)
            if current and current['start'] == original_start:os.kill(pid, signal.SIGCONT)


def pressure(path, kind='some'):
    for line in Path(path).read_text().splitlines():
        if line.startswith(kind+' '):return float(line.split('avg10=')[1].split()[0])
    return 0.


def resources(previous_cpu, gpus):
    ticks = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    total, idle = sum(ticks), ticks[3]+ticks[4]
    busy = None
    if previous_cpu:
        delta = total-previous_cpu[0]
        if delta > 0:busy = 100*(1-(idle-previous_cpu[1])/delta)
    mem = {line.split(':')[0]:int(line.split()[1]) for line in
           Path('/proc/meminfo').read_text().splitlines()}
    gpu = {}
    result = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                             '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
    if result.returncode:raise RuntimeError('Cannot sample GPU resource usage: '+result.stderr[-300:])
    for line in result.stdout.splitlines():
        index, util, used, capacity = map(float, line.split(','))
        if int(index) in gpus:gpu[int(index)]={'utilization_percent':util,'memory_used_mib':used,'memory_total_mib':capacity}
    if set(gpu) != set(gpus):raise ValueError('Configured GPU disappeared')
    return {'cpu_busy_percent':busy,'load1':os.getloadavg()[0],
            'memory_available_gib':mem['MemAvailable']/1024**2,
            'cpu_pressure_percent':pressure('/proc/pressure/cpu'),
            'io_pressure_percent':pressure('/proc/pressure/io','full'),
            'io_some_pressure_percent':pressure('/proc/pressure/io'), 'gpus':gpu}, (total,idle)


def log_time(path):
    if not path.exists():return 0.
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536)); text=f.read().decode(errors='replace')
    values = SIM_TIME.findall(text)
    return float(values[-1]) if values else 0.


def tune(current, maximum, window, previous, dt):
    """Bound growth by observed whole-host pressure and a labelled progress proxy."""
    reason = 'headroom_available'
    if window['memory_available_gib'] < 64 or window['io_pressure_percent'] >= 15:
        return max(1,current-1),'pressure_backoff',True
    if window['gpu_memory_fraction'] >= .85:
        return max(1,current-1),'gpu_memory_backoff',True
    if (previous and previous['proxy_rate'] > 0 and window['proxy_rate'] < .90*previous['proxy_rate']
            and window['cpu_busy_percent'] >= 65):
        return previous['per_gpu'],'progress_proxy_regression',True
    if window['cpu_busy_percent'] >= 85 or window['cpu_pressure_percent'] >= 10:
        return current,'cpu_saturated',True
    if window['gpu_utilization_percent'] >= 95:
        return current,'gpu_saturated',True
    levels = sorted(set([3,4,6,8,10,maximum]))
    next_level = next((v for v in levels if current < v <= maximum),current)
    return next_level,'maximum_reached' if next_level==current else reason,next_level==current


def admission_slots(gpus, desired, active):
    """Fill the least occupied card first when storage admits only a few jobs."""
    occupied=set(active)
    counts=collections.Counter(gpu for gpu,_ in occupied)
    while True:
        available=[g for g in gpus if counts[g]<desired]
        if not available:return
        gpu=min(available,key=lambda g:(counts[g],gpus.index(g)))
        lane=next(i for i in range(desired) if (gpu,i) not in occupied)
        occupied.add((gpu,lane));counts[gpu]+=1
        yield gpu,lane


def load_cli(output):
    snapshot=output/'inputs/project'
    sys.path.insert(0,str(snapshot/'src'))
    s=importlib.util.spec_from_file_location('frozen_quota',snapshot/'scripts/collect_success_quota.py')
    cli=importlib.util.module_from_spec(s);s.loader.exec_module(cli)
    return cli


def storage_module(controller, config, cli):
    path=controller/'collection_nvme.py'
    if cli.sha256(path)!=config.get('storage_sha256'):
        raise ValueError('NVMe storage helper identity changed')
    module=importlib.util.spec_from_file_location('collection_nvme',path)
    value=importlib.util.module_from_spec(module);module.loader.exec_module(value)
    return value


def load_ledger(output, spec, cli):
    """Read the predecessor journal; full revalidation runs concurrently below."""
    rows=[]
    prior=spec.get('prior_run')
    if prior:
        root=Path(prior['root'])
        if any(cli.sha256(root/name)!=digest for name,digest in prior['files'].items()):
            raise ValueError('Prior ledger identity changed')
        old=cli.load_json(root/'run.json');policy=root/'inputs/project'/old['options']['acceptance_policy']
        for category in ('success','failed'):
            for line in (root/(category+'.jsonl')).read_text().splitlines():
                row=json.loads(line)
                if row['accepted']!=(category=='success'):raise ValueError('Prior category mismatch')
                row.setdefault('source_run',str(root));row.setdefault('acceptance_policy',str(policy));rows.append(row)
    for category in ('success','failed'):
        for folder in (output/category).iterdir():
            row=cli.load_json(folder/'result.json')
            if not row or row.get('accepted')!=(category=='success'):raise ValueError('Missing/mismatched terminal result')
            rows.append(row)
    return rows


def verify_ledger(output, request, cli):
    payload=cli.load_json(request);result_path=request.with_name('revalidation_result.json')
    digest=cli.sha256(request);checked=0
    try:
        for row in payload['episodes']:
            locks=output/'storage_locks';locks.mkdir(exist_ok=True)
            lock_name=Path(row['path']).parent.parent.name or hashlib.sha256(row['path'].encode()).hexdigest()
            with (locks/(lock_name+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_SH)
                with cli.h5py.File(row['path'],'r') as f:
                    _,actual=cli.verify_training_acceptance(f,row['path'],policy=row['policy']);cli.check_contract(f)
            if actual!=row['source_sha256']:raise ValueError('Changed historical accepted episode')
            checked+=1
            cli.atomic_json(request.with_name('revalidation_progress.json'),
                            {'checked':checked,'total':len(payload['episodes']),'updated_unix':time.time()})
        cli.atomic_json(result_path,{'passed':True,'checked':checked,'request_sha256':digest,'completed_unix':time.time()})
        return 0
    except Exception as exc:
        cli.atomic_json(result_path,{'passed':False,'checked':checked,'request_sha256':digest,
                                     'error':type(exc).__name__+': '+str(exc)})
        return 2


def start_revalidation(output, controller, rows, policy, cli):
    episodes=[{'path':row['path'],'policy':row.get('acceptance_policy',str(policy)),
               'source_sha256':row['verified']['source_sha256']} for row in rows if row['accepted']]
    request=controller/'revalidation_request.json';cli.atomic_json(request,{'episodes':episodes})
    command=[sys.executable,'-u',str(controller/'controller.py'),'--output',str(output),'--verify-ledger',str(request)]
    log=(controller/'revalidation.log').open('a')
    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True,close_fds=True)
    log.close()
    cli.atomic_json(controller/'revalidation_process.json',{'pid':proc.pid,'command':command})
    return proc,cli.sha256(request),len(episodes)


def supervise(output, controller, config, cli):
    spec=cli.load_json(output/'run.json');snapshot=output/'inputs/project';options=spec['options']
    if (cli.sha256(output/'run.json') != config['base_run_sha256'] or
        cli.fingerprints(snapshot) != spec['files'] or cli.versions()!=spec['versions']):
        raise ValueError('Frozen worker inputs/dependencies changed')
    policy=snapshot/options['acceptance_policy']
    results=load_ledger(output,spec,cli)
    storage=None;fast=None;archive=None;storage_state=None;archiver=None
    if config.get('fast_root'):
        storage=storage_module(controller,config,cli)
        fast=Path(config['fast_root']);fast.mkdir(parents=True,exist_ok=True)
        archive=Path(config['archive_root'])
        command=[sys.executable,'-u',str(controller/'controller.py'),'--output',str(output),
                 '--controller-dir',str(controller),'--archive-only']
        archive.mkdir(parents=True,exist_ok=True)
        marker=cli.load_json(archive/'process.json');info=process_info(marker.get('pid',-1))
        if (info and info['start']==marker.get('proc_start_ticks') and info['state']!='Z'
                and '--archive-only' in info['argv'] and str(output) in info['argv']):
            archiver=AttachedProcess(marker)
        else:
            log=(controller/'archive.log').open('a')
            archiver=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                      start_new_session=True,close_fds=True)
            log.close()
            cli.atomic_json(archive/'process.json',{'pid':archiver.pid,
                'proc_start_ticks':cli.process_identity(archiver.pid),'command':command})
        cli.atomic_json(controller/'archive_process.json',{'pid':archiver.pid,'command':command})
    active={}; maximum=config['max_workers_per_gpu'];gpus=options['gpus']
    for folder in sorted((output/'running').iterdir()):
        claim=cli.load_json(folder/'claim.json');marker=cli.load_json(folder/'process.json')
        info=process_info(marker['pid'])
        if info and info['start']==marker['proc_start_ticks'] and info['state']!='Z':
            if not owned(info,marker,folder,snapshot):raise ValueError('Unowned active worker')
            lane=next(i for i in range(maximum) if (claim['gpu'],i) not in active)
            active[(claim['gpu'],lane)]={**claim,'folder':folder,'marker':marker,
                'process':AttachedProcess(marker),'log':None,'started':time.monotonic()-(time.time()-claim['started_unix'])}
        else:
            code=AttachedProcess(marker).poll()
            results.append(cli.finalize(output,folder,policy,None if code==UNKNOWN_EXIT else code))
    claims=[*results,*active.values()]
    if (len({v['index'] for v in claims}) != len(claims) or len({v['seed'] for v in claims}) != len(claims)):
        raise ValueError('Duplicate episode index/seed')
    index=max((v['index'] for v in claims),default=-1)+1;used={v['seed'] for v in claims}
    successes=sum(v['accepted'] for v in results)
    if successes+len(active)>options['target_success']:raise ValueError('In-flight success quota exceeded')
    verifier,validation_digest,validation_count=start_revalidation(output,controller,results,policy,cli)
    validation_passed=False
    desired=max(config['initial_workers_per_gpu'],max(collections.Counter(v['gpu'] for v in active.values()).values(),default=0))
    desired=min(desired,maximum);started=time.time();last_publish=0.;last_sample=0.;cpu=None
    stage_start=time.monotonic();stage=[];previous=None;history=[];tuning_complete=False
    seen_progress={j['index']:log_time(j['folder']/'collection.log') for j in active.values()}
    phase_advance=0.;sample_wall=time.monotonic();latest={};resource_error=None;infra_streak=0;status='running';error=None

    def publish():
        runtime=copy.deepcopy(spec);runtime['options']['workers_per_gpu']=desired
        jobs={g*maximum+lane:j for (g,lane),j in active.items()}
        state=cli.publish(output,runtime,results,jobs,'draining' if (output/'DRAIN').exists() else status,started,error)
        state.update(scheduler='elastic',controller_directory=str(controller),
            frozen_workers_per_gpu=options['workers_per_gpu'],max_workers_per_gpu=maximum,
            desired_workers_per_gpu=desired,tuning_complete=tuning_complete,resources=latest,
            resource_sampling_error=resource_error,
            historical_revalidation={'passed':validation_passed,'previously_accepted':validation_count,
                'progress':cli.load_json(controller/'revalidation_progress.json')},
            resource_sample_unix=time.time(),scale_history=history[-12:])
        if storage:
            state.update(storage={'fast_root':str(fast),'archive_root':str(archive),
                'capacity':storage_state,'archiver_pid':archiver.pid,
                'archive_status':cli.load_json(archive/'status.json')})
        cli.atomic_json(output/'status.json',state)
        cli.atomic_json(controller/'state.json',state)

    try:
        while successes<options['target_success'] or active or not validation_passed:
            if not validation_passed and verifier.poll() is not None:
                check=cli.load_json(controller/'revalidation_result.json')
                if (verifier.returncode!=0 or not check.get('passed') or
                    check.get('request_sha256')!=validation_digest or check.get('checked')!=validation_count):
                    raise ValueError('Historical source revalidation failed: '+str(check))
                validation_passed=True
            for slot,job in list(active.items()):
                observed=job['process'].poll();code=observed
                if code is None and time.monotonic()-job['started']>config['worker_timeout_s']:
                    stop_owned(job,snapshot);code=-1
                if code is None:continue
                value=log_time(job['folder']/'collection.log')
                phase_advance+=max(0,value-seen_progress.get(job['index'],value))
                seen_progress.pop(job['index'],None)
                if job['log']:job['log'].close()
                result=cli.finalize(output,job['folder'],policy,None if code==UNKNOWN_EXIT else code)
                if observed==UNKNOWN_EXIT:
                    result['exit_code_observed']=False
                    cli.atomic_json(Path(result['directory'])/'result.json',result)
                results.append(result);successes+=int(result['accepted']);del active[slot]
                infra_streak=infra_streak+1 if result['infrastructure_failure'] else 0
                print(json.dumps({k:result.get(k) for k in ('index','gpu','category','error')},ensure_ascii=False),flush=True)
                if infra_streak>=len(gpus):raise RuntimeError('Repeated infrastructure failures')
                publish()
            now=time.monotonic()
            if now-last_sample>=10:
                last_sample=now
                try:
                    latest,cpu=resources(cpu,gpus);resource_error=None
                except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
                    # Monitoring trouble pauses admission, not active recordings.
                    resource_error=type(exc).__name__+': '+str(exc)
                    print('RESOURCE_SAMPLE_RETRY '+resource_error,flush=True)
                    publish();time.sleep(1);continue
                for job in active.values():
                    value=log_time(job['folder']/'collection.log')
                    phase_advance+=max(0,value-seen_progress.get(job['index'],value));seen_progress[job['index']]=value
                if latest['cpu_busy_percent'] is not None:stage.append(latest)
                if stage and now-stage_start>=config['stage_seconds']:
                    window={k:statistics.fmean(s[k] for s in stage) for k in
                        ('cpu_busy_percent','cpu_pressure_percent','io_pressure_percent','memory_available_gib')}
                    window.update(gpu_utilization_percent=statistics.fmean(v['utilization_percent'] for s in stage for v in s['gpus'].values()),
                        gpu_memory_fraction=max(v['memory_used_mib']/v['memory_total_mib'] for s in stage for v in s['gpus'].values()),
                        proxy_rate=phase_advance/max(now-sample_wall,1),per_gpu=desired,
                        proxy_definition='sum of phase-event simulation-time advances / wall seconds; not accepted episodes/hour')
                    before=desired
                    if not tuning_complete:
                        desired,reason,tuning_complete=tune(desired,maximum,window,previous,now-stage_start)
                    elif window['memory_available_gib']<64 or window['io_pressure_percent']>=15 or window['gpu_memory_fraction']>=.85:
                        desired=max(1,desired-1);reason='sustained_pressure_backoff'
                    else:reason='hold_observed_capacity'
                    entry={'unix':time.time(),'from_per_gpu':before,'to_per_gpu':desired,'reason':reason,'window':window}
                    history.append(entry)
                    with (controller/'scale_history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
                    print('SCALE '+json.dumps(entry),flush=True)
                    previous=window;stage=[];phase_advance=0.;sample_wall=now;stage_start=now
            free=cli.shutil.disk_usage(output).free/1024**3
            draining=(output/'DRAIN').exists()
            if draining and not active:status='drained';break
            if storage:
                storage_state=storage.capacity(fast,list(active.values()),config)
                archive_status=cli.load_json(archive/'status.json')
                if archiver.poll() is not None:
                    raise RuntimeError('NVMe archiver exited; pause for explicit recovery')
            admission_ok=(not draining and not resource_error and free>=options['min_free_gib'] and
                (not storage or (storage_state['admit'] and archive_status.get('status')!='error')) and
                (not latest or latest['memory_available_gib']>=64) and
                (not latest or latest['io_pressure_percent']<20))
            if admission_ok:
                for gpu,lane in admission_slots(gpus,desired,active):
                    slot=(gpu,lane)
                    if not cli.quota_slots(options['target_success'],successes,len(active)):break
                    if latest and latest['gpus'][gpu]['memory_used_mib']/latest['gpus'][gpu]['memory_total_mib']>=.85:continue
                    if storage:
                        storage_state=storage.capacity(fast,list(active.values()),config)
                        if not storage_state['admit']:break
                    seed=cli.choose_seed(index,options,used);used.add(seed)
                    name=f'episode_{index:06d}_seed_{seed}'
                    if storage:folder=storage.create_folder(output,fast,name)
                    else:folder=output/'running'/name;folder.mkdir()
                    claim={'index':index,'seed':seed,'gpu':gpu,'slot':gpus.index(gpu)*maximum+lane,
                           'started_unix':time.time(),'scheduler_revision':str(controller)}
                    cli.atomic_json(folder/'claim.json',claim);log=(folder/'collection.log').open('x')
                    proc=subprocess.Popen(cli.command_for(spec,snapshot,folder,seed),cwd=snapshot,
                        env=cli.worker_environment(gpu,snapshot),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    marker={'pid':proc.pid,'proc_start_ticks':cli.process_identity(proc.pid)}
                    cli.atomic_json(folder/'process.json',marker)
                    active[slot]={**claim,'folder':folder,'marker':marker,'process':proc,'log':log,'started':time.monotonic()}
                    seen_progress[index]=0.;index+=1
            elif not active and not draining and not storage:
                raise RuntimeError('Insufficient disk/memory or sustained I/O pressure; explicit resume required')
            if now-last_publish>=10:publish();last_publish=now
            time.sleep(1)
        if status!='drained':status='completed'
        return 0
    except BaseException as exc:
        status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        if verifier.poll() is None:
            verifier.terminate()
            try:verifier.wait(timeout=5)
            except subprocess.TimeoutExpired:verifier.kill();verifier.wait()
        for job in active.values():
            stop_owned(job,snapshot)
            if job['log']:job['log'].close()
        publish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controller-dir',type=Path)
    parser.add_argument('--takeover',action='store_true')
    parser.add_argument('--verify-ledger',type=Path)
    parser.add_argument('--archive-only',action='store_true')
    parser.add_argument('--fast-root',type=Path)
    parser.add_argument('--archive-root',type=Path)
    parser.add_argument('--fast-min-free-gib',type=float,default=128.)
    parser.add_argument('--episode-reserve-gib',type=float,default=16.)
    parser.add_argument('--initial-workers-per-gpu',type=int,default=3)
    parser.add_argument('--max-workers-per-gpu',type=int,default=10)
    parser.add_argument('--stage-seconds',type=float,default=180.)
    parser.add_argument('--worker-timeout-s',type=float,default=21600.)
    args=parser.parse_args();output=args.output.resolve();cli=load_cli(output)
    if args.verify_ledger is not None:return verify_ledger(output,args.verify_ledger,cli)
    if args.controller_dir is None:
        if not (1<=args.initial_workers_per_gpu<=args.max_workers_per_gpu<=12 and args.stage_seconds>=60 and args.worker_timeout_s>0):
            parser.error('Invalid elastic scheduling bounds')
        directory=output/'schedulers'/('elastic_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);directory.mkdir(parents=True)
        source=directory/'controller.py';cli.shutil.copy2(__file__,source)
        config={'base_run_sha256':cli.sha256(output/'run.json'),'controller_sha256':cli.sha256(source),
            'initial_workers_per_gpu':args.initial_workers_per_gpu,'max_workers_per_gpu':args.max_workers_per_gpu,
            'stage_seconds':args.stage_seconds,'worker_timeout_s':args.worker_timeout_s,
            'worker_inputs':'original immutable inputs/project; no simulation or acceptance change',
            'created_unix':time.time()}
        if args.fast_root:
            fast=args.fast_root.resolve();fast.mkdir(parents=True,exist_ok=True)
            if fast.stat().st_dev==output.stat().st_dev:
                parser.error('Fast staging must use a separate filesystem')
            if args.fast_min_free_gib<64 or args.episode_reserve_gib<=0:
                parser.error('Invalid staging capacity reserve')
            helper=directory/'collection_nvme.py'
            cli.shutil.copy2(Path(__file__).with_name('collection_nvme.py'),helper)
            config.update(fast_root=str(fast),archive_root=str((args.archive_root or output/'storage_archive').resolve()),
                fast_min_free_gib=args.fast_min_free_gib,episode_reserve_gib=args.episode_reserve_gib,
                storage_sha256=cli.sha256(helper))
        cli.atomic_json(directory/'controller.json',config)
        command=[sys.executable,'-u',str(source),'--output',str(output),'--controller-dir',str(directory)]
        if args.takeover:command+=['--takeover']
        os.execv(sys.executable,command)
    directory=args.controller_dir.resolve();config=cli.load_json(directory/'controller.json')
    if cli.sha256(Path(__file__))!=config['controller_sha256']:raise ValueError('Scheduler code changed')
    if args.archive_only:
        if not config.get('fast_root'):raise ValueError('Missing NVMe staging configuration')
        storage=storage_module(directory,config,cli)
        try:os.nice(10)
        except OSError:pass
        return storage.archive_loop(output,Path(config['fast_root']),Path(config['archive_root']))
    # Validate before touching a running supervisor.
    spec=cli.load_json(output/'run.json')
    if (cli.sha256(output/'run.json')!=config['base_run_sha256'] or
        cli.fingerprints(output/'inputs/project')!=spec['files'] or cli.versions()!=spec['versions']):
        raise ValueError('Changed frozen worker inputs')
    # A transition lock serializes handoffs only. Older controllers incorrectly
    # held the old lock for their entire lifetime; use the versioned name so a
    # verified takeover of one of those controllers can still proceed.
    with (output/'.elastic_transition_v2.lock').open('a') as takeover_lock:
        fcntl.flock(takeover_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (output/'.manager.lock').open('a') as lock:
            acquire_manager(lock,output,directory,cli,takeover=args.takeover)
            cli.atomic_json(directory/'supervisor.json',{'pid':os.getpid(),'proc_start_ticks':cli.process_identity(os.getpid())})
            fcntl.flock(takeover_lock,fcntl.LOCK_UN)
            from piper_sim.process_lifecycle import termination_guard
            with termination_guard():return supervise(output,directory,config,cli)


if __name__=='__main__':raise SystemExit(main())
