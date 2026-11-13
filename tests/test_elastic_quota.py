"""Supervisor adoption must preserve recordings, identities and quota ownership."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from piper_sim.config import ROOT
from piper_sim.collection_jobs import atomic_json

s=importlib.util.spec_from_file_location('elastic',ROOT/'scripts/collect_elastic_quota.py')
e=importlib.util.module_from_spec(s);s.loader.exec_module(e)


def window(**kwargs):
    return {'memory_available_gib':400.,'io_pressure_percent':0.,'gpu_memory_fraction':.1,
            'proxy_rate':2.,'cpu_busy_percent':30.,'cpu_pressure_percent':0.,
            'gpu_utilization_percent':50.,**kwargs}


@pytest.mark.parametrize('current,target',[(2,3),(3,4),(4,6),(6,8),(8,10)])
def test_growth_uses_bounded_levels(current,target):
    desired,reason,done=e.tune(current,10,window(),None,180.)
    assert desired==target and not done


@pytest.mark.parametrize('changed,reason',[
    ({'memory_available_gib':63.},'pressure_backoff'),
    ({'io_pressure_percent':16.},'pressure_backoff'),
    ({'gpu_memory_fraction':.86},'gpu_memory_backoff'),
    ({'cpu_busy_percent':90.},'cpu_saturated'),
    ({'cpu_pressure_percent':11.},'cpu_saturated'),
    ({'gpu_utilization_percent':97.},'gpu_saturated')])
def test_stop_growth_or_reduce_without_killing_existing_jobs(changed,reason):
    target,actual,done=e.tune(6,10,window(**changed),None,180.)
    assert actual==reason and done and target<=6


def test_saturation_and_progress_regression_are_not_reported_as_speedup():
    target,reason,done=e.tune(8,10,window(proxy_rate=1.,cpu_busy_percent=70.),
        {'proxy_rate':2.,'per_gpu':6},180.)
    assert (target,reason,done)==(6,'progress_proxy_regression',True)
    assert e.tune(10,10,window(),None,180.)==(10,'maximum_reached',True)


def test_disk_limited_admission_balances_cards_before_filling_one_card():
    import itertools
    gpus=list(range(8));active={(g,i):{} for g in gpus for i in range(6)}
    slots=list(itertools.islice(e.admission_slots(gpus,12,active),4))
    assert slots==[(0,6),(1,6),(2,6),(3,6)]
    all_slots=list(e.admission_slots(gpus,12,active))
    assert len(all_slots)==48 and len(set(all_slots))==48
    assert not set(all_slots)&set(active)


def test_unbalanced_adopted_cards_fill_the_least_occupied_first():
    import itertools
    active={(0,i):{} for i in range(8)}
    assert list(itertools.islice(e.admission_slots([0,1,2],12,active),4))==[(1,0),(2,0),(1,1),(2,1)]


def test_attached_process_does_not_trust_reused_pid_or_invent_exit_status(monkeypatch):
    p=e.AttachedProcess({'pid':123,'proc_start_ticks':'original'})
    monkeypatch.setattr(e,'process_info',lambda pid:None)
    assert p.poll()==e.UNKNOWN_EXIT
    monkeypatch.setattr(e,'process_info',lambda pid:{'start':'reused','state':'R','exit':0})
    assert p.poll()==e.UNKNOWN_EXIT
    monkeypatch.setattr(e,'process_info',lambda pid:{'start':'original','state':'R','exit':0})
    assert p.poll() is None
    monkeypatch.setattr(e,'process_info',lambda pid:{'start':'original','state':'Z','exit':512})
    assert p.poll()==2


def test_cleanup_never_signals_a_reused_pid_or_wrong_command(tmp_path,monkeypatch):
    marker={'pid':123,'proc_start_ticks':'original'}
    job={'marker':marker,'folder':tmp_path/'job'}
    monkeypatch.setattr(e.os,'killpg',lambda *a:pytest.fail('Unowned PID must not be signalled'))
    monkeypatch.setattr(e,'process_info',lambda pid:{'start':'reused','state':'R','argv':[]})
    assert not e.signal_owned(job,tmp_path,signal.SIGTERM)
    monkeypatch.setattr(e,'process_info',lambda pid:{'start':'original','state':'R','argv':['other.py']})
    assert not e.signal_owned(job,tmp_path,signal.SIGTERM)


def test_log_progress_is_read_without_opening_recording(tmp_path):
    p=tmp_path/'collection.log'
    assert e.log_time(p)==0
    p.write_text("{'sim_time': 1.25}\n{'sim_time': 2.5e1}\n")
    assert e.log_time(p)==25.


def test_verified_catalog_takeover_preserves_live_worker_session(tmp_path):
    """Exercise real flock/SIGSTOP/catalog/SIGKILL behavior against owned mocks."""
    output=tmp_path/'run';snapshot=output/'inputs/project'
    scripts=snapshot/'scripts';scripts.mkdir(parents=True)
    (output/'success').mkdir();(output/'failed').mkdir()
    controller=output/'scheduler';controller.mkdir()
    folder=output/'running/episode_000000_seed_1';folder.mkdir(parents=True)
    worker=scripts/'demo_clean_table.py';worker.write_text('import time\ntime.sleep(120)\n')
    supervisor=scripts/'collect_success_quota.py'
    supervisor.write_text('''import fcntl,json,os,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);f=r/'running/episode_000000_seed_1'
lock=(r/'.manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('demo_clean_table.py')),'--output',str(f/'recording')],start_new_session=True)
start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
(f/'process.json').write_text(json.dumps({'pid':p.pid,'proc_start_ticks':start}))
(f/'claim.json').write_text(json.dumps({'index':0,'seed':1,'gpu':0,'started_unix':time.time()}))
(r/'status.json').write_text(json.dumps({'manager_pid':os.getpid()}))
while True:time.sleep(1)
''')
    parent=subprocess.Popen([sys.executable,str(supervisor),str(output)],start_new_session=True)
    marker=None
    try:
        deadline=time.monotonic()+10
        while not (output/'status.json').exists():
            if time.monotonic()>deadline:pytest.fail('Mock supervisor did not start')
            time.sleep(.02)
        marker=json.loads((folder/'process.json').read_text())
        cli=SimpleNamespace(load_json=lambda p:json.loads(p.read_text()),atomic_json=atomic_json)
        with (output/'.manager.lock').open('a') as lock:
            e.acquire_manager(lock,output,controller,cli,takeover=True)
            assert parent.wait(timeout=5)==-signal.SIGKILL
            info=e.process_info(marker['pid'])
            assert info['start']==marker['proc_start_ticks'] and info['state']!='Z'
            assert os.getpgid(marker['pid'])==marker['pid']
            receipt=json.loads((controller/'handoff.json').read_text())
            assert marker['pid'] in receipt['preserved_live_pids']
            assert receipt['atomic_claim_catalog_verified'] is True
    finally:
        if parent.poll() is None:parent.kill();parent.wait()
        if marker:
            info=e.process_info(marker['pid'])
            if info and info['start']==marker['proc_start_ticks'] and info['state']!='Z':
                os.killpg(marker['pid'],signal.SIGKILL)


@pytest.mark.parametrize('validation_polls',[0,4])
@pytest.mark.parametrize('steady',[None,1])
def test_supervisor_adopts_inflight_and_reserves_exact_global_quota(tmp_path,monkeypatch,validation_polls,steady):
    s=importlib.util.spec_from_file_location('quota_for_elastic_test',ROOT/'scripts/collect_success_quota.py')
    cli=importlib.util.module_from_spec(s);s.loader.exec_module(cli)
    snapshot=tmp_path/'inputs/project';snapshot.mkdir(parents=True)
    for name in ('success','failed','running'):(tmp_path/name).mkdir()
    controller=tmp_path/'schedulers/test';controller.mkdir(parents=True)
    options={**cli.DEFAULTS,'target_success':4,'gpus':[0],'seed_start':100,'workers_per_gpu':1,'min_free_gib':0.}
    spec={'options':options,'files':{},'versions':{'test':'1'}}
    atomic_json(tmp_path/'run.json',spec)
    config={'base_run_sha256':cli.sha256(tmp_path/'run.json'),'max_workers_per_gpu':3,
            'initial_workers_per_gpu':3,'worker_timeout_s':3600.,'stage_seconds':180.}
    if steady is not None:config['steady_workers_per_gpu']=steady
    folder=tmp_path/'running/episode_000001_seed_101';folder.mkdir()
    atomic_json(folder/'claim.json',{'index':1,'seed':101,'gpu':0,'slot':0,'started_unix':time.time()})
    atomic_json(folder/'process.json',{'pid':123,'proc_start_ticks':'fake'})
    monkeypatch.setattr(cli,'versions',lambda:{'test':'1'})
    monkeypatch.setattr(e,'load_ledger',lambda *a:[{'index':0,'seed':100,'accepted':True,'source_run':'old'}])
    def revalidation(*args):
        atomic_json(controller/'revalidation_result.json',{'passed':True,'checked':1,'request_sha256':'test'})
        class Verifier:
            returncode=0
            calls=0
            def poll(self):
                self.calls+=1
                return 0 if self.calls>=validation_polls else None
        return Verifier(),'test',1
    monkeypatch.setattr(e,'start_revalidation',revalidation)
    monkeypatch.setattr(cli,'assess',lambda *a:{'accepted':True,'error':None,'infrastructure_failure':False})
    monkeypatch.setattr(cli,'process_identity',lambda pid:'fake')
    monkeypatch.setattr(e,'process_info',lambda pid:{'start':'fake','state':'R',
        'argv':[str(snapshot/'scripts/demo_clean_table.py'),str(folder/'recording')]})
    class Adopted:
        def __init__(self,marker):self.pid=marker['pid'];self.calls=0
        def poll(self):self.calls+=1;return None if self.calls==1 else 0
    monkeypatch.setattr(e,'AttachedProcess',Adopted)
    spawned=[]
    class NewProcess:
        def __init__(self,cmd,**kwargs):
            seed=int(cmd[cmd.index('--seed')+1]);spawned.append(seed);self.pid=1000+seed
            assert kwargs['start_new_session'] and kwargs['env']['MUJOCO_EGL_DEVICE_ID']=='0'
        def poll(self):return 0
    monkeypatch.setattr(e.subprocess,'Popen',NewProcess)
    monkeypatch.setattr(e.os,'killpg',lambda *a:pytest.fail('Complete workers must not receive signals'))
    monkeypatch.setattr(e,'resources',lambda *a:({'cpu_busy_percent':10.,'memory_available_gib':400.,
        'io_pressure_percent':0.,'gpus':{0:{'memory_used_mib':100.,'memory_total_mib':24000.}}},(1,1)))
    monkeypatch.setattr(e.time,'sleep',lambda v:None)
    assert e.supervise(tmp_path,controller,config,cli)==0
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['success_count']==4 and state['active']==[] and state['status']=='completed'
    assert state['historical_revalidation']['passed'] is True
    assert spawned==[102,103]
    assert state['frozen_workers_per_gpu']==1 and state['planned_workers']==(steady or 3)
    assert json.loads((tmp_path/'run.json').read_text())==spec
    assert len((tmp_path/'success.jsonl').read_text().splitlines())==4


def test_failed_background_revalidation_cannot_publish_a_success(tmp_path):
    request=tmp_path/'request.json'
    atomic_json(request,{'episodes':[{'path':'/not-trusted.h5','policy':'original','source_sha256':'bad'}]})
    class File:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    def reject(*a,**kw):raise ValueError('source checksum mismatch')
    import hashlib
    cli=SimpleNamespace(load_json=lambda p:json.loads(p.read_text()),
        sha256=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(),atomic_json=atomic_json,
        h5py=SimpleNamespace(File=lambda *a:File()),verify_training_acceptance=reject)
    assert e.verify_ledger(tmp_path,request,cli)==2
    result=json.loads((tmp_path/'revalidation_result.json').read_text())
    assert result['passed'] is False and result['checked']==0
    assert 'source checksum mismatch' in result['error']


def test_pressure_distinguishes_one_io_reader_from_whole_host_stall(tmp_path):
    p=tmp_path/'pressure';p.write_text('some avg10=95.0 avg60=10.0\nfull avg10=0.1 avg60=0.0\n')
    assert e.pressure(p)==95.
    assert e.pressure(p,'full')==.1
