import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter

spec=importlib.util.spec_from_file_location('quota_collector',ROOT/'scripts/collect_success_quota.py')
cli=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_quota_reserves_inflight_slots_and_failure_does_not_fill_quota():
    assert cli.quota_slots(2000,1998,2)==0
    assert cli.quota_slots(2000,1998,1)==1
    assert cli.quota_slots(2000,2000,0)==0


def test_gpu_binding_and_recording_command_keep_absolute_position_teacher(tmp_path):
    before=dict(cli.os.environ)
    env=cli.worker_environment(7,tmp_path)
    assert env['MUJOCO_EGL_DEVICE_ID']=='7' and env['MUJOCO_GL']=='egl'
    assert env['PYTHONPATH']==str(tmp_path/'src') and cli.os.environ==before
    command=cli.command_for({'options':cli.DEFAULTS},tmp_path,tmp_path/'job',43)
    assert '--record' in command and '--rgb' in command
    assert command[command.index('--seed')+1]=='43'
    assert command[command.index('--output')+1]==str(tmp_path/'job/recording')
    assert '--objects' not in command and '--video-speed' not in command
    assert command[command.index('--domain-randomization')+1].endswith('configs/domain_randomization/clean_table.yaml')


def test_entropy_seeds_are_unique_saved_exact_json_integers_and_do_not_repeat(monkeypatch):
    values=iter([19,19,29])
    monkeypatch.setattr(cli.secrets,'randbits',lambda bits:next(values))
    first=cli.choose_seed(0,cli.DEFAULTS,set())
    second=cli.choose_seed(1,cli.DEFAULTS,{first})
    assert (first,second)==(19,29)
    assert first<2**53 and second<2**53


def metadata():
    return {'fps':30.,'action_semantics':'accepted absolute joint targets and full gripper opening',
            'task_scene':{'layout':{'randomized':True}}}


def observation(i):
    return Observation(np.zeros(14),i/30,i/30,{'overhead':np.full((4,4,3),i,np.uint8)},
                       {'overhead':i/30})


def write_episode(folder,meta=None):
    path=folder/'recording/episode.h5'
    with EpisodeWriter(path,meta or metadata(),image_storage='shared',image_compression='gzip') as w:
        for i in range(3):w.append(observation(i),Command(np.zeros(14),i/30,i),observation(i+1))
    return path


@pytest.mark.parametrize('change,message',[
    ({'fps':15.},'sampling'),({'action_semantics':'delta_joint_position'},'semantics'),
    ({'task_scene':{'layout':{'randomized':False}}},'Randomization')])
def test_contract_rejects_wrong_frequency_delta_actions_and_nominal_reset(tmp_path,change,message):
    path=write_episode(tmp_path,{**metadata(),**change})
    with h5py.File(path) as f:
        with pytest.raises(ValueError,match=message):cli.check_contract(f)


def test_forged_success_reports_cannot_bypass_closed_file_training_gate(tmp_path):
    write_episode(tmp_path)
    atomic_json(tmp_path/'recording/report.json',{'training_ready':True})
    atomic_json(tmp_path/'recording/audit.json',{'training_ready':True})
    result=cli.assess(tmp_path,None,0)
    assert not result['accepted'] and not result['training_ready']
    assert result['error'].startswith('FinalReadOnlyVerification:')


def test_terminal_move_keeps_raw_file_sidecars_and_shared_frames_portable(tmp_path,monkeypatch):
    folder=tmp_path/'running/episode_000000_seed_1';folder.mkdir(parents=True)
    (tmp_path/'success').mkdir();(tmp_path/'failed').mkdir()
    path=write_episode(folder);digest=sha256(path)
    audit=b'{"original_recorded_path":"preserved"}\n'
    (folder/'recording/audit.json').write_bytes(audit)
    atomic_json(folder/'claim.json',{'index':0,'seed':1,'gpu':0})
    monkeypatch.setattr(cli,'assess',lambda *args:{'accepted':True,'training_ready':True,
                        'verified':{'source_sha256':digest}})
    result=cli.finalize(tmp_path,folder,None,0)
    moved=Path(result['path'])
    assert not folder.exists() and moved.is_file() and sha256(moved)==digest
    assert moved.with_name('audit.json').read_bytes()==audit
    with h5py.File(moved) as f:
        assert f['next_observation/images/overhead'].is_virtual
        assert all(s.file_name=='.' for s in f['next_observation/images/overhead'].virtual_sources())
        np.testing.assert_array_equal(f['next_observation/images/overhead'][:,0,0,0],[1,2,3])


def test_recovery_never_signals_a_reused_pid(tmp_path,monkeypatch):
    atomic_json(tmp_path/'process.json',{'pid':12345,'proc_start_ticks':'old'})
    monkeypatch.setattr(cli,'process_identity',lambda _: 'new')
    monkeypatch.setattr(cli.os,'killpg',lambda *a:pytest.fail('Must not signal reused PID'))
    cli.recover_owned_child(tmp_path)


@pytest.mark.parametrize('workers_per_gpu',[1,2])
def test_scheduler_warms_up_then_fills_exact_success_quota_retaining_failures(tmp_path,monkeypatch,workers_per_gpu):
    snapshot=tmp_path/'inputs/project';snapshot.mkdir(parents=True)
    for name in ('running','success','failed'):(tmp_path/name).mkdir()
    options={**cli.DEFAULTS,'target_success':3,'seed_start':10,'gpus':[0,1],
             'workers_per_gpu':workers_per_gpu,'min_free_gib':0.}
    run={'options':options,'files':cli.fingerprints(snapshot),'versions':{'test':'1'}}
    monkeypatch.setattr(cli,'versions',lambda:{'test':'1'})
    monkeypatch.setattr(cli.time,'sleep',lambda _:None)
    monkeypatch.setattr(cli,'stop_process_groups',lambda _:None)
    spawned=[]
    class Process:
        def __init__(self,command,**kwargs):
            seed=int(command[command.index('--seed')+1])
            recording=Path(command[command.index('--output')+1]);assert not recording.exists()
            recording.mkdir();(recording/'episode.h5').write_bytes(b'preserved fake recording')
            self.seed=seed;self.pid=10000000+seed
            assert kwargs['start_new_session'] and kwargs['env']['MUJOCO_GL']=='egl'
            spawned.append(seed)
        def poll(self):return 2 if self.seed==10 else 0
    monkeypatch.setattr(cli.subprocess,'Popen',Process)
    def assess(folder,policy,code):
        seed=json.loads((folder/'claim.json').read_text())['seed']
        return {'accepted':seed!=10,'training_ready':seed!=10,'error':'task rejected' if seed==10 else None,
                'infrastructure_failure':False,'verified':None if seed==10 else {'source_sha256':'mock'}}
    monkeypatch.setattr(cli,'assess',assess)
    assert cli.collect(tmp_path,run)==0
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['status']=='completed' and state['success_count']==3 and state['failed_count']==1
    assert state['active']==[] and spawned==[10,11,12,13]
    assert len(list((tmp_path/'success').iterdir()))==3
    assert len(list((tmp_path/'failed').iterdir()))==1
    assert len((tmp_path/'success.jsonl').read_text().splitlines())==3
    assert len((tmp_path/'failed.jsonl').read_text().splitlines())==1


def test_frozen_inputs_change_stops_before_launch(tmp_path,monkeypatch):
    snapshot=tmp_path/'inputs/project';snapshot.mkdir(parents=True)
    run={'options':cli.DEFAULTS,'files':{},'versions':cli.versions()}
    (snapshot/'changed.py').write_text('changed')
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k:pytest.fail('Changed run must not launch'))
    with pytest.raises(ValueError,match='Frozen'):cli.collect(tmp_path,run)


def test_partial_recording_path_is_preserved_on_interruption(tmp_path,monkeypatch):
    folder=tmp_path/'running/episode_000000_seed_1';(folder/'recording').mkdir(parents=True)
    (tmp_path/'failed').mkdir()
    (folder/'recording/episode.h5.partial').write_bytes(b'partial retained')
    atomic_json(folder/'claim.json',{'index':0,'seed':1,'gpu':0})
    monkeypatch.setattr(cli,'assess',lambda *args:{'accepted':False})
    result=cli.finalize(tmp_path,folder,None,None)
    assert Path(result['path']).read_bytes()==b'partial retained'


def test_inheritance_preserves_categories_policy_and_rejects_manifest_changes(tmp_path,monkeypatch):
    root=tmp_path/'old';root.mkdir()
    policy=root/'inputs/project/old_policy.yaml';policy.parent.mkdir(parents=True)
    policy.write_text('old policy')
    atomic_json(root/'run.json',{'options':{'acceptance_policy':'old_policy.yaml'}})
    success={'index':0,'seed':10,'accepted':True,'path':'/old/episode.h5',
             'verified':{'source_sha256':'original'}}
    failed={'index':1,'seed':11,'accepted':False,'error':'task failure'}
    (root/'success.jsonl').write_text(json.dumps(success)+'\n')
    (root/'failed.jsonl').write_text(json.dumps(failed)+'\n')
    spec={'prior_run':{'root':str(root),'files':{name:sha256(root/name)
        for name in ('run.json','success.jsonl','failed.jsonl')}}}
    class File:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(cli.h5py,'File',lambda *a:File())
    checked=[]
    def verify(*args,**kwargs):
        checked.append(kwargs['policy']);return {},'original'
    monkeypatch.setattr(cli,'verify_training_acceptance',verify)
    monkeypatch.setattr(cli,'check_contract',lambda f:None)
    rows=cli.inherited_results(spec)
    assert [r['accepted'] for r in rows]==[True,False]
    assert checked==[str(policy)] and rows[1]['error']=='task failure'
    (root/'failed.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='Prior run'):cli.inherited_results(spec)


def test_drain_does_not_launch_or_change_target(tmp_path,monkeypatch):
    snapshot=tmp_path/'inputs/project';snapshot.mkdir(parents=True)
    for name in ('running','success','failed'):(tmp_path/name).mkdir()
    (tmp_path/'DRAIN').touch()
    run={'options':cli.DEFAULTS,'files':{},'versions':{'test':'1'}}
    monkeypatch.setattr(cli,'versions',lambda:{'test':'1'})
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k:pytest.fail('Drained run must not launch'))
    assert cli.collect(tmp_path,run)==0
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['status']=='drained' and status['success_count']==0 and status['target_success']==2000


def test_inherited_success_fills_same_quota_without_an_extra_worker(tmp_path,monkeypatch):
    snapshot=tmp_path/'inputs/project';snapshot.mkdir(parents=True)
    for name in ('running','success','failed'):(tmp_path/name).mkdir()
    options={**cli.DEFAULTS,'target_success':1}
    run={'options':options,'files':{},'versions':{'test':'1'}}
    row={'index':7,'seed':19,'accepted':True,'path':'/original/episode.h5','source_run':'/original'}
    monkeypatch.setattr(cli,'versions',lambda:{'test':'1'})
    monkeypatch.setattr(cli,'inherited_results',lambda spec:[row])
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k:pytest.fail('Prior success already fills quota'))
    assert cli.collect(tmp_path,run)==0
    assert json.loads((tmp_path/'status.json').read_text())['success_count']==1
    assert json.loads((tmp_path/'success.jsonl').read_text())['path']=='/original/episode.h5'
