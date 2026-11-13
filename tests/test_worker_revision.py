"""Performance rollouts cannot silently change dynamics or old source inputs."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from piper_sim.config import ROOT
from piper_sim.collection_jobs import atomic_json


def modules():
    values=[]
    for name,file in [('elastic_revision','collect_elastic_quota.py'),('quota_revision','collect_success_quota.py')]:
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);values.append(value)
    return values


def setup(tmp_path):
    e,cli=modules();base=tmp_path/'inputs/project';base.mkdir(parents=True)
    cfg={'runtime':{'physics_dt':1/120,'decimation':4},'robot':{'geometry':'authoritative'}}
    (base/'base.yaml').write_text(cli.yaml.safe_dump(cfg));(base/'source.py').write_text('# frozen\n')
    spec={'options':{**cli.DEFAULTS,'config':'base.yaml'},'files':cli.fingerprints(base)}
    atomic_json(tmp_path/'run.json',spec)
    revision=tmp_path/'worker_revisions/sensor3';project=revision/'project';shutil.copytree(base,project)
    candidate=copy.deepcopy(cfg);candidate['runtime']['sensor_postprocess_workers']=3
    (project/'candidate.yaml').write_text(cli.yaml.safe_dump(candidate))
    record={'base_run_sha256':cli.sha256(tmp_path/'run.json'),'options':{**spec['options'],'config':'candidate.yaml'},
        'files':cli.fingerprints(project),'versions':cli.versions()}
    atomic_json(revision/'revision.json',record)
    return e,cli,revision,record


def claim(cli,revision):
    return {'worker_revision':str(revision),'worker_revision_sha256':cli.sha256(revision/'revision.json')}


def test_rollout_uses_explicit_snapshot_preserving_frozen_original(tmp_path):
    e,cli,revision,record=setup(tmp_path)
    before=cli.fingerprints(tmp_path/'inputs/project');cache={}
    snapshot,spec=e.worker_context(tmp_path,claim(cli,revision),cli,cache)
    assert snapshot==revision/'project' and spec['options']['config']=='candidate.yaml'
    assert cli.fingerprints(tmp_path/'inputs/project')==before
    assert e.worker_context(tmp_path,{},cli,cache)[0]==tmp_path/'inputs/project'
    assert e.worker_context(tmp_path,claim(cli,revision),cli,cache)==(snapshot,spec)


@pytest.mark.parametrize('mutation',['dynamics','acceptance','source','bytes','identity'])
def test_rollout_rejects_unrelated_or_unrecorded_changes(tmp_path,mutation):
    e,cli,revision,record=setup(tmp_path);project=revision/'project';request=claim(cli,revision)
    if mutation=='dynamics':
        cfg=cli.yaml.safe_load((project/'candidate.yaml').read_text());cfg['runtime']['decimation']=2
        (project/'candidate.yaml').write_text(cli.yaml.safe_dump(cfg));record['files']=cli.fingerprints(project)
    elif mutation=='acceptance':record['options']['acceptance_policy']='relaxed.yaml'
    elif mutation=='source':
        (project/'source.py').write_text('# changed algorithm\n');record['files']=cli.fingerprints(project)
    elif mutation=='bytes':(project/'candidate.yaml').write_text('unregistered bytes')
    else:record['extra']='marker changed'
    atomic_json(revision/'revision.json',record)
    if mutation!='identity':request=claim(cli,revision)
    with pytest.raises(ValueError):e.worker_context(tmp_path,request,cli,{})


def test_unregistered_external_worker_revision_is_rejected(tmp_path):
    e,cli,revision,record=setup(tmp_path)
    outside=tmp_path/'external';shutil.copytree(revision,outside)
    with pytest.raises(ValueError,match='registered'):e.worker_context(tmp_path,claim(cli,outside),cli,{})


def test_sensor3_collection_config_changes_only_worker_budget():
    from piper_sim.config import load
    from piper_sim.mujoco_asset import configuration_fingerprint
    base=load(ROOT/'configs/clean_table_position_execution_rim25_damped.yaml')
    revised=load(ROOT/'configs/clean_table_position_execution_rim25_damped_sensor3.yaml')
    assert configuration_fingerprint(base)==configuration_fingerprint(revised)
    assert revised['runtime'].pop('sensor_postprocess_workers')==3
    base.pop('_config_path');revised.pop('_config_path')
    assert base==revised


def validation_setup(tmp_path,monkeypatch):
    e,cli=modules();atomic_json(tmp_path/'run.json',{'immutable':'run'})
    previous=tmp_path/'schedulers/previous';previous.mkdir(parents=True)
    script=previous/'controller.py';script.write_text('# frozen validator\n')
    atomic_json(previous/'controller.json',{'base_run_sha256':cli.sha256(tmp_path/'run.json'),
        'controller_sha256':cli.sha256(script)})
    first={'path':'/closed/1.h5','policy':'/policy.yaml','source_sha256':'digest1'}
    request=previous/'revalidation_request.json';atomic_json(request,{'episodes':[first]})
    atomic_json(previous/'revalidation_process.json',{'pid':999999999})
    atomic_json(previous/'revalidation_result.json',{'passed':True,'checked':1,'request_sha256':cli.sha256(request)})
    controller=tmp_path/'schedulers/current';controller.mkdir()
    atomic_json(controller/'handoff.json',{'previous_controller':str(previous)})
    generated=[]
    class Process:
        pid=999999998
        def __init__(self,command,**kwargs):
            request=Path(command[-1]);payload=cli.load_json(request);generated.extend(payload['episodes'])
            atomic_json(request.with_name('revalidation_result.json'),{'passed':True,
                'checked':len(payload['episodes']),'request_sha256':cli.sha256(request)})
        def poll(self):return 0
    monkeypatch.setattr(e.subprocess,'Popen',Process)
    rows=[{'accepted':True,'path':'/closed/1.h5','acceptance_policy':'/policy.yaml','verified':{'source_sha256':'digest1'}},
          {'accepted':True,'path':'/closed/2.h5','acceptance_policy':'/policy.yaml','verified':{'source_sha256':'digest2'}}]
    return e,cli,controller,previous,rows,generated


def test_completed_historical_checks_are_joined_with_only_uncovered_sources(tmp_path,monkeypatch):
    e,cli,controller,previous,rows,generated=validation_setup(tmp_path,monkeypatch)
    verifier,digest,count=e.start_revalidation(tmp_path,controller,rows,Path('/policy.yaml'),cli)
    assert count==2 and [x['path'] for x in generated]==['/closed/2.h5']
    assert verifier.poll()==0
    result=cli.load_json(controller/'revalidation_result.json')
    assert result['passed'] and result['checked']==2 and result['request_sha256']==digest
    parts=cli.load_json(controller/'validation_parts.json')
    assert parts['reused_episodes']==1 and parts['new_episodes']==1


def test_reused_request_mutation_fails_complete_validation(tmp_path,monkeypatch):
    e,cli,controller,previous,rows,generated=validation_setup(tmp_path,monkeypatch)
    verifier,_,_=e.start_revalidation(tmp_path,controller,rows,Path('/policy.yaml'),cli)
    atomic_json(previous/'revalidation_request.json',{'episodes':[]})
    assert verifier.poll()==2
    assert not cli.load_json(controller/'revalidation_result.json')['passed']


def test_changed_source_digest_cannot_reuse_earlier_evidence(tmp_path,monkeypatch):
    e,cli,controller,previous,rows,generated=validation_setup(tmp_path,monkeypatch)
    rows[0]['verified']['source_sha256']='changed source'
    with pytest.raises(ValueError,match='coverage mismatch'):
        e.start_revalidation(tmp_path,controller,rows,Path('/policy.yaml'),cli)
    assert not generated


def test_same_controller_resume_keeps_requests_and_reuses_completed_delta(tmp_path,monkeypatch):
    e,cli,controller,previous,rows,generated=validation_setup(tmp_path,monkeypatch)
    script=controller/'controller.py';script.write_text('# frozen controller\n')
    atomic_json(controller/'controller.json',{'base_run_sha256':cli.sha256(tmp_path/'run.json'),
        'controller_sha256':cli.sha256(script)})
    first,_,_=e.start_revalidation(tmp_path,controller,rows,Path('/policy.yaml'),cli)
    assert first.poll()==0
    manifest=cli.load_json(controller/'validation_parts.json')
    request_hashes={part['request']:cli.sha256(part['request']) for part in manifest['parts']}
    generated.clear()
    second,_,_=e.start_revalidation(tmp_path,controller,rows,Path('/policy.yaml'),cli)
    assert second.poll()==0 and not generated
    assert cli.load_json(controller/'validation_parts.json')['reused_episodes']==2
    assert all(cli.sha256(p)==digest for p,digest in request_hashes.items())


def test_interrupted_checksum_job_rechecks_uncovered_source(tmp_path,monkeypatch):
    e,cli,controller,previous,rows,generated=validation_setup(tmp_path,monkeypatch)
    (previous/'revalidation_result.json').unlink()
    verifier,_,_=e.start_revalidation(tmp_path,controller,rows,Path('/policy.yaml'),cli)
    assert [x['path'] for x in generated]==['/closed/1.h5','/closed/2.h5']
    assert verifier.poll()==0
