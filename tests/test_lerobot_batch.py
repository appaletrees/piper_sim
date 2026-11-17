"""Checkpoint recovery and global LeRobot indices must survive sharded export."""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip('pyarrow')
import pyarrow.parquet as pq
from piper_sim import lerobot_batch as batch
from piper_sim.collection_audit import AUDIT_VERSION, load_acceptance_policy
from piper_sim.config import ROOT
from piper_sim.lerobot_export import _sha256, export_lerobot_v2
from test_lerobot_export import episode

pytestmark=pytest.mark.skipif(not shutil.which('ffmpeg'),reason='CPU encoder is optional')


def accepted(path, index, task='clean table', policy=None):
    episode(path,task=task)
    audit={'passed':True,'training_ready':True,'violations':[],
           'schema_version':AUDIT_VERSION,'acceptance_policy':load_acceptance_policy(policy)[1],
           'checks':{'position_execution':{'passed':True,'violations':[],
                     'position_only_execution_supported':True,'checked_intervals':3}}}
    with h5py.File(path,'r+') as f:
        f.attrs.update(accepted=True,training_ready=True,success=True,
                       dataset_audit=json.dumps(audit),dataset_audit_sidecar='audit.json')
    digest=_sha256(path)
    path.with_name('audit.json').write_text(json.dumps({**audit,'source_sha256':digest}))
    return {'index':index,'path':str(path),'accepted':True,'training_ready':True,
            'verified':{'source_sha256':digest,'frames':3}}


def sources(tmp_path):
    rows=[]
    for i,task in enumerate(['task A','task B','task A']):
        folder=tmp_path/f'source{i}';folder.mkdir()
        rows.append(accepted(folder/'episode.h5',i+12,task))
    manifest=tmp_path/'success.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in reversed(rows)))
    return manifest,rows


def test_batch_checkpoint_recovery_and_merged_indices_match_direct_export(tmp_path):
    manifest,rows=sources(tmp_path);output=tmp_path/'lerobot';policy=ROOT/'configs/collection/quality_policy.yaml'
    result=batch.run(manifest,output,policy,workers=2,min_free_gib=1)
    assert result['episodes']==3 and result['frames']==9 and result['videos']==3
    info=json.loads((output/'meta/info.json').read_text())
    assert info['total_tasks']==2 and info['fps']==30 and info['codebase_version']=='v2.1'
    direct=tmp_path/'direct';export_lerobot_v2([x['path'] for x in rows],direct,require_quality=True)
    for index in range(3):
        file=f'data/chunk-000/episode_{index:06d}.parquet'
        merged=pq.read_table(output/file);expected=pq.read_table(direct/file)
        assert merged.equals(expected)
        extra=f'extras/chunk-000/episode_{index:06d}.jsonl'
        assert (output/extra).read_bytes()==(direct/extra).read_bytes()
    stats=json.loads((output/'meta/stats.json').read_text())
    expected=json.loads((direct/'meta/stats.json').read_text())
    for key in stats:
        for field in stats[key]:np.testing.assert_allclose(stats[key][field],expected[key][field])
    provenance=[json.loads(line) for line in (output/'meta/sim_episodes.jsonl').read_text().splitlines()]
    assert [x['source_manifest_index'] for x in provenance]==[12,13,14]
    work=tmp_path/'lerobot.conversion'
    assert json.loads((work/'status.json').read_text())['status']=='completed'
    before=(work/'shards/000000/meta/export.json').stat().st_mtime_ns
    shutil.rmtree(output)
    batch.run(manifest,output,policy,workers=1,min_free_gib=1)
    assert (work/'shards/000000/meta/export.json').stat().st_mtime_ns==before
    assert all(_sha256(Path(row['path']))==row['verified']['source_sha256'] for row in rows)
    # A missing receipt retains the uncommitted artifacts and re-encodes safely.
    (work/'receipts/000000.json').unlink();batch.export_one(work,0)
    assert batch.verify_shard(work,0,rows[0])
    assert len(list((work/'uncommitted').iterdir()))==1
    # A changed artifact is never accepted as a resumable checkpoint.
    p=work/'shards/000000/data/chunk-000/episode_000000.parquet'
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='Checkpoint integrity'):batch.verify_shard(work,0,rows[0])


def test_manifest_rejects_failed_duplicate_and_missing_sources(tmp_path):
    manifest,rows=sources(tmp_path)
    rows[0]['accepted']=False
    manifest.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='independently verified'):batch.manifest_rows(manifest)
    rows[0]['accepted']=True
    manifest.write_text((json.dumps(rows[0])+'\n')*2)
    with pytest.raises(ValueError,match='Duplicate'):batch.manifest_rows(manifest)
    rows[0]['path']=str(tmp_path/'missing.h5');manifest.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='Missing closed'):batch.manifest_rows(manifest)


def test_failed_quality_keeps_checkpoint_job_but_never_publishes_dataset(tmp_path):
    manifest,rows=sources(tmp_path)
    with h5py.File(rows[0]['path'],'r+') as f:f.attrs['accepted']=False
    output=tmp_path/'bad'
    with pytest.raises(RuntimeError,match='nothing silently skipped'):
        batch.run(manifest,output,ROOT/'configs/collection/quality_policy.yaml',workers=1,min_free_gib=1)
    status=json.loads((tmp_path/'bad.conversion/status.json').read_text())
    assert not output.exists() and status['status']=='failed' and status['failed']
    assert not list((tmp_path/'bad.conversion/receipts').glob('*.json'))


def test_mixed_historical_policies_require_explicit_whitelist(tmp_path):
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    primary=ROOT/'configs/collection/quality_policy.yaml'
    historical=ROOT/'configs/collection/quality_policy_practical_v2.yaml'
    rows=[accepted(first/'episode.h5',1,policy=primary),
          accepted(second/'episode.h5',2,policy=historical)]
    manifest=tmp_path/'success.jsonl';manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    output=tmp_path/'mixed'
    with pytest.raises(ValueError,match='operator whitelist'):
        batch.run(manifest,output,primary,min_free_gib=1)
    result=batch.run(manifest,output,primary,additional_policies=[historical],min_free_gib=1)
    assert result['episodes']==2 and len(result['acceptance_policy_whitelist'])==2
    assert result['quality_required'] is True
