import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from piper_sim import collection_jobs
from piper_sim.collection_jobs import BatchJournal
from piper_sim.config import ROOT


spec = importlib.util.spec_from_file_location('resume_cli', ROOT/'scripts/collect_clean_table.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_resume_keeps_finished_job_and_preserves_interrupted_attempt(tmp_path, monkeypatch):
    root = tmp_path/'batch'
    calls = []
    def run(args, seed, output, **kwargs):
        calls.append((seed, output))
        output.mkdir()
        if seed == 2 and len(calls) == 2:
            (output/'episode.h5.partial').write_bytes(b'incomplete evidence')
            raise KeyboardInterrupt()
        return 0, {'training_ready': False}, {'passed': True, 'violations': [], 'checks': {}}
    monkeypatch.setattr(cli, 'run_episode', run)
    monkeypatch.setattr(collection_jobs, 'batch_identity', lambda args: {'settings': 'fixed'})
    monkeypatch.setattr(collection_jobs, 'validate_completed', lambda r, **kwargs: None)
    argv = ['collect', '--output', str(root), '--states-only', '--seeds', '1', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    partial = calls[-1][1]/'episode.h5.partial'
    assert partial.read_bytes() == b'incomplete evidence'
    monkeypatch.setattr(sys, 'argv', argv+['--resume'])
    assert cli.main() == 0
    assert [seed for seed, path in calls] == [1, 2, 2]
    assert calls[-1][1].name.endswith('_attempt_002')
    assert partial.read_bytes() == b'incomplete evidence'
    records = [json.loads(line) for line in (root/'diagnostics.jsonl').read_text().splitlines()]
    assert [r['seed'] for r in records] == [1, 2]
    assert not (root/'accepted.jsonl').read_text()


def test_identity_change_and_concurrent_resume_are_rejected(tmp_path):
    root = tmp_path/'batch'
    with BatchJournal(root, {'config': 'v1'}):
        with pytest.raises(BlockingIOError):
            with BatchJournal(root, {'config': 'v1'}, resume=True):
                pass
    with pytest.raises(ValueError, match='changed'):
        with BatchJournal(root, {'config': 'v2'}, resume=True):
            pass
    assert json.loads((root/'run.json').read_text()) == {'config': 'v1'}


def test_resume_invalid_evidence_preserves_published_manifest(tmp_path, monkeypatch):
    root = tmp_path/'batch'
    with BatchJournal(root, {}) as journal:
        journal.save(0, {'accepted': True, 'path': 'missing_episode.h5'})
    (root/'accepted.jsonl').write_text('previous manifest\n')
    args = argparse.Namespace(output=root, seeds=[1], validation_fraction=.2)
    def reject(result, **kwargs):
        raise ValueError('checksum mismatch')
    monkeypatch.setattr(collection_jobs, 'validate_completed', reject)
    with BatchJournal(root, {}, resume=True) as journal:
        with pytest.raises(ValueError, match='checksum'):
            cli.collect(args, journal)
    assert (root/'accepted.jsonl').read_text() == 'previous manifest\n'


def test_missing_audit_is_unverified_not_a_physical_failure(tmp_path, monkeypatch):
    root = tmp_path/'batch'
    monkeypatch.setattr(collection_jobs, 'batch_identity', lambda args: {})
    monkeypatch.setattr(cli, 'run_episode', lambda *args, **kwargs: (-1, {}, {}))
    monkeypatch.setattr(sys, 'argv', ['collect', '--output', str(root), '--states-only', '--seeds', '1'])
    assert cli.main() == 2
    assert not (root/'accepted.jsonl').read_text()
    assert not (root/'rejected.jsonl').read_text()
    assert json.loads((root/'unverified.jsonl').read_text())['category'] == 'unverified'
