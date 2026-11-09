"""Publication tests use a stub task verdict; stream and file gates are real."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import yaml

from piper_sim import collection_audit as audit
from piper_sim.collection_jobs import sha256
from piper_sim.config import ROOT
from piper_sim.contracts import Command, Observation
from piper_sim.lerobot_export import _inspect
from piper_sim.recording import EpisodeWriter
from execution_fixtures import declaration, interval


@pytest.fixture
def source(tmp_path, monkeypatch):
    settings = yaml.safe_load((ROOT/'configs/collection/quality_policy.yaml').read_text())
    settings['dataset']['min_frames'] = 6  # Unit fixture, never a production profile.
    policy = tmp_path/'fixture-policy.yaml'
    policy.write_text(yaml.safe_dump(settings))
    cameras = settings['dataset']['required_cameras']
    metadata = {'fps': 30, 'teacher': {'motion_and_quality': settings},
                'camera_rig': {'cameras': {n: {'rgb': {'resolution': [16, 16]}} for n in cameras}}}
    executor = declaration()
    metadata['config'] = {'mujoco': {'position_execution': executor['profile'], 'controller': {'mode': 'zero_order_observed'}}}
    metadata['physics_model'] = {'position_execution': executor, 'controller': {'mode': 'zero_order_observed'}}
    metadata['teacher']['position_execution'] = executor
    path = tmp_path/'episode.h5'
    frames = np.random.default_rng(7).integers(10, 240, (7, 16, 16, 3), dtype=np.uint8)
    q = np.array([0., .2, -.2, 0., 0., 0., .04]*2)
    def observation(i, host):
        return Observation(q, host, i/30, {n: frames[i] for n in cameras},
                           {n: i/30 for n in cameras}, {'label': 'binding fixture 纸杯'})
    with EpisodeWriter(path, metadata, image_storage='shared', extras_storage='compressed') as writer:
        for i in range(6):
            command = q.copy();command[0] = i*.001
            after = observation(i+1, 100.+i+.5)
            after.extras.update(interval(executor, command, i/30, (i+1)/30))
            writer.append(observation(i, 100.+i), Command(command, 100.+i+.2, i), after)
        writer.close(success=True)
    monkeypatch.setattr(audit, 'audit_task', lambda *_: {'passed': True, 'violations': []})
    return path, policy


def inspect(path, policy):
    report = audit.audit_episode(path, policy=policy)
    assert report['passed'] and report['training_ready']
    assert report['source_sha256'] == report['audited_source_sha256'] == sha256(path)
    return report


def public_values(path):
    with h5py.File(path) as f:
        return {name: f[name][:].tobytes() if f[name].dtype.kind != 'O' else f[name][:].tolist()
                for prefix in ('observation', 'next_observation', 'action')
                for name in list_datasets(f[prefix])}


def list_datasets(group):
    names = []
    group.visititems(lambda name, item: names.append(item.name) if isinstance(item, h5py.Dataset) else None)
    return names


def test_correct_publication_preserves_all_public_values_and_portable_vds(source, tmp_path):
    path, policy = source
    original = public_values(path)
    report = inspect(path, policy)
    input_hash = report['source_sha256']
    sidecar = tmp_path/'audit.json'
    audit.save_audit(path, report, sidecar)
    assert report['source_sha256'] == sha256(path) != input_hash
    assert report['audited_source_sha256'] == input_hash
    assert json.loads(sidecar.read_text()) == report
    assert _inspect(path, require_quality=True, policy=policy)['audited_sha256'] == sha256(path)
    assert public_values(path) == original
    moved = tmp_path/'relocated'; moved.mkdir()
    path.rename(moved/path.name); sidecar.rename(moved/sidecar.name)
    assert _inspect(moved/path.name, require_quality=True, policy=policy)['audited_sha256'] == report['source_sha256']
    assert public_values(moved/path.name) == original


def test_report_cannot_be_published_on_a_different_episode_even_identical_bytes(source, tmp_path):
    path, policy = source
    report = inspect(path, policy)
    other = tmp_path/'other.h5'; shutil.copyfile(path, other)
    before = sha256(other)
    with pytest.raises(ValueError, match='different episode'):
        audit.save_audit(other, report, tmp_path/'audit.json')
    assert sha256(other) == before and not (tmp_path/'audit.json').exists()


@pytest.mark.parametrize('change', ['action', 'image', 'metadata', 'extras'])
def test_change_after_audit_is_rejected_before_any_publication_write(source, tmp_path, change):
    path, policy = source
    report = inspect(path, policy); original_report = copy.deepcopy(report)
    with h5py.File(path, 'r+') as f:
        if change == 'action': f['action/position'][2, 0] += .01
        elif change == 'image': f['observation/images/overhead'][2] = 0
        elif change == 'extras':
            # A new public JSON stream is a content change even with valid UTF-8.
            del f['observation/extras_json']
            f['observation/extras_json'] = np.array(['{}']*6, dtype=h5py.string_dtype('utf-8'))
        else: f.attrs['metadata'] = json.dumps({'fps': 30, 'changed': True})
    changed_hash = sha256(path)
    with pytest.raises(ValueError, match='changed after audit'):
        audit.save_audit(path, report, tmp_path/'audit.json')
    assert sha256(path) == changed_hash and report == original_report
    assert not (tmp_path/'audit.json').exists()


@pytest.mark.parametrize('replacement', [False, True])
def test_mutation_or_path_replacement_during_audit_cannot_pass(source, tmp_path, monkeypatch, replacement):
    path, policy = source
    def change_source(*_):
        if replacement:
            other = tmp_path/'replacement.h5'; shutil.copyfile(path, other)
            os.replace(other, path)  # Same bytes; the inspected inode is no longer the path.
        else:
            with path.open('ab') as raw:
                raw.write(b'fault injection after reading'); raw.flush(); os.fsync(raw.fileno())
        return {'passed': True, 'violations': []}
    monkeypatch.setattr(audit, 'audit_task', change_source)
    report = audit.audit_episode(path, policy=policy)
    assert not report['passed'] and not report['training_ready']
    assert 'source_changed_during_audit' in report['violations']


@pytest.mark.parametrize('alias', ['direct', 'hardlink'])
def test_sidecar_cannot_overwrite_the_hdf5(source, tmp_path, alias):
    path, policy = source; report = inspect(path, policy); before = sha256(path)
    output = path if alias == 'direct' else tmp_path/'alias.json'
    if alias == 'hardlink': os.link(path, output)
    with pytest.raises(ValueError, match='separate JSON file'):
        audit.save_audit(path, report, output)
    assert sha256(path) == before


def test_unbound_legacy_report_requires_reaudit(source, tmp_path):
    path, policy = source; report = inspect(path, policy); before = sha256(path)
    report.pop('source_sha256')
    with pytest.raises(ValueError, match='no bound source checksum'):
        audit.save_audit(path, report, tmp_path/'audit.json')
    assert sha256(path) == before


def test_other_reader_prevents_writable_publication(source, tmp_path):
    path, policy = source; report = inspect(path, policy); before = sha256(path)
    with path.open('rb') as reader:
        fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): audit.save_audit(path, report, tmp_path/'audit.json')
    assert sha256(path) == before and not (tmp_path/'audit.json').exists()


def test_atomic_sidecar_failure_denies_export_then_explicit_reaudit_recovers(source, tmp_path, monkeypatch):
    from piper_sim import collection_jobs
    path, policy = source; sidecar = tmp_path/'audit.json'
    audit.save_audit(path, inspect(path, policy), sidecar)
    old_sidecar = sidecar.read_bytes()
    report = inspect(path, policy)
    with monkeypatch.context() as context:
        def fail(*_): raise OSError('injected sidecar write failure')
        context.setattr(collection_jobs, 'atomic_json', fail)
        with pytest.raises(OSError, match='injected'):
            audit.save_audit(path, report, sidecar)
    assert sidecar.read_bytes() == old_sidecar
    with pytest.raises(ValueError, match='changed after its dataset audit'):
        _inspect(path, require_quality=True, policy=policy)
    audit.save_audit(path, inspect(path, policy), sidecar)
    assert _inspect(path, require_quality=True, policy=policy)['audited_sha256'] == sha256(path)


def test_publication_lock_covers_sidecar_commit_and_valid_downgrade(source, tmp_path, monkeypatch):
    from piper_sim import collection_jobs
    path, policy = source; sidecar = tmp_path/'audit.json'
    report = inspect(path, policy)
    original = collection_jobs.atomic_json
    def check_lock(output, value):
        with path.open('rb') as reader:
            with pytest.raises(BlockingIOError): fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
        original(output, value)
    monkeypatch.setattr(collection_jobs, 'atomic_json', check_lock)
    audit.save_audit(path, report, sidecar)
    report['passed'] = report['training_ready'] = False
    report['violations'].append('preflight_rgb_trajectory_mismatch')
    audit.save_audit(path, report, sidecar)
    assert sha256(path) == report['source_sha256']
    with pytest.raises(ValueError, match='passed RGB dataset audit'):
        _inspect(path, require_quality=True, policy=policy)
