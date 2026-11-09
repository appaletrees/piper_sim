"""Reader contract tests; publication fixture stubs task physics, not file gates."""
import json
import fcntl
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

from test_audit_publication import source
from piper_sim import collection_audit as audit
from piper_sim.collection_jobs import sha256
from piper_sim.dataset import sequence_windows
from piper_sim.lerobot_export import _inspect
from piper_sim.training_acceptance import require_self_contained


@pytest.fixture
def accepted(source):
    path, policy = source
    with h5py.File(path, 'r+') as f:
        meta = json.loads(f.attrs['metadata']); meta['task'] = 'stack cups, then clear the table'
        meta['teacher']['private_reward'] = 999
        f.attrs['metadata'] = json.dumps(meta)
    report = audit.audit_episode(path, policy=policy)
    assert report['passed']
    audit.save_audit(path, report, path.with_name('audit.json'))
    return path, policy


def test_raw_diagnostics_require_explicit_opt_out(source):
    path, policy = source
    with pytest.raises(ValueError, match='passed RGB dataset audit'):
        next(sequence_windows(path, horizon=3, policy=policy))
    windows = list(sequence_windows(path, horizon=3, require_quality=False))
    assert len(windows) == 4
    assert all(len(w['state']) == 3 for w in windows)
    with pytest.raises(ValueError, match='policy cannot be applied'):
        next(sequence_windows(path, require_quality=False, policy=policy))


def test_accepted_windows_preserve_every_channel_clock_pixel_and_exclude_privileged_fields(accepted):
    path, policy = accepted
    windows = list(sequence_windows(path, horizon=3, policy=policy))
    assert len(windows) == 4
    expected_keys = {'state', 'action', 'next_state', 'timestamp', 'sim_timestamp', 'next_sim_timestamp',
                     'action_timestamp', 'action_sequence', 'images', 'image_timestamps', 'task'}
    with h5py.File(path) as f:
        for start, window in enumerate(windows):
            assert set(window) == expected_keys
            assert window['task'] == 'stack cups, then clear the table'
            for key, field in {'state': 'observation/state', 'action': 'action/position',
                    'next_state': 'next_observation/state', 'timestamp': 'observation/timestamp',
                    'sim_timestamp': 'observation/sim_time', 'next_sim_timestamp': 'next_observation/sim_time',
                    'action_timestamp': 'action/timestamp', 'action_sequence': 'action/sequence'}.items():
                np.testing.assert_array_equal(window[key], f[field][start:start+3])
            for name, pixels in window['images'].items():
                np.testing.assert_array_equal(pixels, f['observation/images/'+name][start:start+3])
                np.testing.assert_array_equal(window['image_timestamps'][name], window['sim_timestamp'])
            assert not np.array_equal(window['state'], window['action'])
            assert not np.array_equal(window['timestamp'], window['sim_timestamp'])
    spaced = list(sequence_windows(path, horizon=3, stride=2, policy=policy))
    assert [w['action_sequence'].tolist() for w in spaced] == [[0, 1, 2], [2, 3, 4]]
    assert list(sequence_windows(path, horizon=7, policy=policy)) == []


@pytest.mark.parametrize('fault', ['rejected', 'unverified', 'checksum', 'sidecar', 'wrong_policy'])
def test_training_reader_and_export_share_rejection(accepted, tmp_path, fault):
    path, policy = accepted
    if fault == 'sidecar': path.with_name('audit.json').unlink()
    elif fault == 'wrong_policy':
        settings = yaml.safe_load(policy.read_text()); settings['dataset']['min_frames'] += 1
        other = tmp_path/'different-policy.yaml'; other.write_text(yaml.safe_dump(settings)); policy = other
    else:
        with h5py.File(path, 'r+') as f:
            if fault == 'rejected': f.attrs['success'] = False
            elif fault == 'unverified': del f.attrs['training_ready']
            else: f['observation/images/overhead'][2] = 0
    for read in (lambda: next(sequence_windows(path, horizon=3, policy=policy)),
                 lambda: _inspect(path, require_quality=True, policy=policy)):
        with pytest.raises((ValueError, OSError)): read()


@pytest.mark.parametrize('fault', ['missing', 'wrong_count', 'not_supported'])
def test_green_flags_without_complete_execution_evidence_are_rejected(accepted, fault):
    path, policy = accepted
    with h5py.File(path, 'r+') as f:
        report = json.loads(f.attrs['dataset_audit'])
        if fault == 'missing': report['checks'].pop('position_execution')
        elif fault == 'wrong_count': report['checks']['position_execution']['checked_intervals'] -= 1
        else: report['checks']['position_execution']['position_only_execution_supported'] = False
        f.attrs['dataset_audit'] = json.dumps(report)
    # Even matching file checksums and v4 success flags cannot supply a missing
    # required verdict. This is a synthetic publication, not a physical audit.
    path.with_name('audit.json').write_text(json.dumps({**report, 'source_sha256': sha256(path)}))
    for read in (lambda: next(sequence_windows(path, horizon=3, policy=policy)),
                 lambda: _inspect(path, require_quality=True, policy=policy)):
        with pytest.raises(ValueError, match='complete position execution evidence'):
            read()


def test_verification_and_iteration_share_one_locked_handle_and_hash_once(accepted, monkeypatch):
    path, policy = accepted
    calls = []; original = audit._descriptor_sha256
    def counted(descriptor):
        calls.append(descriptor)
        return original(descriptor)
    monkeypatch.setattr(audit, '_descriptor_sha256', counted)
    generator = sequence_windows(path, horizon=2, policy=policy)
    next(generator)
    with path.open('r+b') as writer:
        with pytest.raises(BlockingIOError): fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(list(generator)) == 4
    assert len(calls) == 1
    with path.open('r+b') as writer: fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    generator = sequence_windows(path, horizon=2, policy=policy)
    next(generator); generator.close()
    with path.open('r+b') as writer: fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_path_replacement_cannot_change_data_mid_sequence(accepted, tmp_path):
    path, policy = accepted
    generator = sequence_windows(path, horizon=2, policy=policy)
    next(generator)
    replacement = tmp_path/'replacement.h5'; shutil.copyfile(path, replacement)
    os.replace(replacement, path)
    with pytest.raises(ValueError, match='path changed during sequence reading'):
        next(generator)


@pytest.mark.parametrize('kind', ['external_link', 'soft_link', 'virtual', 'raw_storage'])
def test_external_dependencies_cannot_bypass_a_whole_file_training_checksum(accepted, tmp_path, kind):
    path, policy = accepted
    other = tmp_path/'other.h5'
    with h5py.File(other, 'w') as f: f['values'] = np.arange(6, dtype=np.uint8)
    with h5py.File(path, 'r+') as f:
        if kind == 'external_link': f['outside'] = h5py.ExternalLink(str(other), '/values')
        elif kind == 'soft_link': f['alias'] = h5py.SoftLink('/action/position')
        elif kind == 'virtual':
            layout = h5py.VirtualLayout(shape=(6,), dtype=np.uint8)
            layout[:] = h5py.VirtualSource(str(other), '/values', shape=(6,))
            f.create_virtual_dataset('outside', layout)
        else:
            external = tmp_path/'values.bin'
            f.create_dataset('outside', shape=(6,), dtype=np.uint8,
                             external=[(str(external), 0, h5py.h5f.UNLIMITED)])[:] = np.arange(6)
    fingerprint = sha256(path)
    if kind == 'raw_storage':
        with external.open('r+b') as raw: raw.write(bytes([255]))
    else:
        with h5py.File(other, 'r+') as f: f['values'][0] = 255
    assert sha256(path) == fingerprint
    if kind != 'soft_link':
        # Backing values changed but the container's checksum did not.
        with h5py.File(path) as f: assert int(f['outside'][0]) == 255
    with h5py.File(path) as f, pytest.raises(ValueError, match='HDF5'):
        require_self_contained(f)
    report = audit.audit_episode(path, policy=policy)
    assert not report['passed'] and not report['training_ready']
    assert any('HDF5' in reason for reason in report['violations'])
    for read in (lambda: next(sequence_windows(path, horizon=3, policy=policy)),
                 lambda: _inspect(path, require_quality=True, policy=policy)):
        with pytest.raises(ValueError, match='HDF5'): read()


@pytest.mark.parametrize('options', [{'horizon': 0}, {'stride': -1}, {'horizon': True},
                                   {'stride': 1.5}, {'require_quality': 'false'}])
def test_invalid_window_options_are_rejected_before_reading(options):
    with pytest.raises(ValueError): next(sequence_windows('/missing/episode.h5', **options))


def test_reader_import_does_not_load_simulators_encoders_or_training_frameworks():
    code = "from piper_sim.dataset import sequence_windows; import sys; assert not ({'mujoco','isaaclab','torch','pyarrow','PIL'} & set(sys.modules))"
    subprocess.run([sys.executable, '-c', code], check=True)
