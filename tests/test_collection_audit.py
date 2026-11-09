import json

import h5py
import numpy as np
import pytest
import yaml

from piper_sim.collection_audit import audit_episode, audit_streams, save_audit
from piper_sim.config import ROOT
from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter


def fixture_episode(tmp_path):
    settings = yaml.safe_load((ROOT/'configs/collection/clean_table.yaml').read_text())
    settings['dataset']['min_frames'] = 6
    rig = {'cameras': {n: {'rgb': {'resolution': [16, 16]}}
                       for n in settings['dataset']['required_cameras']}}
    metadata = {'fps': 30, 'camera_rig': rig, 'teacher': {'motion_and_quality': settings}}
    path = tmp_path/'episode.h5'
    rng = np.random.default_rng(42)
    frames = [rng.integers(10, 240, (16, 16, 3), dtype=np.uint8) for _ in range(7)]
    q = np.array([0, .2, -.2, 0, 0, 0, .04]*2)
    def obs(i, host):
        return Observation(q, host, i/30, {n: frames[i] for n in rig['cameras']},
                           {n: i/30 for n in rig['cameras']}, {})
    with EpisodeWriter(path, metadata) as writer:
        for i in range(6):
            writer.append(obs(i, 100+i), Command(q, 100+i+.2, i), obs(i+1, 100+i+.5))
    with h5py.File(path, 'r+') as f: f.attrs['success'] = True
    return path, metadata, settings


def test_all_camera_frames_and_transition_alignment(tmp_path):
    path, meta, settings = fixture_episode(tmp_path)
    with h5py.File(path) as f:
        result = audit_streams(f, meta, settings)
    assert result['passed']
    assert len(result['cameras']) == 3
    assert all(c['checked_images'] == 12 for c in result['cameras'].values())
    with h5py.File(path, 'r+') as f:
        f['observation/images/left_wrist'][3] = f['observation/images/left_wrist'][4]
    with h5py.File(path) as f:
        result = audit_streams(f, meta, settings)
    assert 'rgb_transition_discontinuity/left_wrist' in result['violations']


@pytest.mark.parametrize('field,value,reason', [
    ('observation/state', [0]*14, 'state_transition_discontinuity'),
    ('observation/image_timestamps/overhead', 99., 'rgb_state_desynchronized'),
    ('action/sequence', 99, 'action_sequence_gap'),
    ('action/position', [100]*14, 'action_out_of_bounds'),
    ('action/timestamp', 0., 'action_clock'),
])
def test_corrupted_recording_is_rejected(tmp_path, field, value, reason):
    path, meta, settings = fixture_episode(tmp_path)
    with h5py.File(path, 'r+') as f: f[field][3] = value
    with h5py.File(path) as f, pytest.raises(ValueError, match=reason):
        audit_streams(f, meta, settings)


def test_solid_coloured_and_frozen_camera_are_rejected(tmp_path):
    path, meta, settings = fixture_episode(tmp_path)
    with h5py.File(path, 'r+') as f:
        for prefix in ('observation', 'next_observation'):
            f[prefix+'/images/right_wrist'][:] = [255, 0, 0]
    with h5py.File(path) as f: result = audit_streams(f, meta, settings)
    assert 'uninformative_rgb/right_wrist' in result['violations']
    assert 'frozen_rgb/right_wrist' in result['violations']


def test_success_flag_without_task_evidence_does_not_pass(tmp_path):
    path, _, settings = fixture_episode(tmp_path)
    policy = yaml.safe_load((ROOT/'configs/collection/quality_policy.yaml').read_text())
    policy['dataset']['min_frames'] = settings['dataset']['min_frames']
    policy_path = tmp_path/'test_policy.yaml'
    policy_path.write_text(yaml.safe_dump(policy))
    result = audit_episode(path, policy=policy_path)
    assert not result['passed'] and not result['training_ready']
    assert result['checks']['streams']['passed']
    assert any(v.startswith('task/KeyError') for v in result['violations'])
    save_audit(path, result, tmp_path/'audit.json')
    with h5py.File(path) as f:
        assert not f.attrs['accepted'] and not f.attrs['training_ready']
        assert not json.loads(f.attrs['dataset_audit'])['passed']


def test_incomplete_and_state_only_cannot_be_training_ready(tmp_path):
    path, meta, settings = fixture_episode(tmp_path)
    with h5py.File(path, 'r+') as f:
        for prefix in ('observation', 'next_observation'):
            del f[prefix+'/images']
            del f[prefix+'/image_timestamps']
    with h5py.File(path) as f:
        assert not audit_streams(f, meta, settings)['passed']
        assert audit_streams(f, meta, settings, require_rgb=False)['passed']
    assert not audit_episode(path, require_rgb=False)['training_ready']
    with h5py.File(path, 'r+') as f: f.attrs['complete'] = False
    assert 'incomplete_or_unknown_episode' in audit_episode(path)['violations'][0]


def test_episode_cannot_lower_its_own_production_acceptance_threshold(tmp_path):
    path, meta, _ = fixture_episode(tmp_path)
    # The source declares six frames sufficient. The separate production
    # policy still requires 150 and must reject it regardless of its success flag.
    with h5py.File(path, 'r+') as f:
        meta['teacher']['motion_and_quality']['dataset']['min_frames'] = 1
        meta['teacher']['motion_and_quality']['quality']['measured_jerk_rad_s3'] = 1e20
        f.attrs['metadata'] = json.dumps(meta)
    result = audit_episode(path)
    assert not result['training_ready']
    assert 'streams/ValueError: insufficient_frames' in result['violations']
    assert result['settings']['quality']['measured_jerk_rad_s3'] == 200
    assert result['acceptance_policy']['sha256']


def test_nonfinite_external_policy_is_rejected(tmp_path):
    path, _, _ = fixture_episode(tmp_path)
    policy = yaml.safe_load((ROOT/'configs/collection/quality_policy.yaml').read_text())
    policy['quality']['measured_jerk_rad_s3'] = float('inf')
    supplied = tmp_path/'bad_policy.yaml'; supplied.write_text(yaml.safe_dump(policy))
    result = audit_episode(path, policy=supplied)
    assert not result['passed']
    assert 'invalid policy limit' in result['violations'][0]


@pytest.mark.parametrize('content', ['', '- wrong schema', 'dataset: [unclosed'])
def test_malformed_policy_returns_rejection(tmp_path, content):
    path, _, _ = fixture_episode(tmp_path)
    policy = tmp_path/'malformed.yaml'; policy.write_text(content)
    result = audit_episode(path, policy=policy)
    assert not result['passed'] and not result['training_ready']
    assert result['violations'][0].startswith('ValueError:')
