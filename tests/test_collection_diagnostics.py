import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from piper_sim.collection_jobs import sha256
from piper_sim.collection_audit import load_acceptance_policy
from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter
from test_position_execution import fixture_episode


def test_component_diagnosis_preserves_failed_source_and_does_not_publish(tmp_path):
    spec = importlib.util.spec_from_file_location('diagnose_collection',
        Path(__file__).resolve().parents[1]/'scripts/diagnose_collection.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source, _ = fixture_episode(tmp_path)
    original = sha256(source)
    result = module.diagnose(source, 'configs/collection/quality_policy_tcp.yaml')
    assert result['recorded_success'] is False and result['training_ready'] is False
    assert result['checks']['position_execution']['passed'] is True
    assert result['checks']['task_and_motion']['passed'] is False
    assert result['checks']['streams_without_rgb_requirement']['passed'] is False
    assert result['source_unchanged'] and sha256(source) == original
    assert not source.with_name('audit.json').exists()
    with h5py.File(source) as f:
        assert not f.attrs['success']
        assert 'dataset_audit' not in f.attrs


@pytest.mark.parametrize('spike', [False, True])
def test_closed_failed_motion_is_recomputed_without_task_success(tmp_path, spike):
    spec = importlib.util.spec_from_file_location('diagnose_collection',
        Path(__file__).resolve().parents[1]/'scripts/diagnose_collection.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q = np.array([0., .2, -.2, 0., 0., 0., .04]*2)
    metadata = {'fps': 30, 'collection_initial_target': q.tolist()}
    source = tmp_path/'motion.h5'
    with EpisodeWriter(source, metadata) as writer:
        for i in range(7):
            action = q.copy()
            if spike and i == 4: action[0] += .1
            extras = {'teacher': {'holding': False, 'phase': 'test', 'object': 'synthetic'}}
            writer.append(Observation(q, i+10., i/30), Command(action, i+10.2, i),
                          Observation(q, i+10.5, (i+1)/30, extras=extras))
    original = sha256(source)
    with h5py.File(source) as f:
        assert not f.attrs['success']
        result = module.recorded_motion(f, metadata, load_acceptance_policy()[0])
    assert result['passed'] is not spike
    if spike: assert 'command_arm_jerk' in result['violations']
    assert sha256(source) == original
