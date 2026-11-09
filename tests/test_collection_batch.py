import importlib.util
import json
import sys
from pathlib import Path

from piper_sim.config import ROOT

spec = importlib.util.spec_from_file_location('collection_cli', ROOT/'scripts/collect_clean_table.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def run_batch(tmp_path, monkeypatch, *, state_only=False, reject_preflight=False):
    calls = []
    def run(args, seed, output, *, nominal, rgb):
        calls.append(rgb)
        passed = not reject_preflight
        report = {'training_ready': passed and rgb, 'inventory': {'cup_count': 2}}
        audit = {'passed': passed, 'training_ready': passed and rgb, 'violations': [] if passed else ['slip'],
                 'checks': {'streams': {'rgb': rgb, 'trajectory_sha256': 'same'},
                            'task': {'objects_sha256': 'same'}}}
        return 0, report, audit
    monkeypatch.setattr(cli, 'run_episode', run)
    argv = ['collect', '--output', str(tmp_path/'batch'), '--seeds', '0']
    if state_only: argv += ['--states-only']
    monkeypatch.setattr(sys, 'argv', argv)
    code = cli.main()
    return code, calls, tmp_path/'batch'


def test_default_preflight_then_rgb_publish(monkeypatch, tmp_path):
    code, calls, root = run_batch(tmp_path, monkeypatch)
    assert code == 0 and calls == [False, True]
    accepted = json.loads((root/'accepted.jsonl').read_text())
    assert accepted['training_ready'] and accepted['rgb']
    assert accepted['split'] in ('train', 'validation')
    assert not (root/'diagnostics.jsonl').read_text()


def test_states_only_never_enters_training_manifest(monkeypatch, tmp_path):
    code, calls, root = run_batch(tmp_path, monkeypatch, state_only=True)
    assert code == 0 and calls == [False]
    assert (root/'accepted.jsonl').read_text() == ''
    assert json.loads((root/'diagnostics.jsonl').read_text())['diagnostic_passed']


def test_failed_seed_is_retained_and_never_rendered(monkeypatch, tmp_path):
    code, calls, root = run_batch(tmp_path, monkeypatch, reject_preflight=True)
    assert code == 2 and calls == [False]
    assert (root/'accepted.jsonl').read_text() == ''
    assert json.loads((root/'rejected.jsonl').read_text())['violations'] == ['slip']


def test_nominal_variants_share_one_split():
    assert len({cli.collection_split(seed, True, .2) for seed in range(100)}) == 1
    assert cli.collection_split(42, False, .2) == cli.collection_split(42, False, .2)


def test_preflight_rgb_mismatch_revokes_file_and_manifest(monkeypatch, tmp_path):
    import h5py
    from piper_sim.collection_jobs import sha256
    root = tmp_path/'batch'
    def run(args, seed, output, *, nominal, rgb):
        output.mkdir()
        with h5py.File(output/'episode.h5', 'w') as f:
            f.attrs['accepted'] = f.attrs['training_ready'] = rgb
            f.attrs['metadata'] = json.dumps({'collection_outcome': {'rollout_accepted': True}})
        report = {'training_ready': rgb, 'inventory': {'cup_count': 2}}
        audit = {'passed': True, 'training_ready': rgb, 'violations': [],
                 'episode': str((output/'episode.h5').resolve()), 'source_sha256': sha256(output/'episode.h5'),
                 'checks': {'streams': {'rgb': rgb, 'trajectory_sha256': 'same'},
                            'task': {'objects_sha256': 'changed' if rgb else 'original'}}}
        return 0, report, audit
    monkeypatch.setattr(cli, 'run_episode', run)
    monkeypatch.setattr(sys, 'argv', ['collect', '--output', str(root), '--seeds', '0'])
    assert cli.main() == 2
    assert not (root/'accepted.jsonl').read_text()
    rejected = json.loads((root/'rejected.jsonl').read_text())
    assert rejected['violations'] == ['preflight_rgb_trajectory_mismatch']
    with h5py.File(rejected['path']) as f:
        assert not f.attrs['accepted'] and not f.attrs['training_ready']
        assert not json.loads(f.attrs['dataset_audit'])['passed']
    assert json.loads(Path(rejected['report']).read_text())['error'] == 'preflight_rgb_trajectory_mismatch'
