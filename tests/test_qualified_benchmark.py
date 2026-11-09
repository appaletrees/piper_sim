import importlib.util
import json
from pathlib import Path
import sys
import pytest

from piper_sim.config import ROOT

spec = importlib.util.spec_from_file_location('qualified_benchmark', ROOT/'scripts/benchmark_qualified_collection.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_benchmark_counts_only_revalidated_episodes_and_includes_rejected_jobs(tmp_path, monkeypatch):
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 12345
            assert kwargs['start_new_session']
            folder = Path(command[command.index('--output')+1])
            folder.mkdir()
            (folder/'report.json').write_text(json.dumps({'training_ready': True}))
            (folder/'audit.json').write_text(json.dumps({'training_ready': True, 'violations': [],
                'checks': {'streams': {'frames': 300, 'duration_s': 10.}}}))
            (folder/'episode.h5').write_bytes(b'fake source for orchestrator fault injection')
            (folder/'lerobot_v2/meta').mkdir(parents=True)
            (folder/'lerobot_v2/meta/info.json').write_text('{}')
            Path(command[command.index('--profile-output')+1]).write_text('{}')
        def wait(self, **kwargs): return 0
        def poll(self): return 0
    def validate(path, *, require_quality):
        calls.append((path.parent.name, require_quality))
        if path.parent.name == 'seed_7': raise ValueError('source checksum changed')
    from piper_sim import lerobot_export
    monkeypatch.setattr(lerobot_export, '_inspect', validate)
    monkeypatch.setattr(cli.subprocess, 'Popen', Process)
    monkeypatch.setattr(cli, 'capture_identity', lambda args: {'test': 'frozen'})
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--seeds', '0', '7', '--output', str(tmp_path/'run'),
                                     '--min-free-gib', '.001', '--expected-episode-gib', '.001'])
    cli.main()
    report = json.loads((tmp_path/'run/report.json').read_text())
    assert calls == [('seed_0', True), ('seed_7', True)]
    assert report['qualified_episodes'] == 1 and len(report['jobs']) == 2
    assert report['jobs'][1]['validation_error'] == 'ValueError: source checksum changed'
    assert report['qualified_episodes_per_hour'] == 3600/report['wall_seconds']
    assert report['qualified_transitions_per_wall_second'] == 300/report['wall_seconds']


def test_paired_cohorts_use_same_seeds_and_bounded_pool_and_require_matching_hashes(tmp_path, monkeypatch):
    active, peaks, seen = {}, {}, []
    class Process:
        def __init__(self, command, **kwargs):
            folder = Path(command[command.index('--output')+1])
            self.group = folder.parent.name
            self.seed = int(command[command.index('--seed')+1])
            self.pid = 10000+len(seen)
            seen.append((self.group, self.seed))
            active[self.group] = active.get(self.group, 0)+1
            peaks[self.group] = max(peaks.get(self.group, 0), active[self.group])
        def poll(self): return 0
        def wait(self, **kwargs):
            active[self.group] -= 1
            return 0
    def validate(job, code):
        return {'seed': job['seed'], 'qualified': True, 'frames': 300, 'sim_seconds': 10.,
                'hdf5_bytes': 1000, 'trajectory_sha256': str(job['seed']),
                'objects_sha256': str(job['seed']), 'acceptance_policy_sha256': 'same'}
    monkeypatch.setattr(cli, 'validate_job', validate)
    monkeypatch.setattr(cli.subprocess, 'Popen', Process)
    monkeypatch.setattr(cli, 'capture_identity', lambda args: {'test': 'frozen'})
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--seeds', '0', '2', '7', '--compare-workers', '1', '3',
        '--output', str(tmp_path/'pair'), '--min-free-gib', '.001', '--expected-episode-gib', '.001'])
    assert cli.main() == 0
    assert seen == [(f'workers_{workers}', seed) for workers in (1, 3) for seed in (0, 2, 7)]
    assert peaks == {'workers_1': 1, 'workers_3': 3}
    report = json.loads((tmp_path/'pair/comparison.json').read_text())
    assert report['status'] == 'completed'
    assert report['comparisons'][0]['same_complete_qualified_cohort_and_state_hashes']
    assert report['comparisons'][0]['qualified_throughput_ratio'] == (
        report['groups'][1]['qualified_episodes_per_hour']/report['groups'][0]['qualified_episodes_per_hour'])
    report['groups'][1]['status'] = 'running'
    assert cli.compare_cohorts(report['groups'])[0]['qualified_throughput_ratio'] is None
    report['groups'][1]['status'] = 'completed'
    report['groups'][1]['jobs'][0]['objects_sha256'] = 'changed'
    assert cli.compare_cohorts(report['groups'])[0]['qualified_throughput_ratio'] is None


def test_changed_inputs_stop_before_launch_and_preserve_fingerprint(tmp_path, monkeypatch):
    identities = iter([{'version': 1}, {'version': 2}])
    monkeypatch.setattr(cli, 'capture_identity', lambda args: next(identities))
    def forbidden(*args, **kwargs): raise AssertionError('No worker should be started')
    monkeypatch.setattr(cli.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--seeds', '0', '--output', str(tmp_path/'run'),
        '--min-free-gib', '.001', '--expected-episode-gib', '.001'])
    with pytest.raises(ValueError, match='changed during the run'): cli.main()
    report = json.loads((tmp_path/'run/report.json').read_text())
    assert report['status'] == 'interrupted_or_failed' and not report['jobs']
    assert json.loads((tmp_path/'run/inputs.json').read_text()) == {'version': 1}


def test_worker_timeout_stops_group_and_never_publishes_a_rate(tmp_path, monkeypatch):
    stopped = []
    class Process:
        pid = 12345
        def __init__(self, *args, **kwargs): pass
        def poll(self): return None
    clock = iter([0., 2.])
    monkeypatch.setattr(cli.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(cli.subprocess, 'Popen', Process)
    monkeypatch.setattr(cli, 'stop_process_groups', lambda processes: stopped.extend(processes))
    monkeypatch.setattr(cli, 'capture_identity', lambda args: {'test': 'frozen'})
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--seeds', '0', '--timeout-s', '1',
        '--output', str(tmp_path/'run'), '--min-free-gib', '.001', '--expected-episode-gib', '.001'])
    with pytest.raises(TimeoutError, match='Worker seed 0 exceeded'): cli.main()
    report = json.loads((tmp_path/'run/report.json').read_text())
    assert report['status'] == 'interrupted_or_failed' and not report['jobs']
    assert 'qualified_episodes_per_hour' not in report and len(stopped) == 1


def test_real_input_fingerprint_does_not_require_unused_scipy(monkeypatch):
    from types import SimpleNamespace
    import shutil
    pytest.importorskip('pyarrow')
    if not all(shutil.which(key) for key in ('ffmpeg', 'ffprobe')):
        pytest.skip('Export executables required for real fingerprint smoke test')
    version = cli.importlib.metadata.version
    def without_scipy(name):
        if name.lower() == 'scipy': raise cli.importlib.metadata.PackageNotFoundError(name)
        return version(name)
    monkeypatch.setattr(cli.importlib.metadata, 'version', without_scipy)
    args = SimpleNamespace(config=str(ROOT/'configs/clean_table_workcell.yaml'),
        task_scene=str(ROOT/'configs/tasks/clean_table_rigid.yaml'),
        domain_randomization=str(ROOT/'configs/domain_randomization/clean_table.yaml'),
        collection_profile=str(ROOT/'configs/collection/clean_table.yaml'))
    fingerprint = cli.capture_identity(args)
    assert fingerprint['versions']['Pillow'] and fingerprint['versions']['pyarrow']
    assert fingerprint['export_executables']['ffmpeg']['sha256']
    assert fingerprint['files'][str(ROOT/'scripts/benchmark_qualified_collection.py')]
