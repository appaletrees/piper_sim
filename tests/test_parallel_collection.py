import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from piper_sim.collection_jobs import BatchJournal
from piper_sim.config import ROOT

spec = importlib.util.spec_from_file_location('parallel_collection', ROOT/'scripts/collect_parallel_clean_table.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def args_fixture(path, resume=False):
    return SimpleNamespace(output=path, workers=2, seeds=[7, 9, 43], resume=resume,
        config='workcell.yaml', task_scene='scene.yaml', domain_randomization='dr.yaml',
        collection_profile='profile.yaml', grasp_mode='physical', timeout_s=7200.,
        nominal_fraction=0., validation_fraction=.2, states_only=True,
        skip_preflight=False, video=False, lerobot=False)


def result(seed):
    good = seed != 9
    return {'seed': seed, 'category': 'diagnostics' if good else 'rejected', 'accepted': False,
        'diagnostic_passed': good, 'path': f'/test/seed_{seed}.h5', 'split': 'train',
        'phase_wall_seconds': {'collection': 3.}, 'error': None if good else 'quality rejection'}


@pytest.mark.parametrize('workers', [1, 2, 3])
def test_shards_start_concurrently_validate_all_outcomes_and_resume_does_not_claim_throughput(tmp_path, monkeypatch, workers):
    args = args_fixture(tmp_path/'batch')
    args.workers = workers
    processes, checked = [], []
    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs['start_new_session'] is True
            index = command.index('--seeds')+1
            end = next(i for i in range(index, len(command)) if command[i].startswith('--'))
            self.seeds = [int(s) for s in command[index:end]]
            folder = Path(command[command.index('--output')+1])
            folder.mkdir(exist_ok=True)
            (folder/'run.json').write_text('{}')
            (folder/'summary.json').write_text(json.dumps({'episodes': [result(s) for s in self.seeds]}))
            processes.append(self)
            self.code = None
        def wait(self, **kwargs):
            assert len(processes) % workers == 0  # Every shard launched before waiting.
            self.code = 0 if any(s != 9 for s in self.seeds) else 2
            return self.code
        def poll(self): return self.code
    monkeypatch.setattr(cli.subprocess, 'Popen', Process)
    monkeypatch.setattr(cli, 'validate_completed', lambda r: checked.append(r['seed']))
    import piper_sim.collection_jobs as jobs
    monkeypatch.setattr(jobs, 'validate_completed', lambda r: None)
    with BatchJournal(args.output, {'same': 'identity'}) as journal:
        assert cli.collect(args, journal) == 0
    expected = [args.seeds[index::workers] for index in range(workers)]
    assert [p.seeds for p in processes] == expected
    assert checked == [seed for shard in expected for seed in shard]
    summary = json.loads((args.output/'summary.json').read_text())
    assert [r['seed'] for r in summary['episodes']] == [7, 9, 43]
    assert summary['diagnostic_passed'] == 2 and summary['accepted'] == 0 and summary['rejected'] == 1
    before = (args.output/'jobs/000000.json').read_bytes()
    args.resume = True
    with BatchJournal(args.output, {'same': 'identity'}, resume=True) as journal:
        assert cli.collect(args, journal) == 0
    assert (args.output/'jobs/000000.json').read_bytes() == before
    evidence = json.loads((args.output/'invocation_002.json').read_text())
    assert evidence['status'] == 'completed' and evidence['accepted_episodes_per_hour'] is None


def test_child_claims_are_not_published_when_revalidation_fails(tmp_path, monkeypatch):
    args = args_fixture(tmp_path/'batch')
    args.seeds, args.workers = [7], 1
    class Process:
        def __init__(self, command, **kwargs):
            folder = Path(command[command.index('--output')+1]); folder.mkdir()
            (folder/'summary.json').write_text(json.dumps({'episodes': [result(7)]}))
        def wait(self, **kwargs): return 0
        def poll(self): return 0
    monkeypatch.setattr(cli.subprocess, 'Popen', Process)
    def reject(_): raise ValueError('Source checksum changed')
    monkeypatch.setattr(cli, 'validate_completed', reject)
    with BatchJournal(args.output, {}) as journal:
        with pytest.raises(ValueError, match='checksum'):
            cli.collect(args, journal)
    assert not (args.output/'jobs/000000.json').exists()
    assert (args.output/'diagnostics.jsonl').read_text() == ''
    assert json.loads((args.output/'invocation_001.json').read_text())['status'] == 'interrupted_or_failed'


def test_cleanup_signals_only_owned_running_process_groups(monkeypatch):
    signals = []
    class Process:
        def __init__(self, pid, finished): self.pid, self.finished, self.waited = pid, finished, False
        def poll(self): return 0 if self.finished else None
        def wait(self, timeout=None):
            if not self.waited:
                self.waited = True
                raise cli.subprocess.TimeoutExpired('owned process', timeout)
            return -9
    running, finished = Process(12345, False), Process(54321, True)
    monkeypatch.setattr(cli.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    cli.stop_owned_workers([(None, None, None, running), (None, None, None, finished)])
    assert signals == [(12345, cli.signal.SIGTERM), (12345, cli.signal.SIGKILL)]


def test_worker_command_keeps_parent_export_and_uses_child_resume(tmp_path):
    args = args_fixture(tmp_path)
    args.lerobot = True
    (tmp_path/'run.json').write_text('{}')
    command = cli.worker_command(args, [7, 43], tmp_path)
    assert '--resume' in command and '--states-only' in command and '--lerobot' not in command
    assert command[command.index('--seeds')+1:command.index('--config')] == ['7', '43']
