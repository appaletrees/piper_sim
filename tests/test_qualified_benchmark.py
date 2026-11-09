import importlib.util
import json
from pathlib import Path
import sys

from piper_sim.config import ROOT

spec = importlib.util.spec_from_file_location('qualified_benchmark', ROOT/'scripts/benchmark_qualified_collection.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_benchmark_counts_only_revalidated_episodes_and_includes_rejected_jobs(tmp_path, monkeypatch):
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            folder = Path(command[command.index('--output')+1])
            folder.mkdir()
            (folder/'report.json').write_text(json.dumps({'training_ready': True}))
            (folder/'audit.json').write_text(json.dumps({'training_ready': True, 'violations': [],
                'checks': {'streams': {'frames': 300, 'duration_s': 10.}}}))
            (folder/'episode.h5').write_bytes(b'fake source for orchestrator fault injection')
            (folder/'lerobot_v2/meta').mkdir(parents=True)
            (folder/'lerobot_v2/meta/info.json').write_text('{}')
        def wait(self, **kwargs): return 0
        def poll(self): return 0
    def validate(path, *, require_quality):
        calls.append((path.parent.name, require_quality))
        if path.parent.name == 'seed_7': raise ValueError('source checksum changed')
    from piper_sim import lerobot_export
    monkeypatch.setattr(lerobot_export, '_inspect', validate)
    monkeypatch.setattr(cli.subprocess, 'Popen', Process)
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--seeds', '0', '7', '--output', str(tmp_path/'run'),
                                     '--min-free-gib', '.001'])
    cli.main()
    report = json.loads((tmp_path/'run/report.json').read_text())
    assert calls == [('seed_0', True), ('seed_7', True)]
    assert report['qualified_episodes'] == 1 and len(report['jobs']) == 2
    assert report['jobs'][1]['validation_error'] == 'ValueError: source checksum changed'
    assert report['qualified_episodes_per_hour'] == 3600/report['wall_seconds']
    assert report['qualified_transitions_per_wall_second'] == 300/report['wall_seconds']
