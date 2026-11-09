"""Exercise real process groups, including a child that ignores SIGTERM."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from piper_sim.config import ROOT


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group supervision')
def test_sigterm_records_interruption_and_stops_owned_descendants(tmp_path):
    child = tmp_path/'child.py'
    ready = tmp_path/'ready.json'
    child.write_text('''import json, os, signal, subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],
    stdout=subprocess.PIPE, text=True)
assert grandchild.stdout.readline().strip() == 'ready'
with open(sys.argv[1], 'w') as f: json.dump([os.getpid(), grandchild.pid], f)
time.sleep(60)
''')
    parent = tmp_path/'parent.py'
    parent.write_text(f'''import importlib.util, pathlib, sys
from types import SimpleNamespace
from piper_sim.collection_jobs import BatchJournal
spec=importlib.util.spec_from_file_location('cli', {str(ROOT/'scripts/collect_parallel_clean_table.py')!r})
cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
cli.worker_command=lambda *a: [sys.executable, {str(child)!r}, {str(ready)!r}]
args=SimpleNamespace(output=pathlib.Path({str(tmp_path/'batch')!r}), workers=1,
    seeds=[7], resume=False, validation_fraction=.2, lerobot=False)
with cli.termination_guard():
    with BatchJournal(args.output, {{}}) as journal: cli.collect(args,journal)
''')
    process = subprocess.Popen([sys.executable, str(parent)], env={**os.environ, 'PYTHONPATH': str(ROOT/'src')})
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    owned = []
    try:
        deadline = time.monotonic()+10
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.02)
        assert ready.exists(), 'Parent/child failed to start'
        owned = json.loads(ready.read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 128+signal.SIGTERM
        evidence = json.loads((tmp_path/'batch/invocation_001.json').read_text())
        assert evidence['status'] == 'interrupted_or_failed'
        assert 'SIGTERM' in evidence['error']
        assert not (tmp_path/'batch/jobs/000000.json').exists()
        assert unrelated.poll() is None
        for pid in owned:
            stat = Path(f'/proc/{pid}/stat')
            deadline = time.monotonic()+3
            while stat.exists() and stat.read_text().split(') ')[1][0] != 'Z' and time.monotonic() < deadline:
                time.sleep(.02)
            assert not stat.exists() or stat.read_text().split(') ')[1][0] == 'Z'
    finally:
        if process.poll() is None: process.kill()
        process.wait()
        if owned:
            try: os.killpg(owned[0], signal.SIGKILL)
            except ProcessLookupError: pass
        unrelated.terminate(); unrelated.wait()
