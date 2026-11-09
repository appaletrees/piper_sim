#!/usr/bin/env python3
"""Run independent resumable collection shards, then validate and merge manifests.

Each shard uses the unchanged serial collector and its own durable journal. The
parent owns process groups, so interruption stops only its collectors/recorders.
No frame, time, physics, quality-policy or split semantics are changed.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from piper_sim.collection_jobs import BatchJournal, atomic_json, batch_identity, sha256, validate_completed
from piper_sim.config import ROOT


class CollectionTerminated(SystemExit):
    def __init__(self, signum):
        super().__init__(128+signum)
        self.signum = signum

    def __str__(self):
        return signal.Signals(self.signum).name


@contextmanager
def termination_guard():
    """Let SIGTERM unwind journals and owned process groups, just like Ctrl-C."""
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, _frame):
        # A repeated termination request must not interrupt descendant cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise CollectionTerminated(signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def worker_command(args, seeds, output):
    command = [sys.executable, str(ROOT/'scripts/collect_clean_table.py'), '--output', str(output),
               '--seeds', *map(str, seeds)]
    for name in ('config', 'task_scene', 'domain_randomization', 'collection_profile', 'grasp_mode',
                 'timeout_s', 'nominal_fraction', 'validation_fraction'):
        command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    for name in ('states_only', 'skip_preflight', 'video'):
        if getattr(args, name): command.append('--'+name.replace('_', '-'))
    if (output/'run.json').is_file(): command.append('--resume')
    return command


def stop_owned_workers(jobs):
    """Workers start new sessions; signal their own group including recorder children."""
    # A failed group leader may have exited while its recorder is still alive.
    owned = [process for _, _, _, process in jobs if process.poll() not in (0, 2)]
    for process in owned:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    for process in owned:
        if process.poll() is None:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: pass
        # Waiting for the leader alone misses descendants which ignore SIGTERM.
        # Only groups created by this invocation are addressed.
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def export_completed(args, results):
    from piper_sim.lerobot_export import export_lerobot_v2
    for split in ('train', 'validation'):
        paths = [r['path'] for r in results if r['accepted'] and r['split'] == split]
        if not paths: continue
        output = args.output/('lerobot_v2_'+split)
        checkpoint = args.output/('export_'+split+'.json')
        inputs = {str(path): sha256(path) for path in paths}
        if output.exists():
            expected = json.loads(checkpoint.read_text())
            files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
            if expected != {'inputs': inputs, 'files': files}:
                raise ValueError('Existing export changed: '+str(output))
        else:
            export_lerobot_v2(paths, output, require_quality=True)
            atomic_json(checkpoint, {'inputs': inputs, 'files': {
                str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})


def collect(args, journal):
    # Revalidate previous successes before touching published parent manifests.
    previous = [journal.completed(index) for index in range(len(args.seeds))]
    journal.publish([r for r in previous if r is not None], validation_fraction=args.validation_fraction)
    jobs, results = [], {i: r for i, r in enumerate(previous) if r is not None}
    started = time.monotonic()
    attempt = 1
    while (args.output/f'invocation_{attempt:03d}.json').exists(): attempt += 1
    evidence = {'status': 'running', 'workers': args.workers, 'seeds': args.seeds,
        'resumed': args.resume, 'started_unix': time.time(), 'completed_shards': [],
        'throughput_note': 'Fresh invocations include startup, every attempt, validation and optional export. '
            'Resume reuses earlier work, so its rate is deliberately omitted. Single cohorts are not success distributions.'}
    evidence_path = args.output/f'invocation_{attempt:03d}.json'
    atomic_json(evidence_path, evidence)
    try:
        for worker in range(min(args.workers, len(args.seeds))):
            seeds = args.seeds[worker::args.workers]
            folder = args.output/f'worker_{worker:02d}'
            if folder.exists() and not (folder/'run.json').exists():
                raise ValueError('Incomplete shard initialization; preserve it and choose a new parent output: '+str(folder))
            log = (args.output/f'worker_{worker:02d}_invocation_{attempt:03d}.log').open('w')
            try:
                process = subprocess.Popen(worker_command(args, seeds, folder), stdout=log,
                    stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, 'MUJOCO_GL': 'egl'}, start_new_session=True)
            except BaseException:
                log.close()
                raise
            jobs.append((seeds, folder, log, process))
        for seeds, folder, log, process in jobs:
            code = process.wait()
            log.close()
            if code not in (0, 2): raise RuntimeError(f'Collector exited {code}: {folder}')
            child = json.loads((folder/'summary.json').read_text())
            rows = child['episodes']
            if len(rows) != len(seeds) or {r['seed'] for r in rows} != set(seeds):
                raise ValueError('Missing or duplicate shard outcomes: '+str(folder))
            for result in rows:
                validate_completed(result)
                index = args.seeds.index(result['seed'])
                journal.save(index, result)
                results[index] = result
            ordered = [results[index] for index in sorted(results)]
            journal.publish(ordered, validation_fraction=args.validation_fraction)
            evidence['completed_shards'].append({'path': str(folder), 'returncode': code})
            atomic_json(evidence_path, evidence)
        ordered = [results[index] for index in range(len(args.seeds))]
        if args.lerobot: export_completed(args, ordered)
        seconds = time.monotonic()-started
        accepted = sum(r['accepted'] for r in ordered)
        evidence.update(status='completed', wall_seconds=seconds, accepted=accepted,
            attempted=len(ordered), diagnostic_passed=sum(r['diagnostic_passed'] for r in ordered),
            accepted_episodes_per_hour=None if args.resume else accepted*3600/seconds,
            export_requested=args.lerobot)
        return 0 if any(r['accepted'] or r['diagnostic_passed'] for r in ordered) else 2
    except BaseException as error:
        evidence.update(status='interrupted_or_failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        stop_owned_workers(jobs)
        for _, _, log, _ in jobs: log.close()
        evidence['elapsed_including_cleanup_s'] = time.monotonic()-started
        atomic_json(evidence_path, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--config', default='configs/clean_table_workcell.yaml')
    parser.add_argument('--task-scene', default='configs/tasks/clean_table_rigid.yaml')
    parser.add_argument('--domain-randomization', default='configs/domain_randomization/clean_table.yaml')
    parser.add_argument('--collection-profile', default='configs/collection/clean_table.yaml')
    parser.add_argument('--grasp-mode', choices=['physical'], default='physical')
    parser.add_argument('--states-only', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--lerobot', action='store_true')
    parser.add_argument('--timeout-s', type=float, default=7200.)
    parser.add_argument('--nominal-fraction', type=float, default=0.)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume: parser.error('Choose a new output or --resume')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be distinct and nonnegative')
    if not 0 <= args.nominal_fraction <= 1 or not 0 <= args.validation_fraction < 1 or args.timeout_s <= 0:
        parser.error('Invalid fraction or timeout')
    if args.lerobot and args.states_only: parser.error('--lerobot requires RGB')
    # Relative paths must refer to the same inputs in parent and worker processes.
    for name in ('config', 'task_scene', 'domain_randomization', 'collection_profile'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    identity = batch_identity(args)
    identity['files'][str(Path(__file__).resolve())] = sha256(__file__)
    with termination_guard():
        with BatchJournal(args.output, identity, resume=args.resume) as journal:
            return collect(args, journal)


if __name__ == '__main__':
    raise SystemExit(main())
