#!/usr/bin/env python3
"""Measure qualified RGB throughput with a bounded queue or paired worker counts.

Every complete/failed recording and its audit is retained. No preflight is run;
wall time includes startup, all attempts, final audits, exports and parent checks.
Paired cohorts run sequentially using identical seeds and frozen source inputs.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from piper_sim.collection_jobs import atomic_json, batch_identity, sha256
from piper_sim.config import ROOT
from piper_sim.process_lifecycle import termination_guard, stop_process_groups


def capture_identity(args):
    value = batch_identity(args)
    from piper_sim.lerobot_export import check_dependencies
    check_dependencies()
    value['versions'].update({key: importlib.metadata.version(key) for key in ('Pillow', 'pyarrow', 'scipy')})
    value['export_executables'] = {key: {'path': str(Path(shutil.which(key)).resolve()),
        'sha256': sha256(shutil.which(key))} for key in ('ffmpeg', 'ffprobe')}
    value['runtime'] = {'python': sys.version, 'executable': sys.executable,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'environment': {key: os.environ.get(key) for key in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'MUJOCO_GL')}}
    for name in ('benchmark_qualified_collection.py', 'profile_collection.py'):
        path = ROOT/'scripts'/name
        value['files'][str(path)] = sha256(path)
    return value


def worker_command(args, seed, output, profile):
    command = [sys.executable, str(ROOT/'scripts/profile_collection.py'), '--profile-output', str(profile),
        '--record', '--rgb', '--lerobot', '--seed', str(seed), '--output', str(output)]
    for key in ('config', 'task_scene', 'domain_randomization', 'collection_profile'):
        command += ['--'+key.replace('_', '-'), getattr(args, key)]
    return command


def validate_job(job, code):
    folder = job['folder']
    entry = {'seed': job['seed'], 'returncode': code, 'path': str(folder.resolve()), 'qualified': False,
             'profile': str(job['profile'].resolve())}
    try:
        rollout = json.loads((folder/'report.json').read_text())
        audit = json.loads((folder/'audit.json').read_text())
        entry.update(error=rollout.get('error'), violations=audit.get('violations'))
        if code == 0 and rollout.get('training_ready') and audit.get('training_ready'):
            from piper_sim.lerobot_export import _inspect
            _inspect(folder/'episode.h5', require_quality=True)
            # The worker only returns zero after atomic export publication.
            if not (folder/'lerobot_v2/meta/info.json').is_file() or not job['profile'].is_file():
                raise ValueError('Missing completed LeRobot export or profile')
            streams, task = audit['checks']['streams'], audit['checks'].get('task', {})
            entry.update(qualified=True, frames=streams['frames'], sim_seconds=streams['duration_s'],
                hdf5_bytes=(folder/'episode.h5').stat().st_size,
                trajectory_sha256=streams.get('trajectory_sha256'), objects_sha256=task.get('objects_sha256'),
                acceptance_policy_sha256=audit.get('acceptance_policy', {}).get('sha256'))
    except (OSError, ValueError, KeyError) as error:
        entry['validation_error'] = type(error).__name__+': '+str(error)
    return entry


def run_cohort(args, output, workers, verify_inputs, identity):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'inputs.json', identity)
    report = {'status': 'running', 'workers': workers, 'seeds': args.seeds, 'scope': __doc__,
        'started_unix': time.time(), 'jobs': [], 'running_jobs': [],
        'limitations': ['Engineering regression cohort, not distribution-wide success.',
            'Other applications may share this host. Process and total-device GPU telemetry are distinguished.',
            'No state preflight; final consumer training-loader checks are separate from acquisition/export timing.']}
    jobs, active, pending = [], [], iter(args.seeds)
    exhausted = False
    def publish():
        report['running_jobs'] = [{'seed': j['seed'], 'pid': j['process'].pid} for j in active]
        atomic_json(output/'report.json', report)
    publish()
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                seed = next(pending, None)
                if seed is None:
                    exhausted = True
                    break
                verify_inputs()
                folder, profile = output/f'seed_{seed}', output/f'seed_{seed}_profile.json'
                log = (output/f'seed_{seed}.log').open('w')
                try:
                    process = subprocess.Popen(worker_command(args, seed, folder, profile), stdout=log,
                        stderr=subprocess.STDOUT, cwd=ROOT,
                        env={**os.environ, 'MUJOCO_GL': 'egl'}, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                job = {'seed': seed, 'folder': folder, 'profile': profile, 'log': log, 'process': process,
                       'started_monotonic': time.monotonic()}
                jobs.append(job); active.append(job)
                publish()
            for job in active:
                if job['process'].poll() is None and time.monotonic()-job['started_monotonic'] > args.timeout_s:
                    raise TimeoutError(f"Worker seed {job['seed']} exceeded {args.timeout_s:g} seconds")
            ready = [job for job in active if job['process'].poll() is not None]
            if not ready:
                if active: time.sleep(.05)
                continue
            for job in ready:
                code = job['process'].wait()
                job['log'].close()
                active.remove(job)
                entry = validate_job(job, code)
                report['jobs'].append(entry)
                report['jobs'].sort(key=lambda row: args.seeds.index(row['seed']))
                publish()
                print(json.dumps(entry), flush=True)
        verify_inputs()
        wall = time.perf_counter()-started
        accepted = [job for job in report['jobs'] if job['qualified']]
        report.update(status='completed', wall_seconds=wall, qualified_episodes=len(accepted),
            qualified_episodes_per_hour=len(accepted)*3600/wall,
            qualified_transitions_per_wall_second=sum(job['frames'] for job in accepted)/wall,
            qualified_hdf5_bytes=sum(job['hdf5_bytes'] for job in accepted))
    except BaseException as error:
        report.update(status='interrupted_or_failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        stop_process_groups([job['process'] for job in jobs])
        for job in jobs: job['log'].close()
        report['running_jobs'] = []
        report['elapsed_including_cleanup_s'] = time.perf_counter()-started
        atomic_json(output/'report.json', report)
    return report


def compare_cohorts(reports):
    baseline = reports[0]
    comparisons = []
    fields = ('trajectory_sha256', 'objects_sha256', 'acceptance_policy_sha256', 'frames', 'sim_seconds')
    for report in reports[1:]:
        old = {row['seed']: row for row in baseline['jobs']}
        new = {row['seed']: row for row in report['jobs']}
        same = (baseline.get('status') == report.get('status') == 'completed' and
                set(old) == set(new) == set(baseline['seeds']) and
                all(old[s]['qualified'] and new[s]['qualified'] and
                    all(old[s].get(key) is not None and old[s][key] == new[s].get(key) for key in fields)
                    for s in old))
        comparisons.append({'baseline_workers': baseline['workers'], 'candidate_workers': report['workers'],
            'same_complete_qualified_cohort_and_state_hashes': same,
            'qualified_throughput_ratio': report['qualified_episodes_per_hour']/baseline['qualified_episodes_per_hour']
                if same and baseline['qualified_episodes_per_hour'] else None,
            'scope': 'One paired cohort. No distribution-wide throughput or pixel-equality claim.'})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--workers', type=int, choices=range(1, 4))
    group.add_argument('--compare-workers', type=int, nargs='+', choices=range(1, 4))
    parser.add_argument('--min-free-gib', type=float, default=24.)
    parser.add_argument('--expected-episode-gib', type=float, default=10.,
                        help='Planning allowance per full RGB episode including export; actual usage may differ')
    parser.add_argument('--timeout-s', type=float, default=7200., help='Per worker timeout, including audit and export')
    for key, default in [('config', 'configs/clean_table_workcell.yaml'),
                         ('task-scene', 'configs/tasks/clean_table_rigid.yaml'),
                         ('domain-randomization', 'configs/domain_randomization/clean_table.yaml'),
                         ('collection-profile', 'configs/collection/clean_table.yaml')]:
        parser.add_argument('--'+key, default=default)
    args = parser.parse_args()
    args.output = args.output.resolve()
    counts = args.compare_workers or [args.workers or min(2, len(args.seeds))]
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Use distinct nonnegative seeds')
    if len(set(counts)) != len(counts) or max(counts) > len(args.seeds):
        parser.error('Worker counts must be distinct and no larger than the seed cohort')
    if args.compare_workers and len(counts) < 2: parser.error('Comparison requires at least two worker counts')
    if not all(math.isfinite(v) and v > 0 for v in (args.min_free_gib, args.expected_episode_gib)):
        parser.error('Disk allowances must be finite and positive')
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0: parser.error('Invalid worker timeout')
    if args.output.exists(): parser.error('Choose a new output directory; previous attempts are retained')
    ancestor = next(path for path in args.output.parents if path.exists())
    required = max(args.min_free_gib, 5+len(counts)*len(args.seeds)*args.expected_episode_gib)
    if shutil.disk_usage(ancestor).free < required*1024**3:
        parser.error(f'Insufficient disk: estimated allowance requires {required:g} GiB')
    for key in ('config', 'task_scene', 'domain_randomization', 'collection_profile'):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    identity = capture_identity(args)
    def verify_inputs():
        if capture_identity(args) != identity:
            raise ValueError('Benchmark source, assets, configuration or dependencies changed during the run')
    with termination_guard():
        if not args.compare_workers:
            report = run_cohort(args, args.output, counts[0], verify_inputs, identity)
            return 0 if report['qualified_episodes'] == len(args.seeds) else 2
        args.output.mkdir(parents=True)
        atomic_json(args.output/'inputs.json', identity)
        started = time.perf_counter()
        result = {'status': 'running', 'seeds': args.seeds, 'worker_counts': counts,
                  'estimated_allowance_gib': required, 'groups': [],
                  'dataset_note': 'Groups deliberately repeat the same seed families for benchmarking. '
                    'Do not concatenate them as independent training data or split copies across train/validation. '
                    'Use one cohort and the production family split when curating a dataset.'}
        atomic_json(args.output/'comparison.json', result)
        try:
            for count in counts:
                verify_inputs()
                result['groups'].append(run_cohort(args, args.output/f'workers_{count}', count, verify_inputs, identity))
                atomic_json(args.output/'comparison.json', result)
            result.update(status='completed', comparisons=compare_cohorts(result['groups']))
        except BaseException as error:
            result.update(status='interrupted_or_failed', error=type(error).__name__+': '+str(error))
            raise
        finally:
            result['total_wall_seconds_including_group_setup'] = time.perf_counter()-started
            atomic_json(args.output/'comparison.json', result)
        return 0 if all(r['qualified_episodes'] == len(args.seeds) for r in result['groups']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
