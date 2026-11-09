#!/usr/bin/env python3
"""Measure end-to-end qualified throughput of independent physical RGB workers.

Each worker retains its complete or failed recording, audit and profiler output.
No preflight is run in this benchmark; all attempts, final audits and exports are
included in wall time. Quality is revalidated before any job counts as qualified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-free-gib', type=float, default=24.)
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Use one or two distinct nonnegative seeds')
    if args.min_free_gib <= 0 or shutil.disk_usage(args.output.parent).free < args.min_free_gib*1024**3:
        parser.error('Insufficient disk reserve for the requested benchmark')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'workers': len(args.seeds), 'seeds': args.seeds,
        'scope': __doc__, 'started_unix': time.time(), 'jobs': [],
        'limitations': ['Single concurrent cohort, not a distribution-wide success estimate.',
            'Other applications may share this host. Compare only like-for-like cohorts for speedup claims.',
            'Final consumer training-loader checks are separate from this acquisition/export timing.']}
    jobs = []
    started = time.perf_counter()
    try:
        for seed in args.seeds:
            folder = args.output/f'seed_{seed}'
            profile = args.output/f'seed_{seed}_profile.json'
            log = (args.output/f'seed_{seed}.log').open('w')
            command = [sys.executable, 'scripts/profile_collection.py', '--profile-output', str(profile),
                '--record', '--rgb', '--lerobot', '--seed', str(seed),
                '--domain-randomization', 'configs/domain_randomization/clean_table.yaml', '--output', str(folder)]
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'MUJOCO_GL': 'egl'})
            except BaseException:
                log.close()
                raise
            jobs.append((seed, folder, profile, log, process))
        for seed, folder, profile, log, process in jobs:
            code = process.wait()
            log.close()
            entry = {'seed': seed, 'returncode': code, 'path': str(folder.resolve()), 'qualified': False}
            try:
                rollout = json.loads((folder/'report.json').read_text())
                audit = json.loads((folder/'audit.json').read_text())
                entry.update(error=rollout.get('error'), violations=audit.get('violations'))
                if code == 0 and rollout.get('training_ready') and audit.get('training_ready'):
                    from piper_sim.lerobot_export import _inspect
                    _inspect(folder/'episode.h5', require_quality=True)
                    # --lerobot returns success only after its atomic export publication.
                    if not (folder/'lerobot_v2/meta/info.json').is_file():
                        raise ValueError('Missing completed LeRobot export')
                    entry.update(qualified=True, frames=audit['checks']['streams']['frames'],
                        sim_seconds=audit['checks']['streams']['duration_s'],
                        hdf5_bytes=(folder/'episode.h5').stat().st_size)
            except (OSError, ValueError, KeyError) as error:
                entry['validation_error'] = type(error).__name__+': '+str(error)
            report['jobs'].append(entry)
            print(json.dumps(entry), flush=True)
        wall = time.perf_counter()-started
        accepted = [job for job in report['jobs'] if job['qualified']]
        report.update(status='completed', wall_seconds=wall, qualified_episodes=len(accepted),
            qualified_episodes_per_hour=len(accepted)*3600/wall,
            qualified_transitions_per_wall_second=sum(job['frames'] for job in accepted)/wall,
            qualified_hdf5_bytes=sum(job['hdf5_bytes'] for job in accepted))
    except BaseException as error:
        report.update(status='interrupted_or_failed', error=type(error).__name__+': '+str(error))
        for _, _, _, _, process in jobs:
            if process.poll() is None: process.terminate()
        for _, _, _, _, process in jobs:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
    finally:
        for _, _, _, log, _ in jobs: log.close()
        report['elapsed_including_cleanup_s'] = time.perf_counter()-started
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
