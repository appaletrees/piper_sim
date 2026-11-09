#!/usr/bin/env python3
"""Measure one versus two independent MuJoCo RGB collectors, without a teacher.

Records remain diagnostic and success=false. This is a concurrency feasibility
measurement, never qualified task throughput. No shared output or hardware I/O.
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.frames < 150 or args.repeats < 1:
        parser.error('Use at least 150 frames and one repeat')
    if shutil.disk_usage(args.output.parent).free < 10*1024**3:
        parser.error('Reserve at least 10 GiB for diagnostic recordings')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'training_ready': False,
        'scope': 'Single baseline then two concurrent headless MuJoCo collectors; 3x640x480/30Hz output.',
        'limitations': 'Diagnostic motions only. Includes child startup, reset/warmup, close and stream audits. '
                       'One group per worker count; other applications and an independent full RGB task share this host.',
        'groups': []}
    processes = []
    try:
        for workers in (1, 2):
            started = time.perf_counter()
            processes = []
            logs = []
            for worker in range(workers):
                folder = args.output/f'workers_{workers}_job_{worker}'
                log = folder.with_suffix('.log').open('w')
                logs.append(log)
                command = [sys.executable, 'scripts/benchmark_backend.py', '--backend', 'mujoco', '--rgb',
                    '--frames', str(args.frames), '--repeats', str(args.repeats), '--output', str(folder)]
                processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                    env={**os.environ, 'MUJOCO_GL': 'egl'}), folder))
            codes = [process.wait() for process, _ in processes]
            wall = time.perf_counter()-started
            for log in logs: log.close()
            children = [json.loads((folder/'report.json').read_text()) for _, folder in processes]
            if any(code != 0 for code in codes) or any(c['status'] != 'diagnostic_passed' for c in children):
                raise RuntimeError('A child failed; no throughput result is certified')
            frames = sum(c['frames_per_repeat']*len(c['repeats']) for c in children)
            group = {'workers': workers, 'wall_seconds': wall, 'recorded_transitions': frames,
                'end_to_end_diagnostic_transitions_per_s': frames/wall,
                'output_bytes': sum(r['bytes'] for c in children for r in c['repeats']),
                'children': children}
            report['groups'].append(group)
            print(json.dumps({k: v for k, v in group.items() if k != 'children'}), flush=True)
        report['observed_aggregate_ratio'] = (report['groups'][1]['end_to_end_diagnostic_transitions_per_s']/
                                               report['groups'][0]['end_to_end_diagnostic_transitions_per_s'])
        report['status'] = 'diagnostic_passed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        for process, _ in processes:
            if process.poll() is None: process.terminate()
        for process, _ in processes:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
