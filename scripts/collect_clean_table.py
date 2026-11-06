#!/usr/bin/env python3
"""Batch privileged collection. Only full-task quality passes enter accepted.jsonl."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--domain-randomization', default='configs/domain_randomization/clean_table.yaml')
    parser.add_argument('--collection-profile', default='configs/collection/clean_table.yaml')
    parser.add_argument('--grasp-mode', choices=['assisted', 'physical'], default='assisted')
    parser.add_argument('--states-only', action='store_true', help='Fast motion audit; not an RGB training dataset')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--lerobot', action='store_true', help='Export accepted RGB episodes only')
    parser.add_argument('--timeout-s', type=float, default=3600.)
    parser.add_argument('--nominal-fraction', type=float, default=0., help='Fraction of seeds using the fixed nominal domain')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new run directory')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    if args.lerobot and args.states_only:
        parser.error('--lerobot requires RGB')
    if not 0 <= args.nominal_fraction <= 1 or args.timeout_s <= 0:
        parser.error('nominal-fraction must be in [0,1], timeout-s must be positive')
    args.output.mkdir(parents=True)
    results = []
    start = time.monotonic()
    for index, seed in enumerate(args.seeds):
        from piper_sim.randomization import stream_seed
        nominal = stream_seed(seed, 'collection/nominal')/2**64 < args.nominal_fraction
        output = args.output/f'episode_{index:06d}_seed_{seed}'
        command = [sys.executable, str(Path(__file__).with_name('demo_clean_table.py')),
                   '--record', '--seed', str(seed), '--output', str(output),
                   '--grasp-mode', args.grasp_mode,
                   '--collection-profile', args.collection_profile]
        if not nominal:
            command += ['--domain-randomization', args.domain_randomization]
        if not args.states_only: command.append('--rgb')
        if args.video: command.append('--video')
        with (args.output/f'episode_{index:06d}.log').open('w') as log:
            try:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_s)
                code = run.returncode
            except subprocess.TimeoutExpired:
                code = -1
        report_path = output/'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        accepted = code == 0 and report.get('accepted', False)
        result = {'seed': seed, 'nominal': nominal, 'accepted': accepted, 'returncode': code,
                  'error': report.get('error', 'timeout/process failure' if code else None),
                  'path': str((output/'episode.h5').resolve()), 'report': str(report_path.resolve()),
                  'grasp_mode': args.grasp_mode, 'rgb': not args.states_only,
                  'training_ready': False}
        results.append(result)
        with (args.output/('accepted.jsonl' if accepted else 'rejected.jsonl')).open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
        summary = {'attempted': len(results), 'accepted': sum(x['accepted'] for x in results),
                   'elapsed_seconds': time.monotonic()-start, 'episodes': results,
                   'note': 'Quality pass applies to selected grasp mode; assisted is not physical transfer validation.'}
        (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    paths = [r['path'] for r in results if r['accepted']]
    if args.lerobot and paths:
        from piper_sim.lerobot_export import export_lerobot_v2
        export_lerobot_v2(paths, args.output/'lerobot_v2')
    return 0 if paths else 2


if __name__ == '__main__':
    raise SystemExit(main())
