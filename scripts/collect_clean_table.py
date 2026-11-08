#!/usr/bin/env python3
"""Screen seeds, collect RGB, then publish only independently audited episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def collection_split(seed, nominal, fraction):
    from piper_sim.randomization import stream_seed
    # Fixed nominal resets are one trajectory family even with different seeds.
    key = 'collection/nominal_split_v1' if nominal else 'collection/split_v1'
    value = stream_seed(0 if nominal else seed, key)/2**64
    return 'validation' if value < fraction else 'train'


def run_episode(args, seed, output, *, nominal, rgb):
    command = [sys.executable, str(Path(__file__).with_name('demo_clean_table.py')),
               '--record', '--seed', str(seed), '--output', str(output),
               '--config', args.config, '--task-scene', args.task_scene,
               '--grasp-mode', args.grasp_mode, '--collection-profile', args.collection_profile]
    if not nominal: command += ['--domain-randomization', args.domain_randomization]
    if rgb: command.append('--rgb')
    if rgb and args.video: command.append('--video')
    with output.with_suffix('.log').open('w') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=args.timeout_s).returncode
        except subprocess.TimeoutExpired:
            code = -1
    try:
        report = json.loads((output/'report.json').read_text())
        audit = json.loads((output/'audit.json').read_text())
    except (OSError, ValueError):
        report, audit = {}, {}
    return code, report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--config', default='configs/clean_table_workcell.yaml')
    parser.add_argument('--task-scene', default='configs/tasks/clean_table_rigid.yaml')
    parser.add_argument('--domain-randomization', default='configs/domain_randomization/clean_table.yaml')
    parser.add_argument('--collection-profile', default='configs/collection/clean_table.yaml')
    parser.add_argument('--grasp-mode', choices=['physical'], default='physical')
    parser.add_argument('--states-only', action='store_true', help='Diagnostics only; never accepted.jsonl')
    parser.add_argument('--skip-preflight', action='store_true', help='Record directly; final RGB audit still mandatory')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--lerobot', action='store_true', help='Export accepted RGB episodes only, separately by split')
    parser.add_argument('--timeout-s', type=float, default=7200., help='Per phase timeout including full RGB audit')
    parser.add_argument('--nominal-fraction', type=float, default=0.)
    parser.add_argument('--validation-fraction', type=float, default=.2,
                        help='Stable seed-family split; keep the same value across runs')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output already exists; use a new run directory')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    if args.lerobot and args.states_only: parser.error('--lerobot requires RGB')
    if not 0 <= args.nominal_fraction <= 1 or args.timeout_s <= 0 or not 0 <= args.validation_fraction < 1:
        parser.error('Invalid fraction or timeout')
    args.output.mkdir(parents=True)
    # Empty files are intentional: absence of accepted episodes is explicit.
    for name in ('accepted', 'rejected', 'diagnostics'):
        (args.output/(name+'.jsonl')).touch()
    results, start = [], time.monotonic()
    from piper_sim.randomization import stream_seed
    for index, seed in enumerate(args.seeds):
        nominal = stream_seed(seed, 'collection/nominal')/2**64 < args.nominal_fraction
        split = collection_split(seed, nominal, args.validation_fraction)
        output = args.output/f'episode_{index:06d}_seed_{seed}'
        preflight, preflight_path, mismatch = None, None, False
        if not args.states_only and not args.skip_preflight:
            preflight_path = output.with_name(output.name+'_preflight')
            code, report, preflight = run_episode(args, seed, preflight_path, nominal=nominal, rgb=False)
        if preflight is None or (code == 0 and preflight.get('passed', False)):
            code, report, audit = run_episode(args, seed, output, nominal=nominal, rgb=not args.states_only)
            if preflight and audit.get('passed'):
                mismatch = (preflight['checks']['streams']['trajectory_sha256'] !=
                            audit['checks']['streams']['trajectory_sha256'] or
                            preflight['checks']['task']['objects_sha256'] != audit['checks']['task']['objects_sha256'])
                if mismatch:
                    from piper_sim.collection_audit import save_audit
                    audit['passed'] = audit['training_ready'] = False
                    audit['violations'].append('preflight_rgb_trajectory_mismatch')
                    save_audit(output/'episode.h5', audit, output/'audit.json')
                    report.update(accepted=False, training_ready=False, audit_passed=False,
                                  error='preflight_rgb_trajectory_mismatch', audit_violations=audit['violations'])
                    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        else:
            output, audit = preflight_path, preflight
        accepted = bool(not args.states_only and code == 0 and report.get('training_ready')
                        and audit.get('training_ready') and not mismatch)
        diagnostic = bool(args.states_only and code == 0 and audit.get('passed'))
        result = {'seed': seed, 'nominal': nominal, 'split': split, 'accepted': accepted,
                  'training_ready': accepted, 'diagnostic_passed': diagnostic, 'returncode': code,
                  'error': report.get('error') or (None if accepted or diagnostic else 'collection_rejected'),
                  'violations': audit.get('violations', ['missing_audit']),
                  'path': str((output/'episode.h5').resolve()), 'report': str((output/'report.json').resolve()),
                  'audit': str((output/'audit.json').resolve()), 'rgb': bool(audit.get('checks', {}).get('streams', {}).get('rgb')),
                  'preflight': str(preflight_path.resolve()) if preflight_path else None,
                  'inventory': report.get('inventory'), 'grasp_mode': args.grasp_mode}
        results.append(result)
        category = 'accepted' if accepted else 'diagnostics' if diagnostic else 'rejected'
        with (args.output/(category+'.jsonl')).open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
        summary = {'attempted': len(results), 'accepted': sum(x['accepted'] for x in results),
                   'diagnostic_passed': sum(x['diagnostic_passed'] for x in results),
                   'rejected': sum(not x['accepted'] and not x['diagnostic_passed'] for x in results),
                   'elapsed_seconds': time.monotonic()-start, 'episodes': results,
                   'validation_fraction': args.validation_fraction,
                   'note': 'Acceptance requires a complete physical task and closed-file RGB audit. All failures retained; finite seeds are not a distribution-wide success estimate.'}
        (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.lerobot:
        from piper_sim.lerobot_export import export_lerobot_v2
        for split in ('train', 'validation'):
            paths = [r['path'] for r in results if r['accepted'] and r['split'] == split]
            if paths: export_lerobot_v2(paths, args.output/('lerobot_v2_'+split), require_quality=True)
    return 0 if any(r['accepted'] or r['diagnostic_passed'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
