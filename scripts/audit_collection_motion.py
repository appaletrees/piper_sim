#!/usr/bin/env python3
"""Recompute 30 Hz motion metrics from recorded applied actions and measured states."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from piper_sim.collection_audit import load_acceptance_policy
from piper_sim.experts.quality import MotionQuality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--policy', type=Path, help='Independent operator-selected acceptance policy')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists')
    settings, policy = load_acceptance_policy(args.policy)
    with h5py.File(args.episode) as f:
        meta = json.loads(f.attrs['metadata'])
        monitor = MotionQuality(settings, 1/meta['fps'])
        times = f['next_observation/sim_time'][:]
        commands = f['action/position'][:]
        measured = f['next_observation/state'][:]
        if 'collection_initial_target' in meta:
            monitor.add(float(f['observation/sim_time'][0]), meta['collection_initial_target'],
                        f['observation/state'][0], context={'phase': 'prepared'})
        inferred_holds = 0
        for i, (t, q, actual) in enumerate(zip(times, commands, measured)):
            context = json.loads(f['next_observation/extras_json'][i]).get('teacher', {})
            if 'holding' in context:
                holding = bool(context['holding'])
            else:
                holding = i > 0 and np.array_equal(q, commands[i-1])
                inferred_holds += 1
            monitor.add(t, q, actual, holding=holding, context=context)
        result = monitor.report()
        result['source'] = str(args.episode.resolve())
        result['acceptance_policy'] = policy
        result['inferred_hold_frames'] = inferred_holds
        result['training_ready'] = False
        result['scope'] = 'Independent-policy motion diagnostic, including failed source episodes; no task, contact or image acceptance. Holds use recorded flags when present, otherwise repeated commands.'
    text = json.dumps(result, indent=2, ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__': main()
