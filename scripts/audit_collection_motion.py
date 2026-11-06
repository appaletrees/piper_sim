#!/usr/bin/env python3
"""Recompute 30 Hz motion metrics from recorded applied actions and measured states."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import yaml

from piper_sim.config import ROOT
from piper_sim.experts.quality import MotionQuality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with h5py.File(args.episode) as f:
        meta = json.loads(f.attrs['metadata'])
        settings = meta.get('teacher', {}).get('motion_and_quality') or yaml.safe_load(
            (ROOT/'configs/collection/clean_table.yaml').read_text())
        monitor = MotionQuality(settings, 1/meta['fps'])
        times = f['next_observation/sim_time'][:]
        commands = f['action/position'][:]
        measured = f['next_observation/state'][:]
        for i, (t, q, actual) in enumerate(zip(times, commands, measured)):
            context = json.loads(f['next_observation/extras_json'][i]).get('teacher', {})
            holding = i > 0 and np.array_equal(q, commands[i-1])
            monitor.add(t, q, actual, holding=holding, context=context)
        result = monitor.report()
        result['source'] = str(args.episode.resolve())
        result['scope'] = 'Motion audit only; does not re-certify task success, collision or image content. Holds inferred from repeated commands.'
    text = json.dumps(result, indent=2, ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__': main()
