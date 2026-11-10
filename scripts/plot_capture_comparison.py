#!/usr/bin/env python3
"""Compare three original grasp/lift traces without resampling or time scaling."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_preload_comparison import read_trace
from piper_sim.collection_jobs import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('window', type=Path)
    parser.add_argument('--object', default='chopstick_b')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex='col', sharey='row')
    records = []
    cases = [('Zero target, then relax', args.baseline),
             ('Geometric target: dropped during lift', args.geometry),
             ('Geometric target + preload window', args.window)]
    for col, (label, directory) in enumerate(cases):
        z, record = read_trace(directory, args.object)
        records.append({'label': label, **record})
        time = z['time_from_lift']
        axes[0, col].plot(time, z['qpos_before'][:, 6:].sum(axis=1)*1000, label='Measured opening')
        axes[0, col].plot(time, z['control'][:, 6]*1000, label='Internal reference', alpha=.8)
        axes[1, col].plot(time, z['actuator_force'][:, 6], label='Gripper actuator')
        axes[2, col].plot(time, z['qvel_before'][:, 6:].sum(axis=1), label='Native opening velocity')
        axes[3, col].plot(time, z['acceleration'], label='Native velocity difference / dt')
        axes[0, col].set_title(label, fontsize=10)
        axes[3, col].set_xlabel('Time from this run\'s lift onset (s)')
        for row, unit in enumerate(('Opening (mm)', 'Force (N)', 'Velocity (m/s)', 'Acceleration (m/s²)')):
            axes[row, col].axvline(0, ls='--', lw=.8, color='gray')
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(unit)
    fig.suptitle(args.object+' | original native samples, aligned at each lift onset\n'
                 'Fixed controller/material; uncalibrated simulation; no complete episode certified')
    fig.tight_layout(rect=(0, 0, 1, .95))
    destination = args.output/'comparison.png'
    fig.savefig(destination, dpi=150);plt.close(fig)
    atomic_json(args.output/'plot.json', {'scope': __doc__, 'sources': records,
        'object': args.object, 'training_ready': False, 'image_sha256': sha256(destination),
        'script_sha256': sha256(__file__),
        'reader_sha256': sha256(Path(__file__).with_name('plot_preload_comparison.py'))})


if __name__ == '__main__':
    main()
