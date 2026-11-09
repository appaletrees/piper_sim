#!/usr/bin/env python3
"""Plot original native samples around each release acceleration peak."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='Recorded native release trace')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.trace, allow_pickle=False) as z:
        dt = z['time_after']-z['time_before']
        acceleration = (z['qvel_after'][:, 6:].sum(axis=1)-z['qvel_before'][:, 6:].sum(axis=1))/dt
        groups = [(target, phase) for target, phase in (('bowl', 'unload_gripper'),
                  ('plate', 'release'), ('side_dish', 'release'))
                  if np.any((z['target'] == target) & (z['phase'] == phase))]
        if not groups:
            raise ValueError('No release samples to plot')
        fig, axes = plt.subplots(4, len(groups), figsize=(max(8, 5*len(groups)), 10), sharex='col', squeeze=False)
        for column, (target, phase) in enumerate(groups):
            ids = np.flatnonzero((z['target'] == target) & (z['phase'] == phase))
            peak = ids[np.argmax(np.abs(acceleration[ids]))]
            ids = ids[np.abs(z['time_before'][ids]-z['time_before'][peak]) <= .04]
            t = (z['time_before'][ids]-z['time_before'][peak])*1000
            axes[0, column].plot(t, z['control'][ids, 6]*1000, label='Command opening')
            axes[0, column].plot(t, z['qpos_before'][ids, 6:].sum(axis=1)*1000, label='Measured opening')
            axes[1, column].plot(t, z['qvel_before'][ids, 6:].sum(axis=1), label='Native opening velocity')
            # bias[0] can contain an experimental compliance force as well as
            # kv*vref. Dividing it by kv would invent a velocity reference.
            # Only the measured velocity is unambiguous in this trace format.
            axes[2, column].plot(t, z['actuator_force'][ids, 6], label='Tendon actuator force')
            axes[2, column].plot(t, z['constraint_force_by_type'][ids, 7, 6:].mean(axis=1),
                                 label='Mean jaw contact force')
            axes[3, column].plot(t, acceleration[ids], label='Integrated velocity difference')
            axes[3, column].set_xlabel('Time from acceleration peak (ms)')
            axes[0, column].set_title(f'{target} / {phase}\nsolver time {z["time_before"][peak]:.6f} s')
            for row, label in enumerate(('Opening (mm)', 'Velocity (m/s)', 'Force (N)', 'Acceleration (m/s²)')):
                axes[row, column].set_ylabel(label)
                axes[row, column].axvline(0, color='grey', lw=.7, linestyle='--')
                axes[row, column].grid(alpha=.2)
                axes[row, column].legend(fontsize=7, loc='best')
        frequency = 1/float(np.median(dt))
        fig.suptitle(f'{args.label}: {frequency:.0f} Hz physics\n'
                     'Diagnostic of simulation; hardware parameters remain uncalibrated', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(args.output, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
