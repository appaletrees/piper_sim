#!/usr/bin/env python3
"""Compare original native grasp/lift traces, aligned only at lift onset.

No sample interpolation, time scaling or source mutation. This diagnostic is
not a calibrated hardware measurement or a training acceptance verdict.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256


def read_trace(directory, target):
    metadata = json.loads((directory/'trace.json').read_text())
    source = directory/'trace.npz'
    digest = sha256(source)
    if digest != metadata['trace_sha256']:
        raise ValueError('Trace hash mismatch')
    with np.load(source, allow_pickle=False) as archive:
        z = {key: archive[key] for key in ('time_before', 'time_after', 'qpos_before',
            'qvel_before', 'qvel_after', 'control', 'actuator_force', 'target', 'phase')}
    selected = z['target'] == target
    if not np.any(selected & (z['phase'] == 'lift')):
        raise ValueError('No recorded lift for selected object')
    z = {key: value[selected] for key, value in z.items()}
    dt = z['time_after']-z['time_before']
    if (not np.allclose(dt, metadata['integration_dt'], atol=1e-8, rtol=0)
            or not np.allclose(z['time_before'][1:], z['time_after'][:-1], atol=1e-8, rtol=0)):
        raise ValueError('Continuous native trace required; never bridge missing samples')
    onset = z['time_before'][z['phase'] == 'lift'][0]
    z['time_from_lift'] = z['time_before']-onset
    z['acceleration'] = (z['qvel_after'][:, 6:].sum(axis=1)-z['qvel_before'][:, 6:].sum(axis=1))/dt
    if sha256(source) != digest:
        raise ValueError('Source changed')
    return z, {'source': str(source), 'sha256': digest, 'samples': len(dt),
               'lift_onset_sim_time_s': float(onset), 'integration_dt': metadata['integration_dt']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--object', default='chopstick_b')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex='col', sharey='row')
    records = []
    for col, (label, directory) in enumerate((('Original zero target', args.baseline),
                                              ('Position preload reduction', args.candidate))):
        z, record = read_trace(directory, args.object);records.append(record)
        time = z['time_from_lift']
        axes[0, col].plot(time, z['qpos_before'][:, 6:].sum(axis=1)*1000, label='Measured opening')
        axes[0, col].plot(time, z['control'][:, 6]*1000, label='Internal position reference', alpha=.8)
        axes[1, col].plot(time, z['actuator_force'][:, 6], label='Gripper actuator force')
        axes[2, col].plot(time, z['qvel_before'][:, 6:].sum(axis=1), label='Native opening velocity')
        axes[3, col].plot(time, z['acceleration'], label='Native velocity difference / dt')
        axes[0, col].set_title(label)
        axes[3, col].set_xlabel('Time from this rollout\'s lift onset (s)')
        for row, unit in enumerate(('Opening (mm)', 'Force (N)', 'Velocity (m/s)', 'Acceleration (m/s²)')):
            axes[row, col].axvline(0, ls='--', lw=.8, color='gray')
            axes[row, col].grid(alpha=.2);axes[row, col].legend(fontsize=8)
            if col == 0: axes[row, col].set_ylabel(unit)
    fig.suptitle(args.object+' | native samples; uncalibrated simulation\n'
                 'Identical fixed servo/material; different recorded position commands', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    destination = args.output/'comparison.png';fig.savefig(destination, dpi=150);plt.close(fig)
    atomic_json(args.output/'plot.json', {'scope': __doc__, 'sources': records,
        'image_sha256': sha256(destination), 'script_sha256': sha256(__file__), 'training_ready': False})


if __name__ == '__main__':
    main()
