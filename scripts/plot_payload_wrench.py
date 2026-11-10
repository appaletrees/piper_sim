#!/usr/bin/env python3
"""Plot original payload wrench samples without smoothing or time rescaling.

Use system Python with matplotlib. Inputs are audit_payload_wrench output
directories. Each column has its own time origin; this is not a matched-state
causal comparison when controller parameters differ for the whole episode.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def segments(times, phases, targets):
    """Do not connect separate native intervals or separate task phases."""
    dt = np.diff(times)
    if not len(dt) or np.any(dt <= 0):
        raise ValueError('At least two strictly increasing native timestamps required')
    breaks = (dt > 1.5 * np.median(dt)) | (phases[1:] != phases[:-1]) | (targets[1:] != targets[:-1])
    return np.split(np.arange(len(times)), np.flatnonzero(breaks) + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', required=True, help='Label=output_directory')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fig, axes = plt.subplots(5, len(args.case), figsize=(5.2 * len(args.case), 12), squeeze=False)
    evidence = []
    for col, item in enumerate(args.case):
        label, directory = item.split('=', 1)
        directory = Path(directory)
        report = json.loads((directory / 'audit.json').read_text())
        source = directory / 'wrench.npz'
        before = digest(source)
        if before != report['series_sha256']:
            raise ValueError('Wrench series hash mismatch')
        with np.load(source, allow_pickle=False) as archive:
            z = dict(archive)
        times = z['time_before']
        groups = segments(times, z['phase'], z['target'])
        t = times - times[0]
        traces = [
            [('plate', np.rad2deg(z['tilt_from_up_rad']))],
            [('plate', np.linalg.norm(z['angular_velocity_world_rad_s'], axis=1))],
            [('fingers', z['finger_contact_normal_sum_N'])],
            [(k, z[k + '_force_N'][:, 2]) for k in ('fingers', 'external_support', 'other_robot')],
            [(k, np.linalg.norm(z[k + '_torque_COM_Nm'], axis=1))
             for k in ('fingers', 'external_support', 'other_robot')],
        ]
        labels = ['Tilt from up [deg]', 'Angular speed [rad/s]', 'Summed finger normal [N]',
                  'World Z contact force [N]', 'Contact torque about COM [Nm]']
        for row, (series, ylabel) in enumerate(zip(traces, labels)):
            ax = axes[row, col]
            for j, (name, values) in enumerate(series):
                for i, group in enumerate(groups):
                    ax.plot(t[group], values[group], color=f'C{j}', lw=.8,
                            label=name if i == 0 else None)
            if row == 3:
                ax.axhline(report['weight_N'], color='black', ls=':', lw=.8, label='weight magnitude')
            for group in groups[1:]:
                ax.axvline(t[group[0]], color='.75', lw=.5)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.18)
            if row >= 3:
                ax.legend(fontsize=7, loc='best')
        axes[0, col].set_title(label)
        axes[-1, col].set_xlabel('Seconds since first traced step (1x simulation time)')
        for group in groups:
            axes[0, col].text(t[group[0]], 1.01, str(z['phase'][group[0]]),
                             transform=axes[0, col].get_xaxis_transform(), fontsize=7, rotation=30)
        if digest(source) != before:
            raise ValueError('Source changed during plotting')
        evidence.append({'label': label, 'directory': str(directory), 'series_sha256': before,
                         'recording_sha256': report['source_sha256'], 'trace_sha256': report['trace_sha256'],
                         'native_samples': len(times), 'first_last_time': times[[0, -1]].tolist(),
                         'segments': [{'phase': str(z['phase'][g[0]]), 'samples': len(g),
                                       'first_last_time': times[g[[0, -1]]].tolist()} for g in groups]})
    fig.suptitle('Original native solver samples; no filtering or time rescaling', fontsize=13)
    fig.tight_layout(rect=(0, .015, 1, .96))
    fig.text(.5, .005, 'Contact torque alone is not a dynamics balance or hardware calibration certificate.',
             ha='center', fontsize=9)
    args.output.mkdir(parents=True)
    fig.savefig(args.output / 'comparison.png', dpi=150)
    plt.close(fig)
    (args.output / 'plot.json').write_text(json.dumps({
        'scope': __doc__, 'script_sha256': digest(__file__), 'cases': evidence,
        'plot_sha256': digest(args.output / 'comparison.png'), 'training_ready': False,
        'processing': 'Every original sample; time origin shifted independently; lines broken at phase or sampling gaps.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
