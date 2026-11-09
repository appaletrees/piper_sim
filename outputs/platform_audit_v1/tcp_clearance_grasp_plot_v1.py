"""Plot the same recorded grasp phase before and after the optional planner."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
source, output = base/'tcp_clearance_grasp_compare_v1.json', base/'tcp_clearance_grasp_compare_v1.png'
if output.exists(): raise FileExistsError(output)
runs = json.loads(source.read_text())['runs']
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9), layout='constrained')
for label, color in [('before', '#b74736'), ('after', '#177968')]:
    rows = runs[label]['rows']; times = [r['relative_time_s'] for r in rows]
    axes[0].plot(times, [r['measured_width_m']*1000 for r in rows], label=label+' measured', color=color)
    axes[0].plot(times, [r['command_width_m']*1000 for r in rows], label=label+' command', color=color, ls=':', alpha=.7)
    for ax, key in [(axes[1], 'acceleration_m_s2'), (axes[2], 'jerk_m_s3')]:
        ax.plot(times, [r[key] for r in rows], label=label, color=color)
    axes[3].plot(times, [r['reconstructed_neighbor_overlap_m']*1000 for r in rows], label=label, color=color)
axes[0].set_ylabel('Aperture (mm)'); axes[0].legend(fontsize=9, ncol=2)
for ax, bound, title in [(axes[1], 1., 'Acceleration (m/s²)'), (axes[2], 50., 'Jerk (m/s³)')]:
    ax.axhline(bound, color='#777', ls='--', lw=1); ax.axhline(-bound, color='#777', ls='--', lw=1)
    ax.set_ylabel(title)
axes[3].set_ylabel('Neighbor overlap (mm)'); axes[3].set_xlabel('Simulation time since start of grasp phase (s)')
axes[3].legend(fontsize=9)
for ax in axes:
    ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Chopstick approach | seed 1340830695\nMeasured aperture response and reconstructed neighboring-stick contact')
fig.supxlabel('30 Hz state differences; original quality limits unchanged. Local diagnostic, not sensor RGB or a force replay.', fontsize=9)
fig.savefig(output, dpi=150)
print(output)
