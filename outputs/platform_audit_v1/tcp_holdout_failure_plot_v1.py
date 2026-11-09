"""Plot recorded aperture and kinematic neighbour contact around the rejected grasp."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent
source=base/'tcp_holdout_failure_seed1340830695_v1.json'
output=base/'tcp_holdout_failure_seed1340830695_v1.png'
if output.exists(): raise FileExistsError(output)
evidence=json.loads(source.read_text()); rows=evidence['rows']
times=[r['sim_time'] for r in rows]
peak=next(r for r in rows if r['transition_index']==evidence['peak_transition'])
fig,axes=plt.subplots(4,1,sharex=True,figsize=(10,9),layout='constrained')
axes[0].plot(times,[r['command_width_m']*1000 for r in rows],label='Applied aperture target',color='#166b9a')
axes[0].plot(times,[r['measured_width_m']*1000 for r in rows],'.-',label='Measured full aperture',color='#c64a35')
axes[0].set_ylabel('Aperture (mm)'); axes[0].legend(loc='lower left',fontsize=9)
for ax,key,bound,label in [(axes[1],'measured_interval_acceleration_m_s2',1.,'Acceleration (m/s²)'),
                           (axes[2],'measured_interval_jerk_m_s3',50.,'Jerk (m/s³)')]:
    ax.plot(times,[r[key] for r in rows],'.-',color='#c64a35')
    ax.axhline(bound,color='#777777',ls='--',lw=1);ax.axhline(-bound,color='#777777',ls='--',lw=1)
    ax.set_ylabel(label)
depth=[]
for row in rows:
    contacts=[-c['signed_distance_m']*1000 for c in row['reconstructed_finger_contacts']
              if 'task/chopstick_b' in c['bodies']]
    depth.append(max([0.,*contacts]))
axes[3].plot(times,depth,'.-',color='#4a7651',label='Reconstructed finger / neighbour overlap')
axes[3].set_ylabel('Overlap (mm)');axes[3].set_xlabel('Recorded simulation time (s)')
axes[3].legend(loc='upper right',fontsize=9)
for ax in axes:
    ax.grid(alpha=.2); ax.axvline(peak['sim_time'],color='#333333',ls=':',lw=1)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Rejected new layout: chopstick approach\nSeed 1340830695 | unchanged 10 mm target; actual jaw deflection',fontsize=13)
fig.supxlabel('30 Hz recorded-state differences; dashed limits are unchanged. Geometry is reconstructed, not a force replay.',fontsize=9)
fig.savefig(output,dpi=160)
print(output)
