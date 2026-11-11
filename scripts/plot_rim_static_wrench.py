#!/usr/bin/env python3
"""Plot model-only static requirements alongside original native servo force."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads(args.results.read_text());rows=report['results']
    original=[r for r in rows if r['label'].startswith('rim_alignment/')]
    source=max(original,key=lambda r:r['time_before'])
    sweep=[source]+[r for r in rows if r['label'].startswith('rigid_contact_hypothesis/')]
    x=[0.]+[abs(float(r['label'].split('/')[-1])) for r in sweep[1:]]
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    ax[0].plot(x,[r['optimistic_motor_lower_bound_N'] for r in sweep],'o-',label='Frozen-contact static minimum')
    ax[0].axhline(5,color='C1',ls='--',label='Original 5 N capacity')
    ax[0].axhline(25,color='C2',ls='--',label='Experimental 25 N capacity')
    ax[0].set(xlabel='Rigid rotation towards level [deg]',ylabel='Tendon actuator force magnitude [N]',
              title='Unchanged contact manifold: planning hypothesis')
    with np.load(args.trace/'trace.npz',allow_pickle=False) as z:
        selected=np.isin(z['phase'],['rim_alignment','transfer'])
        t=z['time_before'][selected];force=abs(z['actuator_force'][selected,-1])
        origin=float(z['time_before'][z['phase']=='transfer'][0])
        ax[1].plot(t-origin,force,lw=.7,label='Original native servo force (all samples)')
    native=[r for r in rows if r['label'].startswith(('rim_alignment/','transfer/')) and r.get('static_elliptic_solution')]
    ax[1].scatter([r['time_before']-origin for r in native],
        [r['optimistic_motor_lower_bound_N'] for r in native],s=15,color='C3',label='Sparse static minimum (not dynamic force)')
    ax[1].set(xlabel='Seconds from transfer start [s]',ylabel='Tendon actuator force magnitude [N]',
              title='Original failed rollout: no filtering or speed change')
    for a in ax:a.grid(alpha=.2);a.legend(fontsize=7)
    fig.suptitle('Static model feasibility does not certify dynamic carrying',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.text(.5,.02,'No compliance/inertial wrench or contact evolution prediction. New executor remains uncalibrated.',ha='center',fontsize=8)
    args.output.mkdir(parents=True);fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    (args.output/'plot.json').write_text(json.dumps({'scope':__doc__,'training_ready':False,
        'results_sha256':digest(args.results),'trace_sha256':digest(args.trace/'trace.npz'),
        'script_sha256':digest(__file__),'plot_sha256':digest(args.output/'comparison.png'),
        'processing':'All original native actuator samples; sparse static queries as markers; virtual rigid contact sweep explicitly separate.'},indent=2)+'\n')


if __name__=='__main__':main()
