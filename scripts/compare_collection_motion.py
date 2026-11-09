#!/usr/bin/env python3
"""Compare real simulation timing and stop density; never retime video or data."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def cadence(path):
    with h5py.File(path) as f:
        metadata = json.loads(f.attrs['metadata'])
        fps = float(metadata['fps'])
        times = np.r_[f['observation/sim_time'][0], f['next_observation/sim_time'][:]]
        states = np.vstack([f['observation/state'][0], f['next_observation/state'][:]])
        commands = np.vstack([metadata['collection_initial_target'], f['action/position'][:]])
        arm = np.r_[0:6, 7:13]
        stopped = np.ones(len(times)-1, dtype=bool)
        for samples in (states, commands):
            v = abs(np.diff(samples, axis=0))*fps
            stopped &= (v[:, arm].max(1) < .02) & (v[:, [6,13]].max(1) < .002)
        edges = np.diff(np.r_[False, stopped, False].astype(int))
        lengths = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
        sustained = lengths[lengths/fps >= .25]
        phases = {}
        for raw, stationary in zip(f['next_observation/extras_json'], stopped):
            phase = json.loads(raw)['teacher']['phase']
            stat = phases.setdefault(phase, {'seconds':0., 'stationary_seconds':0.})
            stat['seconds'] += 1/fps
            stat['stationary_seconds'] += int(stationary)/fps
        return {'episode':str(Path(path).resolve()), 'frames':len(times)-1,
                'simulation_duration_s':float(times[-1]-times[0]), 'fps':fps,
                'stationary_fraction':float(stopped.mean()),
                'stationary_bouts_at_least_250ms':len(sustained),
                'longest_stationary_bout_s':float(max(lengths, default=0)/fps),
                'arm_joint_travel_rad':np.sum(abs(np.diff(states[:,arm],axis=0)),axis=0).tolist(),
                'phases':phases, 'task_success':bool(f.attrs['success']),
                'audited_training_ready':bool(f.attrs.get('training_ready',False)),
                'definition':'Both applied targets and measured motion below 0.02 rad/s for all arm joints and 0.002 m/s for both grippers; intentional contact holds included.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file')
    before, after = cadence(args.baseline), cadence(args.candidate)
    result={'baseline':before,'candidate':after,
            'duration_reduction_fraction':1-after['simulation_duration_s']/before['simulation_duration_s'],
            'note':'Different controllers can produce different physical contacts. All comparisons use real simulation timestamps; no frames removed or video sped up.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
