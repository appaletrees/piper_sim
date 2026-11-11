#!/usr/bin/env python3
"""Read-only geometry preflight from a closed recording, never dynamics replay."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import yaml

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.experts.clean_table import SortFailure
from piper_sim.experts.rigid_table import RigidTableTeacher
from piper_sim.experts.rim_transport import search
from piper_sim.simulation import SimulationSession
from probe_grasp_aperture import state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--collection-profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshot',choices=('transfer','final'),default='transfer')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hash = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs['complete']:
            raise ValueError('Closed recording required')
        meta = json.loads(f.attrs['metadata'])
        if args.snapshot=='final':
            extras=json.loads(f['next_observation/extras_json'][-1])
        else:
            event = next(e for e in meta['collection_outcome']['teacher_events']
                         if e['object'] == 'plate' and e['phase'] == 'transfer')
            times = f['observation/sim_time'][:]
            i = int(np.argmin(abs(times - event['sim_time'])))
            if abs(times[i] - event['sim_time']) > 1e-8:
                raise ValueError('Exact transfer boundary required')
            extras = json.loads(f['observation/extras_json'][i])
    cfg = copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
    settings = yaml.safe_load(args.collection_profile.read_text())
    b = MujocoBackend(cfg, cameras=False)
    result = {'scope': __doc__, 'source': str(args.source), 'source_sha256': source_hash,
              'script_sha256': sha256(__file__), 'collection_profile_sha256': sha256(args.collection_profile),
              'training_ready': False, 'passed': False}
    result['snapshot']=args.snapshot
    try:
        layout = meta['task_scene']['layout']
        b.reset(layout['seed'], layout['randomized'])
        for side, arm in extras['teacher']['physics']['arms'].items():
            b.data.qpos[b.indices[side]['qpos']] = arm['joint_position']
        for name, pose in extras['objects'].items():
            adr = b.model.joint('task/' + name + '/free').qposadr[0]
            b.data.qpos[adr:adr+7] = pose['position'] + pose['quaternion_wxyz']
        mujoco.mj_forward(b.model, b.data)
        t = RigidTableTeacher(SimulationSession(b), settings=settings)
        t.q = np.asarray(extras['controller_interval']['command'])
        t.current, t.phase = 'plate', 'transfer'
        t.track_grasp('right', 'plate')
        initial, command = state(b), t.q.copy()
        diagnostics={}
        try:
            plan = search(t, 'right', 'plate', 'basket', settings['pacing']['rim_transport'],diagnostics=diagnostics)
            result.update(passed=True, evidence=plan['evidence'], transfer_commands=len(plan['transfer']),
                          lower_commands=len(plan['lower']))
        except SortFailure as exc:
            result['error'] = str(exc)
            result['diagnostics']=diagnostics
        if not np.array_equal(initial, state(b)) or not np.array_equal(command, t.q):
            raise RuntimeError('Geometry query changed isolated integration state or command')
        result['isolated_state_and_command_unchanged'] = True
    finally:
        b.close()
    if sha256(args.source) != source_hash:
        raise ValueError('Source recording changed')
    args.output.mkdir(parents=True)
    if result['passed']:
        with (args.output / 'commands.npz').open('xb') as f:
            np.savez_compressed(f, transfer=plan['transfer'], lower=plan['lower'])
        result['commands_sha256'] = sha256(args.output / 'commands.npz')
    atomic_json(args.output / 'preflight.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
