"""Compare the rejected and revised grasp, without modifying either recording."""
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256

base = Path(__file__).resolve().parent
output = base/'tcp_clearance_grasp_compare_v1.json'
if output.exists(): raise FileExistsError(output)
sources = {
    'before': Path('/dev/shm/piper_tcp_holdout_v1/worker_02/episode_000001_seed_1340830695/episode.h5'),
    'after': Path('/dev/shm/piper_tcp_clearance_v1/worker_00/episode_000000_seed_1340830695/episode.h5')}
result = {'training_ready': False, 'seed': 1340830695, 'script_sha256': sha256(__file__), 'runs': {}}
for label, source in sources.items():
    digest = sha256(source)
    report = json.loads(source.with_name('report.json').read_text())
    events = [e for e in report['events'] if e.get('object') == 'chopstick_a']
    begin = next(e['sim_time'] for e in events if e['phase'] == 'grasp')
    end = next(e['sim_time'] for e in events if e['phase'] == 'lift')
    with h5py.File(source, 'r') as f:
        assert f.attrs['complete'] and not f.attrs['training_ready']
        meta = json.loads(f.attrs['metadata'])
        times = f['next_observation/sim_time'][:]
        aperture = np.r_[f['observation/state'][0, 13], f['next_observation/state'][:, 13]].astype(float)
        derivatives = {order: np.r_[[np.nan]*(order-1), np.diff(aperture, n=order)*meta['fps']**order]
                       for order in (1, 2, 3)}
        cfg = copy.deepcopy(meta['config'])
        cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
        backend = MujocoBackend(cfg, cameras=False)
        rows = []
        try:
            layout = meta['task_scene']['layout']; backend.reset(layout['seed'], layout['randomized'])
            if meta['teacher'].get('multiccd'):
                backend.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
            for i in np.flatnonzero((times >= begin-.1) & (times <= end+.1)):
                extras = json.loads(f['next_observation/extras_json'][i])
                for side, arm in extras['teacher']['physics']['arms'].items():
                    backend.data.qpos[backend.indices[side]['qpos']] = arm['joint_position']
                for name, pose in extras['objects'].items():
                    adr = backend.model.joint('task/'+name+'/free').qposadr[0]
                    backend.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
                mujoco.mj_kinematics(backend.model, backend.data)
                mujoco.mj_collision(backend.model, backend.data)
                depth = 0.
                for contact in backend.data.contact:
                    names = [backend.model.body(int(backend.model.geom_bodyid[g])).name for g in contact.geom]
                    if ('task/chopstick_b' in names and any(n in ('right/link7', 'right/link8') for n in names)):
                        depth = max(depth, -float(contact.dist))
                rows.append({'frame': int(i), 'relative_time_s': float(times[i]-begin),
                    'sim_time': float(times[i]), 'command_width_m': float(f['action/position'][i, 13]),
                    'measured_width_m': float(aperture[i+1]),
                    'acceleration_m_s2': float(derivatives[2][i]), 'jerk_m_s3': float(derivatives[3][i]),
                    'reconstructed_neighbor_overlap_m': depth})
        finally: backend.close()
    assert sha256(source) == digest
    result['runs'][label] = {'source': str(source), 'source_sha256': digest, 'source_unchanged': True,
        'online_quality': {key: report['quality'][key] for key in ('passed', 'metrics', 'thresholds', 'violations')},
        'task_summary': report['summary'], 'candidate_selection': [e for e in events if e['phase'] == 'grasp_candidate_selected'],
        'rows': rows, 'peak_window_neighbor_overlap_m': max(r['reconstructed_neighbor_overlap_m'] for r in rows)}
result['limitations'] = ['Known development seed, not an independent reliability estimate.',
    '30 Hz finite differences and native-position reconstruction; no original contact-force replay or substep clearance proof.',
    'Only grasp planning changed. Per-target material switching and lack of hardware calibration remain.',
    'State-only diagnostics on volatile tmpfs; not a durable RGB training release.']
atomic_json(output, result)
print({label: {'quality_passed': r['online_quality']['passed'],
              'acceleration_peak': r['online_quality']['metrics']['measured_gripper_acceleration'],
              'jerk_peak': r['online_quality']['metrics']['measured_gripper_jerk'],
              'neighbor_overlap_m': r['peak_window_neighbor_overlap_m']} for label, r in result['runs'].items()})
