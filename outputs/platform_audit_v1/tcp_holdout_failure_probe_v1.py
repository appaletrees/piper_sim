"""Read-only native grasp-event and neighbouring-object geometry diagnostic."""
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT

base=ROOT/'outputs/platform_audit_v1'
output=base/'tcp_holdout_failure_seed1340830695_v1.json'
if output.exists(): raise FileExistsError(output)
source=Path('/dev/shm/piper_tcp_holdout_v1/worker_02/episode_000001_seed_1340830695/episode.h5')
digest=sha256(source)
report=json.loads(source.with_name('report.json').read_text())
quality=report['quality']
assert not quality['passed'] and report['summary']['success']
peak=quality['peaks']['measured_gripper_acceleration']
assert peak['channel']==13 and peak['context']=={'phase':'grasp','object':'chopstick_a'}
with h5py.File(source,'r') as f:
    assert f.attrs['complete'] and not f.attrs['training_ready']
    meta=json.loads(f.attrs['metadata'])
    changed=[p for p,d in meta['provenance'].items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=d]
    assert not changed
    times=f['next_observation/sim_time'][:]
    center=int(np.argmin(abs(times-peak['sim_time'])))
    opening=np.r_[f['observation/state'][0,13],f['next_observation/state'][:,13]].astype(float)
    derivatives={order:np.r_[[np.nan]*(order-1),np.diff(opening,n=order)*meta['fps']**order]
                 for order in (1,2,3)}
    cfg=copy.deepcopy(meta['config'])
    cfg['task_scene']['domain_randomization']=meta['task_scene'].get('domain_profile')
    backend=MujocoBackend(cfg,cameras=False)
    rows=[]
    try:
        layout=meta['task_scene']['layout'];backend.reset(layout['seed'],layout['randomized'])
        model,data=backend.model,backend.data
        if meta['teacher'].get('multiccd'):
            model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        for i in range(center-12,center+13):
            extra=json.loads(f['next_observation/extras_json'][i]);teacher=extra['teacher']
            native=teacher['physics']['arms']['right']
            for side,arm in teacher['physics']['arms'].items():
                data.qpos[backend.indices[side]['qpos']]=arm['joint_position']
            for name,pose in extra['objects'].items():
                address=model.joint('task/'+name+'/free').qposadr[0]
                data.qpos[address:address+7]=pose['position']+pose['quaternion_wxyz']
            mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
            contacts=[]
            for contact in data.contact:
                names=[model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
                if not any(n in ('right/link7','right/link8') for n in names): continue
                contacts.append({'bodies':names,'geoms':[model.geom(int(g)).name for g in contact.geom],
                                 'signed_distance_m':float(contact.dist)})
            rows.append({'transition_index':i,'sim_time':float(times[i]),
                         'phase':teacher['phase'],'object':teacher['object'],
                         'command_width_m':float(f['action/position'][i,13]),
                         'measured_width_m':float(opening[i+1]),
                         'measured_interval_velocity_m_s':float(derivatives[1][i]),
                         'measured_interval_acceleration_m_s2':float(derivatives[2][i]),
                         'measured_interval_jerk_m_s3':float(derivatives[3][i]),
                         'native_width_m':sum(native['joint_position'][6:]),
                         'native_endpoint_velocity_m_s':sum(native['joint_velocity'][6:]),
                         'recorded_target_finger_contacts':native['target_finger_contacts'],
                         'recorded_robot_collision_pair':teacher['robot_collision_pair'],
                         'recorded_robot_penetration_m':teacher['robot_penetration_m'],
                         'reconstructed_finger_contacts':contacts})
    finally: backend.close()
assert sha256(source)==digest
peak_row=next(row for row in rows if row['transition_index']==center)
assert abs(peak_row['measured_interval_acceleration_m_s2']-quality['metrics']['measured_gripper_acceleration'])<1e-4
assert abs(peak_row['measured_interval_jerk_m_s3']-quality['metrics']['measured_gripper_jerk'])<1e-3
result={'training_ready':False,'source':str(source),'source_sha256':digest,'source_unchanged':True,
        'analysis_sha256':sha256(__file__),'changed_source_provenance':changed,
        'phase':'grasp','object':'chopstick_a','peak_transition':center,'rows':rows,
        'online_quality':{k:quality[k] for k in ('passed','violations','metrics','thresholds','peaks')},
        'observed':'Right aperture target stays 10 mm around the spike; recorded and reconstructed finger contact with neighbouring chopstick_b accompanies the aperture deflection before target pinch.',
        'inference':'Investigate non-target clearance during slender-object contact approach; correlation and pose reconstruction alone do not identify a unique cause.',
        'limitations':['No integration or state reset in a live rollout; scratch geometry only.',
                       'Forces and velocities come from recorded native snapshots; geometric overlaps are not force estimates.',
                       'Public derivatives use float32 recorded positions at 30 Hz and can differ slightly from online float64 values.',
                       'Failure is retained; no acceptance limit or scene geometry was changed.']}
atomic_json(output,result)
print(json.dumps({'peak_transition':center,'recorded_acceleration':peak_row['measured_interval_acceleration_m_s2'],
                  'recorded_jerk':peak_row['measured_interval_jerk_m_s3'],
                  'recorded_pair':peak_row['recorded_robot_collision_pair'],
                  'peak_finger_contacts':peak_row['reconstructed_finger_contacts']},indent=2))
