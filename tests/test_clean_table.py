import copy
import json
import time

import numpy as np
import pytest
from piper_sim.config import load
from piper_sim.tasks import create_task


def task():
    return create_task('configs/tasks/clean_table.yaml',load())


def test_video_objects_scale_stacks_and_reproducible_randomization():
    t=task()
    assert len(t.names)==11
    assert t.specs['plate']['diameter_m']==.190
    assert t.specs['basket']['size_m']==[.290,.235,.115]
    assert t.original_table_bounds[1,0]==pytest.approx(.645)
    assert t.table_bounds[1,0]==pytest.approx(1.395)
    assert len(t.parts['bowl'])>17  # both radial and angular convex decomposition
    original=copy.deepcopy(t.layout)
    p=original['poses']
    assert p['plate']['position'][2]<p['bowl']['position'][2]<p['spoon']['position'][2]
    for seed in range(100):
        a=t.reset(seed,True)
        assert t._valid_layout(a['poses']) and a==t.reset(seed,True)
    assert t.reset(2,False)['poses']==original['poses']
    assert t.reset(2,True)['poses']!=original['poses']
    bad=copy.deepcopy(t.config);bad['objects']['bowl']['support']='spoon'
    with pytest.raises(ValueError,match='Cyclic'):create_task(bad,load())
    bad=copy.deepcopy(t.config);bad['environment']['rear_table']['position_m'][0]+=.01
    with pytest.raises(ValueError,match='adjoin'):create_task(bad,load())


def test_metric_requires_full_volume_correct_bin_and_dwell():
    t=task();poses=copy.deepcopy(t.layout['poses'])
    assert not t.metrics(poses)['summary']['success']
    target=poses['basket'];poses['plate']={
        'position':(np.array(target['position'])+[0,0,.005]).tolist(),
        'quaternion_wxyz':target['quaternion_wxyz'][:]}
    assert t.metrics(poses,.2)['plate']['contained']
    assert not t.metrics(poses)['plate']['success']  # reads cannot accumulate time
    assert t.metrics(poses,.31)['plate']['success']
    poses['plate']['position'][2]+=.2
    assert not t.metrics(poses)['plate']['contained']
    poses['plate']['position'][2]-=.2
    poses['plate']['position'][1]+=.08  # centre inside, rim outside
    assert not t.metrics(poses)['plate']['contained']
    # A moving target bin must be evaluated in its current local frame.
    poses['basket']['position'][0]+=.2
    assert not t.metrics(poses)['plate']['success']


def test_task_imports_no_physics_engine():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',
        "import sys; from piper_sim.config import load; from piper_sim.tasks import create_task; "
        "create_task('configs/tasks/clean_table.yaml',load()); "
        "assert not any(n in sys.modules for n in ('mujoco','torch','isaaclab','piper_sdk'))"],check=True)


def test_mujoco_stability_open_containers_and_recording(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.contracts import Command
    from piper_sim.recording import EpisodeWriter
    import h5py
    cfg=load();cfg['task_scene']='configs/tasks/clean_table.yaml'
    b=MujocoBackend(cfg)
    try:
        assert b.model.nu==14 and b.model.ncam==3
        for _ in range(90):b.step()
        obs=b.read();t=b.task
        for n in ('placemat','basket','trash_bin'):
            assert obs.extras['objects'][n]['position'][2]==pytest.approx(t.table_top,abs=.002)
        assert obs.extras['objects']['bowl']['position'][2]>t.table_top+.007
        # A real free-body drop through each opening must settle on its floor.
        for n,target in [('plate','basket'),('paper','trash_bin')]:
            p=obs.extras['objects'][target];j=b.model.joint('task/'+n+'/free');q=j.qposadr[0]
            b.data.qpos[q:q+7]=[* (np.array(p['position'])+[0,0,.04]),*p['quaternion_wxyz']]
            b.data.qvel[j.dofadr[0]:j.dofadr[0]+6]=0
        for _ in range(90):b.step()
        obs=b.read()
        assert obs.extras['task']['metrics']['plate']['success']
        assert obs.extras['task']['metrics']['paper']['success']
        path=tmp_path/'sorting.h5'
        with EpisodeWriter(path,{'task_scene':t.metadata()}) as w:
            action=Command(b.control_targets(),time.monotonic(),0)
            b.apply(action);b.step();w.append(obs,action,b.read())
        with h5py.File(path) as f:
            extras=json.loads(f['observation/extras_json'][0])
            assert extras['task']['name']=='clean_table'
            assert len(extras['objects'])==11
        b.reset(18,True);a=copy.deepcopy(b.task.layout)
        b.reset(18,True);assert b.task.layout==a
    finally:b.close()


def test_vessel_contact_sections_leave_inner_cavity_empty():
    from piper_sim.tasks.clean_table_meshes import convex_profile_parts
    profile=np.array([(.01,0),(.48,0),(.54,.06),(.58,.15),(.80,.54),(.98,.93),
                      (1,.98),(.99,1),(.94,1),(.91,.93),(.72,.51),(.48,.13),(.01,.13)])
    parts=convex_profile_parts(profile)
    def inside(point,polygon):
        d=np.roll(polygon,-1,axis=0)-polygon
        p=point-polygon
        return np.all(d[:,0]*p[:,1]-d[:,1]*p[:,0]>=-1e-10)
    assert len(parts)>1
    # A single full-section hull fills this interior point above the bowl floor.
    assert not any(inside(np.array([.5,.5]),profile[ids]) for ids in parts)
    assert any(inside(np.array([.3,.05]),profile[ids]) for ids in parts)
