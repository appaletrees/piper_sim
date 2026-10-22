import copy
import json
import time
import numpy as np
import pytest

from piper_sim.config import ROOT, load
from piper_sim.tasks.connector import ConnectorTask


def task():
    return ConnectorTask('configs/tasks/connector.yaml',load())


def test_cad_scale_volume_holes_and_two_photo_objects():
    t=task();a=t.assets
    np.testing.assert_allclose(np.diff(a['bounds_m'],axis=0)[0],[.17,.06,.033],atol=1e-9)
    assert a['collision_volume_m3']==pytest.approx(a['source_volume_m3'],abs=1e-10)
    assert len(a['holes'])==6
    assert len(t.parts['black_charger'])==4 and len(t.parts['yellow_plug'])==6
    assert sum(p['name'].startswith('pin_') for p in t.parts['black_charger'])==2
    assert sum(p['name'].startswith('pin_') for p in t.parts['yellow_plug'])==3


def test_seeded_layouts_stay_separate_and_reproducible():
    t=task();old=t.reset(0)
    for seed in range(100):
        a=t.reset(seed)
        assert t._valid_layout(a['poses'])
        assert a==t.reset(seed)
    assert t.reset(1)!=old
    assert t.reset(2,False)['poses']==t.reset(3,False)['poses']
    config=copy.deepcopy(t.config);config['objects']=['black_charger']
    assert tuple(ConnectorTask(config,load()).parts)==('black_charger',)


def test_seating_requires_all_pins_inside_correct_holes_orientation_and_hold_time():
    t=task();t.reset(0,False)
    poses=copy.deepcopy(t.layout['poses'])
    assert not any(m['seated'] for m in t.metrics(poses).values())
    for name in t.names: poses[name]=t.target_pose(name)
    assert all(m['seated'] and not m['success'] for m in t.metrics(poses,.1).values())
    assert all(m['success'] for m in t.metrics(poses,.21).values())
    # Within the coarse 1 mm centre tolerance but pin corners hit a 0.5 mm clearance.
    poses['black_charger']['position'][0]+=.0009
    metric=t.metrics(poses)['black_charger']
    assert not metric['all_pins_in_matching_holes'] and not metric['success']
    poses['black_charger']=t.target_pose('black_charger');poses['black_charger']['quaternion_wxyz']=[0,1,0,0]
    assert not t.metrics(poses)['black_charger']['seated']
    poses['yellow_plug']['quaternion_wxyz']=[0,0,0,1]
    assert not t.metrics(poses)['yellow_plug']['seated']


def test_task_does_not_import_an_engine_or_robot():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',
        "import sys; from piper_sim.tasks.connector import ConnectorTask; "
        "from piper_sim.config import load; ConnectorTask('configs/tasks/connector.yaml',load()); "
        "assert not any(x in sys.modules for x in ('mujoco','torch','isaaclab','piper_sdk'))"],check=False)
    assert result.returncode==0


def test_two_plugs_physically_enter_source_holes_and_wrong_placement_is_blocked(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.recording import EpisodeWriter
    from piper_sim.contracts import Command
    import h5py
    cfg=load();cfg['task_scene']='configs/tasks/connector.yaml'
    b=MujocoBackend(cfg)
    try:
        assert b.model.nu==14 and b.model.ncam==3
        b.reset(2,False)
        def place(name,dx):
            p=b.task.target_pose(name);p['position'][0]+=dx;p['position'][2]+=.005 if dx==0 else .021
            j=b.model.joint(f'task/{name}/free');i=j.qposadr[0]
            b.data.qpos[i:i+7]=[*p['position'],*p['quaternion_wxyz']]
        for n in b.task.names: place(n,0)
        for _ in range(45):b.step()
        before=b.read()
        assert all(m['success'] for m in before.extras['task']['metrics'].values())
        path=tmp_path/'task.h5'
        with EpisodeWriter(path,{'task_scene':b.task.metadata()}) as writer:
            action=Command(b.control_targets(),time.monotonic(),0)
            b.apply(action);b.step();writer.append(before,action,b.read())
        with h5py.File(path) as f:
            extras=json.loads(f['observation/extras_json'][0])
            assert set(extras['objects'])=={'board','black_charger','yellow_plug'}
            assert extras['task']['metrics']['black_charger']['success']
        b.reset(2,False)
        for n in b.task.names:place(n,.010)
        for _ in range(45):b.step()
        extras=b.read().extras
        assert not any(m['success'] for m in extras['task']['metrics'].values())
        # At least one pin rests on top of solid material, rather than crossing it.
        assert extras['objects']['black_charger']['position'][2]>.065
    finally:b.close()
