"""Random layout, real open-cup contacts and existing observation/recording contract."""
import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from piper_sim.config import ROOT, load
from piper_sim.tasks import create_task


def config(count=2):
    c=yaml.safe_load((ROOT/'configs/tasks/paper_cups.yaml').read_text())
    c['cup']['count']=count
    return c


def test_photo_dimensions_open_geometry_and_reproducible_clearance():
    t=create_task(config(),load('configs/paper_cups_workcell.yaml'))
    np.testing.assert_allclose(t.bounds['cup_00'][1]-t.bounds['cup_00'][0],[.08,.08,.095],atol=1e-8)
    for seed in range(1000):
        a=t.reset(seed,True)
        assert a==t.reset(seed,True)
        xy=np.array([p['position'][:2] for p in a['poses'].values()])
        assert np.linalg.norm(xy[0]-xy[1])>=.115
        assert np.all(xy>=t.center_bounds[0]) and np.all(xy<=t.center_bounds[1])
    assert t.reset(1,True)!=t.reset(2,True)
    assert t.reset(1,False)['poses']==t.reset(2,False)['poses']
    for part in t.parts['cup_00']:
        points=np.asarray(part['points'])
        assert np.isfinite(points).all()
        if part['collision'] and 'floor' not in part['name']:
            # Every convex wall/lip piece lies strictly outside the central
            # cavity. A single convex hull spanning the mouth fails this check.
            center=points[:,:2].mean(0);axis=center/np.linalg.norm(center)
            assert np.min(points[:,:2]@axis)>.026
    assert len([p for p in t.parts['cup_00'] if p['collision']])==65
    assert t.metadata()['config']['cup']['mass_kg']==.008


def test_count_configuration_and_tilt_metrics():
    t=create_task(config(6),load('configs/paper_cups_workcell.yaml'))
    assert len(t.names)==6 and len(t.reset(7,False)['poses'])==6
    poses=copy.deepcopy(t.layout['poses']);poses['cup_00']['quaternion_wxyz']=[2**-.5,2**-.5,0,0]
    metrics=t.metrics(poses)
    assert not metrics['cup_00']['upright']
    assert metrics['summary']['upright_count']==5
    assert not metrics['summary']['success_defined'] and not metrics['summary']['success']


@pytest.mark.parametrize('change', [
    {'cup':{'count':0}}, {'cup':{'wall_m':.05}},
    {'randomization':{'center_bounds_xy_m':[[.0,-.2],[.3,.2]]}},
    {'randomization':{'min_gap_m':-.01}},
])
def test_invalid_specs_fail_before_engine_creation(change):
    c=config()
    for k,v in change.items():c[k].update(v)
    with pytest.raises(ValueError):create_task(c,load('configs/paper_cups_workcell.yaml'))


def test_cups_settle_and_mouth_admits_small_body(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.simulation import SimulationSession
    cfg=load('configs/paper_cups_workcell.yaml');cfg['task_scene']=config()
    def probe(xml):
        root=ET.fromstring(xml)
        b=ET.SubElement(root.find('worldbody'),'body',name='probe',pos='0 0 1')
        ET.SubElement(b,'freejoint',name='probe/free')
        ET.SubElement(b,'inertial',pos='0 0 0',mass='.0005',diaginertia='3.2e-9 3.2e-9 3.2e-9')
        # Default soft sphere settings deliberately differ from thin cup settings.
        ET.SubElement(b,'geom',type='sphere',size='.004')
        return ET.tostring(root,encoding='unicode')
    b=MujocoBackend(cfg,model_xml_transform=probe)
    try:
        b.reset(42,False);q=b.model.joint('probe/free').qposadr[0]
        pos=b.read().extras['objects']['cup_00']['position']
        b.data.qpos[q:q+3]=np.array(pos)+[0,0,.13]
        for _ in range(120):b.step()
        obs=b.read()
        assert obs.extras['task']['metrics']['summary']['upright_count']==2
        assert obs.extras['task']['metrics']['summary']['on_table_count']==2
        assert abs(b.data.body('probe').xpos[2]-.034)<.001
        for seed in (7,23):
            b.reset(seed,True)
            for _ in range(60):b.step()
            assert b.read().extras['task']['metrics']['summary']['on_table_count']==2
        s=SimulationSession(b,record_dir=tmp_path)
        s.reset(8,True);s.start_recording('cups',b.task.config['instruction'])
        for _ in range(3):s.step()
        rec=s.stop_recording(False)
        import h5py
        with h5py.File(rec['path']) as f:
            metadata=json.loads(f.attrs['metadata'])
            assert metadata['task_scene']['layout']['seed']==8
            extra=json.loads(f['observation/extras_json'][0])
            assert sorted(extra['objects'])==['cup_00','cup_01']
            assert extra['task']['name']=='paper_cups'
    finally:b.close()
