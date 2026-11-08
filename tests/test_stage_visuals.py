"""Appearance regressions must not change the robot or grasp dynamics."""
import copy
import json
import numpy as np
import yaml
from piper_sim.config import ROOT, load
from piper_sim.tasks import create_task


def config():
    c=load('configs/clean_table_workcell.yaml')
    c['task_scene']=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    return c


def test_stage_geometry_and_attachment_contract():
    c=config();task=create_task(c['task_scene'],c)
    assert task.stage_visual_metadata['collision'] is False
    assert task.stage_visual_metadata['generator'].startswith('Blender 4.5')
    parents={p['parent'] for p in task.stage_visuals['parts']}
    assert parents=={'world','left/link2','left/link3','left/link6','right/link2','right/link3','right/link6'}
    for p in task.stage_visuals['parts']:
        xyz,normal,uv,faces=(np.asarray(p[k]) for k in ('points','normals','uv','faces'))
        assert xyz.shape==normal.shape and uv.shape==(len(xyz),2)
        assert np.isfinite(np.concatenate((xyz,normal,uv),axis=1)).all()
        assert faces.min()>=0 and faces.max()<len(xyz)
        np.testing.assert_allclose(np.linalg.norm(normal,axis=1),1,atol=2e-5)
    # New material names have independent RNG streams, so scene layout and
    # dynamics samples remain identical to the previous scene for the same seed.
    old=copy.deepcopy(c['task_scene']);old['appearance'].pop('stage_library')
    old['domain_randomization']=c['task_scene']['domain_randomization']='configs/domain_randomization/clean_table.yaml'
    a,b=create_task(old,c),create_task(c['task_scene'],c)
    for seed in (0,42,43):
        assert a.reset(seed,True)['poses']==b.reset(seed,True)['poses']
        assert a.domain_parameters['objects']==b.domain_parameters['objects']
        assert a.domain_parameters['cameras']==b.domain_parameters['cameras']


def test_stage_changes_no_physics_and_follows_live_links():
    import mujoco
    from piper_sim.backends.mujoco import MujocoBackend
    c=config();old=copy.deepcopy(c);old['task_scene']['appearance'].pop('stage_library')
    backends=[]
    try:
        a,b=(MujocoBackend(x) for x in (old,c));backends=[a,b]
        for field in ('body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat',
                      'jnt_range','jnt_pos','jnt_axis','actuator_gainprm','actuator_biasprm',
                      'actuator_ctrlrange','cam_pos','cam_quat','cam_fovy','cam_intrinsic'):
            np.testing.assert_array_equal(getattr(a.model,field),getattr(b.model,field),err_msg=field)
        # Original contacts remain present, even when a background's appearance
        # is hidden; the new visual geoms cannot contribute inertia or contacts.
        for g in range(a.model.ngeom):
            name=a.model.geom(g).name;h=b.model.geom(name).id
            for field in ('geom_contype','geom_conaffinity','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp'):
                np.testing.assert_array_equal(getattr(a.model,field)[g],getattr(b.model,field)[h],err_msg=name+' '+field)
        for p in b.task.stage_visuals['parts']:
            g=b.model.geom('stage_visual/'+p['name']).id
            assert b.model.geom_contype[g]==b.model.geom_conaffinity[g]==0
            assert b.model.body(int(b.model.geom_bodyid[g])).name==p['parent']
        import xml.etree.ElementTree as ET
        xml=ET.fromstring(b.task_adapter.compose_model(ROOT/'assets/mujoco/workcell.xml',b.task))
        for name,material in b.task.stage_visuals['mujoco_material_bindings'].items():
            g=xml.find(f"worldbody/.//geom[@name='{name}']")
            assert 'rgba' not in g.attrib
            assert g.get('material')=='clean/'+material
        q=np.array(c['task_scene']['reference']['joint14_rad_fraction']);q[[6,13]]*=.095
        for x in backends:x.set_observation_state(q)
        for _ in range(30):a.step();b.step()
        np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-9,rtol=0)
        np.testing.assert_allclose(a.data.cam_xpos,b.data.cam_xpos,atol=1e-10)
        np.testing.assert_allclose(a.data.cam_xmat,b.data.cam_xmat,atol=1e-10)
    finally:
        for b in backends:b.close()
