"""Visual authoring must preserve the calibrated workcell and grasp physics."""
import copy
import json
import numpy as np
import pytest
import yaml
from piper_sim.config import ROOT, load
from piper_sim.tasks import create_task


def configs():
    enhanced=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    original=copy.deepcopy(enhanced);original['appearance'].pop('asset_library')
    return original,enhanced


def test_visual_library_preserves_contacts_dimensions_and_layout():
    old,new=(create_task(c,load()) for c in configs())
    assert new.visual_asset_metadata['generator'].startswith('Blender 4.5')
    for n in old.names:
        np.testing.assert_array_equal(old.bounds[n],new.bounds[n])
        np.testing.assert_array_equal(old.surface_points[n],new.surface_points[n])
        assert [p for p in old.parts[n] if p['collision']]==[p for p in new.parts[n] if p['collision']]
        visuals=[p for p in new.parts[n] if not p['hidden']]
        assert visuals and all(p['name'].startswith('blender_') for p in visuals)
        for p in visuals:
            xyz,uv,normals,faces=(np.array(p[k]) for k in ('points','uv','normals','faces'))
            assert xyz.shape==normals.shape and uv.shape==(len(xyz),2)
            assert np.isfinite(np.concatenate([xyz,normals,uv],axis=1)).all()
            assert faces.min()>=0 and faces.max()<len(xyz)
            np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=2e-5)
    for seed in (0,1,42,43):
        assert old.reset(seed,True)['poses']==new.reset(seed,True)['poses']
    changed=copy.deepcopy(configs()[1]);changed['objects']['cup_00']['height_m']+=.001
    with pytest.raises(ValueError,match='dimensions changed'):
        create_task(changed,load())


def test_blender_contacts_and_cameras_compile_to_same_physics():
    pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    backends=[]
    try:
        for task in configs():
            c=load('configs/clean_table_workcell.yaml');c['task_scene']=task
            backends.append(MujocoBackend(c))
        a,b=backends
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_range','actuator_gainprm',
                      'actuator_biasprm','actuator_ctrlrange','cam_pos','cam_quat','cam_fovy'):
            np.testing.assert_array_equal(getattr(a.model,field),getattr(b.model,field),err_msg=field)
        for name,parts in a.task.parts.items():
            for p in parts:
                if not p['collision']:continue
                full='task/'+name+'/'+p['name'];ga,gb=a.model.geom(full).id,b.model.geom(full).id
                for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp'):
                    np.testing.assert_allclose(getattr(a.model,field)[ga],getattr(b.model,field)[gb],atol=1e-12,err_msg=full+' '+field)
        for _ in range(20):a.step();b.step()
        np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-8,rtol=0)
    finally:
        for b in backends:b.close()


def test_baked_maps_have_real_variation_and_portable_references():
    from PIL import Image
    path=ROOT/'assets/blender/clean_table/manifest.json'
    manifest=json.loads(path.read_text())
    for spec in manifest['materials'].values():
        for field in ('texture','detail_texture','normal_texture','roughness_texture'):
            if field in spec:
                assert not spec[field].startswith('/')
                assert (path.parent/spec[field]).is_file()
    for family in ('paper','plastic','porcelain','rubber','table','metal'):
        for channel in ('albedo','roughness','normal'):
            pixels=np.array(Image.open(path.parent/'textures'/f'{family}_{channel}.png'))
            assert pixels.shape==(1024,1024,3)
            assert np.ptp(pixels[:,:,0])>0,(family,channel)
