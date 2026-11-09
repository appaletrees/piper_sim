"""Rigid inventory, domain reset, real cup nesting and sequential task gates."""
import copy,json
import numpy as np
import pytest,yaml
from piper_sim.config import ROOT,load
from piper_sim.tasks import create_task


def config(randomized=False):
    c=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    if randomized:c['domain_randomization']='configs/domain_randomization/clean_table.yaml'
    return c


def test_rigid_inventory_and_multi_domain_sampling():
    t=create_task(config(True),load())
    assert not {'towel','paper','placemat','trash_bin'} & set(t.names)
    counts,tables,colors,positions=set(),set(),set(),set()
    for seed in range(80):
        layout=t.reset(seed,True)
        assert layout==t.reset(seed,True) and t._valid_layout(layout['poses'])
        counts.add(layout['content']['cup_count'])
        p=t.domain_parameters;tables.add(p['tabletop_material'])
        colors.add(tuple(p['materials']['cup_00_print']['color_rgb']))
        positions.add(tuple(layout['poses']['charger']['position']))
        assert len(t.required)==8+layout['content']['cup_count']
        json.dumps(t.metadata());json.dumps(t.metrics(layout['poses']))
    assert counts=={2,3} and len(tables)==3 and len(colors)>=3 and len(positions)==80
    t.reset(3,False);assert 'cup_02' not in t.active_names and t.domain_parameters is None


def test_stack_task_requires_nesting_on_table_before_final_containment():
    t=create_task(config(),load());poses=copy.deepcopy(t.layout['poses'])
    # Full-volume containment alone cannot certify the requested sequence.
    binpose=np.array(poses['basket']['position'])
    for i,n in enumerate(('cup_00','cup_01')):
        poses[n]={'position':(binpose+[0,0,.003+i*.011]).tolist(),'quaternion_wxyz':[1,0,0,0]}
    assert t.stack_metrics(poses)['nested']
    assert not t.stack_metrics(poses)['on_table']
    assert not t.metrics(poses,1.)['cup_stack']['stacked_on_table_first']
    for i,n in enumerate(('cup_00','cup_01')):poses[n]['position']=[.25,0,t.table_top+i*.011]
    assert t.metrics(poses,.8)['cup_stack']['stacked_on_table_first']
    poses['cup_01']['position'][0]+=.025
    assert not t.stack_metrics(poses)['nested']
    poses['cup_01']['position'][0]-=.025;poses['cup_01']['position'][2]+=.04
    assert not t.stack_metrics(poses)['nested']


def test_runtime_reset_hides_and_excludes_inactive_slots_and_restores_properties():
    mujoco=pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    c=load('configs/clean_table_workcell.yaml');c['task_scene']=config(True)
    b=MujocoBackend(c)
    try:
        nominal=[b.model.cam_pos.copy(),b.model.cam_quat.copy(),b.model.cam_fovy.copy()]
        original=b.model.geom_matid.copy()
        table_mesh=b.model.geom_dataid[b.model.geom('cad_static_0').id]
        assert b.model.mesh_texcoordnum[table_mesh]>0
        g=b.model.geom('cad_static_0').id
        start=b.model.mesh_vertadr[table_mesh];count=b.model.mesh_vertnum[table_mesh]
        world=b.model.mesh_vert[start:start+count]@b.data.geom_xmat[g].reshape(3,3).T+b.data.geom_xpos[g]
        table=c['scene']['table'];center=np.asarray(table['position']);half=np.asarray(table['size'])/2
        np.testing.assert_allclose(world.min(0),center-half,atol=1e-7)
        np.testing.assert_allclose(world.max(0),center+half,atol=1e-7)
        assert 'texture' not in b.task.materials['cup_00_print']
        assert 'texture' in b.task.materials['cup_print']  # independent badge
        captures={}
        for seed in range(12):
            b.reset(seed,True);n=b.task.layout['content']['cup_count']
            material=b.model.material('clean/'+b.task.domain_parameters['tabletop_material']).id
            assert b.model.geom_matid[b.model.geom('cad_static_0').id]==material
            for cup,color in b.task.layout['content']['cup_colors'].items():
                np.testing.assert_allclose(b.model.mat_rgba[b.model.material('clean/'+cup+'_print').id,:3],color)
            captures[n]=seed
            assert len(b.read().extras['objects'])==9+n
            idx=b.model.body('task/cup_02').id;g=b.model.geom_bodyid==idx
            assert np.any(b.model.geom_contype[g])==(n==3)
            if n==2:assert not b.model.geom_rgba[g,3].any()
        assert set(captures)=={2,3}
        b.reset(captures[3],True);state=b.model.mat_rgba.copy();layout=copy.deepcopy(b.task.layout)
        b.reset(captures[2],True);b.reset(captures[3],True)
        np.testing.assert_array_equal(state,b.model.mat_rgba);assert b.task.layout==layout
        b.reset(0,False);np.testing.assert_array_equal(original,b.model.geom_matid)
        for a,z in zip(nominal,(b.model.cam_pos,b.model.cam_quat,b.model.cam_fovy)):np.testing.assert_array_equal(a,z)
        assert not any(b.model.eq_type==mujoco.mjtEq.mjEQ_WELD)
    finally:b.close()


def test_open_cups_nest_by_contact_without_constraints():
    mujoco=pytest.importorskip('mujoco')
    from piper_sim.backends.mujoco import MujocoBackend
    c=load('configs/clean_table_workcell.yaml');c['task_scene']=config();c['task_scene']['nominal_cup_count']=3
    b=MujocoBackend(c)
    try:
        # Explicit drop fixture, not a teacher or a successful robot rollout.
        for i,n in enumerate(('cup_00','cup_01','cup_02')):
            q=b.model.joint('task/'+n+'/free').qposadr[0]
            b.data.qpos[q:q+7]=[.215,0,.025+i*.025,1,0,0,0]
        for _ in range(150):b.step()
        check=b.task.stack_metrics(b.read().extras['objects'])
        assert check['nested'],check
        assert b.task.stack_on_table_verified
        assert not b.read().extras['task']['metrics']['summary']['success']
    finally:b.close()


def test_stick_grasp_ignores_roll_but_tracks_actual_center_and_long_axis():
    from types import SimpleNamespace
    from piper_sim.experts.rigid_table import RigidTableTeacher
    from piper_sim.experts.mujoco_ik import grasp_rotation
    from piper_sim.transforms import pose_matrix
    task=create_task(config(),load())
    teacher=RigidTableTeacher.__new__(RigidTableTeacher)
    teacher.b=SimpleNamespace(task=task)
    teacher.pacing={}
    # Recorded seed-42 failure: neighbouring pickup rolled this stick by 90 deg.
    pose={'position':[.3682261156,-.1521210331,.0279317443],
          'quaternion_wxyz':[.7070912207,-.7070912207,-.0046910072,.0046910072]}
    teacher.object_pose=lambda name:pose
    side,p,r,width=teacher.plan_grasp('chopstick_b')
    t=pose_matrix(pose['position'],pose['quaternion_wxyz'])
    center=t[:3,3]+t[:3,:3]@[0,0,.003]
    np.testing.assert_allclose(p,center+[0,0,.002])
    np.testing.assert_allclose(r[:,2],[0,0,-1],atol=1e-12)
    assert side=='right' and np.isclose(width,.010,atol=1e-6)
    assert abs(np.dot(r[:,1],t[:3,0]))<.02
    # A 180-degree roll about the same physical center must give the same grasp.
    pose['position']=(center+[0,0,.003]).tolist()
    pose['quaternion_wxyz']=[0.,1.,0.,0.]
    _,p2,r2,_=teacher.plan_grasp('chopstick_b')
    np.testing.assert_allclose(p2,center+[0,0,.002])
    np.testing.assert_allclose(r2,grasp_rotation(0,-1))
