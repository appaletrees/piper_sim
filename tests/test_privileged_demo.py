from types import SimpleNamespace
import numpy as np
import pytest

mujoco=pytest.importorskip('mujoco')
from piper_sim.experts.grasp_assist import ContactGraspAssist, finger_contact_evidence
from piper_sim.experts.mujoco_ik import PiperIK, grasp_rotation


def test_assist_requires_contact_and_preserves_activation_pose():
    xml='''<mujoco><option gravity="0 0 0"/><worldbody>
    <body name="left/link6" pos="0 0 .5">
      <body name="left/link7" pos="0 -.025 0"><geom type="box" size=".03 .005 .03"/></body>
      <body name="left/link8" pos="0 .025 0"><geom type="box" size=".03 .005 .03"/></body>
    </body>
    <body name="task/paper" pos=".3 0 .5"><freejoint/><geom type="box" size=".025 .023 .025" mass=".02"/></body>
    </worldbody><equality><weld name="demo_grasp/left/paper" body1="left/link6" body2="task/paper" active="false"/></equality></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    b=SimpleNamespace(model=model,data=data);assist=ContactGraspAssist(b)
    mujoco.mj_forward(model,data)
    with pytest.raises(RuntimeError,match='No finger contact'):assist.engage('left','paper')
    assert not data.eq_active[0]
    data.qpos[:2]=[0,.02];mujoco.mj_forward(model,data)
    with pytest.raises(RuntimeError,match='both sides'):assist.engage('left','paper')
    assert not data.eq_active[0]  # touching one finger is not a pinch
    # Test fixture creates an actual opposed-finger contact, outside any episode.
    data.qpos[:2]=0;mujoco.mj_forward(model,data)
    q,v=data.qpos.copy(),data.qvel.copy()
    evidence = finger_contact_evidence(b, 'left', 'paper')
    assert all(e['normal_force_N'] > .005 and e['contact_count'] > 0 for e in evidence.values())
    result=assist.engage('left','paper')
    assert result['contact_fingers']==['7','8']
    np.testing.assert_array_equal(q,data.qpos);np.testing.assert_array_equal(v,data.qvel)
    mujoco.mj_forward(model,data)
    equality_rows=data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert np.max(abs(data.efc_pos[equality_rows]))<1e-9
    assist.release();assert not data.eq_active[0]


def test_ik_is_bounded_and_does_not_change_live_state():
    from piper_sim.config import load
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.contracts import LOWER,UPPER
    b=MujocoBackend(load())
    try:
        before=b.data.qpos.copy();ik=PiperIK(b,'right')
        q,ep,er=ik.solve([.35,-.16,.06],grasp_rotation(0,0),[.4,2,-1.3,0,.9,.4])
        assert ep<.001 and er<.01
        assert np.all(q>=LOWER[:6]) and np.all(q<=UPPER[:6])
        np.testing.assert_array_equal(before,b.data.qpos)
        _,ep,_=ik.solve([3,0,2],grasp_rotation(0,0),q)
        assert ep>.5
    finally:b.close()


def test_physics_evidence_is_read_only_and_tcp_velocity_matches_fk():
    from piper_sim.config import load
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.experts.physics_evidence import physics_snapshot
    b = MujocoBackend(load())
    try:
        # Synthetic fixture velocity; no rollout or hardware connection.
        ids = b.indices['right']['dof']
        b.data.qvel[ids[:6]] = [.02, -.01, .03, -.04, .01, .02]
        mujoco.mj_forward(b.model, b.data)
        arrays = {key: getattr(b.data, key).copy() for key in
                  ('qpos', 'qvel', 'qfrc_applied', 'ctrl', 'eq_active')}
        result = physics_snapshot(b)
        for key, expected in arrays.items():
            np.testing.assert_array_equal(getattr(b.data, key), expected)
        arm = result['arms']['right']
        np.testing.assert_array_equal(arm['joint_position'], b.data.qpos[b.indices['right']['qpos']])
        np.testing.assert_array_equal(arm['combined_drive_generalized_force'],
                                     b.data.qfrc_actuator[ids]+b.data.qfrc_applied[ids])
        before = np.array(arm['tcp_position_world_m'])
        mujoco.mj_integratePos(b.model, b.data.qpos, b.data.qvel, 1e-6)
        mujoco.mj_forward(b.model, b.data)
        after = np.array(physics_snapshot(b)['arms']['right']['tcp_position_world_m'])
        np.testing.assert_allclose((after-before)/1e-6, arm['tcp_linear_velocity_world_m_s'], atol=1e-8)
    finally:
        b.close()
