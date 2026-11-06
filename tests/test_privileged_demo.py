from types import SimpleNamespace
import numpy as np
import pytest

mujoco=pytest.importorskip('mujoco')
from piper_sim.experts.grasp_assist import ContactGraspAssist
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
    # Test fixture creates an actual opposed-finger contact, outside any episode.
    data.qpos[0]=0;mujoco.mj_forward(model,data)
    q,v=data.qpos.copy(),data.qvel.copy()
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
