import pytest

mujoco = pytest.importorskip('mujoco')
from piper_sim.experts.collision import carried_obstacle_clearance


def test_signed_payload_clearance_includes_near_contacts_and_ignores_internal_robot_visuals():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="1 1 .1" pos="0 0 -1"/>
      <body name="task/stick" pos="0 0 .1"><geom type="sphere" size=".02"/></body>
      <body name="task/other" pos=".045 0 .1"><geom type="sphere" size=".02"/></body>
      <body name="task/stacked" pos=".01 0 .1"><geom type="sphere" size=".02"/></body>
      <body name="right/link7" pos=".01 0 .1"><geom type="sphere" size=".02"/></body>
      <geom name="visual" type="sphere" size="10" contype="0" conaffinity="0"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    distance, pair = carried_obstacle_clearance(model, data, ['stick', 'stacked'], .02)
    # The stacked member is also a payload and is nearer to the obstacle.
    assert distance == pytest.approx(-.005)
    assert pair == ['task/stacked', 'task/other']
    model.body_pos[model.body('task/stacked').id, 0] = -.10
    mujoco.mj_forward(model, data)
    distance, pair = carried_obstacle_clearance(model, data, ['stick', 'stacked'], .02)
    assert distance == pytest.approx(.005)
    assert pair == ['task/stick', 'task/other']
    distance, pair = carried_obstacle_clearance(model, data, ['stick', 'stacked'], .003)
    assert distance == .003 and pair is None
    # A fresh position query must not reuse cached world coordinates.
    model.body_pos[model.body('task/other').id, 0] = .038
    mujoco.mj_forward(model, data)
    distance, _ = carried_obstacle_clearance(model, data, ['stick', 'stacked'], .003)
    assert distance == pytest.approx(-.002)
