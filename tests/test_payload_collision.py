import pytest
mujoco = pytest.importorskip("mujoco")
from piper_sim.experts.collision import carried_obstacle_penetration


@pytest.mark.parametrize('other,carried,blocked', [
    ('task/basket', ['cup'], True),
    ('task/cup2', ['cup','cup2'], False),
    ('right/link7', ['cup'], False),
])
def test_payload_scene_collision_and_intended_contact_exclusions(other, carried, blocked):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="task/cup" pos="0 0 .2"><freejoint/><geom type="box" size=".03 .03 .03"/></body>
      <body name="{other}" pos=".04 0 .2"><geom type="box" size=".03 .03 .03"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth, pair = carried_obstacle_penetration(model, data, carried)
    assert (depth > .004) == blocked
    assert bool(pair) == blocked
