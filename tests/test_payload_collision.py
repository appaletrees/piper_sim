import pytest
mujoco = pytest.importorskip("mujoco")
from piper_sim.experts.collision import carried_obstacle_penetration, robot_penetration


@pytest.mark.parametrize('first,second,target,blocked', [
    ('left/link2', 'left/link6', None, True),
    ('left/link2', 'right/link6', None, True),
    ('left/base_link', 'left/link4', None, True),
    ('left/link7', 'task/cup', 'cup', False),
    ('left/link6', 'task/cup', 'cup', True),
    ('left/link7', 'task/basket', 'cup', True),
    ('task/cup', 'task/basket', None, False),
])
def test_nonadjacent_self_interarm_and_target_contacts(first, second, target, blocked):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="{first}" pos="0 0 .2"><freejoint/><geom type="box" size=".03 .03 .03"/></body>
      <body name="{second}" pos=".04 0 .2"><geom type="box" size=".03 .03 .03"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth, pair = robot_penetration(model, data, target)
    assert (depth > .004) == blocked
    assert bool(pair) == blocked
    data.qpos[0] = -.5
    mujoco.mj_forward(model, data)
    assert robot_penetration(model, data, target) == (0., None)


def test_collision_topology_cache_releases_model():
    import gc
    import weakref
    from piper_sim.experts.collision import _MODEL_INDICES
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    data = mujoco.MjData(model)
    robot_penetration(model, data)
    ref = weakref.ref(model)
    assert model in _MODEL_INDICES
    del data, model
    gc.collect()
    assert ref() is None


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
