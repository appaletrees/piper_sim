import mujoco
import numpy as np
import pytest

from piper_sim.experts.support_evidence import payload_support


@pytest.mark.parametrize('reverse', [False, True])
def test_force_sign_matches_weight_for_either_contact_order(reverse):
    # Collider type order puts the payload second against a plane and first
    # against a box. Assert the actual solver ordering, not just XML ordering.
    floor = ('type="box" size="1 1 .01" pos="0 0 -.01"' if reverse else
             'type="plane" size="1 1 .1"')
    m = mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".001"/>
    <worldbody><geom name="floor" {floor}/>
    <body name="task/item" pos="0 0 .06"><freejoint/><geom name="payload" type="sphere" size=".05" mass=".2"/></body></worldbody>
    </mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, nstep=1500);mujoco.mj_forward(m, d)
    assert m.geom(int(d.contact[0].geom[0 if reverse else 1])).name == 'payload'
    before = (d.qpos.copy(), d.qvel.copy(), d.efc_force.copy(), d.qfrc_constraint.copy())
    result = payload_support(m, d, 'item')
    assert result['support_fraction_of_weight'] == pytest.approx(1., abs=1e-5)
    assert result['support_force_world_N'][2] == pytest.approx(.2*9.81, abs=1e-5)
    for original, after in zip(before, (d.qpos, d.qvel, d.efc_force, d.qfrc_constraint)):
        np.testing.assert_array_equal(original, after)


def test_robot_support_is_excluded_and_airborne_payload_has_zero_support():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="right/link7"><geom type="box" size="1 1 .01"/></body>
    <body name="task/item" pos="0 0 .06"><freejoint/><geom size=".05" mass=".2"/></body>
    </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, nstep=1000);mujoco.mj_forward(m, d)
    assert d.ncon > 0
    assert payload_support(m, d, 'item')['support_fraction_of_weight'] == 0
    d.qpos[2] = 1.;mujoco.mj_forward(m, d)
    assert not payload_support(m, d, 'item')['contacts']
