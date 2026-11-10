"""Validate wrench decoding on a separate known-load MuJoCo fixture.

The artificial applied torque is confined to this unit fixture. Collection
traces reject any externally applied payload wrench.
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_payload_wrench import wrenches_about_com


@pytest.mark.parametrize('integrator', ['Euler', 'implicitfast'])
def test_native_contact_wrenches_balance_known_load_at_offset_com(integrator):
    model = mujoco.MjModel.from_xml_string(f'''
        <mujoco><option timestep=".001" integrator="{integrator}"/>
        <default><geom condim="6" friction="1 .1 .05"/></default>
        <worldbody><geom type="plane" size="1 1 .1"/>
          <body name="payload" pos="0 0 .051"><freejoint/>
            <inertial pos=".025 0 0" mass="1" diaginertia=".01 .01 .01"/>
            <geom type="box" size=".1 .1 .05"/>
          </body>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    applied_torque = np.array([.03, .02, .1])
    data.xfrc_applied[1, 3:] = applied_torque
    for _ in range(2000):
        mujoco.mj_step(model, data)
    frames, local, positions, signs = [], [], [], []
    for index, contact in enumerate(data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        frames.append(contact.frame.reshape(3, 3).copy())
        local.append(force)
        positions.append(contact.pos.copy())
        signs.append(model.geom_bodyid[contact.geom2] == 1)
    assert len(local) == 4
    assert np.linalg.norm(np.asarray(local)[:, 3:]) > .01  # includes direct contact torque
    # In this fixture acceleration has settled; soft-contact friction can still
    # permit slow creep. Do not claim static sticking from wrench balance alone.
    assert np.max(np.abs(data.qacc)) < 1e-7
    force, torque = wrenches_about_com(np.asarray(frames), np.asarray(local), signs,
                                      positions, np.tile(data.xipos[1], (len(local), 1)))
    np.testing.assert_allclose(force.sum(axis=0), -model.opt.gravity, atol=2e-7)
    np.testing.assert_allclose(torque.sum(axis=0), -applied_torque, atol=2e-7)
