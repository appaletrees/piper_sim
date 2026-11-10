from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from trace_contact_substeps import ContactTrace


@pytest.mark.parametrize('integrator', ['Euler', 'implicitfast'])
def test_payload_pose_and_velocity_share_contact_force_evaluation_time(integrator):
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0" timestep=".001" integrator="{integrator}"/>
      <worldbody><body pos=".2 0 0"><joint type="slide" axis="1 0 0"/>
      <inertial pos=".05 0 0" mass="1" diaginertia=".1 .1 .1"/><geom size=".01"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);d.qpos[0]=.1;d.qvel[0]=.2
    trace=ContactTrace(m,[0],[0],[],[],payload_bodies=[1])
    row=trace.before(d);mujoco.mj_step(m,d);trace.after(d,row,'test','payload')
    assert d.qpos[0] == pytest.approx(.1002)
    assert row['payload_origin_world'][0,0] == pytest.approx(.3)
    assert row['payload_com_world'][0,0] == pytest.approx(.35)
    np.testing.assert_allclose(row['payload_velocity_world'][0],[0.,0.,0.,.2,0.,0.])
    assert row['time_before'] == 0. and row['time_after'] == .001
