import copy

import mujoco
import numpy as np
import pytest

from piper_sim.experts.geometric_capture import closure_target, validate_profile


PROFILE = {'kinds': ['chopstick', 'pen'], 'scan_step_m': .0001, 'overtravel_m': .0003}


def fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="finger1" pos=".001 0 0"><joint name="jaw1" type="slide" axis="1 0 0"/>
        <geom type="box" size=".001 .01 .01" margin=".002"/></body>
      <body name="finger2" pos="-.001 0 0"><joint name="jaw2" type="slide" axis="-1 0 0"/>
        <geom type="box" size=".001 .01 .01" margin=".002"/></body>
      <body name="object"><freejoint/><geom type="box" size=".003 .003 .003"/></body>
      </worldbody></mujoco>''')
    live, scratch = mujoco.MjData(model), mujoco.MjData(model)
    live.qpos[:2] = .004
    live.qvel[:] = .012
    live.qfrc_applied[:] = .123
    mujoco.mj_forward(model, live)
    return model, live, scratch


def state(m, d):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    out = np.empty(mujoco.mj_stateSize(m, kind))
    mujoco.mj_getState(m, d, out, kind)
    return out


def query(m, d, scratch):
    return closure_target(m, d, scratch, jaw_qpos=[0, 1],
        finger_bodies=[m.body('finger1').id, m.body('finger2').id],
        object_body=m.body('object').id, profile=PROFILE)


def test_scan_uses_overlap_not_margin_or_solver_force_and_preserves_live_state():
    m, live, scratch = fixture()
    before = state(m, live)
    parameters = m.geom_solref.copy(), m.geom_solimp.copy(), m.geom_friction.copy()
    result = query(m, live, scratch)
    assert .00589 <= result['bilateral_width_m'] <= .00601
    assert result['target_width_m'] == pytest.approx(result['bilateral_width_m']-.0003)
    assert not result['already_bilateral_at_start']
    assert result['scan_samples'] > 1  # positive-distance margin exists at the first sample
    np.testing.assert_array_equal(state(m, live), before)
    for actual, expected in zip((m.geom_solref, m.geom_solimp, m.geom_friction), parameters):
        np.testing.assert_array_equal(actual, expected)


def test_live_scratch_alias_rejected_before_any_write():
    m, live, _ = fixture()
    before = state(m, live)
    with pytest.raises(ValueError, match='Independent scratch'):
        query(m, live, live)
    np.testing.assert_array_equal(state(m, live), before)


def test_missing_object_contact_fails_without_moving_live_object():
    m, live, scratch = fixture()
    live.qpos[3] = 1.  # the free object's y coordinate
    before = state(m, live)
    with pytest.raises(RuntimeError, match='No bilateral'):
        query(m, live, scratch)
    np.testing.assert_array_equal(state(m, live), before)


@pytest.mark.parametrize('field,value', [
    ('kinds', []), ('kinds', ['pen', 'pen']), ('scan_step_m', True),
    ('scan_step_m', float('nan')), ('scan_step_m', 0.),
    ('overtravel_m', -.001), ('overtravel_m', .003)])
def test_invalid_profile(field, value):
    p = copy.deepcopy(PROFILE);p[field] = value
    with pytest.raises(ValueError): validate_profile(p)
