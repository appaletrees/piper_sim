import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('force_probe_test', scripts/'probe_release_force_floor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_applies_native_force_bound_and_restores_without_other_model_writes(probe, monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".0010416666666666667" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody><body><joint name="jaw" type="slide"/><geom size=".02" mass="1"/></body></worldbody>
      <actuator><position name="right/gripper" joint="jaw" kp="1000" kv="120" forcerange="-5 5"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    data.ctrl[0] = -.1
    teacher = SimpleNamespace(b=SimpleNamespace(model=model, data=data), releasing=('right', 'bowl'))
    tracer = probe.tracing.ContactTrace(model, [0], [0], [0], [0])
    gains, biases, friction = model.actuator_gainprm.copy(), model.actuator_biasprm.copy(), model.geom_friction.copy()
    records = []

    def fake_step(teacher):
        for _ in range(32):
            # Exercise the exact hook used immediately before each integration.
            row = tracer.before(data)
            mujoco.mj_step(model, data)
            assert row['force_range'][0, 0]-1e-12 <= data.actuator_force[0] <= 5+1e-12
            records.append((data.time, *row['force_range'][0]))

    monkeypatch.setattr(probe.RigidTableTeacher, 'step', fake_step)
    monkeypatch.setattr(probe.RigidTableTeacher, 'metadata', lambda teacher: {})
    metadata = {}

    def fake_main():
        for frame in range(40):
            teacher.releasing = ('right', 'bowl') if frame < 20 else None
            probe.RigidTableTeacher.step(teacher)
        metadata.update(probe.RigidTableTeacher.metadata(teacher))
        return 0

    monkeypatch.setattr(probe.tracing, 'main', fake_main)
    monkeypatch.setattr(sys, 'argv', ['probe', '--force-ramp-seconds', '.4'])
    assert probe.main() == 0
    rows = np.asarray(records)
    assert rows[0, 1] == -5
    np.testing.assert_array_equal(rows[:, 2], 5)
    assert np.all(np.diff(rows[:640, 1]) >= -1e-12)
    assert np.all(np.diff(rows[640:, 1]) <= 1e-12)
    np.testing.assert_array_equal(rows[400:640, 1], 0)
    np.testing.assert_array_equal(model.actuator_forcerange[0], [-5, 5])
    for expected, actual in ((gains, model.actuator_gainprm), (biases, model.actuator_biasprm),
                             (friction, model.geom_friction)):
        np.testing.assert_array_equal(expected, actual)
    assert data.ctrl[0] == -.1
    assert [e['event'] for e in metadata['release_force_floor_transitions']] == ['begin_unload', 'begin_restore', 'restored']
    assert probe.RigidTableTeacher.step is fake_step


@pytest.mark.parametrize('options', [['--force-ramp-seconds', '0'], ['--rgb'], ['--trace-phases', 'release']])
def test_probe_rejects_invalid_or_unobserved_modes(probe, monkeypatch, options):
    before = probe.RigidTableTeacher.step
    monkeypatch.setattr(sys, 'argv', ['probe', *options])
    with pytest.raises(ValueError):
        probe.main()
    assert probe.RigidTableTeacher.step is before
