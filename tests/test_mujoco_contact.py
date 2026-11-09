import numpy as np
import pytest
import yaml

from piper_sim.config import ROOT, load
from piper_sim.mujoco_contact import MujocoFingerContact, load_finger_profile, require_contact_engine

mujoco = pytest.importorskip('mujoco')


def profile():
    return load_finger_profile('configs/physics/mujoco_fixed_fingers.yaml')[0]


def contact_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
    <worldbody>
      <body name="left/link7"><geom type="sphere" size=".02"/></body>
      <body name="left/link8" pos="0 1 0"><geom type="sphere" size=".02"/></body>
      <body name="right/link7" pos="1 0 0"><geom type="sphere" size=".02"/></body>
      <body name="right/link8" pos="1 1 0"><geom type="sphere" size=".02"/></body>
      <body name="task/left_object" pos="0 0 .039"><freejoint/><geom type="sphere" size=".02" friction=".65 .003 .0001"/></body>
      <body name="task/right_object" pos="1 0 .039"><freejoint/><geom type="sphere" size=".02" friction=".65 .003 .0001"/></body>
    </worldbody></mujoco>''')


def test_actual_contact_friction_tracks_per_arm_reset_draws_and_is_reproducible():
    model = contact_model(); data = mujoco.MjData(model)
    material = MujocoFingerContact(model, profile())
    before = data.qpos.copy()
    material.apply(42, True)
    first = model.geom_friction.copy(); metadata = material.metadata()
    mujoco.mj_forward(model, data)
    assert len(data.contact) == 2
    for contact in data.contact:
        body = model.body(int(model.geom_bodyid[contact.geom[0]])).name
        side = body.split('/')[0]
        coefficients = metadata['arms'][side]['friction']
        np.testing.assert_allclose(contact.friction,
            [coefficients[0], coefficients[0], coefficients[1], coefficients[2], coefficients[2]])
        np.testing.assert_array_equal(contact.solref, profile()['solref'])
        np.testing.assert_array_equal(contact.solimp, profile()['solimp'])
        assert .8 <= metadata['arms'][side]['friction_scale'] <= 1.2
    assert metadata['arms']['left']['friction_scale'] != metadata['arms']['right']['friction_scale']
    material.apply(43, True); assert not np.array_equal(model.geom_friction, first)
    material.apply(42, True); np.testing.assert_array_equal(model.geom_friction, first)
    assert material.metadata() == metadata
    material.apply(42, False)
    for ids in material.indices.values():
        np.testing.assert_array_equal(model.geom_friction[ids], [profile()['friction']]*len(ids))
    np.testing.assert_array_equal(data.qpos, before)


def test_invalid_material_and_unsupported_engine_fail_without_silent_fallback():
    for key, value in [('friction', [-1, 0, 0]), ('friction_scale_range', [1.2, .8]),
                       ('solref', [0, 1]), ('solimp', [.99, .95, .001, .5, 2]),
                       ('solimp', [.99, .99999, .001, .5, 2]), ('extra', 1)]:
        config = profile(); config[key] = value
        with pytest.raises(ValueError): load_finger_profile(config)
    short = profile(); short['solref'][0] = .001
    with pytest.raises(ValueError, match='at least two'): MujocoFingerContact(contact_model(), short)
    model = contact_model(); model.geom_priority[-1] = 2
    with pytest.raises(ValueError, match='priorities'): MujocoFingerContact(model, profile())
    model = contact_model(); model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    with pytest.raises(ValueError, match='contact override'): MujocoFingerContact(model, profile())
    require_contact_engine({}, 'isaac')
    with pytest.raises(ValueError, match='no equivalent Isaac'):
        require_contact_engine({'mujoco': {'finger_contact': profile()}}, 'isaac')


def test_runtime_mixing_override_cannot_silently_replace_material():
    model = contact_model(); material = MujocoFingerContact(model, profile())
    model.geom_priority[-1] = 1
    with pytest.raises(RuntimeError, match='priorities'): material.assert_unchanged()
    model.geom_priority[-1] = 0
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    with pytest.raises(ValueError, match='contact override'): material.assert_unchanged()
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    material.assert_unchanged()


def test_backend_material_survives_teacher_and_reset_and_is_recorded(monkeypatch):
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.experts.clean_table import CleanTableTeacher
    from piper_sim.experts.rigid_table import RigidTableTeacher
    from piper_sim.simulation import SimulationSession
    cfg = load('configs/clean_table_workcell.yaml')
    cfg['mujoco'] = {'finger_contact': profile()}
    cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
    backend = MujocoBackend(cfg, cameras=False)
    try:
        backend.reset(42, True)
        first = backend.model.geom_friction.copy()
        session = SimulationSession(backend)
        teacher = RigidTableTeacher(session)
        assert teacher.release_preload_m == .003
        assert teacher.rim_release_noslip_iterations == 0
        backend.finger_contact.assert_unchanged()
        # Changing motor force/gain by object must not change material.
        monkeypatch.setattr(CleanTableTeacher, 'sort_one', lambda self, name: None)
        for name in ('plate', 'side_dish', 'pen'):
            teacher.sort_one(name); backend.finger_contact.assert_unchanged()
        meta = session.metadata()['physics_model']['finger_contact']
        assert meta['randomized'] and meta['parameter_origin'] == 'uncalibrated_configuration'
        assert teacher.metadata()['finger_contact'] == meta
        assert teacher.metadata()['per_target_contact_material_switching'] is False
        backend.reset(43, True); backend.reset(42, True)
        np.testing.assert_array_equal(backend.model.geom_friction, first)
        assert session.metadata()['physics_model']['finger_contact'] == meta
        backend.model.geom_friction[backend.finger_contact.all_indices[0], 0] *= 2
        time = backend.sim_time
        with pytest.raises(RuntimeError, match='changed during episode'): backend.step()
        assert backend.sim_time == time
        backend.reset(42, True); backend.step()
    finally:
        backend.close()


@pytest.mark.parametrize('release_iterations', [0, 10])
def test_rim_release_solver_retained_only_during_configured_opening(monkeypatch, release_iterations):
    from types import SimpleNamespace
    from piper_sim.experts.clean_table import CleanTableTeacher
    from piper_sim.experts.rigid_table import RigidTableTeacher
    teacher = object.__new__(RigidTableTeacher)
    teacher.b = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(noslip_iterations=0)),
        task=SimpleNamespace(specs={'dish': {'kind': 'side_dish'}, 'pen': {'kind': 'pen'}}))
    teacher.stack_carried = False
    teacher.rim_release_noslip_iterations = release_iterations
    monkeypatch.setattr(CleanTableTeacher, 'step', lambda self: self.b.model.opt.noslip_iterations)
    for carried, releasing, expected in [
        (('right', 'dish'), None, 10),
        (None, ('right', 'dish'), release_iterations),
        (None, None, 0),
        (None, ('right', 'pen'), 0),
    ]:
        teacher.carried, teacher.releasing = carried, releasing
        assert teacher.step() == expected


def test_http_reset_and_recording_keep_effective_material_metadata(tmp_path):
    import json
    import os
    import socket
    import subprocess
    import sys
    import time
    import urllib.error
    import h5py
    from piper_sim.client import SimulationClient

    with socket.socket() as address:
        address.bind(('127.0.0.1', 0))
        port = address.getsockname()[1]
    log = tmp_path/'server.log'
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, str(ROOT/'scripts/run_scene.py'),
            '--backend', 'mujoco', '--headless', '--steps', '600', '--api-port', str(port),
            '--config', 'configs/clean_table_fixed_contact.yaml',
            '--task-scene', 'configs/tasks/clean_table_rigid.yaml',
            '--record-dir', str(tmp_path)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'MUJOCO_GL': 'egl'})
        client = SimulationClient(f'http://127.0.0.1:{port}', timeout=5.)
        try:
            deadline = time.monotonic()+30
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    client.metadata(); break
                except urllib.error.URLError:
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(.05)
            client.reset(seed=42, randomize=True)
            material = client.metadata()['physics_model']['finger_contact']
            assert material['seed'] == 42 and material['randomized']
            client.start_recording('material', task='stationary metadata diagnostic')
            deadline = time.monotonic()+5
            while client.recording_status()['frames'] < 3:
                assert time.monotonic() < deadline
                time.sleep(.02)
            result = client.stop_recording(success=False)
            with h5py.File(result['path']) as recording:
                assert recording.attrs['complete']
                meta = json.loads(recording.attrs['metadata'])
                assert meta['physics_model']['finger_contact'] == material
            client.reset(seed=43, randomize=True)
            assert client.metadata()['physics_model']['finger_contact']['arms'] != material['arms']
            client.reset(seed=42, randomize=True)
            assert client.metadata()['physics_model']['finger_contact'] == material
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
