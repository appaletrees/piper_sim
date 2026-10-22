import json
import time
import urllib.error
import urllib.request
import numpy as np
import pytest
from piper_sim.client import SimulationClient
from piper_sim.control import CommandServer
from piper_sim.contracts import Observation, UPPER


def client_for(server):
    return SimulationClient(f'http://127.0.0.1:{server.server.server_port}')


def post(client, path, body):
    return json.loads(client._request(path, body)[0])


def test_atomic_observation_png_metadata_and_95mm_range():
    server = CommandServer(0)
    client = client_for(server)
    q = [0, .8, -.6, 0, 0, 0, .095]*2
    pixels = np.arange(36, dtype=np.uint8).reshape(3,4,3)
    obs = Observation(q, time.monotonic(), .5, {'overhead': pixels}, {'overhead': .5}, {'objects': {'x': 1}})
    try:
        server.publish(obs, {'applied_sequence': 7})
        # Modifying the owner's buffers must not alter a published frame.
        pixels[:] = 0
        obs.extras['objects']['x'] = 99
        bundle = client.observation()
        assert bundle['state']['sim_time'] == bundle['state']['image_timestamps']['overhead'] == .5
        assert bundle['state']['extras']['objects']['x'] == 1
        np.testing.assert_array_equal(bundle['images']['overhead'], np.arange(36).reshape(3,4,3))
        rgb, frame = client.image('overhead')
        np.testing.assert_array_equal(rgb, bundle['images']['overhead'])
        assert frame == bundle['state']['frame_id']
        assert client.metadata()['limits']['upper'][6] == .095
        assert client.send_target(q) == 0
        assert client.send_target(q) == 1
        q[6] = .096
        with pytest.raises(urllib.error.HTTPError) as error: client.send_target(q)
        assert error.value.code == 400
        assert client.state()['last_received_sequence'] == 1
    finally: server.close()


def test_reset_and_recording_are_owner_thread_mailboxes():
    server = CommandServer(0, allow_reset=True, allow_recording=True)
    client = client_for(server)
    try:
        server.publish(Observation([0]*14, time.monotonic()))
        client.send_target([0]*14, sequence=5)
        request = post(client, '/reset', {'seed': 42, 'randomize': True})
        assert server.reset_id == 0
        assert server.poll_reset() == request
        assert server.poll_reset() is None
        assert server.sample() is None
        server.complete_reset(request['reset_id'])
        server.publish(Observation([0]*14, time.monotonic()))
        assert client.state()['reset_id'] == request['reset_id']
        # Sequence high-water survives reset; old packets cannot revive old targets.
        with pytest.raises(urllib.error.HTTPError): client.send_target([0]*14, sequence=5)
        record = post(client, '/recording/start', {'name': 'episode_001', 'task': 'insert plug'})
        assert not client.recording_status()['active']
        assert server.poll_recording() == record
        with pytest.raises(urllib.error.HTTPError) as error: post(client, '/reset', {})
        assert error.value.code == 409
        server.update_recording(request_id=record['request_id'], active=True)
        with pytest.raises(urllib.error.HTTPError) as error: post(client, '/reset', {})
        assert error.value.code == 409
        stop = post(client, '/recording/stop', {'success': True})
        server.update_recording(request_id=stop['request_id'], active=False)
        assert not client.recording_status()['active']
        for body in ({'seed': True}, {'randomize': 'yes'}, {'seed': -1}):
            with pytest.raises(urllib.error.HTTPError) as error: post(client, '/reset', body)
            assert error.value.code == 400
        with pytest.raises(urllib.error.HTTPError): post(client, '/recording/start', {'name': '../outside'})
    finally: server.close()


def test_readonly_limits_and_lifecycle_disabled():
    upper = UPPER.copy(); upper[[6,13]] = .07
    server = CommandServer(0, allow_commands=False, upper=upper)
    client = client_for(server)
    try:
        assert client.metadata()['limits']['upper'][6] == .07
        for endpoint in ('/target', '/reset', '/recording/start'):
            with pytest.raises(urllib.error.HTTPError) as error: post(client, endpoint, {})
            assert error.value.code == 403
    finally: server.close()


def test_agilex_units_randomization_and_episode_boundaries(tmp_path):
    pytest.importorskip('mujoco')
    import h5py
    from piper_sim.simulation import SimulationSession
    from piper_sim.agilex import AgilexAdapter
    from piper_sim.lerobot_alignment import connector_plugin_state_to_si
    with SimulationSession.mujoco(task_scene='configs/tasks/connector.yaml', record_dir=tmp_path) as sim:
        adapter = AgilexAdapter(sim)
        adapter.reset(seed=10, randomize=True)
        layout = sim.backend.task.layout.copy()
        adapter.reset(seed=11, randomize=True)
        assert sim.backend.task.layout != layout
        adapter.reset(seed=10, randomize=True)
        assert sim.backend.task.layout == layout
        adapter.set_target_joint_angles([10, 15, -20, 0, 0, 0], arm='left')
        adapter.set_gripper_open_value(1., arm='left')
        adapter.set_gripper_open_value(.5, arm='right')
        adapter.send()
        assert sim.target[0] == pytest.approx(np.deg2rad(10))
        assert sim.target[6] == .095 and sim.target[13] == .0475
        adapter.start_recording('test_episode', 'test units')
        with pytest.raises(RuntimeError, match='Stop recording'): adapter.reset(seed=11)
        for _ in range(90): sim.step()
        adapter.refresh()
        assert adapter.get_current_gripper_open_value('left') == pytest.approx(1, abs=.015)
        np.testing.assert_allclose(connector_plugin_state_to_si(adapter.get_observation()['observation.state']),
                                   sim.read().position, atol=1e-7)
        result = adapter.stop_recording(success=True)
        with h5py.File(result['path']) as f:
            assert f.attrs['complete'] and f.attrs['success']
            assert f['action/position'].shape == (90,14)
            assert f['action/position'][0,6] == pytest.approx(.095)
            assert json.loads(f.attrs['metadata'])['task_scene']['layout']['seed'] == 10
        adapter.reset(seed=12)
        with pytest.raises(FileExistsError): adapter.start_recording('test_episode')


def test_agilex_camera_names_and_color_order():
    from piper_sim.agilex import AgilexAdapter
    server = CommandServer(0)
    try:
        pixels = np.full((2,2,3), [255,20,10], dtype=np.uint8)
        server.publish(Observation([0]*14, time.monotonic(), 0, {'left_wrist': pixels}, {'left_wrist': 0}))
        adapter = AgilexAdapter(client_for(server))
        np.testing.assert_array_equal(adapter.capture_frames()['cam_left_wrist']['rgb'][0,0], [10,20,255])
        np.testing.assert_array_equal(adapter.get_observation()['observation.images.cam_left_wrist_rgb'][0,0], [255,20,10])
    finally: server.close()
