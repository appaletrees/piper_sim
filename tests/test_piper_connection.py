import time
import numpy as np
import pytest

from piper_sim.config import load
from piper_sim.contracts import Command
from piper_sim.piper_connection import PiperConnection, check_can_interfaces, state_dict
from test_hardware import FakeSDK


def connection(**kwargs):
    FakeSDK.instances = []
    return PiperConnection(load()["hardware"], sdk_factory=FakeSDK, **kwargs)


def test_explicit_connect_context_and_default_read_only():
    robot = connection()
    assert not robot.connected
    assert FakeSDK.instances == []
    with pytest.raises(ConnectionError):
        robot.read()
    with robot:
        assert robot.connected
        info = state_dict(robot.read())
        assert info["left"]["gripper_width_m"] == .04
        assert len(info["right"]["joints_rad"]) == 6
        with pytest.raises(PermissionError):
            robot.set_gripper_widths(left_m=.04, right_m=.04)
    assert not robot.connected
    assert all([c[0] for c in s.calls] == ["connect", "disconnect"] for s in FakeSDK.instances)


def test_one_sdk_gripper_command_per_arm_and_full_width_units():
    with connection(allow_actuation=True) as robot:
        robot.arm()
        accepted = robot.set_gripper_widths(left_m=.04, right_m=.04)
        np.testing.assert_allclose(accepted.position[[6, 13]], .04)
        for sdk in FakeSDK.instances:
            assert [c for c in sdk.calls if c[0] == "gripper"] == [("gripper", 40000, 500, 1, 0)]
        time.sleep(.01)
        second = robot.set_gripper_widths(left_m=.045, right_m=.035)
        assert second.sequence == 1
        assert .04 < second.position[6] < .045
        assert .035 < second.position[13] < .04
        for i, sdk in enumerate(FakeSDK.instances):
            assert sdk.calls[-1] == ("gripper", round(second.position[7*i+6]*1e6), 500, 1, 0)
    assert all(("stop", 1, 0, 0) in sdk.calls for sdk in FakeSDK.instances)


def test_external_command_timestamp_is_not_renewed():
    with connection(allow_actuation=True) as robot:
        robot.arm()
        q = robot.read().position
        with pytest.raises(ValueError, match="Stale"):
            robot.apply(Command(q, time.monotonic()-1., 0))
        assert not robot.armed
        assert all(("stop", 1, 0, 0) in sdk.calls for sdk in FakeSDK.instances)


def test_feedback_timeout_closes_both_connections_without_commands():
    class NoFeedback(FakeSDK):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stale = True
    FakeSDK.instances = []
    robot = PiperConnection(load()["hardware"], sdk_factory=NoFeedback, timeout_s=.02)
    with pytest.raises(TimeoutError, match="fresh feedback"):
        robot.connect()
    assert not robot.connected
    assert all([c[0] for c in s.calls] == ["connect", "disconnect"] for s in FakeSDK.instances)


def test_can_preflight_rejects_missing_down_and_duplicate(monkeypatch):
    import piper_sim.piper_connection as module
    cfg = load()["hardware"]
    monkeypatch.setattr(module, "inspect_can_interfaces", lambda: [])
    with pytest.raises(ConnectionError, match="missing"):
        check_can_interfaces(cfg)
    monkeypatch.setattr(module, "inspect_can_interfaces", lambda: [
        {"name": cfg["can_left"], "up": True}, {"name": cfg["can_right"], "up": False}])
    with pytest.raises(ConnectionError, match="down"):
        check_can_interfaces(cfg)
    cfg["can_right"] = cfg["can_left"]
    with pytest.raises(ValueError, match="distinct"):
        check_can_interfaces(cfg)
