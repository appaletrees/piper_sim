from types import SimpleNamespace as NS
import time
import threading
import numpy as np
import pytest
from piper_sim.backends.piper import PiperBackend
from piper_sim.config import load
from piper_sim.contracts import Command


class FakeSDK:
    instances = []

    def __init__(self, name, **kwargs):
        self.name = name
        self.calls = []
        self.stale = False
        self.fail_joint = False
        self.enabled = True
        self.gripper_range = (0., .07)
        self.gripper_feedback = 40000
        self.feedback_lock = threading.RLock()
        self.frame_times = {i: time.monotonic()
                            for i in [0x2A5, 0x2A6, 0x2A7, 0x2A8, *range(0x261, 0x267)]}
        self.instances.append(self)

    def ConnectPort(self, **kwargs):
        self.calls.append(("connect", kwargs))

    def DisconnectPort(self):
        self.calls.append(("disconnect",))

    def GetArmJointMsgs(self):
        return NS(time_stamp=time.time() - (5 if self.stale else 0),
                  joint_state=NS(joint_1=0, joint_2=45000, joint_3=-45000,
                                 joint_4=0, joint_5=0, joint_6=0))

    def GetArmGripperMsgs(self):
        return NS(time_stamp=time.time(), gripper_state=NS(grippers_angle=self.gripper_feedback))

    def GetSDKGripperRangeParam(self):
        return self.gripper_range

    def SetSDKGripperRangeParam(self, low, high):
        self.gripper_range = (low, high)

    def GetArmLowSpdInfoMsgs(self):
        return NS(time_stamp=time.time(), **{
            f"motor_{i}": NS(foc_status=NS(driver_enable_status=self.enabled))
            for i in range(1, 7)})

    def MotionCtrl_2(self, *args):
        self.calls.append(("motion", *args))

    def JointCtrl(self, *args):
        if self.fail_joint:
            raise OSError("CAN failure")
        self.calls.append(("joints", *args))

    def GripperCtrl(self, *args):
        self.calls.append(("gripper", *args))

    def MotionCtrl_1(self, *args):
        self.calls.append(("stop", *args))


def create(actuate=False):
    FakeSDK.instances = []
    return PiperBackend(load()["hardware"], allow_actuation=actuate, sdk_factory=FakeSDK)


def test_read_only_never_sends_motion_and_converts_si():
    p = create()
    q = p.read().position
    assert q[1] == pytest.approx(np.pi/4)
    assert q[6] == pytest.approx(.04)
    with pytest.raises(PermissionError):
        p.arm()
    p.close()
    assert all([c[0] for c in a.calls] == ["connect", "disconnect"] for a in FakeSDK.instances)
    assert FakeSDK.instances[0].calls[0][1] == {"piper_init": False}


def test_forward_units_and_fault_stops_both_arms():
    p = create(True)
    p.arm()
    p.apply(Command(p.read().position, time.monotonic(), 0))
    assert ("joints", 0, 45000, -45000, 0, 0, 0) in FakeSDK.instances[0].calls
    assert ("gripper", 40000, 500, 1, 0) in FakeSDK.instances[0].calls
    FakeSDK.instances[1].fail_joint = True
    time.sleep(.002)
    with pytest.raises(OSError):
        p.apply(Command(p.read().position, time.monotonic(), 1))
    assert all(("stop", 1, 0, 0) in a.calls for a in FakeSDK.instances)
    assert not p.armed
    p.close()


def test_watchdog_and_stale_feedback():
    p = create(True)
    p.arm()
    deadline = time.monotonic() + 1.
    while p.armed and time.monotonic() < deadline:
        time.sleep(.01)
    assert not p.armed
    assert p.fault == "command watchdog timeout"
    p.close()
    p = create()
    FakeSDK.instances[0].stale = True
    with pytest.raises(RuntimeError, match="stale"):
        p.read()
    p.close()


def test_disabled_motors_block_arm():
    p = create(True)
    FakeSDK.instances[1].enabled = False
    with pytest.raises(RuntimeError, match="enabled"):
        p.arm()
    p.close()
    assert not any(c[0] == "motion" for a in FakeSDK.instances for c in a.calls)


def test_one_joint_pair_missing_blocks_even_if_aggregate_is_fresh():
    p = create(True)
    FakeSDK.instances[0].frame_times[0x2A5] = time.monotonic()-1
    with pytest.raises(RuntimeError, match="individual"):
        p.arm()
    p.close()


def test_sdk_range_is_reported_and_oversized_target_rejected():
    p = create(True)
    try:
        assert p.upper[6] == p.upper[13] == .07
        assert p.sdk_gripper_ranges == {'left': [0., .07], 'right': [0., .07]}
        p.arm()
        q = p.read().position
        q[[6,13]] = .095
        with pytest.raises(ValueError, match='bounds'):
            p.apply(Command(q, time.monotonic(), 0))
        assert not any(c[0] == 'joints' for a in FakeSDK.instances for c in a.calls)
    finally:
        p.close()


def test_explicit_confirmed_95mm_sdk_range_and_units():
    cfg = load()['hardware']; cfg['gripper_max_width_m'] = .095
    p = PiperBackend(cfg, allow_actuation=True, sdk_factory=FakeSDK)
    try:
        for sdk in p.arms.values(): sdk.gripper_feedback = 95000
        p.arm()
        p.apply(Command(p.read().position, time.monotonic(), 0))
        assert p.upper[6] == .095
        assert all(('gripper', 95000, 500, 1, 0) in a.calls for a in p.arms.values())
    finally:
        p.close()
