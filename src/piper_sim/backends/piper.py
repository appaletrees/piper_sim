"""Optional CAN adapter. Construction is read-only; actuation requires explicit arming.

This adapter does not import the legacy controller (which starts control threads).
"""
import sys
import threading
import time
import numpy as np
from ..contracts import Observation, ARMS, LOWER, UPPER, GRIPPER_MAX_WIDTH_M
from ..safety import CommandGuard


class PiperBackend:
    def __init__(self, cfg, *, allow_actuation=False, sdk_factory=None):
        self.cfg = cfg
        self.allow_actuation = allow_actuation
        self.armed = False
        self.closed = False
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.last_send = 0.
        self.fault = None
        self.guard = CommandGuard(max_age=cfg["command_max_age"],
                                  joint_speed=cfg["joint_speed"],
                                  gripper_speed=cfg["gripper_speed"])
        if cfg["can_left"] == cfg["can_right"]:
            raise ValueError("The two arms must use different CAN interfaces")
        if sdk_factory is None:
            sys.path.insert(0, cfg["sdk_path"])
            from piper_sdk import C_PiperInterface_V2
            from .sdk_tracking import tracked_sdk
            sdk_factory = tracked_sdk(C_PiperInterface_V2)
        self.arms = {}
        self.lower, self.upper = LOWER.copy(), UPPER.copy()
        self.sdk_gripper_ranges = {}
        try:
            for side in ARMS:
                sdk = sdk_factory(cfg[f"can_{side}"], start_sdk_joint_limit=True,
                                  start_sdk_gripper_limit=True)
                self.arms[side] = sdk
                requested = cfg.get("gripper_max_width_m")
                if requested is not None:
                    if not np.isfinite(requested) or not 0 < requested <= GRIPPER_MAX_WIDTH_M:
                        raise ValueError("Hardware gripper_max_width_m must be in (0, 0.095]")
                    # Only changes this SDK instance's software clamp; no CAN/firmware command.
                    sdk.SetSDKGripperRangeParam(0., requested)
                low, high = sdk.GetSDKGripperRangeParam()
                if not np.isfinite([low, high]).all() or not 0 <= low < high:
                    raise RuntimeError(f"{side}: invalid SDK gripper range")
                self.sdk_gripper_ranges[side] = [float(low), float(high)]
                index = 6 if side == "left" else 13
                self.lower[index], self.upper[index] = low, min(high, GRIPPER_MAX_WIDTH_M)
                sdk.ConnectPort(piper_init=False)
            self.guard = CommandGuard(max_age=cfg["command_max_age"], joint_speed=cfg["joint_speed"],
                                      gripper_speed=cfg["gripper_speed"], lower=self.lower, upper=self.upper)
        except BaseException:
            self.close()
            raise
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _fresh_frames(self, sdk, ids):
        now = time.monotonic()
        if any(not 0 <= now-sdk.frame_times.get(frame, -float("inf")) <=
               self.cfg["feedback_max_age"] for frame in ids):
            raise RuntimeError("Stale or missing individual CAN feedback frame")

    def read(self):
        values, stamps = [], {}
        wall_now = time.time()
        for side, sdk in self.arms.items():
            with sdk.feedback_lock:
                self._fresh_frames(sdk, (0x2A5, 0x2A6, 0x2A7, 0x2A8))
                joints, gripper = sdk.GetArmJointMsgs(), sdk.GetArmGripperMsgs()
                for name, msg in (("joints", joints), ("gripper", gripper)):
                    stamp = float(msg.time_stamp)
                    if not -.05 <= wall_now - stamp <= self.cfg["feedback_max_age"] or stamp <= 0:
                        raise RuntimeError(f"{side} {name}: stale or missing CAN feedback")
                    stamps[f"{side}.{name}"] = stamp
                values.extend(np.deg2rad([getattr(joints.joint_state, f"joint_{i}") / 1000.
                                          for i in range(1, 7)]))
                values.append(gripper.gripper_state.grippers_angle / 1_000_000.)
        return Observation(values, time.monotonic(), extras={"sdk_wall_timestamps": stamps})

    def arm(self):
        if not self.allow_actuation:
            raise PermissionError("Hardware actuation is disabled")
        with self.lock:
            if self.closed or self.fault:
                raise RuntimeError("Reconnect after a stop/fault")
            self.read()  # require feedback before any control command
            # Existing physical enable is required; never implicitly power the arms.
            for side, sdk in self.arms.items():
                with sdk.feedback_lock:
                    self._fresh_frames(sdk, range(0x261, 0x267))
                    low = sdk.GetArmLowSpdInfoMsgs()
                    if not all(getattr(low, f"motor_{i}").foc_status.driver_enable_status
                               for i in range(1, 7)):
                        raise RuntimeError(f"{side}: all motors must already be enabled by operator")
            self.last_send = time.monotonic()
            self.armed = True

    def apply(self, command):
        with self.lock:
            if not self.armed or self.closed or self.fault:
                raise PermissionError("Hardware is not armed")
            try:
                current = self.read()
                accepted = self.guard.accept(command, current.position, time.monotonic())
                for index, sdk in enumerate(self.arms.values()):
                    q = accepted.position[index*7:index*7+7]
                    sdk.MotionCtrl_2(0x01, 0x01, self.cfg["speed_percent"], 0x00)
                    sdk.JointCtrl(*(int(round(v)) for v in np.rad2deg(q[:6])*1000))
                    sdk.GripperCtrl(int(round(q[6]*1_000_000)),
                                    self.cfg["gripper_effort"], 0x01, 0)
                self.last_send = time.monotonic()
                return accepted
            except BaseException:
                self._stop("command or feedback fault")
                raise

    def _stop(self, reason):
        # Called under RLock. Stop both sides even if one send fails.
        self.fault = reason
        if self.armed:
            self.armed = False
            for sdk in self.arms.values():
                try:
                    sdk.MotionCtrl_1(0x01, 0, 0)
                    sdk.GripperCtrl(0, 0, 0x00, 0)
                except Exception:
                    pass

    def _watch(self):
        while not self.stop_event.wait(.02):
            with self.lock:
                if self.armed and time.monotonic() - self.last_send > self.cfg["command_max_age"]:
                    self._stop("command watchdog timeout")

    def close(self):
        with self.lock:
            if self.closed:
                return
            self._stop("closed")
            self.closed = True
            self.stop_event.set()
            for sdk in self.arms.values():
                try:
                    sdk.DisconnectPort()
                except Exception:
                    pass
