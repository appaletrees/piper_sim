"""Explicit real-arm connection API; no simulator or legacy control thread imports.

Each arm exposes six joint angles and ONE gripper opening in SI units. The SDK
adapter converts that opening to one GripperCtrl value (0.001 mm units).
"""
import copy
import math
from pathlib import Path
import re
import time

import numpy as np

from .config import load
from .contracts import Command


def inspect_can_interfaces():
    """Read Linux SocketCAN interface state without opening a bus or changing it."""
    result = []
    for path in sorted(Path("/sys/class/net").iterdir()):
        try:
            if int((path/"type").read_text()) == 280:  # ARPHRD_CAN
                result.append({"name": path.name,
                    "up": bool(int((path/"flags").read_text(), 16) & 1)})
        except (FileNotFoundError, ValueError):
            continue
    return result


def check_can_interfaces(cfg):
    names = [cfg["can_left"], cfg["can_right"]]
    if names[0] == names[1]:
        raise ValueError("Left and right arms need distinct CAN interfaces")
    available = {i["name"]: i for i in inspect_can_interfaces()}
    for name in names:
        if not re.fullmatch(r"[\w.:-]+", name) or name not in available:
            raise ConnectionError(f"SocketCAN interface {name!r} is missing; "
                                  f"available: {list(available)}")
        if not available[name]["up"]:
            raise ConnectionError(f"SocketCAN interface {name!r} is down")


def state_dict(observation):
    """Readable state for the CLI or another application's status display."""
    q = observation.position
    return {"timestamp": observation.timestamp,
            **{side: {"joints_rad": q[i*7:i*7+6].tolist(), "gripper_width_m": float(q[i*7+6])}
               for i, side in enumerate(("left", "right"))}}


class PiperConnection:
    """Dual-arm SDK connection. Default is read-only, including disconnect.

    Actuation requires allow_actuation=True, already-enabled physical motors,
    arm(), and continuously refreshed targets. This API never automatically
    enables motors, homes joints, or configures firmware master/slave mode.
    """
    def __init__(self, hardware_config=None, *, can_left=None, can_right=None,
                 allow_actuation=False, timeout_s=5., sdk_factory=None):
        self.config = copy.deepcopy(hardware_config if hardware_config is not None else load()["hardware"])
        for key, value in (("can_left", can_left), ("can_right", can_right)):
            if value is not None:
                self.config[key] = value
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = timeout_s
        self.allow_actuation = allow_actuation
        self.sdk_factory = sdk_factory
        self._backend = None
        self._sequence = 0
        self._last_target = None

    @property
    def connected(self):
        return self._backend is not None and not self._backend.closed

    @property
    def armed(self):
        return self.connected and self._backend.armed

    def connect(self):
        if self.connected:
            return self
        if self.sdk_factory is None:
            check_can_interfaces(self.config)
        from .backends.piper import PiperBackend
        backend = PiperBackend(self.config, allow_actuation=self.allow_actuation,
                               sdk_factory=self.sdk_factory)
        deadline = time.monotonic()+self.timeout_s
        try:
            while True:
                try:
                    backend.read()  # all joint-pair and gripper frames must be fresh
                    break
                except RuntimeError as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("No complete fresh feedback from both Piper arms; "
                                           "check CAN mapping, bitrate and feedback IDs") from error
                    time.sleep(.01)
        except BaseException:
            backend.close()
            raise
        self._backend = backend
        self._sequence = 0
        self._last_target = None
        return self

    def _require_connection(self):
        if not self.connected:
            raise ConnectionError("Call connect() or use 'with PiperConnection(...)' first")
        return self._backend

    def read(self):
        return self._require_connection().read()

    def arm(self):
        """Allow commands after fresh feedback and physical enable are verified."""
        self._require_connection().arm()

    def apply(self, command):
        """Forward a timestamped command without renewing its age or sequence."""
        accepted = self._require_connection().apply(command)
        self._sequence = max(self._sequence, command.sequence+1)
        self._last_target = accepted.position.copy()
        return accepted

    def send(self, *, left_joints_rad, left_gripper_width_m,
             right_joints_rad, right_gripper_width_m):
        """Send one target per arm. Returns the accepted, slew-limited Command."""
        left = np.asarray(left_joints_rad, dtype=float)
        right = np.asarray(right_joints_rad, dtype=float)
        if left.shape != (6,) or right.shape != (6,):
            raise ValueError("Each arm needs exactly six joint angles")
        command = Command([*left, left_gripper_width_m, *right, right_gripper_width_m],
                          time.monotonic(), self._sequence)
        return self.apply(command)

    def set_gripper_widths(self, *, left_m, right_m):
        """One opening per gripper; hold the last accepted arm targets."""
        q = self._last_target if self._last_target is not None else self.read().position
        return self.send(left_joints_rad=q[:6], left_gripper_width_m=left_m,
                         right_joints_rad=q[7:13], right_gripper_width_m=right_m)

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()
