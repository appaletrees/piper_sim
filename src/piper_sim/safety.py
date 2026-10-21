"""Bounds, freshness, alignment and slew limits shared by command consumers."""
import math
import numpy as np
from .contracts import Command, LOWER, UPPER, vector


class CommandGuard:
    def __init__(self, max_age=.15, joint_speed=.5, gripper_speed=.03,
                 max_tracking_error=.35):
        if not all(math.isfinite(x) and x > 0 for x in
                   (max_age, joint_speed, gripper_speed, max_tracking_error)):
            raise ValueError("Guard parameters must be finite and positive")
        self.max_age = max_age
        self.speed = np.array([joint_speed] * 6 + [gripper_speed])[
            np.arange(14) % 7]
        self.max_tracking_error = max_tracking_error
        self.previous = None
        self.previous_time = None
        self.sequence = -1

    def accept(self, command: Command, measured, now: float) -> Command:
        q = vector(command.position)
        measured = vector(measured)
        if not math.isfinite(now) or not math.isfinite(command.timestamp):
            raise ValueError("Non-finite timestamp")
        if not -.01 <= now - command.timestamp <= self.max_age:
            raise ValueError("Stale or future command")
        if command.sequence <= self.sequence:
            raise ValueError("Out-of-order command")
        if (q < LOWER - 1e-6).any() or (q > UPPER + 1e-6).any():
            raise ValueError("Command outside URDF / configured gripper bounds")
        if self.previous is None:
            # A master must first align with the follower. Never jump to home.
            if (np.abs(q - measured) > np.array([.15]*6 + [.01])[
                    np.arange(14) % 7]).any():
                raise ValueError("Master/follower are not aligned at handover")
            self.previous = measured.copy()
            self.previous_time = now - .01
        if now <= self.previous_time:
            raise ValueError("Clock did not advance")
        if (np.abs(self.previous - measured) >
                np.array([self.max_tracking_error]*6 + [.025])[np.arange(14) % 7]).any():
            raise ValueError("Follower tracking error")
        dt = min(now - self.previous_time, .05)
        bounded = self.previous + np.clip(q - self.previous, -self.speed*dt, self.speed*dt)
        self.previous, self.previous_time, self.sequence = bounded, now, command.sequence
        return Command(bounded, command.timestamp, command.sequence)
