"""Shared SI-unit contract. Left/right are viewed from behind the robot."""
from dataclasses import dataclass, field
from typing import Protocol
import math
import numbers
import numpy as np

ARMS = ("left", "right")
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
FEATURE_NAMES = tuple(f"{side}.{name}" for side in ARMS
                      for name in (*JOINT_NAMES, "gripper_width"))
LOWER = np.array([-2.618, 0, -2.967, -1.745, -1.22, -2.0944, 0] * 2)
UPPER = np.array([2.618, 3.14, 0, 1.745, 1.22, 2.0944, .07] * 2)


def vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError("Expected 14 finite values: left[6 rad + width m], right[6 rad + width m]")
    return result.copy()


@dataclass(frozen=True)
class Command:
    position: np.ndarray
    timestamp: float  # monotonic seconds on the local host
    sequence: int

    def __post_init__(self):
        object.__setattr__(self, "position", vector(self.position))
        if not math.isfinite(self.timestamp):
            raise ValueError("Non-finite command timestamp")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, numbers.Integral) or self.sequence < 0:
            raise ValueError("Sequence must be a nonnegative integer")


@dataclass
class Observation:
    position: np.ndarray
    timestamp: float  # monotonic seconds; never compare across hosts
    sim_time: float | None = None
    images: dict[str, np.ndarray] = field(default_factory=dict)
    image_timestamps: dict[str, float] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        self.position = vector(self.position)


class Backend(Protocol):
    def read(self) -> Observation: ...
    def apply(self, command: Command) -> Command: ...
    def close(self) -> None: ...
