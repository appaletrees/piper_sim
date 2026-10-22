"""Synchronous engine-independent simulation I/O and episode lifecycle.

The caller owns stepping. No wall-clock controller thread and no CAN imports.
For Isaac, construct its backend after AppLauncher, then pass it to this class.
"""
from pathlib import Path
import time
import numpy as np
from .contracts import Command, FEATURE_NAMES, LOWER, UPPER, vector
from .recording import EpisodeWriter


class SimulationSession:
    def __init__(self, backend, record_dir="outputs/api_episodes"):
        self.backend = backend
        self.record_dir = Path(record_dir)
        self.writer = None
        self.sequence = 0
        self.target = np.clip(backend.read().position, LOWER, UPPER)

    @classmethod
    def mujoco(cls, config=None, *, task_scene=None, cameras=False, viewer=False, **kwargs):
        from .config import load
        from .backends.mujoco import MujocoBackend
        cfg = load(config)
        if task_scene is not None:
            import yaml
            cfg["task_scene"] = yaml.safe_load(Path(task_scene).read_text()) if isinstance(task_scene, (str, Path)) else task_scene
        return cls(MujocoBackend(cfg, cameras=cameras, viewer=viewer), **kwargs)

    def metadata(self):
        b = self.backend
        from .transforms import load_camera_rig
        return {"backend": type(b).__name__, "mode": "sim", "features": list(FEATURE_NAMES),
                "units": ["rad"]*6+["m"]+["rad"]*6+["m"],
                "limits": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
                "fps": 1/(b.dt*b.decimation), "config": b.cfg,
                "camera_rig": b.rig if hasattr(b, "rig") else load_camera_rig(b.cfg),
                "task_scene": b.task.metadata() if b.task else None,
                "state_timestamp_clock": "host_monotonic", "image_timestamp_clock": "simulation"}

    def read(self):
        return self.backend.read()

    def send_target(self, position):
        q = vector(position)
        if np.any(q < LOWER) or np.any(q > UPPER): raise ValueError("Target outside model limits")
        self.target = q

    def step(self, position=None):
        if position is not None: self.send_target(position)
        before = self.read()
        action = self.backend.apply(Command(self.target, time.monotonic(), self.sequence))
        self.sequence += 1
        self.backend.step()
        after = self.read()
        if self.writer: self.writer.append(before, action, after)
        return after

    def reset(self, seed=None, randomize=None):
        if self.writer: raise RuntimeError("Stop recording before reset")
        self.backend.reset(seed=seed, randomize=randomize)
        self.target = np.array(self.backend.cfg['robot']['home']*2)
        self.sequence = 0
        return self.read()

    def start_recording(self, name, task="dual Piper demonstration"):
        import re
        if self.writer: raise RuntimeError("Already recording")
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name): raise ValueError("Invalid episode name")
        self.writer = EpisodeWriter(self.record_dir/(name+'.h5'), {**self.metadata(), "task": task,
            "action_semantics": "applied absolute joint targets and full gripper opening"})

    def stop_recording(self, success=False):
        if not self.writer: raise RuntimeError("Not recording")
        self.writer.close(success=success)
        result = {"path": str(self.writer.path if self.writer.count else self.writer.partial),
                  "frames": self.writer.count}
        self.writer = None
        return result

    def close(self, complete=True):
        if self.writer:
            self.writer.close(complete=complete)
            self.writer = None
        self.backend.close()

    def __enter__(self): return self
    def __exit__(self, exc_type, *_): self.close(complete=exc_type is None)
