"""Joint-control compatibility with example_agilex's DataManager/PiperController.

Explicit refresh/send calls keep one coherent read and one dual-arm write per
iteration. Joint setters use DEGREES; gripper setters use 0..1 = 0..95 mm.
The endpoint may be SimulationClient or SimulationSession. No legacy imports.
"""
import numpy as np
from .contracts import GRIPPER_MAX_WIDTH_M
from .lerobot_alignment import CAMERA_MAPPING, connector_plugin_state_to_si


def si_to_agilex(position):
    q = np.asarray(position, dtype=float).copy()
    q[np.r_[0:6, 7:13]] = np.rad2deg(q[np.r_[0:6, 7:13]])
    q[[6, 13]] /= GRIPPER_MAX_WIDTH_M
    return q


class AgilexAdapter:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.meta = endpoint.metadata()
        self.refresh()
        self.target = np.clip(self.observation.position, self.meta['limits']['lower'],
                              self.meta['limits']['upper'])

    @staticmethod
    def _offset(arm):
        if arm not in ('left', 'right'): raise ValueError("arm must be left or right")
        return 0 if arm == 'left' else 7

    def refresh(self):
        self.observation = self.endpoint.read()
        return self.observation

    def get_current_joint_angles(self, arm='right'):
        i = self._offset(arm)
        return np.rad2deg(self.observation.position[i:i+6])

    def get_current_gripper_open_value(self, arm='right'):
        # Retain tiny physical overshoot in measurements; never silently clip data.
        return float(self.observation.position[self._offset(arm)+6]/GRIPPER_MAX_WIDTH_M)

    def get_target_joint_angles(self, arm='right'):
        i = self._offset(arm)
        return np.rad2deg(self.target[i:i+6])

    def get_target_gripper_open_value(self, arm='right'):
        return float(self.target[self._offset(arm)+6]/GRIPPER_MAX_WIDTH_M)

    get_gripper_open_value = get_target_gripper_open_value

    def set_target_joint_angles(self, joint_angles, arm='right'):
        q = np.asarray(joint_angles, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all(): raise ValueError("Expected six finite degrees")
        i = self._offset(arm)
        target = self.target.copy(); target[i:i+6] = np.deg2rad(q)
        self._set_target(target)

    def set_gripper_open_value(self, value, arm='right'):
        if not np.isfinite(value) or not 0 <= value <= 1: raise ValueError("Gripper must be in [0,1]")
        target = self.target.copy(); target[self._offset(arm)+6] = value*GRIPPER_MAX_WIDTH_M
        self._set_target(target)

    set_target_gripper_open_value = set_gripper_open_value

    def _set_target(self, target):
        if (np.any(target < self.meta['limits']['lower']) or np.any(target > self.meta['limits']['upper'])):
            raise ValueError("Target outside endpoint limits")
        self.target = target.copy()

    def send_action(self, action):
        """14 values in connector_plugin order: six degrees + normalized gripper, left then right."""
        self._set_target(connector_plugin_state_to_si(action))
        return self.send()

    def send(self):
        """Flush both arms once. HTTP users repeat at 30 Hz; local users then call step()."""
        return self.endpoint.send_target(self.target)

    def capture_frames(self):
        """Like MultiCameraManager: cam_* names, BGR uint8 under 'rgb', depth=None."""
        return {legacy.removesuffix('_rgb'): {
            'rgb': self.observation.images[name][..., ::-1].copy(), 'depth': None,
            'timestamp': self.observation.image_timestamps[name], 'color_order': 'BGR'}
            for legacy, name in CAMERA_MAPPING.items() if name in self.observation.images}

    def get_observation(self):
        """Policy data: degrees/normalized state + actual RGB HWC images (no implicit resizing)."""
        return {'observation.state': si_to_agilex(self.observation.position).astype(np.float32),
                **{'observation.images.'+legacy: self.observation.images[name].copy()
                   for legacy, name in CAMERA_MAPPING.items() if name in self.observation.images},
                'sim_time': self.observation.sim_time, 'extras': self.observation.extras}

    def reset(self, seed=None, randomize=None):
        self.endpoint.reset(seed=seed, randomize=randomize)
        self.meta = self.endpoint.metadata()
        self.refresh()
        self.target = np.clip(self.observation.position, self.meta['limits']['lower'], self.meta['limits']['upper'])
        return self.get_observation()

    def start_recording(self, name, task='dual Piper demonstration'):
        return self.endpoint.start_recording(name, task)

    def stop_recording(self, success=False):
        return self.endpoint.stop_recording(success)
