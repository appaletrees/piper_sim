"""Explicit clean-table radian/normalized-gripper LeRobot v2.1 reader.

Separate from the degree-based connector_plugin adapter. Playback is a
kinematic visual audit, never a command path or a physical policy rollout.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

from .contracts import ARMS, FEATURE_NAMES, LOWER, UPPER, vector

CAMERAS = {"image": "overhead", "wrist_image_l": "left_wrist", "wrist_image_r": "right_wrist"}
PROFILE = "piper_rad15_gripper_fraction95mm_v2_1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rad15_to_si(values):
    """Retain twelve radian joints; convert two SDK fractions to full stroke m.

    The auxiliary coordinate is not a joint, a stop flag, or a physics input.
    No clipping, temporal shift, or degree conversion is performed.
    """
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim not in (1, 2) or raw.shape[-1] != 15 or not np.isfinite(raw).all():
        raise ValueError('Expected finite 15D state/actions under the explicit rad15 profile')
    q = raw[..., :14].copy()
    if np.any(q[..., [6, 13]] < 0) or np.any(q[..., [6, 13]] > 1):
        raise ValueError('Expected normalized grippers in [0,1]')
    q[..., [6, 13]] *= .095
    return q


def limit_report(positions):
    q = np.asarray(positions)
    excess = np.maximum(np.maximum(LOWER - q, q - UPPER), 0)
    return {
        'frames_outside_exact_limits': int(np.any(excess > 0, axis=1).sum()),
        'clipping_applied': False,
        'features': [{'feature': name, 'frames': int(np.count_nonzero(excess[:, i])),
                      'max_excess': float(excess[:, i].max()),
                      'unit': 'm' if i in (6, 13) else 'rad'}
                     for i, name in enumerate(FEATURE_NAMES) if np.any(excess[:, i] > 0)],
    }


@dataclass
class RecordedEpisode:
    root: Path
    parquet: Path
    index: int
    fps: float
    timestamps: np.ndarray
    state_raw: np.ndarray
    action_raw: np.ndarray
    state_si: np.ndarray
    action_si: np.ndarray
    videos: dict[str, Path]
    resolution: tuple[int, int]

    def __len__(self):
        return len(self.timestamps)


def load_rad15_episode(dataset, episode=0):
    """Load and validate saved frame correspondence; source files stay read-only.

    Shape alone does NOT establish units. Calling this named reader selects the
    explicit contract documented by phoenix-mwa/configs/data/piper_sources.json
    and docs/PIPER_DATA.md for the user-provided clean-table dataset.
    """
    import pyarrow.parquet as pq
    root = Path(dataset).expanduser().resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('codebase_version') != 'v2.1' or info.get('robot_type') != 'piper':
        raise ValueError('Expected Piper LeRobot v2.1')
    if type(episode) is not int or not 0 <= episode < info['total_episodes']:
        raise ValueError('Episode index outside dataset')
    if info.get('fps') != 30:
        raise ValueError('The declared rad15 recording contract requires 30 fps')
    features = info['features']
    for field in ('state', 'actions'):
        if features[field].get('shape') != [15] or features[field].get('dtype') != 'float32':
            raise ValueError(f'Expected float32 {field}[15]')
    dimensions = []
    for key in CAMERAS:
        f = features[key]
        if f.get('dtype') != 'video' or len(f['shape']) != 3 or f['shape'][2] != 3:
            raise ValueError(f'Expected HWC RGB video: {key}')
        dimensions.append(tuple(f['shape'][:2]))
    if len(set(dimensions)) != 1:
        raise ValueError('This comparison requires equal native camera resolutions')
    chunk = episode // info['chunks_size']

    def inside(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Dataset path escapes root')
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    parquet = inside(info['data_path'].format(episode_chunk=chunk, episode_index=episode))
    table = pq.read_table(parquet)
    frames = np.asarray(table['frame_index'])
    n = len(frames)
    if not n or not np.array_equal(frames, np.arange(n)):
        raise ValueError('Nonconsecutive saved frame indices')
    if not np.all(np.asarray(table['episode_index']) == episode):
        raise ValueError('Episode identity mismatch')
    timestamps = np.asarray(table['timestamp'], dtype=np.float64)
    # float32 timestamps accumulate representation error, not sampling drift.
    if not np.isfinite(timestamps).all() or not np.allclose(timestamps, frames / 30, rtol=1e-7, atol=1e-5):
        raise ValueError('Saved timestamp does not match frame_index / 30')
    states = np.asarray(table['state'].to_pylist(), dtype=np.float64)
    actions = np.asarray(table['actions'].to_pylist(), dtype=np.float64)
    if states.shape != (n, 15) or actions.shape != (n, 15):
        raise ValueError('Unexpected numeric array shape')
    videos = {key: inside(info['video_path'].format(episode_chunk=chunk, episode_index=episode, video_key=key))
              for key in CAMERAS}
    return RecordedEpisode(root, parquet, episode, 30., timestamps, states, actions,
                           rad15_to_si(states), rad15_to_si(actions), videos,
                           (dimensions[0][1], dimensions[0][0]))


def place_kinematic_observation(backend, position):
    """Exact recorded MuJoCo pose, including reported limit overshoot.

    This private replay path deliberately does not use apply()/step(). Those
    control APIs retain strict limits. mj_forward computes poses only; model
    limits, CAD transforms, initial task objects and simulation time stay intact.
    """
    import mujoco
    q = vector(position)
    for i, side in enumerate(ARMS):
        a = q[7*i:7*i+7]
        ids = backend.indices[side]
        backend.data.qpos[ids['qpos']] = [*a[:6], a[6]/2, a[6]/2]
        backend.data.qvel[ids['dof']] = 0
    mujoco.mj_forward(backend.model, backend.data)
    backend._images = None
