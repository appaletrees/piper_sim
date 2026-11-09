"""Curated sequence reader; raw diagnostics require an explicit opt-out.

Model-specific normalization belongs downstream. Privileged extras are never
included in a window; next_state is a transition target, not a policy input.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from .contracts import FEATURE_NAMES


def validate_episode(path):
    """Structural diagnostic only: this does not grant training acceptance."""
    with h5py.File(path, "r") as f:
        return _validate_open_episode(f)


def _validate_open_episode(f):
    if not f.attrs.get("complete", False):
        raise ValueError("Episode is incomplete")
    if (f.attrs.get('schema_version') != 'piper.transitions.v1'
            or tuple(json.loads(f.attrs['features'])) != FEATURE_NAMES
            or json.loads(f.attrs['units']) != ['rad']*6+['m']+['rad']*6+['m']):
        raise ValueError('Unexpected episode schema, feature order or units')
    n = int(f.attrs.get("num_transitions", 0))
    if n < 1:
        raise ValueError("Empty episode")
    for name in ("observation/state", "action/position", "next_observation/state"):
        a = f[name][:]
        if a.shape != (n, 14) or not np.isfinite(a).all():
            raise ValueError(f"Invalid {name}")
    for name in ("observation/timestamp", "action/timestamp", "next_observation/timestamp"):
        t = f[name][:]
        if t.shape != (n,) or not np.isfinite(t).all() or (np.diff(t) < 0).any():
            raise ValueError(f"Invalid timestamps: {name}")
    def check(name, obj):
        if isinstance(obj, h5py.Dataset) and (not obj.shape or obj.shape[0] != n):
            raise ValueError(f"Inconsistent length: {name}")
    f.visititems(check)
    return {"transitions": n, "metadata": json.loads(f.attrs["metadata"]),
            "cameras": list(f.get("observation/images", {})),
            "features": json.loads(f.attrs["features"])}


def sequence_windows(path, horizon=16, stride=1, *, require_quality=True, policy=None):
    """Yield complete windows from an accepted RGB episode, with SI actions.

    Set require_quality=False only for raw diagnostics, never as a quality grant.
    timestamp is source host monotonic time; sim_timestamp and image_timestamps
    are simulation time. No padding, frame skipping within a window, or extras.
    Close the generator on early exit to release its HDF5 read lock.
    """
    if type(horizon) is not int or type(stride) is not int or horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive integers")
    if type(require_quality) is not bool:
        raise ValueError('require_quality must be boolean')
    if not require_quality and policy is not None:
        raise ValueError('A policy cannot be applied when require_quality=False')
    path = Path(path).resolve()
    with h5py.File(path, "r") as f:
        if require_quality:
            from .training_acceptance import verify_training_acceptance
            from .collection_audit import _same_open_file
            verify_training_acceptance(f, path, policy=policy)
        info = _validate_open_episode(f)
        for start in range(0, info["transitions"]-horizon+1, stride):
            if require_quality and not _same_open_file(path, f.id.get_vfd_handle()):
                raise ValueError('Episode path changed during sequence reading')
            selection = slice(start, start+horizon)
            yield {
                "state": f["observation/state"][selection],
                "action": f["action/position"][selection],
                "next_state": f["next_observation/state"][selection],
                "timestamp": f["observation/timestamp"][selection],
                "sim_timestamp": f["observation/sim_time"][selection],
                "next_sim_timestamp": f["next_observation/sim_time"][selection],
                "action_timestamp": f["action/timestamp"][selection],
                "action_sequence": f["action/sequence"][selection],
                "images": {name: f[f"observation/images/{name}"][selection]
                           for name in info["cameras"]},
                "image_timestamps": {name: f[f"observation/image_timestamps/{name}"][selection]
                                     for name in info["cameras"]},
                "task": info["metadata"].get("task", ""),
            }
