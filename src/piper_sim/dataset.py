"""Model-neutral sequence reader. Model-specific normalization belongs downstream."""
import json
import h5py
import numpy as np


def validate_episode(path):
    with h5py.File(path, "r") as f:
        if not f.attrs.get("complete", False):
            raise ValueError("Episode is incomplete")
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
            if isinstance(obj, h5py.Dataset) and obj.shape[0] != n:
                raise ValueError(f"Inconsistent length: {name}")
        f.visititems(check)
        return {"transitions": n, "metadata": json.loads(f.attrs["metadata"]),
                "cameras": list(f.get("observation/images", {})),
                "features": json.loads(f.attrs["features"])}


def sequence_windows(path, horizon=16, stride=1):
    """Never crosses episode boundaries; returns raw SI state/action plus RGB sequences."""
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    info = validate_episode(path)
    with h5py.File(path, "r") as f:
        for start in range(0, info["transitions"]-horizon+1, stride):
            selection = slice(start, start+horizon)
            yield {
                "state": f["observation/state"][selection],
                "action": f["action/position"][selection],
                "next_state": f["next_observation/state"][selection],
                "timestamp": f["observation/timestamp"][selection],
                "images": {name: f[f"observation/images/{name}"][selection]
                           for name in info["cameras"]},
                "task": info["metadata"].get("task", ""),
            }
