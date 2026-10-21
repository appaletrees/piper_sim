"""Append-only episode writer; stores transitions (obs_t, applied action_t, obs_t+1)."""
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from .contracts import FEATURE_NAMES, Command, Observation


class EpisodeWriter:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.f = h5py.File(self.partial, "x")
        self.f.attrs.update(schema_version="piper.transitions.v1",
                            features=json.dumps(FEATURE_NAMES),
                            units=json.dumps(["rad"]*6+["m"]+["rad"]*6+["m"]),
                            metadata=json.dumps(metadata), created_unix=time.time(),
                            complete=False)
        self.count = 0
        self.camera_names = None
        self.failed = False

    def _append(self, name, value):
        value = np.asarray(value)
        if name not in self.f:
            self.f.create_dataset(name, shape=(0, *value.shape),
                                  maxshape=(None, *value.shape), dtype=value.dtype,
                                  chunks=(1, *value.shape), compression="lzf")
        ds = self.f[name]
        if ds.shape[1:] != value.shape:
            raise ValueError(f"Shape changed for {name}")
        ds.resize(self.count + 1, axis=0)
        ds[self.count] = value

    def append(self, before: Observation, action: Command, after: Observation):
        names = tuple(sorted(before.images))
        if names != tuple(sorted(after.images)):
            raise ValueError("Camera set changed within transition")
        if self.camera_names is not None and names != self.camera_names:
            raise ValueError("Camera set changed during episode")
        if after.timestamp < before.timestamp:
            raise ValueError("Observation time reversed")
        payload = {}
        for prefix, obs in (("observation", before), ("next_observation", after)):
            payload[f"{prefix}/state"] = obs.position.astype(np.float32)
            payload[f"{prefix}/timestamp"] = obs.timestamp
            payload[f"{prefix}/sim_time"] = np.nan if obs.sim_time is None else obs.sim_time
            payload[f"{prefix}/extras_json"] = np.asarray(
                json.dumps(obs.extras), dtype=h5py.string_dtype("utf-8"))
            for name, pixels in obs.images.items():
                if "/" in name or not name:
                    raise ValueError("Invalid camera name")
                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                    raise ValueError("Images must be HxWx3 RGB uint8")
                payload[f"{prefix}/images/{name}"] = pixels
                payload[f"{prefix}/image_timestamps/{name}"] = obs.image_timestamps[name]
        payload["action/position"] = action.position.astype(np.float32)
        payload["action/timestamp"] = action.timestamp
        payload["action/sequence"] = np.int64(action.sequence)
        for key, value in payload.items():
            if key in self.f and self.f[key].shape[1:] != np.asarray(value).shape:
                raise ValueError(f"Shape changed for {key}")
        try:
            for key, value in payload.items():
                self._append(key, value)
        except BaseException:
            self.failed = True
            raise
        self.camera_names = names
        self.count += 1
        self.f.attrs["num_transitions"] = self.count
        self.f.flush()

    def close(self, success=False, complete=True):
        if not self.f.id.valid:
            return
        complete = complete and not self.failed and self.count > 0
        self.f.attrs.update(complete=complete, success=success)
        self.f.close()
        if complete:
            os.replace(self.partial, self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(complete=exc_type is None)
