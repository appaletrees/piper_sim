"""Append-only episode writer; stores transitions (obs_t, applied action_t, obs_t+1)."""
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from .contracts import FEATURE_NAMES, Command, Observation


class EpisodeWriter:
    def __init__(self, path, metadata, *, image_compression="lzf",
                 image_storage="duplicated", flush_every=1):
        if image_compression not in ("lzf", "gzip"):
            raise ValueError("image_compression must be lzf or gzip")
        self.image_compression = image_compression
        if image_storage not in ("duplicated", "shared"):
            raise ValueError("image_storage must be duplicated or shared")
        if type(flush_every) is not int or flush_every < 1:
            raise ValueError("flush_every must be a positive integer")
        self.image_storage = image_storage
        self.flush_every = flush_every
        self._last_images = {}
        self._last_image_timestamps = {}
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
                            complete=False, flush_every=flush_every)
        self.count = 0
        self.camera_names = None
        self.failed = False

    def _append(self, name, value, *, index=None):
        index = self.count if index is None else index
        value = np.asarray(value)
        if name not in self.f:
            compression = self.image_compression if ("/images/" in name or
                name.startswith("recording_terminal_images/")) else "lzf"
            self.f.create_dataset(name, shape=(0, *value.shape),
                                  maxshape=(None, *value.shape), dtype=value.dtype,
                                  chunks=(1, *value.shape), compression=compression,
                                  compression_opts=1 if compression == "gzip" else None)
        ds = self.f[name]
        if ds.shape[1:] != value.shape:
            raise ValueError(f"Shape changed for {name}")
        ds.resize(index + 1, axis=0)
        ds[index] = value

    def append(self, before: Observation, action: Command, after: Observation):
        if self.failed:
            raise RuntimeError("Writer failed; this partial episode cannot be continued")
        try:
            self._append_transition(before, action, after)
        except BaseException:
            # A caller catching a validation or I/O error must not publish a
            # silently truncated episode on normal context-manager exit.
            self.failed = True
            raise

    def _append_transition(self, before, action, after):
        names = tuple(sorted(before.images))
        if names != tuple(sorted(after.images)):
            raise ValueError("Camera set changed within transition")
        if self.camera_names is not None and names != self.camera_names:
            raise ValueError("Camera set changed during episode")
        if not np.isfinite([before.timestamp, after.timestamp]).all():
            raise ValueError("Non-finite observation timestamp")
        if after.timestamp < before.timestamp:
            raise ValueError("Observation time reversed")
        payload = {}
        for prefix, obs in (("observation", before), ("next_observation", after)):
            payload[f"{prefix}/state"] = obs.position.astype(np.float32)
            payload[f"{prefix}/timestamp"] = obs.timestamp
            payload[f"{prefix}/sim_time"] = np.nan if obs.sim_time is None else obs.sim_time
            payload[f"{prefix}/extras_json"] = np.asarray(
                json.dumps(obs.extras, allow_nan=False), dtype=h5py.string_dtype("utf-8"))
            for name, pixels in obs.images.items():
                if "/" in name or not name:
                    raise ValueError("Invalid camera name")
                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                    raise ValueError("Images must be HxWx3 RGB uint8")
                if not np.isfinite(obs.image_timestamps[name]):
                    raise ValueError("Non-finite image timestamp")
                if self.image_storage == "duplicated" or prefix == "observation":
                    payload[f"{prefix}/images/{name}"] = pixels
                payload[f"{prefix}/image_timestamps/{name}"] = obs.image_timestamps[name]
        if self.image_storage == "shared":
            for name in names:
                if before.images[name].shape != after.images[name].shape:
                    raise ValueError("Image shape changed within transition")
                if self.count and (not np.array_equal(before.images[name], self._last_images[name]) or
                                  before.image_timestamps[name] != self._last_image_timestamps[name]):
                    raise ValueError("Noncontiguous RGB frames: "+name)
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
        if self.image_storage == "shared":
            # Own immutable snapshots: callers/backends may reuse image buffers.
            self._last_images = {k: v.copy() for k, v in after.images.items()}
            self._last_image_timestamps = dict(after.image_timestamps)
        self.count += 1
        self.f.attrs["num_transitions"] = self.count
        if self.count % self.flush_every == 0:
            self.f.flush()

    def _finalize_shared_images(self):
        """Same self-contained layout as offline compaction; no external links.

        Store N obs frames plus one terminal frame. next_obs is a shifted view.
        The terminal dataset has logical length N for v1 reader compatibility;
        only its final chunk is allocated. No frames or timestamps are removed.
        """
        for name, pixels in self._last_images.items():
            self._append('recording_terminal_images/'+name, pixels, index=self.count-1)
            shared = self.f['observation/images/'+name]
            terminal = self.f['recording_terminal_images/'+name]
            layout = h5py.VirtualLayout(shape=shared.shape, dtype=shared.dtype)
            source = h5py.VirtualSource('.', shared.name, shape=shared.shape)
            final = h5py.VirtualSource('.', terminal.name, shape=terminal.shape)
            if self.count > 1:
                layout[:-1] = source[1:]
            layout[-1] = final[-1]
            self.f.create_virtual_dataset('next_observation/images/'+name, layout)
        if self._last_images:
            self.f.attrs['image_storage'] = 'shared_transition_frames_v1'

    def close(self, success=False, complete=True):
        if not self.f.id.valid:
            return
        complete = complete and not self.failed and self.count > 0
        try:
            if complete and self.image_storage == "shared":
                self._finalize_shared_images()
            self.f.attrs.update(complete=complete, success=bool(success and complete))
        except BaseException:
            self.failed = True
            self.f.attrs['complete'] = False
            raise
        finally:
            self.f.close()
            self._last_images.clear()
        if complete:
            os.replace(self.partial, self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(complete=exc_type is None)
