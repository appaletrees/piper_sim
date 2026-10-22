"""Versioned loopback API: immutable observations, bounded command/reset mailboxes.

HTTP threads never call a simulator, CAN SDK, renderer or reset. The owning
simulation thread samples commands, performs resets, and publishes snapshots.
"""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time
from urllib.parse import urlsplit

import numpy as np
from .contracts import Command, FEATURE_NAMES, LOWER, UPPER, vector


class CommandServer:
    def __init__(self, port, *, allow_commands=True, allow_reset=False, allow_recording=False,
                 metadata=None, lower=None, upper=None):
        self.lock = threading.Lock()
        self.latest = None
        self.state = None
        self.images = {}
        self.frame_id = -1
        self.received_sequence = -1
        self.reset_id = 0
        self._next_reset_id = 1
        self._reset_pending = None
        self._reset_delivered = False
        self.recording = {"active": False, "frames": 0, "path": None, "error": None, "request_id": 0}
        self._record_pending = None
        self._next_record_id = 1
        self.lower = vector(LOWER if lower is None else lower)
        self.upper = vector(UPPER if upper is None else upper)
        self.metadata = {
            "schema_version": "piper.sim.http.v1", "features": list(FEATURE_NAMES),
            "units": ["rad"]*6+["m"]+["rad"]*6+["m"],
            "limits": {"lower": self.lower.tolist(), "upper": self.upper.tolist()},
            "allow_commands": allow_commands, "allow_reset": allow_reset,
            "allow_recording": allow_recording,
            "action_semantics": "absolute joint targets; one full gripper opening per arm",
            "observation_bundle": "/observation.npz", **copy.deepcopy(metadata or {})}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, data=b"", content_type="application/json", headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                for key, value in (headers or {}).items(): self.send_header(key, str(value))
                self.end_headers()
                try: self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError): pass

            def failure(self, status, message):
                self.reply(status, json.dumps({"error": str(message)}).encode())

            def body(self):
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096: raise ValueError("Invalid content length")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict): raise ValueError("Expected a JSON object")
                return value

            def do_POST(self):
                path = urlsplit(self.path).path
                if path not in ("/target", "/reset", "/recording/start", "/recording/stop"):
                    self.failure(404, "Unknown endpoint"); return
                if path == "/target" and not allow_commands:
                    self.failure(403, "This connection is read-only"); return
                if path == "/reset" and not allow_reset:
                    self.failure(403, "Reset is available only in pure simulation without active recording"); return
                if path.startswith("/recording/") and not allow_recording:
                    self.failure(403, "Recording lifecycle unavailable on this connection"); return
                try:
                    body = self.body()
                    if path == "/target":
                        command = Command(vector(body["position"]), time.monotonic(), body["sequence"])
                        if np.any(command.position < owner.lower) or np.any(command.position > owner.upper):
                            raise ValueError("Target outside advertised limits")
                        with owner.lock:
                            if command.sequence <= owner.received_sequence:
                                raise ValueError("Sequence must increase")
                            owner.latest = command
                            owner.received_sequence = int(command.sequence)
                        # Receipt only: applied/rejected sequence is published by the simulation owner.
                        self.reply(204)
                    elif path == "/reset":
                        seed = body.get("seed")
                        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
                            raise ValueError("seed must be a nonnegative integer below 2**63")
                        randomize = body.get("randomize")
                        if randomize is not None and type(randomize) is not bool:
                            raise ValueError("randomize must be boolean")
                        with owner.lock:
                            if owner.recording["active"] or owner._record_pending:
                                self.failure(409, "Stop recording before reset"); return
                            if owner._reset_pending is not None:
                                self.failure(409, "A reset is already pending"); return
                            request = {"reset_id": owner._next_reset_id, "seed": seed, "randomize": randomize}
                            owner._next_reset_id += 1
                            owner._reset_pending = request
                            owner._reset_delivered = False
                        self.reply(202, json.dumps(request).encode())
                    else:
                        import re
                        operation = path.rsplit('/', 1)[1]
                        name = body.get("name", "episode")
                        task = body.get("task", "dual Piper demonstration")
                        success = body.get("success", False)
                        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
                            raise ValueError("name must contain 1..80 letters, digits, underscores or hyphens")
                        if not isinstance(task, str) or len(task) > 1000 or type(success) is not bool:
                            raise ValueError("Invalid task or success")
                        with owner.lock:
                            if owner._record_pending or owner._reset_pending:
                                self.failure(409, "Another lifecycle request is pending"); return
                            if (operation == "start") == owner.recording["active"]:
                                self.failure(409, "Recording is already started or stopped"); return
                            request = {"request_id": owner._next_record_id, "operation": operation,
                                       "name": name, "task": task, "success": success}
                            owner._next_record_id += 1
                            owner._record_pending = request
                        self.reply(202, json.dumps(request).encode())
                except (ValueError, KeyError, TypeError) as error:
                    self.failure(400, error)

            def do_GET(self):
                path = urlsplit(self.path).path
                with owner.lock:
                    state, images = owner.state, owner.images  # immutable published objects
                    metadata = copy.deepcopy(owner.metadata)
                    received_sequence = owner.received_sequence
                    recording = dict(owner.recording)
                if path == "/recording":
                    self.reply(200, json.dumps(recording).encode()); return
                if path == "/metadata":
                    self.reply(200, json.dumps(metadata).encode()); return
                if state is None:
                    self.failure(503, "No observation published yet"); return
                headers = {"X-Observation-Id": state["frame_id"]}
                if path == "/state":
                    result = dict(state, last_received_sequence=received_sequence)
                    self.reply(200, json.dumps(result).encode(), headers=headers)
                elif path == "/observation.npz":
                    # One frame's state + timestamps + extras + RGB, never mixed across HTTP calls.
                    bundle = {"state_json": np.asarray(json.dumps(state)),
                              "position": np.asarray(state["position"], dtype=np.float64)}
                    bundle.update({"rgb_"+name: pixels for name, pixels in images.items()})
                    buffer = BytesIO(); np.savez(buffer, **bundle)
                    self.reply(200, buffer.getvalue(), "application/octet-stream", headers)
                elif path.startswith("/cameras/") and path.endswith(".png"):
                    name = path[len("/cameras/"):-4]
                    if name not in images:
                        self.failure(404, "Camera unavailable; start with --enable_cameras"); return
                    from PIL import Image
                    buffer = BytesIO(); Image.fromarray(images[name]).save(buffer, format="PNG")
                    self.reply(200, buffer.getvalue(), "image/png", headers)
                else: self.failure(404, "Unknown endpoint")

            def log_message(self, *_): pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def sample(self, max_age=.15):
        with self.lock:
            if self.latest is None: return None
            if time.monotonic() - self.latest.timestamp > max_age:
                raise TimeoutError("Command stream expired")
            return self.latest

    def poll_reset(self):
        """Only the simulation owner may consume/execute a reset request."""
        with self.lock:
            if self._reset_pending is None or self._reset_delivered: return None
            self._reset_delivered = True
            self.latest = None
            return dict(self._reset_pending)

    def complete_reset(self, reset_id):
        with self.lock:
            if self._reset_pending is None or reset_id != self._reset_pending["reset_id"]:
                raise ValueError("Unexpected reset acknowledgement")
            self.reset_id = reset_id
            self._reset_pending = None
            self._reset_delivered = False

    def update_metadata(self, **values):
        with self.lock: self.metadata.update(copy.deepcopy(values))

    def poll_recording(self):
        with self.lock:
            return dict(self._record_pending) if self._record_pending else None

    def update_recording(self, *, request_id=None, **values):
        with self.lock:
            self.recording.update(values)
            if request_id is not None:
                self.recording["request_id"] = request_id
                self._record_pending = None

    def publish(self, observation, control=None):
        # Copy before publishing. Neither rendering nor encoding holds the mailbox lock.
        images = {k: v.copy() for k, v in observation.images.items()}
        state = {"schema_version": "piper.sim.observation.v1",
                 "position": observation.position.tolist(), "timestamp": observation.timestamp,
                 "sim_time": observation.sim_time, "image_timestamps": dict(observation.image_timestamps),
                 "extras": copy.deepcopy(observation.extras), "control": copy.deepcopy(control or {}),
                 "cameras": {name: {"shape": list(p.shape), "encoding": "RGB8",
                                    "url": f"/cameras/{name}.png"} for name, p in images.items()}}
        with self.lock:
            self.frame_id += 1
            state.update(frame_id=self.frame_id, reset_id=self.reset_id,
                         last_received_sequence=self.received_sequence)
            self.state, self.images = state, images

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
