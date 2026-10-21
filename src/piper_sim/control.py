"""Loopback-only command ingress, independent of Isaac and hardware adapters."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from .contracts import Command, vector


class CommandServer:
    def __init__(self, port, *, allow_commands=True):
        self.lock = threading.Lock()
        self.latest = None
        self.state = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/target":
                    self.send_error(404)
                    return
                if not allow_commands:
                    self.send_error(403, "This connection is read-only")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Invalid content length")
                    body = json.loads(self.rfile.read(length))
                    command = Command(vector(body["position"]), time.monotonic(), body["sequence"])
                    with owner.lock:
                        if owner.latest and command.sequence <= owner.latest.sequence:
                            raise ValueError("Sequence must increase")
                        owner.latest = command
                    self.send_response(204)
                    self.end_headers()
                except (ValueError, KeyError, TypeError) as error:
                    self.send_error(400, str(error))

            def do_GET(self):
                if self.path != "/state":
                    self.send_error(404)
                    return
                with owner.lock:
                    data = json.dumps(owner.state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def sample(self, max_age=.15):
        with self.lock:
            if self.latest is None:
                return None
            if time.monotonic() - self.latest.timestamp > max_age:
                raise TimeoutError("Command stream expired")
            return self.latest

    def publish(self, observation):
        with self.lock:
            self.state = {"position": observation.position.tolist(),
                          "timestamp": observation.timestamp}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
