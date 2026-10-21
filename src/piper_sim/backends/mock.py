import time
from ..contracts import Observation, vector


class MockBackend:
    """Deterministic test backend; not a physics simulator or training data source."""
    def __init__(self, initial):
        self.position = vector(initial)

    def read(self):
        return Observation(self.position.copy(), time.monotonic())

    def apply(self, command):
        self.position = vector(command.position)
        return command

    def close(self):
        pass
