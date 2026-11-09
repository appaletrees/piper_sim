"""Owned CPU sensor snapshots; keys must identify the actual camera capture."""
import copy


class CameraSnapshotCache:
    def __init__(self):
        self.clear()

    def clear(self):
        self.key = None
        self.snapshot = None

    def read(self, key, capture):
        if self.snapshot is None or key != self.key:
            images, poses = capture()
            # Publish only after a successful capture. Own every array/list so
            # neither the renderer nor a caller can corrupt a future read.
            snapshot = ({name: pixels.copy() for name, pixels in images.items()},
                        copy.deepcopy(poses))
            self.snapshot, self.key = snapshot, key
        images, poses = self.snapshot
        return ({name: pixels.copy() for name, pixels in images.items()}, copy.deepcopy(poses))
