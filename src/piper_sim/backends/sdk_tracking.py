"""Track individual CAN feedback frames: SDK aggregate timestamps are insufficient."""
import threading
import time


def tracked_sdk(base):
    class TrackedSDK(base):
        # Do not reuse a legacy controller's singleton on the same CAN channel.
        _instances = {}
        _lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            self.feedback_lock = threading.RLock()
            self.frame_times = {}
            super().__init__(*args, **kwargs)

        def ParseCANFrame(self, message):
            with self.feedback_lock:
                super().ParseCANFrame(message)
                if message is not None and not message.is_error_frame and message.dlc == 8:
                    self.frame_times[message.arbitration_id] = time.monotonic()

    return TrackedSDK
