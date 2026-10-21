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
            # This supplied SDK logs failed sends instead of raising from
            # JointCtrl/GripperCtrl. Promote transport failures at its CAN
            # boundary so PiperBackend can stop both sides and report a fault.
            port = getattr(self, f"_{base.__name__}__arm_can", None)
            if port is None:
                raise RuntimeError("Unsupported Piper SDK: CAN transport is unavailable")
            original_send = port.SendCanMessage

            def checked_send(*args, **kwargs):
                status = original_send(*args, **kwargs)
                if status is not port.CAN_STATUS.SEND_MESSAGE_SUCCESS:
                    raise OSError(f"Piper CAN send failed: {status}")
                return status

            port.SendCanMessage = checked_send

        def ParseCANFrame(self, message):
            with self.feedback_lock:
                super().ParseCANFrame(message)
                if message is not None and not message.is_error_frame and message.dlc == 8:
                    self.frame_times[message.arbitration_id] = time.monotonic()

    return TrackedSDK
