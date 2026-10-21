from types import SimpleNamespace as NS
import pytest
from piper_sim.backends.sdk_tracking import tracked_sdk


class LoggingOnlySDK:
    """Reproduce the supplied SDK's transport-status (not exception) contract."""
    def __init__(self):
        success = object()
        self.status = success
        self.sent = []
        self.__arm_can = NS(CAN_STATUS=NS(SEND_MESSAGE_SUCCESS=success),
                            SendCanMessage=self._send)

    def _send(self, address, payload):
        self.sent.append((address, payload))
        return self.status

    def GripperCtrl(self, width):
        self.__arm_can.SendCanMessage(0x159, width)

    def ParseCANFrame(self, message):
        pass


def test_transport_failure_cannot_be_silently_reported_as_success():
    sdk = tracked_sdk(LoggingOnlySDK)()
    sdk.GripperCtrl(40000)
    assert sdk.sent == [(0x159, 40000)]
    sdk.status = "CAN_SEND_FAILED"
    with pytest.raises(OSError, match="CAN_SEND_FAILED"):
        sdk.GripperCtrl(30000)


def test_only_valid_full_feedback_frames_update_freshness():
    sdk = tracked_sdk(LoggingOnlySDK)()
    sdk.ParseCANFrame(NS(is_error_frame=False, dlc=8, arbitration_id=0x2A8))
    before = dict(sdk.frame_times)
    sdk.ParseCANFrame(NS(is_error_frame=True, dlc=8, arbitration_id=0x2A5))
    sdk.ParseCANFrame(NS(is_error_frame=False, dlc=4, arbitration_id=0x2A6))
    assert sdk.frame_times == before
