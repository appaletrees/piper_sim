import copy

import numpy as np
import pytest

from piper_sim.lerobot_alignment import (
    CAMERA_MAPPING, CONNECTOR_PLUGIN_NAMES, bounds_violations,
    connector_plugin_state_to_si, feedback_quality, validate_connector_plugin_info,
)


def info():
    features = {key: {"dtype": "float32", "shape": [14],
                      "names": list(CONNECTOR_PLUGIN_NAMES)}
                for key in ("observation.state", "action")}
    features["source_timestamp"] = {"dtype": "float64", "shape": [1]}
    for camera in CAMERA_MAPPING:
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [3, 480, 640]}
    return {"codebase_version": "v2.1", "fps": 30, "features": features}


def test_real_recording_units_preserve_gripper_beyond_old_model_limit():
    # Actual episode 0 frame 0 values; 0.97 is normalized stroke, not metres
    # and not a 70 mm normalization. Refuse to silently lose this discrepancy.
    raw = [0, 0, 0, 0, 0, 0, .7915789484977722,
           43.79999923706055, 107.05400085449219, -90.23600006103516,
           -4.507999897003174, 69.87799835205078, 23.275999069213867,
           .9705263376235962]
    si = connector_plugin_state_to_si(raw)
    np.testing.assert_allclose(si[[6, 13]], [.0752, .0922], atol=3e-9)
    assert si[7] == pytest.approx(.7644541990577108)
    assert {v["index"] for v in bounds_violations(si, .07)} == {6, 13}
    assert bounds_violations(si, .095) == []
    assert raw[6] > .79  # No in-place mutation.


def test_reordered_or_resized_dataset_is_not_assigned_connector_units():
    correct = info()
    validate_connector_plugin_info(correct)
    for kind in ("order", "shape", "fps"):
        bad = copy.deepcopy(correct)
        if kind == "order":
            names = bad["features"]["observation.state"]["names"]
            names[0], names[7] = names[7], names[0]
        elif kind == "shape":
            bad["features"]["observation.images.cam_external_rgb"]["shape"] = [3, 800, 1280]
        else:
            bad["fps"] = 60
        with pytest.raises(ValueError):
            validate_connector_plugin_info(bad)


@pytest.mark.parametrize("gripper", [-.01, 1.01, float("nan")])
def test_malformed_normalized_stroke_is_rejected(gripper):
    raw = np.zeros(14)
    raw[6] = gripper
    with pytest.raises(ValueError):
        connector_plugin_state_to_si(raw)


def test_zero_cache_is_not_declared_a_physical_zero_pose():
    states = np.zeros((3, 14))
    states[:, 6] = .79158
    states[:, 7] = [15, 20, 25]
    states[1, 1] = .023  # observed tiny variation does not establish valid feedback
    quality = feedback_quality(states)
    assert quality["arms"]["left"]["status"] == "suspicious_near_zero_feedback"
    assert quality["arms"]["right"]["status"] == "reported_feedback_unverified"
    assert not quality["arms"]["left"]["use_as_zero_pose_calibration"]
    assert not quality["physical_pose_verified"]
