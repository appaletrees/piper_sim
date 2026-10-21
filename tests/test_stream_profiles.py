import numpy as np
import pytest
from piper_sim.camera_models import load_profiles, nominal_rgb, stream_rgb
from piper_sim.config import load


def test_vga_crop_preserves_vertical_rays_and_labels_uncertainty():
    profiles, _ = load_profiles(load())
    for profile in profiles["models"].values():
        native = nominal_rgb(profile)
        vga = stream_rgb(profile, {"mode": "crop_nominal", "resolution": [640, 480]})
        k = np.array(vga["intrinsic_matrix"])
        assert vga["resolution"] == [640, 480]
        assert "unverified" in vga["calibration_status"]
        assert vga["fov_deg"][0] < native["fov_deg"][0]
        assert abs(vga["fov_deg"][1]-native["fov_deg"][1]) < 1e-10
        assert k[0, 2] == 320 and k[1, 2] == 240


def test_asymmetric_calibration_covers_all_output_rays():
    profiles, _ = load_profiles(load())
    k = [[620., 0, 332.], [0, 615., 229.], [0, 0, 1]]
    stream = {"mode": "calibrated_pinhole", "resolution": [640, 480],
              "intrinsic_matrix": k, "source": "synthetic calibration test"}
    result = stream_rgb(profiles["models"]["D435i"], stream)
    np.testing.assert_array_equal(result["intrinsic_matrix"], k)
    focal = result["render_focal_px"]
    for axis, size in enumerate([640, 480]):
        for edge in [0, size]:
            source_pixel = (edge-k[axis][2])*focal/k[axis][axis]+size/2
            assert -1e-10 <= source_pixel <= size+1e-10
    stream["distortion_coefficients"] = [.01, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="rectification"):
        stream_rgb(profiles["models"]["D435i"], stream)
