import math
import numpy as np
import pytest
from piper_sim.camera_models import load_profiles, nominal_rgb
from piper_sim.config import load


@pytest.mark.parametrize("model,resolution,fov", [
    ("D435i", [1920, 1080], [69.4, 42.5]), ("D455", [1280, 800], [90, 65])])
def test_nominal_rgb_projection_and_uniform_downsampling(model, resolution, fov):
    profiles, _ = load_profiles(load())
    full = nominal_rgb(profiles["models"][model])
    half = nominal_rgb(profiles["models"][model], .5)
    k = np.array(full["intrinsic_matrix"])
    assert full["resolution"] == resolution
    # Pixel edges subtend the published full field in each axis, including fy.
    for axis in range(2):
        half_angle = math.degrees(math.atan((resolution[axis]/2)/k[axis, axis]))
        assert abs(2*half_angle-fov[axis]) < 1e-10
    np.testing.assert_allclose(half["intrinsic_matrix"], np.diag([.5, .5, 1])@k)
    np.testing.assert_allclose(half["effective_aperture_mm"], full["effective_aperture_mm"])
    # Each output edge must sample an existing ray, without black padding or a
    # narrower field. This also catches using max(fx, fy) for the covering view.
    focal = full["render_focal_px"]
    for axis in range(2):
        ray = resolution[axis]/(2*k[axis, axis])
        source_offset = focal*ray
        assert source_offset <= resolution[axis]/2+1e-10
        assert abs(source_offset/focal-ray) < 1e-12
    assert full["calibration_status"] == "public_nominal_fov_not_device_calibration"


@pytest.mark.parametrize("scale", [0, -1, 2, float("nan"), .333])
def test_reject_crop_or_invalid_scale(scale):
    profiles, _ = load_profiles(load())
    with pytest.raises(ValueError):
        nominal_rgb(profiles["models"]["D435i"], scale)
