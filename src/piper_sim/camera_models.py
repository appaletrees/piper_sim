"""Engine-independent public RealSense RGB specifications and nominal projection.

Public FOVs are rounded, toleranced specifications, not device calibration. Keep
fx and fy independent rather than silently replacing the published vertical FOV.
"""
import copy
import hashlib
import json
import math
from .config import ROOT


def load_profiles(cfg):
    path = ROOT / cfg["cameras"]["rgb_profiles"]
    raw = path.read_bytes()
    profiles = json.loads(raw)
    if profiles["schema_version"] != 1:
        raise ValueError("Unsupported RGB profile schema")
    return profiles, hashlib.sha256(raw).hexdigest()


def nominal_rgb(profile, scale=1.0):
    """Full-sensor FOV, optionally uniformly downsampled (never a hardware ROI)."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("image_scale must be finite and in (0, 1]")
    native_width, native_height = profile["native_resolution"]
    dimensions = [native_width*scale, native_height*scale]
    if any(abs(x-round(x)) > 1e-8 or x < 1 for x in dimensions):
        raise ValueError("image_scale must produce integer dimensions without changing aspect ratio")
    width, height = map(round, dimensions)
    hfov, vfov = profile["fov_deg"]
    if not 0 < hfov < 180 or not 0 < vfov < 180:
        raise ValueError("Invalid RGB FOV")
    fx = width / (2*math.tan(math.radians(hfov)/2))
    fy = height / (2*math.tan(math.radians(vfov)/2))
    focal_mm = profile["focal_length_mm"]
    # The fixed Sim 5.1/Lab 2.3 render path enforces square pixels even with the
    # OpenCV schema. Render a covering pinhole frustum, then sample calibrated
    # rays on the GPU. min(fx, fy) covers both requested fields without padding.
    render_focal_px = min(fx, fy)
    result = copy.deepcopy(profile)
    result.update({
        "resolution": [width, height], "image_scale": scale,
        "intrinsic_matrix": [[fx, 0., width/2], [0., fy, height/2], [0., 0., 1.]],
        "effective_aperture_mm": [width*focal_mm/fx, height*focal_mm/fy],
        "render_focal_px": render_focal_px,
        "render_aperture_mm": [width*focal_mm/render_focal_px, height*focal_mm/render_focal_px],
        "projection_implementation": "covering_pinhole_then_gpu_ray_resampling",
        "calibration_status": "public_nominal_fov_not_device_calibration",
        "distortion_model": "opencv_pinhole",
        "distortion_coefficients": [0.]*12,
        "distortion_status": "ideal_pinhole; factory coefficients unavailable",
        "exposure_model": "instantaneous; hardware shutter type is metadata only",
        "depth_of_field": "disabled; factory focus distance unavailable",
    })
    return result
