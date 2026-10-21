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
        "pixel_coordinates": "opencv_integer_centers",
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


def stream_rgb(profile, stream):
    """Explicit stream K, or a labelled full-height centre-crop hypothesis.

    A 4:3 RealSense mode is not a nonuniform resize of its full sensor. Actual
    mode intrinsics must come from the device; crop_nominal is only a baseline
    for comparing historical recordings which did not save their calibration.
    """
    width, height = stream["resolution"]
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (width, height)):
        raise ValueError("Stream resolution must contain two positive integers")
    result = nominal_rgb(profile)
    mode = stream["mode"]
    if mode == "crop_nominal":
        native_w, native_h = profile["native_resolution"]
        scale = height/native_h
        if scale > 1 or width > native_w*scale:
            raise ValueError("Centre crop cannot exceed the full-height sensor image")
        fx = result["intrinsic_matrix"][0][0]*scale
        fy = result["intrinsic_matrix"][1][1]*scale
        cx, cy = width/2, height/2
        status = "unverified_full_height_center_crop_hypothesis"
    elif mode == "calibrated_pinhole":
        k = stream["intrinsic_matrix"]
        if (len(k) != 3 or any(len(row) != 3 for row in k)
                or any(not math.isfinite(v) for row in k for v in row)
                or k[0][1] != 0 or k[1][0] != 0 or list(k[2]) != [0, 0, 1]):
            raise ValueError("Expected a finite zero-skew 3x3 camera matrix")
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        if fx <= 0 or fy <= 0 or not 0 < cx < width or not 0 < cy < height:
            raise ValueError("Invalid pinhole intrinsics")
        if any(stream.get("distortion_coefficients", [])):
            raise ValueError("Nonzero lens distortion needs rectification before pinhole matching")
        if not stream.get("source"):
            raise ValueError("Calibrated intrinsics require a provenance source")
        status = "provided_pinhole_intrinsics"
    else:
        raise ValueError(f"Unsupported RGB stream mode: {mode}")
    # Off-centre principal points also need a covering raw frustum.
    focal = min(fx*width/(2*max(cx, width-cx)), fy*height/(2*max(cy, height-cy)))
    focal_mm = result["focal_length_mm"]
    result.update({
        "resolution": [width, height], "image_scale": None,
        "intrinsic_matrix": [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
        "effective_aperture_mm": [width*focal_mm/fx, height*focal_mm/fy],
        "render_focal_px": focal,
        "render_aperture_mm": [width*focal_mm/focal, height*focal_mm/focal],
        "fov_deg": [math.degrees(math.atan(cx/fx)+math.atan((width-cx)/fx)),
                    math.degrees(math.atan(cy/fy)+math.atan((height-cy)/fy))],
        "calibration_status": status, "stream_profile": copy.deepcopy(stream),
    })
    return result
