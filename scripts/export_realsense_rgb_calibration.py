#!/usr/bin/env python3
"""Read RealSense RGB intrinsics without starting streaming or controlling a robot.

SDK config.resolve selects a profile without applying it to the device:
https://realsenseai.github.io/librealsense/python_docs/_generated/pyrealsense2.config.html
The EEPROM intrinsics are specific to the requested resolution, not derived by
rescaling the product's advertised FOV. Mount extrinsics are not available here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def camera_mapping(value):
    try:
        name, serial = value.split("=", 1)
        if name not in ("overhead", "left_wrist", "right_wrist") or not serial:
            raise ValueError
        return name, serial
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use overhead=SERIAL, left_wrist=SERIAL, or right_wrist=SERIAL") from exc


def collect_intrinsics(camera_serials, width=640, height=480, fps=30):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("Install optional pyrealsense2 in the environment used for this script") from exc
    context = rs.context()
    devices = {device.get_info(rs.camera_info.serial_number): device
               for device in context.query_devices()}
    missing = set(camera_serials.values()) - devices.keys()
    if missing:
        raise RuntimeError(f"Requested RealSense devices are not connected: {sorted(missing)}")
    result = {
        "schema_version": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_status": "device_eeprom_rgb_profile",
        "stream_started": False, "mount_extrinsics_measured": False,
        "historical_dataset_calibration_verified": False,
        "image_rotation_degrees": 0, "cameras": {},
    }
    for name, serial in camera_serials.items():
        device = devices[serial]
        configuration = rs.config()
        configuration.enable_device(serial)
        configuration.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # Intentionally no pipeline.start(), sensor.open(), sensor.start(),
        # set_option(), hardware_reset(), or any CAN/robot SDK import.
        pipeline = rs.pipeline(context)
        resolved = configuration.resolve(rs.pipeline_wrapper(pipeline))
        profile = resolved.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsic = profile.get_intrinsics()
        if (intrinsic.width, intrinsic.height, profile.fps()) != (width, height, fps):
            raise RuntimeError(f"Resolved an unexpected stream profile for {name}")
        result["cameras"][name] = {
            "serial": serial, "model": device.get_info(rs.camera_info.name),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "resolution": [width, height], "fps": fps, "format": str(profile.format()),
            "intrinsic_matrix": [[intrinsic.fx, 0., intrinsic.ppx],
                                 [0., intrinsic.fy, intrinsic.ppy], [0., 0., 1.]],
            "distortion_model": str(intrinsic.model),
            "distortion_coefficients": list(intrinsic.coeffs),
            "intrinsics_source": "pyrealsense2.video_stream_profile.get_intrinsics",
            "pixel_coordinates": "unrotated raw color profile",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=camera_mapping, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if len(dict(args.camera)) != len(args.camera) or len({serial for _, serial in args.camera}) != len(args.camera):
        parser.error("Camera names and serials must each be unique")
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("Width, height and fps must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new file to retain earlier calibration evidence")
    result = collect_intrinsics(dict(args.camera), args.width, args.height, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
