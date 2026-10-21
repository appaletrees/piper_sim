#!/usr/bin/env python3
"""Acceptance checks for a recorded pure-simulation --demo episode, not mock data."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from piper_sim.dataset import validate_episode
from piper_sim.config import ROOT
from piper_sim.kinematics import forward
from piper_sim.contracts import JOINT_NAMES
from piper_sim.transforms import pose_matrix

parser = argparse.ArgumentParser()
parser.add_argument("episode")
parser.add_argument("--output", default="outputs/sim_validation")
args = parser.parse_args()
info = validate_episode(args.episode)
if info["metadata"].get("mode") != "sim" or info["metadata"].get("source") != "isaaclab":
    raise ValueError("This check requires a pure Isaac simulation recording")
if info["transitions"] < 60:
    raise ValueError("Record at least 60 transitions")
if set(info["cameras"]) != {"overhead", "left_wrist", "right_wrist"}:
    raise ValueError("Three cameras are required")
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
with h5py.File(args.episode, "r") as f:
    if not info["metadata"].get("demo"):
        raise ValueError("Use --demo to exercise both arms and grippers for acceptance")
    action_range = np.ptp(f["action/position"][:], axis=0)
    if (action_range < np.tile([.01]*6+[.005], 2)).any():
        raise ValueError("The acceptance episode must exercise all joints and both grippers")
    errors = np.abs(f["action/position"][:] - f["next_observation/state"][:])
    limits = np.array([.1]*6+[.01])*np.ones((2, 7))
    if (np.quantile(errors, .95, axis=0) > limits.reshape(14)).any():
        raise ValueError("Joint tracking exceeds 0.1 rad / 0.01 m at the 95th percentile")
    time_delta = np.diff(f["observation/sim_time"][:])
    np.testing.assert_allclose(time_delta, 1/info["metadata"]["fps"], atol=1e-6)
    engine_time = np.array([json.loads(value)["physics_time"]
                            for value in f["next_observation/extras_json"][:]])
    np.testing.assert_allclose(np.diff(engine_time), 1/info["metadata"]["fps"], atol=1e-6)
    cfg = info["metadata"]["config"]
    extra = json.loads(f["next_observation/extras_json"][-1])
    expected_z = cfg["scene"]["table"]["position"][2] + cfg["scene"]["table"]["size"][2]/2 \
        + cfg["scene"]["object"]["size"][2]/2
    object_z = extra["object"]["position_world"][2]
    if abs(object_z - expected_z) > .01:
        raise ValueError("Test object did not settle on the tabletop")
    variances = {}
    for name in info["cameras"]:
        pixels = f[f"next_observation/images/{name}"][-1]
        variances[name] = float(pixels.std())
        if variances[name] < 3:
            raise ValueError(f"{name}: blank or nearly uniform camera output")
        Image.fromarray(pixels).save(output / f"{name}.png")
    # A nonuniform image can still show the inside of a mount or an incorrect pose.
    # Verify the recorded poses against scene configuration and actual joint feedback.
    rig = info["metadata"]["camera_rig"]
    position_errors, rotation_errors = [], []
    for states, raw_extra in zip(f["next_observation/state"][:], f["next_observation/extras_json"][:]):
        poses = json.loads(raw_extra)["cameras"]
        for name, mount in rig["cameras"].items():
            expected = pose_matrix(mount["position"], mount["rotation_wxyz_ros"])
            if name != "overhead":
                side = name.split("_")[0]
                index = ("left", "right").index(side)
                base = pose_matrix(cfg["robot"][side]["position"], cfg["robot"][side]["rotation"])
                q = dict(zip(JOINT_NAMES, states[7*index:7*index+6]))
                expected = forward(ROOT/cfg["robot"]["urdf"], q, base)["link6"] @ expected
            actual = pose_matrix(poses[name]["position_world"], poses[name]["quaternion_wxyz_ros"])
            distance = float(np.linalg.norm(actual[:3, 3]-expected[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ expected[:3, :3])-1)/2, -1, 1)))
            position_errors.append(distance)
            rotation_errors.append(angle)
            if distance > .001 or angle > .005:
                raise ValueError(f"{name}: measured camera pose does not follow CAD/FK: {distance}m, {angle}rad")
    pixels = f["next_observation/images/overhead"][-1].astype(float)
    red_pixels = int(((pixels[..., 0] > 100) &
                      (pixels[..., 0] > 1.5*pixels[..., 1]) &
                      (pixels[..., 0] > 1.5*pixels[..., 2])).sum())
    if red_pixels < 10:
        raise ValueError("Overhead camera cannot see the red test object; inspect exported PNGs")
    result = {"episode": str(Path(args.episode).resolve()), "transitions": info["transitions"],
              "tracking_error_p95": np.quantile(errors, .95, axis=0).tolist(),
              "object_z": object_z, "expected_object_z": expected_z, "image_std": variances,
              "overhead_red_pixels": red_pixels, "camera_positions_match_fk": True,
              "camera_max_position_error_m": max(position_errors),
              "camera_max_rotation_error_rad": max(rotation_errors),
              "camera_source_sha256": rig["source_sha256"]}
    result["action_range"] = action_range.tolist()
    result["physics_time_matches_control_clock"] = True
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
