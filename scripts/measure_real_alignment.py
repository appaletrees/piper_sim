#!/usr/bin/env python3
"""Reproduce numeric CAD-to-recording diagnostics without reading real images.

This script performs no fitting, rendering, hardware access or source mutation.
It reports residuals against independent numeric annotations, never a calibrated
physical alignment. Optional local PNG hashes verify annotation provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from piper_sim.config import ROOT, load
from piper_sim.kinematics import forward
from piper_sim.transforms import load_camera_rig, pose_matrix, transform_points


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(points, world_from_camera, intrinsic):
    camera_points = transform_points(points, np.linalg.inv(world_from_camera))
    if not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 0).any():
        raise ValueError("Diagnostic geometry must be finite and in front of the camera")
    pixels = camera_points @ intrinsic.T
    return pixels[:, :2] / pixels[:, 2, None]


def mesh_points(path):
    """Read visual points in their USD asset root coordinates, including xforms."""
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open CAD gripper asset: {path}")
    cache = UsdGeom.XformCache()
    points = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            points.append(transform_points(local, transform))
    if not points:
        raise ValueError(f"No meshes in CAD gripper asset: {path}")
    return np.concatenate(points)


def camera_intrinsic(rig, name, landmarks):
    rgb = rig["cameras"][name]["rgb"]
    if rgb["resolution"] != landmarks["resolution"]:
        raise ValueError(f"{name}: camera and annotation resolutions differ; use the alignment config")
    if any(rgb.get("distortion_coefficients", [])):
        raise ValueError("These diagnostic annotations require a pinhole or rectified projection")
    return np.asarray(rgb["intrinsic_matrix"], dtype=float)


def verify_reference_images(landmarks, directory):
    if directory is None:
        return {"status": "not_requested", "required_for_numeric_diagnostics": False}
    images = []
    for ref in landmarks["reference_images"]:
        name = ref["local_basename"]
        if Path(name).name != name:
            raise ValueError("Reference image names must be plain basenames")
        actual = digest(directory / name)
        if actual != ref["sha256"]:
            raise ValueError(f"Reference image hash mismatch: {name}")
        images.append({"basename": name, "sha256": actual, "matched": True})
    return {"status": "matched", "required_for_numeric_diagnostics": False, "images": images}


def measure(config, landmarks_path, gripper_manifest, reference_images=None):
    cfg = load(config)
    rig = load_camera_rig(cfg)
    landmarks = json.loads(landmarks_path.read_text())
    manifest = json.loads(gripper_manifest.read_text())
    if landmarks["schema_version"] != 1 or manifest["schema_version"] != 1:
        raise ValueError("Unsupported annotation or CAD gripper manifest schema")
    if manifest["source_sha256"] != rig["source_sha256"]:
        raise ValueError("Camera and gripper assets were generated from different CAD revisions")
    image_verification = verify_reference_images(landmarks, reference_images)
    provenance = {
        "config": {"name": Path(cfg["_config_path"]).name, "sha256": digest(cfg["_config_path"])},
        "landmarks": {"name": landmarks_path.name, "sha256": digest(landmarks_path)},
        "camera_rig_sha256": digest(ROOT / cfg["cameras"]["rig_manifest"]),
        "rgb_profiles_sha256": rig["rgb_profiles_sha256"],
        "gripper_manifest_sha256": digest(gripper_manifest),
        "urdf_sha256": digest(ROOT / cfg["robot"]["urdf"]),
        "source_cad_sha256": rig["source_sha256"],
        "gripper_assets": [],
    }

    table_annotation = landmarks["overhead_table_edges"]
    camera = rig["cameras"][table_annotation["camera"]]
    intrinsic = camera_intrinsic(rig, table_annotation["camera"], landmarks)
    pose = pose_matrix(camera["position"], camera["rotation_wxyz_ros"])
    size = np.asarray(cfg["scene"]["table"]["size"], dtype=float)
    center = np.asarray(cfg["scene"]["table"]["position"], dtype=float)
    lower, upper = center-size/2, center+size/2
    table_results = []
    for edge in table_annotation["edges"]:
        if edge["world_table_axis"] not in ("positive_x", "negative_x"):
            raise ValueError("Unsupported table edge association")
        x = upper[0] if edge["world_table_axis"] == "positive_x" else lower[0]
        world = np.array([[x, upper[1], upper[2]], [x, lower[1], upper[2]]])
        pixels = project(world, pose, intrinsic)
        if abs(pixels[1, 0]-pixels[0, 0]) < 1e-8:
            raise ValueError("Cannot measure a horizontal table edge from a vertical projection")
        roi_x = (edge["roi_xyxy"][0]+edge["roi_xyxy"][2]-1)/2
        t = (roi_x-pixels[0, 0])/(pixels[1, 0]-pixels[0, 0])
        predicted_y = float(pixels[0, 1]+t*(pixels[1, 1]-pixels[0, 1]))
        residual = predicted_y-edge["observed_y_px"]
        table_results.append({
            "name": edge["name"], "source_world_endpoints_m": world.tolist(),
            "predicted_endpoints_px": pixels.tolist(), "evaluation_x_px": roi_x,
            "predicted_y_px": predicted_y, "observed_y_px": edge["observed_y_px"],
            "signed_vertical_residual_px": residual, "absolute_vertical_residual_px": abs(residual),
            "annotation_uncertainty_px": table_annotation["uncertainty_px"],
            "within_annotation_uncertainty": abs(residual) <= table_annotation["uncertainty_px"],
            "predicted_y_inside_image": 0 <= predicted_y < landmarks["resolution"][1],
        })

    wrist_annotation = landmarks["wrist_gripper_tips"]
    side = wrist_annotation["arm"]
    camera = rig["cameras"][wrist_annotation["camera"]]
    intrinsic = camera_intrinsic(rig, wrist_annotation["camera"], landmarks)
    width = float(wrist_annotation["gripper_width_m"])
    if not np.isfinite(width) or width < 0:
        raise ValueError("Diagnostic gripper opening must be finite and nonnegative")
    # Six-axis zero is only a computational reference: common arm transforms
    # cancel from all camera-relative finger projections. It is not a claim
    # about the recorded arm's measured joint state.
    fk = forward(ROOT/cfg["robot"]["urdf"], {"joint7": width/2, "joint8": width/2})
    pose = fk["link6"] @ pose_matrix(camera["position"], camera["rotation_wxyz_ros"])
    wrist_results = []
    for tip in wrist_annotation["tips"]:
        link = tip["link"]
        record = manifest["arms"][side]["links"][link]
        if record["parent_link"] != link:
            raise ValueError("CAD gripper manifest has inconsistent parent links")
        path = ROOT/record["asset"]
        points = transform_points(mesh_points(path), fk[link])
        pixels = project(points, pose, intrinsic)
        index = int(np.argmin(pixels[:, 1]))
        predicted = pixels[index]
        observed = tip["observed_pixel"]
        residual = float(predicted[1]-observed[1])
        wrist_results.append({
            "name": tip["name"], "link": link,
            "predicted_uppermost_vertex_px": predicted.tolist(), "observed_pixel": observed,
            "signed_vertical_residual_px": residual, "absolute_vertical_residual_px": abs(residual),
            "annotation_uncertainty_px": wrist_annotation["uncertainty_px"],
            "within_annotation_uncertainty": abs(residual) <= wrist_annotation["uncertainty_px"],
        })
        provenance["gripper_assets"].append({"name": path.name, "sha256": digest(path)})

    return {
        "schema_version": 1, "dataset": landmarks["dataset"], "passed": False,
        "status": "diagnostic_only_physical_alignment_not_validated",
        "calibration_performed": False, "fitted_parameters": [],
        "pass_limitations": ["Manual silhouette/edge associations are not a calibration target.",
                             "Unknown device calibration and geometry associations prevent physical alignment certification."],
        "inputs": provenance, "optional_reference_image_verification": image_verification,
        "annotation_limitations": landmarks["limitations"],
        "camera_models": {name: {"intrinsic_matrix": rig["cameras"][name]["rgb"]["intrinsic_matrix"],
                                  "calibration_status": rig["cameras"][name]["rgb"]["calibration_status"]}
                          for name in (table_annotation["camera"], wrist_annotation["camera"])},
        "overhead": {"episode": table_annotation["episode"], "frame": table_annotation["frame"],
                     "stream": table_annotation["stream"], "edges": table_results,
                     "association_limits": table_annotation["association_limits"],
                     "additional_observation_frames": table_annotation["additional_observation_frames"]},
        "wrist": {"episode": wrist_annotation["episode"], "frame": wrist_annotation["frame"],
                  "stream": wrist_annotation["stream"], "gripper_width_m": width,
                  "width_source": wrist_annotation["width_source"], "tips": wrist_results,
                  "arm_pose_role": "computational reference only; common six-axis transforms cancel",
                  "association_limits": wrist_annotation["association_limits"]},
        "max_absolute_vertical_residual_px": max(
            item["absolute_vertical_residual_px"] for item in table_results+wrist_results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/connector_alignment.yaml")
    parser.add_argument("--landmarks", type=Path, default=ROOT/"configs/alignment_landmarks.json")
    parser.add_argument("--gripper-manifest", type=Path, default=ROOT/"assets/generated/cad_gripper/manifest.json")
    parser.add_argument("--reference-images", type=Path,
                        help="Optionally verify local reference PNG hashes; images are never copied")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for diagnostic JSON")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    result = measure(args.config, args.landmarks, args.gripper_manifest, args.reference_images)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir/"alignment_measurements.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(output), "passed": result["passed"], "status": result["status"],
                      "max_absolute_vertical_residual_px": result["max_absolute_vertical_residual_px"]}, indent=2))
