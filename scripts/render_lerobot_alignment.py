#!/usr/bin/env python3
"""Render saved observations without control, physics integration or hardware.

This is a visual audit, not trajectory replay: observed SDK gripper strokes can
exceed the URDF's dynamics limits and are never silently clipped or sent to CAN.
"""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config")
parser.add_argument("--original-gripper", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.output.exists():
    parser.error("Use a new output directory; existing evidence is never overwritten")
manifest = json.loads(args.manifest.read_text())
dataset_root = Path(manifest["dataset_root"]).resolve()
if args.output.resolve() == dataset_root or args.output.resolve().is_relative_to(dataset_root):
    parser.error("Render outputs must be outside the read-only dataset")
if manifest.get("unit_contract") != "connector_plugin_v2_1_degrees_normalized95mm":
    parser.error("Unsupported dataset unit contract")
args.output.mkdir(parents=True)
launcher = AppLauncher(args)
app = launcher.app
backend = None
try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from pxr import UsdGeom
    from isaacsim.core.utils.stage import get_current_stage
    from piper_sim.backends.isaac import IsaacBackend
    from piper_sim.config import load, ROOT
    from piper_sim.transforms import load_camera_rig, pose_matrix
    from piper_sim.kinematics import forward
    from piper_sim.contracts import JOINT_NAMES, vector

    cfg = load(args.config)
    cfg["cameras"]["streams"] = {name: {"mode": "crop_nominal", "resolution": [640, 480]}
                                     for name in manifest["camera_mapping"].values()}
    # A caller-supplied calibrated stream always takes precedence over the
    # explicitly unverified fallback above.
    cfg["cameras"]["streams"].update(load(args.config)["cameras"].get("streams", {}))
    cfg["appearance"] = {"cad_gripper": not args.original_gripper, "photo_materials": True}
    backend = IsaacBackend(cfg, args.device, cameras=True)
    stage = get_current_stage()
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Object")).MakeInvisible()
    rig = load_camera_rig(cfg)
    report = {"schema_version": 1, "mode": "static_observation_visual_audit",
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "physics_advanced_during_capture": False, "hardware_accessed": False,
              "state_clipping_applied": False,
              "physical_alignment_verified": False,
              "gripper_stroke_is_measured_aperture": False,
              "feedback_quality": manifest.get("episode_feedback_quality", {}),
              "rig": rig, "samples": []}
    for sample in manifest["samples"]:
        capture_physics_time = backend.sim.current_time
        q = vector(sample["state_si"])
        if np.max(np.abs(q[[6, 13]])) > .09501 or np.min(q[[6, 13]]) < 0:
            raise ValueError("Unexpected SDK stroke outside this dataset's conversion contract")
        for i, (side, arm) in enumerate(backend.arms.items()):
            positions = arm.data.default_joint_pos.clone()
            a = q[i*7:i*7+7]
            positions[0, backend.indices[side]] = torch.tensor(
                [*a[:6], a[6]/2, a[6]/2], device=backend.sim.device, dtype=torch.float32)
            arm.write_joint_state_to_sim(positions, torch.zeros_like(positions))
            arm.set_joint_position_target(positions)
            arm.update(backend.dt)
        for _ in range(16):
            backend.sim.render()
        for camera in backend.cameras.values():
            camera.update(backend.dt*backend.decimation, force_recompute=True)
        observation = backend.read()
        if abs(backend.sim.current_time-capture_physics_time) > 1e-9:
            raise ValueError("Physics advanced during static visual capture")
        np.testing.assert_allclose(observation.position, q, atol=2e-6)
        errors = {}
        for name, camera in rig["cameras"].items():
            predicted = pose_matrix(camera["position"], camera["rotation_wxyz_ros"])
            if name != "overhead":
                side = name.split("_")[0]
                i = 0 if side == "left" else 1
                base = pose_matrix(cfg["robot"][side]["position"], cfg["robot"][side]["rotation"])
                predicted = forward(ROOT/cfg["robot"]["urdf"],
                    dict(zip(JOINT_NAMES, q[i*7:i*7+6])), base)["link6"] @ predicted
            actual = observation.extras["cameras"][name]
            measured = pose_matrix(actual["position_world"], actual["quaternion_wxyz_ros"])
            errors[name] = float(np.max(np.abs(measured-predicted)))
            if errors[name] > 2e-5:
                raise ValueError(f"{name}: rendered pose did not follow the saved joints")
        stem = f"episode_{sample['episode_index']:06d}_frame_{sample['frame_index']:06d}"
        canvas = Image.new("RGB", (640*3, 510*2), "white")
        draw = ImageDraw.Draw(canvas)
        for col, name in enumerate(("overhead", "left_wrist", "right_wrist")):
            real_path = Path(sample["images"][name]["path"])
            if hashlib.sha256(real_path.read_bytes()).hexdigest() != sample["images"][name]["sha256"]:
                raise ValueError(f"Source reference image changed: {real_path}")
            real = Image.open(real_path).convert("RGB")
            rendered = Image.fromarray(observation.images[name])
            if real.size != rendered.size:
                raise ValueError("Reference and simulated stream dimensions differ")
            rendered.save(args.output/f"{stem}_{name}.png")
            canvas.paste(real, (col*640, 30))
            canvas.paste(rendered, (col*640, 540))
            draw.text((col*640+8, 8), f"REAL {name} | {stem}", fill="black")
            draw.text((col*640+8, 518), f"CAD + UNVERIFIED stream K | {name}", fill="black")
        canvas.save(args.output/f"{stem}_comparison.png")
        report["samples"].append({"episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"], "state_si": q.tolist(),
            "state_max_error": float(np.max(np.abs(observation.position-q))),
            "physics_time_before_and_after": [capture_physics_time, backend.sim.current_time],
            "camera_matrix_max_errors": errors, "cameras": observation.extras["cameras"],
            "state_bounds_violations": sample["state_bounds_violations"]})
    if not args.headless:
        backend.sim.set_camera_view([1.65, -2.4, 1.45], [.12, 0, .4])
        for _ in range(8):
            backend.sim.render()
        backend.save_overview(args.output/"overview.png")
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"rendered_samples": len(report["samples"]), "output": str(args.output)}))
except BaseException:
    import traceback
    traceback.print_exc()
    (args.output/"failure.txt").write_text(traceback.format_exc())
    raise
finally:
    if backend:
        backend.close()
    app.close(wait_for_replicator=False)
