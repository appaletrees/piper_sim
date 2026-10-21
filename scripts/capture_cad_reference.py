#!/usr/bin/env python3
"""Capture the CAD assembly pose in Isaac, with no hardware and no physics drift."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--output", type=Path, default=Path("outputs/cad_restore_acceptance"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    parser.error("Reference viewport captures require the GUI")
args.enable_cameras = True
app = AppLauncher(args).app
backend = None
try:
    import numpy as np
    from PIL import Image
    from piper_sim.backends.isaac import IsaacBackend
    from piper_sim.config import load
    from piper_sim.transforms import load_camera_rig, pose_matrix
    cfg = load(args.config)
    cfg["robot"]["home"][:6] = [0.]*6
    backend = IsaacBackend(cfg, args.device, cameras=True)
    backend.reset()
    # Render after writing exact joint zero, without integrating gravity.
    for _ in range(12):
        backend.sim.render()
    for camera in backend.cameras.values():
        camera.update(backend.dt*backend.decimation, force_recompute=True)
    observation = backend.read()
    rig = load_camera_rig(cfg)
    pose_errors = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, pose in observation.extras["cameras"].items():
        expected = np.array(rig["cameras"][name]["reference_world_pose"])
        actual = pose_matrix(pose["position_world"], pose["quaternion_wxyz_ros"])
        error = float(np.linalg.norm(expected[:3, 3]-actual[:3, 3]))
        assert error < 1e-6, (name, error)
        pose_errors[name] = error
        path = args.output/f"reference_{name}.png"
        if path.exists():
            raise FileExistsError(path)
        Image.fromarray(observation.images[name]).save(path)
    for name, eye, target in [
        ("overview", [1.65, -2.4, 1.45], [.05, 0, .48]),
        ("left_mount", [.42, .02, .47], [.06, .27, .285]),
        ("right_mount", [.42, -.55, .47], [.06, -.33, .285]),
        ("central_mount", [.36, -.33, .94], [0, 0, .756]),
    ]:
        backend.sim.set_camera_view(eye, target)
        for _ in range(12):
            backend.sim.render()
        backend.save_overview(args.output/f"{name}.png")
    (args.output/"reference_poses.json").write_text(json.dumps({
        "reference_joint_positions_rad": [0.]*6, "source_sha256": rig["source_sha256"],
        "max_position_errors_m": pose_errors, "cameras": observation.extras["cameras"],
        "note": "Exact reset pose rendered without physics integration; not a dynamic episode",
    }, indent=2))
    print(json.dumps(pose_errors, indent=2))
finally:
    if backend:
        backend.close()
    else:
        from isaaclab.sim import SimulationContext
        context = SimulationContext.instance()
        if context is not None:
            context._disable_app_control_on_stop_handle = True
    app.close(wait_for_replicator=False)
