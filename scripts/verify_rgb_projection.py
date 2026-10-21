#!/usr/bin/env python3
"""Validate rendered RGB projection using coloured fiducials at known 3D points.

Uses the real mounted sensors, renderer and frame readback. No physics advances
while measuring, and the test markers never enter a training episode.
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--output", type=Path, default=Path("outputs/rgb_projection"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app
backend = None
try:
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom
    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage
    from piper_sim.backends.isaac import IsaacBackend
    from piper_sim.config import load
    from piper_sim.transforms import load_camera_rig, pose_matrix, transform_points

    cfg = load(args.config)
    backend = IsaacBackend(cfg, args.device, cameras=True)
    rig = load_camera_rig(cfg)
    stage = get_current_stage()
    observation = backend.read()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output/"projection.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    results = {}
    for name, mount in rig["cameras"].items():
        rgb = mount["rgb"]
        width, height = rgb["resolution"]
        k = np.array(rgb["intrinsic_matrix"])
        pose = observation.extras["cameras"][name]
        transform = pose_matrix(pose["position_world"], pose["quaternion_wxyz_ros"])
        expected = np.array([[.15*width, .25*height], [.5*width, .75*height], [.85*width, .4*height]])
        camera_points = np.column_stack(((expected[:, 0]-k[0, 2])/k[0, 0],
                                          (expected[:, 1]-k[1, 2])/k[1, 1], np.ones(3))) * .2
        points = transform_points(camera_points, transform)
        group = f"/World/Projection_{name}"
        UsdGeom.Xform.Define(stage, group)
        for i, point in enumerate(points):
            color = tuple(float(i == channel) for channel in range(3))
            shape = sim_utils.SphereCfg(radius=.0015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color))
            shape.func(f"{group}/marker_{i}", shape, translation=tuple(point))
        for _ in range(16):
            backend.sim.render()
        sensor = backend.cameras[name]
        sensor.update(backend.dt*backend.decimation, force_recompute=True)
        pixels = sensor.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(pixels).save(args.output/f"projection_{name}.png")
        assert list(pixels.shape) == [height, width, 3]
        np.testing.assert_allclose(sensor.data.intrinsic_matrices[0].cpu().numpy(), k, atol=1e-4)
        measured = []
        for i, (u, v) in enumerate(expected):
            x0, x1 = int(u)-40, int(u)+41
            y0, y1 = int(v)-40, int(v)+41
            patch = pixels[y0:y1, x0:x1].astype(float)
            others = np.max(np.delete(patch, i, axis=2), axis=2)
            mask = (patch[..., i] > 100) & (patch[..., i] > 1.8*others)
            ys, xs = np.nonzero(mask)
            if len(xs) < 8:
                raise ValueError(f"{name}: marker {i} not visible at expected pixel {u, v}")
            measured.append([float(xs.mean()+x0), float(ys.mean()+y0)])
        errors = np.linalg.norm(np.array(measured)-expected, axis=1)
        results[name] = {"model": mount["model"], "resolution": [width, height],
                         "intrinsic_matrix": k.tolist(), "expected_pixels": expected.tolist(),
                         "measured_pixels": measured, "errors_px": errors.tolist()}
        if errors.max() > 2.:
            raise ValueError(f"{name}: rendered optical projection error {errors.max():.3f}px > 2px")
        UsdGeom.Imageable(stage.GetPrimAtPath(group)).MakeInvisible()
    report = {"passed": True, "tolerance_px": 2., "cameras": results,
              "max_error_px": max(max(value["errors_px"]) for value in results.values()),
              "source_sha256": rig["source_sha256"], "rgb_profiles_sha256": rig["rgb_profiles_sha256"]}
    report_path.write_text(json.dumps(report, indent=2))
    (args.output/"projection_failure.txt").unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
except BaseException:
    # Kit's fast shutdown may exit before Python can display a pending exception.
    import traceback
    traceback.print_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"projection_failure.txt").write_text(traceback.format_exc())
    raise
finally:
    if backend:
        backend.close()
    else:
        from isaaclab.sim import SimulationContext
        context = SimulationContext.instance()
        if context is not None:
            context._disable_app_control_on_stop_handle = True
    app.close(wait_for_replicator=False)
