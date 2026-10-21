#!/usr/bin/env python3
"""Check actual OpenGL pixels against the configured K, at all three CAD mounts.

Markers are viewer-only geometry: they never change dynamics or recordings.
Run with MUJOCO_GL=egl for headless GPU rendering on Linux.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.config import load


def check(backend, output):
    results = {}
    for name, mount in backend.rig["cameras"].items():
        width, height = mount["rgb"]["resolution"]
        k = np.array(mount["rgb"]["intrinsic_matrix"])
        cid = backend.model.camera(name).id
        rotation = backend.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        expected = np.array([[.15*width, .25*height], [.5*width, .75*height], [.85*width, .4*height]])
        rays = np.column_stack(((expected[:, 0]-k[0, 2])/k[0, 0],
                                (expected[:, 1]-k[1, 2])/k[1, 1], np.ones(3)))
        points = .2*rays @ rotation.T + backend.data.cam_xpos[cid]
        with mujoco.Renderer(backend.model, height=height, width=width) as renderer:
            renderer.update_scene(backend.data, camera=name)
            for i, point in enumerate(points):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                radius = max(.0015, 5*.2/min(k[0, 0], k[1, 1]))
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, radius), point, np.eye(3).reshape(-1),
                    np.array([float(c == i) for c in range(3)]+[1.]))
                geom.emission = 1
                renderer.scene.ngeom += 1
            pixels = renderer.render().copy()
        Image.fromarray(pixels).save(output/f"projection_{name}.png")
        measured = []
        for i, (u, v) in enumerate(expected):
            x0, y0 = int(u)-50, int(v)-50
            patch = pixels[y0:int(v)+51, x0:int(u)+51].astype(float)
            others = np.max(np.delete(patch, i, axis=2), axis=2)
            ys, xs = np.nonzero((patch[..., i] > 100) & (patch[..., i] > 1.8*others))
            if len(xs) < 8:
                raise ValueError(f"{name}: marker {i} missing near {u, v}")
            measured.append([float(xs.mean()+x0), float(ys.mean()+y0)])
        errors = np.linalg.norm(np.array(measured)-expected, axis=1)
        results[name] = {"model": mount["model"], "resolution": [width, height],
            "intrinsic_matrix": k.tolist(), "expected_pixels": expected.tolist(),
            "measured_pixels": measured, "errors_px": errors.tolist()}
    maximum = max(max(c["errors_px"]) for c in results.values())
    report = {"passed": maximum <= 2., "engine": "mujoco", "version": mujoco.__version__,
        "tolerance_px": 2., "max_error_px": maximum, "cameras": results,
        "source_sha256": backend.rig["source_sha256"],
        "rgb_profiles_sha256": backend.rig["rgb_profiles_sha256"]}
    (output/"projection.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError(f"Rendered projection error {maximum:.3f}px exceeds 2px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, default=Path("outputs/mujoco_projection"))
    args = parser.parse_args()
    if (args.output/"projection.json").exists():
        raise FileExistsError(args.output/"projection.json")
    args.output.mkdir(parents=True, exist_ok=True)
    backend = MujocoBackend(load(args.config))
    try:
        check(backend, args.output)
    finally:
        backend.close()
