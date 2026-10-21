#!/usr/bin/env python3
"""Re-read source STEP and compare every exported vertex at the CAD reference pose."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from convert_workcell_cad import instances, tessellate, bounds
from piper_sim.config import ROOT, load
from piper_sim.kinematics import forward
from piper_sim.transforms import load_camera_rig, pose_matrix, transform_points


def verify(config=None):
    cfg = load(config)
    rig = load_camera_rig(cfg)
    asset = ROOT/cfg["scene"]["cad_usd"]
    manifest = json.loads(asset.with_suffix(".json").read_text())
    step = Path(manifest["source"])
    assert hashlib.sha256(step.read_bytes()).hexdigest() == manifest["sha256"] == rig["source_sha256"]
    stages = {"static": Usd.Stage.Open(str(asset))}
    reference = {"static": np.eye(4)}
    for side in ("left", "right"):
        stages[side] = Usd.Stage.Open(str(ROOT/rig["mounts"][side]["usd"]))
        base = pose_matrix(cfg["robot"][side]["position"], cfg["robot"][side]["rotation"])
        reference[side] = forward(ROOT/cfg["robot"]["urdf"], {}, base)["link6"]
    report = {"source_sha256": manifest["sha256"], "parts": [],
              "tessellation_tolerance_mm": manifest["tessellation_tolerance_mm"],
              "camera_selection": rig["selection"], "source_flange_comparison": rig["alignment"]}
    source = list(instances(step))
    for part in manifest["parts"]:
        if part["omission_reason"]:
            continue
        index, target = part["occurrence"], part["target"]
        path, shape = source[index]
        assert path == part["path"]
        source_points, source_faces = tessellate(shape, manifest["tessellation_tolerance_mm"])
        # Independent explicit axis mapping: CAD (X,Y,Z) mm -> world (Z,X,Y) m.
        expected = source_points[:, [2, 0, 1]] / 1000 + [-.055, 0, .025]
        mesh = UsdGeom.Mesh(stages[target].GetPrimAtPath(part["usd_prim"]))
        actual = transform_points(np.array(mesh.GetPointsAttr().Get()), reference[target])
        np.testing.assert_array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), source_faces)
        error = np.linalg.norm(expected-actual, axis=1).max()
        assert error < 1e-7, (path, error)
        exact_bounds = bounds(shape)[:, [2, 0, 1]] / 1000 + [-.055, 0, .025]
        usd_bounds = np.array([actual.min(0), actual.max(0)])
        extent_error = float(np.abs(usd_bounds-exact_bounds).max()*1000)
        assert extent_error <= manifest["tessellation_tolerance_mm"]+.002, (path, extent_error)
        report["parts"].append({
            "occurrence": index, "name": path[-1], "target": target,
            "vertex_count": len(actual), "max_vertex_error_mm": float(error*1000),
            "max_analytic_bbox_error_mm": extent_error,
        })
    assert len(report["parts"]) == 72
    report["checked_meshes"] = len(report["parts"])
    report["max_vertex_error_mm"] = max(p["max_vertex_error_mm"] for p in report["parts"])
    report["max_analytic_bbox_error_mm"] = max(p["max_analytic_bbox_error_mm"] for p in report["parts"])
    report["passed"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, default=Path("outputs/cad_restore_acceptance/cad_alignment.json"))
    args = parser.parse_args()
    result = verify(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "parts"}, indent=2))
