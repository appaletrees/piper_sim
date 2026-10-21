#!/usr/bin/env python3
"""Export the supplied STEP gripper as articulated visual meshes.

This changes visual geometry only. Joint limits, collision shapes and inertial
parameters remain the responsibility of the robot description. The single CAD
opening is a reference pose, not evidence of a hardware travel limit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from convert_workcell_cad import (
    CAD_ROTATION, CAD_TRANSLATION, SOURCE_SHA256, instances, tessellate,
)
from piper_sim.config import ROOT, load
from piper_sim.kinematics import forward
from piper_sim.transforms import pose_matrix, transform_points


# Original occurrence IDs, before the left arm's repeated assembly (+56).
GROUPS = {"link6": [37, 38, 39, 40, 41, 42],
          "link7": [43, 44, 45, 46], "link8": [33, 34, 35, 36]}


def opening_reference(points_by_occurrence, link6_world, shift=0):
    """Return the parallel support-plane gap along the gripper's sliding axis.

    The tiny assembly skew is retained. At joint zero the two innermost support
    planes coincide; they need not coincide with the URDF Y=0 plane exactly.
    """
    local = {i: transform_points(p, np.linalg.inv(link6_world))
             for i, p in points_by_occurrence.items()}
    negative_inner = max(local[i+shift][:, 1].max() for i in GROUPS["link7"])
    positive_inner = min(local[i+shift][:, 1].min() for i in GROUPS["link8"])
    width = float(positive_inner-negative_inner)
    if not 0.04 < width < 0.08:
        raise ValueError(f"Unexpected source CAD gripper opening: {width} m")
    return width, float((negative_inner+positive_inner)/2)


def write_visual(stage, path, points, triangles, metallic=False):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(len(triangles), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
        triangles.astype(np.int32).reshape(-1)))
    color = (.42, .44, .46) if metallic else (.025, .028, .032)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    material = UsdShade.Material.Define(stage, path+"/Material")
    shader = UsdShade.Shader.Define(stage, path+"/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.35 if metallic else .55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(.7 if metallic else .05)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def prepare(step=ROOT/"demoa_asm_20261020.stp", output=ROOT/"assets/generated/cad_gripper",
            config=None):
    from pxr import Usd, UsdGeom
    cfg = load(config)
    if hashlib.sha256(step.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("STEP revision changed; re-identify gripper occurrences")
    parts = list(instances(step))
    if len(parts) != 168:
        raise ValueError("Unexpected STEP assembly structure")
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "source_sha256": SOURCE_SHA256,
              "source": str(step), "units": "metres", "tessellation_tolerance_mm": .1,
              "scope": "visuals only; upstream collision, inertia and joint limits unchanged",
              "material_source": "approximate gray/black appearance from user photos; no measured reflectance",
              "opening_definition": "distance between innermost support planes along link6 Y; width=joint7+joint8",
              "arms": {}, "max_roundtrip_error_m": 0.}
    for side, shift in (("right", 0), ("left", 56)):
        base = pose_matrix(cfg["robot"][side]["position"], cfg["robot"][side]["rotation"])
        reference = forward(ROOT/cfg["robot"]["urdf"], {}, base)
        world, faces = {}, {}
        for ids in GROUPS.values():
            for i in ids:
                p, f = tessellate(parts[i+shift][1], .1)
                world[i+shift] = p @ CAD_ROTATION.T*.001+CAD_TRANSLATION
                faces[i+shift] = f
        width, center = opening_reference(world, reference["link6"], shift)
        reference_joints = {"joint7": width/2, "joint8": width/2}
        reference = forward(ROOT/cfg["robot"]["urdf"], reference_joints, base)
        arm = {"source_reference_width_m": width,
               "reference_joint_positions_m": reference_joints,
               "closed_contact_center_link6_y_m": center,
               "hardware_travel_limit_inferred": False, "links": {}}
        for link, ids in GROUPS.items():
            path = output/f"{side}_{link}.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, "Z")
            root = UsdGeom.Xform.Define(stage, "/CadGripper")
            stage.SetDefaultPrim(root.GetPrim())
            record = {"asset": str(path.resolve().relative_to(ROOT)), "parent_link": link,
                      "reference_link_world": reference[link].tolist(), "parts": []}
            for i in ids:
                index = i+shift
                local = transform_points(world[index], np.linalg.inv(reference[link]))
                rebuilt = transform_points(local.astype(np.float32), reference[link])
                error = float(np.linalg.norm(rebuilt-world[index], axis=1).max())
                result["max_roundtrip_error_m"] = max(result["max_roundtrip_error_m"], error)
                write_visual(stage, f"/CadGripper/part_{index:03d}", local, faces[index],
                             metallic=i in (34, 35, 40, 44, 45))
                record["parts"].append({"occurrence": index, "source_path": parts[index][0],
                    "vertices": len(local), "triangles": len(faces[index]),
                    "source_world_bounds_m": [world[index].min(0).tolist(), world[index].max(0).tolist()],
                    "local_bounds_m": [local.min(0).tolist(), local.max(0).tolist()],
                    "roundtrip_error_m": error})
            stage.GetRootLayer().Export(str(path))
            arm["links"][link] = record
        local_base = np.concatenate([transform_points(world[i+shift], np.linalg.inv(reference["link6"]))
                                     for i in GROUPS["link6"]])
        arm["fixed_base_width_m"] = float(np.ptp(local_base[:, 1]))
        result["arms"][side] = arm
    (output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"manifest": str(output/"manifest.json"),
                      "source_reference_width_m": {s: a["source_reference_width_m"]
                                                   for s, a in result["arms"].items()},
                      "max_roundtrip_error_m": result["max_roundtrip_error_m"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=Path, default=ROOT/"demoa_asm_20261020.stp")
    parser.add_argument("--output", type=Path, default=ROOT/"assets/generated/cad_gripper")
    parser.add_argument("--config")
    args = parser.parse_args()
    prepare(args.step, args.output, args.config)
