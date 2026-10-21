#!/usr/bin/env python3
"""Convert the supplied STEP workcell and its moving wrist camera assemblies.

The selected revision is hash-checked: occurrence numbers disambiguate duplicate
part names in this particular assembly. Geometry is never resized or re-modelled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.gp import gp_Trsf


def label_name(label):
    attr = TDataStd_Name()
    return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else ""


def instances(step):
    doc = TDocStd_Document(TCollection_ExtendedString("piper-workcell"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(str(step)) != IFSelect_RetDone or not reader.Transfer(doc):
        raise RuntimeError(f"STEP import failed: {step}")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    free = TDF_LabelSequence()
    shapes.GetFreeShapes(free)

    def walk(label, location, path):
        referred = TDF_Label()
        if shapes.IsReference_s(label):
            location = location.Multiplied(shapes.GetLocation_s(label))
            shapes.GetReferredShape_s(label, referred)
        else:
            referred = label
        name = label_name(referred) or label_name(label) or "unnamed"
        path = [*path, name]
        if shapes.IsAssembly_s(referred):
            children = TDF_LabelSequence()
            shapes.GetComponents_s(referred, children)
            for i in range(1, children.Length()+1):
                yield from walk(children.Value(i), location, path)
        else:
            shape = shapes.GetShape_s(referred).Moved(location)
            if not shape.IsNull():
                yield path, shape

    for i in range(1, free.Length()+1):
        yield from walk(free.Value(i), TopLoc_Location(), [])


def bounds(shape):
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return np.asarray(box.Get()).reshape(2, 3)


def tessellate(shape, tolerance):
    mesh = BRepMesh_IncrementalMesh(shape, tolerance, False, .25, True)
    mesh.Perform()
    vertices, faces = [], []
    offset = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            for i in range(1, triangulation.NbNodes()+1):
                p = triangulation.Node(i).Transformed(transform)
                vertices.append((p.X(), p.Y(), p.Z()))
            for i in range(1, triangulation.NbTriangles()+1):
                a, b, c = triangulation.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                faces.append((offset+a-1, offset+b-1, offset+c-1))
            offset += triangulation.NbNodes()
        explorer.Next()
    return np.asarray(vertices), np.asarray(faces)


SOURCE_SHA256 = "6c7f1281549c60c1f634d6e99a0b886dca66c7cd0305109f3e4cc6d735399e2d"
CAD_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
CAD_TRANSLATION = np.array([-.055, 0., .025])
WRIST_PARTS = {
    "right": [47, 48, 49, 50, 52, 53, 54, 55, 56, 57],
    "left": [103, 104, 105, 106, 108, 109, 110, 111, 112, 113],
}
OMITTED = {
    51: "duplicate right D435; occurrence 47 selected",
    107: "duplicate left D435; occurrence 103 selected",
    122: "alternative central D435 bracket M_2; D455 bracket M_3 selected",
    152: "alternative central D435 body; D455 selected",
    153: "alternative central D435 mounting screw",
    154: "alternative central D435 mounting screw",
}


def cylinder_features(shape, radius):
    """Read analytic cylindrical surfaces in evaluated assembly coordinates."""
    result = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        surface = BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current()))
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            if abs(cylinder.Radius() - radius) < 1e-5:
                origin, axis = cylinder.Location(), cylinder.Axis().Direction()
                result.append((np.array([origin.X(), origin.Y(), origin.Z()]),
                               np.array([axis.X(), axis.Y(), axis.Z()]),
                               surface.LastVParameter()))
        explorer.Next()
    return result


def optical_reference(shape, model):
    """Virtual projection frame at stereo aperture midpoint, NOT RGB calibration."""
    candidates = cylinder_features(shape, 8.105 if model == "D435" else 9.35)
    apertures = []
    for center, axis, end in candidates:
        entrance = center + axis*end
        if not any(np.linalg.norm(entrance-p) < .001 for p in apertures):
            apertures.append(entrance)
    if len(apertures) != 2:
        raise ValueError(f"{model}: expected two stereo apertures, got {len(apertures)}")
    # Image right is toward negative CAD X / negative world Y.
    apertures.sort(key=lambda p: p[0])
    x_axis = apertures[0]-apertures[1]
    x_axis /= np.linalg.norm(x_axis)
    z_axis = candidates[0][1]
    if z_axis[2] < 0:
        z_axis = -z_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rotation = CAD_ROTATION @ np.column_stack([x_axis, y_axis, z_axis])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = CAD_ROTATION @ np.mean(apertures, axis=0)*.001 + CAD_TRANSLATION
    return pose, {"stereo_aperture_centers_cad_mm": [p.tolist() for p in apertures],
                  "stereo_aperture_spacing_mm": float(np.linalg.norm(apertures[0]-apertures[1])),
                  "front_normal_cad": z_axis.tolist()}


def quaternion(rotation):
    # Gf uses row-vector matrices; the rest of this project uses column vectors.
    from pxr import Gf
    q = Gf.Matrix3d(*rotation.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    return [q.GetReal(), *q.GetImaginary()]


def camera_body_dimensions(shape, world_pose):
    """Exact BRep bounds in its front-face frame; values remain in millimetres."""
    matrix = np.eye(4)
    matrix[:3, :3] = world_pose[:3, :3].T @ CAD_ROTATION
    matrix[:3, 3] = world_pose[:3, :3].T @ (CAD_TRANSLATION-world_pose[:3, 3])*1000
    trsf = gp_Trsf()
    trsf.SetValues(*matrix[:3].reshape(-1).tolist())
    return np.ptp(bounds(shape.Moved(TopLoc_Location(trsf))), axis=0)


def new_stage(path, root_name):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, f"/{root_name}", path


def write_mesh(stage, path, points, faces, color, *, moving=False, guide=False):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).reshape(-1)))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if guide:
        mesh.CreatePurposeAttr("guide")
    else:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        if moving:
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    material = UsdShade.Material.Define(stage, path + "/Material")
    shader = UsdShade.Shader.Define(stage, path + "/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def convert(step, output, tolerance=.1, config=None):
    from piper_sim.config import ROOT, load
    from piper_sim.kinematics import forward
    from piper_sim.transforms import pose_matrix, transform_points
    cfg = load(config)
    if not 0 < tolerance <= .1:
        raise ValueError("The source-alignment export requires 0 < tessellation tolerance <= 0.1 mm")
    digest = hashlib.sha256(step.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("STEP revision changed; re-identify camera occurrences before conversion")
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = list(instances(step))
    if len(parts) != 168:
        raise ValueError("Unexpected STEP leaf occurrence count")
    stages = {"static": new_stage(output, "Workcell")}
    reference_poses = {}
    rig = {
        "schema_version": 1, "source_sha256": digest,
        "selection": {"overhead": "D455", "left_wrist": "D435", "right_wrist": "D435"},
        "reference_joint_positions_rad": [0.]*6,
        "geometry_units": "metres", "geometry_scale_from_cad_mm": .001,
        "calibration_status": "CAD geometry aligned; RGB optics uncalibrated",
        "projection_reference": "midpoint of stereo entrance apertures; +Z front, +X image right (ROS)",
        "intrinsics": {"focal_length_mm": 24., "horizontal_aperture_mm": 36.,
                       "status": "generic pinhole placeholder, not RealSense calibration"},
        "cameras": {}, "mounts": {}, "alignment": {},
    }
    for side in ("left", "right"):
        base = pose_matrix(cfg["robot"][side]["position"], cfg["robot"][side]["rotation"])
        reference_poses[side] = forward(ROOT/cfg["robot"]["urdf"], {}, base)["link6"]
        mount_path = output.parent / f"{side}_wrist.usda"
        stages[side] = new_stage(mount_path, "CameraRig")
        rig["mounts"][side] = {
            "usd": str(mount_path.resolve().relative_to(ROOT)),
            "parent_prim": f"/World/{side}/link6",
            "source_occurrences": WRIST_PARTS[side],
            "reference_link6_world": reference_poses[side].tolist(),
        }
        flange = cylinder_features(parts[26 if side == "right" else 82][1], 17.5)[0]
        cad_flange = flange[0]+flange[1]*flange[2]
        flange_world = CAD_ROTATION @ cad_flange*.001 + CAD_TRANSLATION
        rig["alignment"][side] = {
            "cad_flange_world_m": flange_world.tolist(),
            "urdf_flange_world_m": reference_poses[side][:3, 3].tolist(),
            "source_urdf_flange_difference_mm": float(
                np.linalg.norm(flange_world-reference_poses[side][:3, 3])*1000),
        }
    manifest = {
        "source": str(step.resolve()), "sha256": digest,
        "cad_to_world": {"rotation": CAD_ROTATION.tolist(), "scale": .001,
                         "translation_m": CAD_TRANSLATION.tolist()},
        "tessellation_tolerance_mm": tolerance,
        "color_note": "neutral display materials; STEP contains no source surface colors",
        "parts": [],
    }
    previews = {key: {} for key in stages}
    counts = dict.fromkeys(stages, 0)
    max_roundtrip_error = 0.
    for index, (path, shape) in enumerate(parts):
        is_robot = any(name.startswith("PIPER2025") for name in path)
        side = next((s for s, ids in WRIST_PARTS.items() if index in ids), None)
        reason = OMITTED.get(index)
        if is_robot and side is None and reason is None:
            reason = "frozen CAD robot replaced by articulated source URDF"
        entry = {"occurrence": index, "path": path,
                 "cad_bounds_mm": bounds(shape).tolist(),
                 "excluded_robot": is_robot and side is None,
                 "omission_reason": reason}
        manifest["parts"].append(entry)
        if reason:
            continue
        points_cad, faces = tessellate(shape, tolerance)
        if not len(faces):
            raise RuntimeError(f"No triangles for occurrence {index}: {path}")
        world = points_cad @ CAD_ROTATION.T*.001 + CAD_TRANSLATION
        points = transform_points(world, np.linalg.inv(reference_poses[side])) if side else world
        # Quantize exactly as USD point3f storage, then verify reconstruction.
        stored = points.astype(np.float32)
        rebuilt = transform_points(stored, reference_poses[side]) if side else stored
        error = float(np.linalg.norm(rebuilt-world, axis=1).max())
        max_roundtrip_error = max(max_roundtrip_error, error)
        target = side or "static"
        stage, root, _ = stages[target]
        name = re.sub("[^a-zA-Z0-9_]", "_", path[-1])
        mesh_path = f"{root}/part_{index:03d}_{name}"
        guide = index == 125
        camera_body = index in (47, 103, 124)
        color = (.07, .08, .09) if camera_body else (
            (.33, .36, .40) if "TABLE" in name else (.55, .57, .60))
        write_mesh(stage, mesh_path, points, faces, color, moving=bool(side), guide=guide)
        entry.update({"target": target, "usd_prim": mesh_path, "guide": guide,
                      "triangles": len(faces), "world_bounds_m": [world.min(0).tolist(), world.max(0).tolist()],
                      "roundtrip_max_error_m": error})
        if not guide:
            preview = previews[target]
            n = len(preview)//3
            preview[f"points_{n}"] = stored
            preview[f"triangles_{n}"] = faces.astype(np.int32)
            preview[f"color_{n}"] = np.asarray(color, dtype=np.float32)
        counts[target] += 1
        if camera_body:
            model = "D455" if index == 124 else "D435"
            world_pose, evidence = optical_reference(shape, model)
            local_pose = np.linalg.inv(reference_poses[side]) @ world_pose if side else world_pose
            name = f"{side}_wrist" if side else "overhead"
            # Measure housing in its analytic front frame (bounding box is rotated in CAD).
            camera_points = transform_points(world, np.linalg.inv(world_pose))
            rig["cameras"][name] = {
                "model": model, "source_occurrence": index,
                "prim_path": f"/World/{side}/link6/CameraRig/Camera" if side else "/World/Overhead",
                "position": local_pose[:3, 3].tolist(), "rotation_wxyz_ros": quaternion(local_pose[:3, :3]),
                "reference_world_pose": world_pose.tolist(),
                "housing_brep_dimensions_mm": camera_body_dimensions(shape, world_pose).tolist(),
                "housing_dimensions_in_camera_frame_mm": (np.ptp(camera_points, axis=0)*1000).tolist(),
                **evidence,
            }
    for key, (stage, _, path) in stages.items():
        stage.GetRootLayer().Export(str(path))
        np.savez_compressed(path.with_suffix(".npz"), **previews[key])
    manifest["static_mesh_count"] = counts["static"]
    manifest["wrist_mesh_count"] = {s: counts[s] for s in ("left", "right")}
    manifest["guide_mesh_count"] = 1
    manifest["max_geometry_roundtrip_error_m"] = max_roundtrip_error
    rig["max_geometry_roundtrip_error_m"] = max_roundtrip_error
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    (output.parent/"camera_rig.json").write_text(json.dumps(rig, indent=2))
    print(json.dumps({"meshes": counts, "max_roundtrip_error_m": max_roundtrip_error,
                      "cameras": {k: v["housing_dimensions_in_camera_frame_mm"]
                                  for k, v in rig["cameras"].items()}}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=Path, default=Path("demoa_asm_20261020.stp"))
    parser.add_argument("--output", type=Path, default=Path("assets/generated/workcell/workcell.usda"))
    parser.add_argument("--tolerance-mm", type=float, default=.1)
    parser.add_argument("--config")
    args = parser.parse_args()
    convert(args.step, args.output, args.tolerance_mm, args.config)
