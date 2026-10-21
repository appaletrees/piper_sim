"""Optional visual replacements; rendering choices do not alter robot physics."""
import json
from pathlib import Path
import re

from piper_sim.config import ROOT


def apply_cad_gripper_visuals(stage, manifest_path=None, world_root="/World"):
    """Attach original STEP parts to articulated links and hide old STL visuals.

    Call after spawning the robots and before simulation starts. The source CAD
    reference opening has already been removed from the exported local meshes,
    so joint7/joint8 continue to represent half of the commanded opening.
    """
    from pxr import UsdGeom
    manifest_path = Path(manifest_path or ROOT/"assets/generated/cad_gripper/manifest.json")
    if not manifest_path.is_absolute():
        manifest_path = ROOT/manifest_path
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported CAD gripper visual manifest")
    # Validate everything before changing the stage.
    changes, hides = [], []
    for side, arm in manifest["arms"].items():
        old_base = stage.GetPrimAtPath(f"{world_root}/{side}/gripper_base/visuals")
        if not old_base.IsValid():
            raise ValueError("URDF gripper_base visual prim not found; re-check fixed-link merge layout")
        hides.append(old_base)
        for link, record in arm["links"].items():
            parent = f"{world_root}/{side}/{link}"
            if not stage.GetPrimAtPath(parent).IsValid():
                raise ValueError(f"Required articulated link missing: {parent}")
            asset = ROOT/record["asset"]
            if not asset.is_file():
                raise FileNotFoundError(asset)
            if link != "link6":
                old = stage.GetPrimAtPath(parent+"/visuals")
                if not old.IsValid():
                    raise ValueError(f"Original visual missing: {parent}/visuals")
                hides.append(old)
            changes.append((parent+"/CadVisual", asset))
    for path, asset in changes:
        visual = UsdGeom.Xform.Define(stage, path)
        visual.GetPrim().GetReferences().AddReference(str(asset))
    for old in hides:
        UsdGeom.Imageable(old).MakeInvisible()
    return manifest


def apply_workcell_materials(stage, config_path=None, world_root="/World"):
    """Bind photo-inspired materials without editing geometry or physics.

    Call after robot, CAD workcell and camera-mount creation, before rendering.
    Strong parent bindings override imported STL materials while preserving
    instanceability. Collision branches and articulation attributes are untouched.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    config_path = Path(config_path or ROOT/"configs/appearance.json")
    if not config_path.is_absolute():
        config_path = ROOT/config_path
    config = json.loads(config_path.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported appearance configuration")
    materials = {}
    scope_path = world_root+"/AppearanceMaterials"
    UsdGeom.Scope.Define(stage, scope_path)
    for name, values in config["materials"].items():
        material = UsdShade.Material.Define(stage, scope_path+"/"+name)
        shader = UsdShade.Shader.Define(stage, scope_path+"/"+name+"/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*values["diffuse_color"]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(values["metallic"])
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(values["roughness"])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material
    bound = {}

    def bind(prim, name):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            materials[name], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        bound[str(prim.GetPath())] = name

    for side in ("left", "right"):
        for link, name in config["robot_links"].items():
            visual = stage.GetPrimAtPath(f"{world_root}/{side}/{link}/visuals")
            if visual.IsValid():
                bind(visual, name)
    part_materials = {i: name for name, ids in config["cad_parts"].items() for i in ids}
    roots = [world_root+"/Workcell", *(f"{world_root}/{s}/link6/CameraRig" for s in ("left", "right"))]
    for path in roots:
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            continue
        # The workcell importer does not instance these CAD part meshes. Do not
        # de-instance or modify any unexpected proxy geometry to force a color.
        for prim in Usd.PrimRange(root):
            match = re.fullmatch(r"part_(\d+)_.*", prim.GetName())
            if not match or not prim.IsA(UsdGeom.Mesh):
                continue
            if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            index = int(match.group(1))
            bind(prim, part_materials.get(index, config["remaining_cad_parts"]))
    for name, material in (("table", "table_white"), ("mast", "aluminum")):
        primitive = stage.GetPrimAtPath(f"{world_root}/{name}")
        if primitive.IsValid():
            bind(primitive, material)
    return {"source": config["source"], "calibration_status": config["calibration_status"],
            "camera_front_panel": config["camera_front_panel"], "bindings": bound}
