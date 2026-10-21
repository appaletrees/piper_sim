import json
from pathlib import Path
import numpy as np
import pytest
from piper_sim.config import ROOT, load
from piper_sim.transforms import load_camera_rig, pose_matrix, transform_points

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")


def test_static_cad_topology_collision_and_units():
    cfg = load()
    path = ROOT/cfg["scene"]["cad_usd"]
    manifest = json.loads(path.with_suffix(".json").read_text())
    stage = Usd.Stage.Open(str(path))
    assert stage
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Z"
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == manifest["static_mesh_count"] == 52
    assert any(p["excluded_robot"] for p in manifest["parts"])
    for prim in meshes:
        guide = UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide"
        assert prim.HasAPI(UsdPhysics.CollisionAPI) != guide
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get())
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        assert np.isfinite(points).all()
        assert indices.min() >= 0 and indices.max() < len(points)
        assert counts.sum() == len(indices)


def test_cad_table_matches_contact_height_and_config():
    cfg = load()
    manifest = json.loads((ROOT/cfg["scene"]["cad_usd"]).with_suffix(".json").read_text())
    table = next(p for p in manifest["parts"] if p["path"][-1] == "TABLE_DEMOA")
    bounds = np.array(table["world_bounds_m"])
    np.testing.assert_allclose(bounds[1]-bounds[0], cfg["scene"]["table"]["size"], atol=1e-8)
    np.testing.assert_allclose(bounds.mean(0), cfg["scene"]["table"]["position"], atol=1e-8)


def test_selected_camera_bodies_brackets_and_reference_dimensions():
    cfg = load()
    rig = load_camera_rig(cfg)
    assert rig["selection"] == {"overhead": "D455", "left_wrist": "D435i", "right_wrist": "D435i"}
    expected_centers = {"overhead": [.0189, 0, .75366764],
                        "left_wrist": [.09384029, .27000935, .32523133],
                        "right_wrist": [.09384029, -.32999065, .32523133]}
    for name, camera in rig["cameras"].items():
        world = np.array(camera["reference_world_pose"])
        np.testing.assert_allclose(world[:3, 3], expected_centers[name], atol=2e-6)
        depression = 60 if name == "overhead" else 15
        np.testing.assert_allclose(world[2, 2], -np.sin(np.deg2rad(depression)), atol=1e-5)
        expected_width = 124 if name == "overhead" else 90
        assert abs(camera["housing_dimensions_in_camera_frame_mm"][0]-expected_width) < .1
        assert abs(camera["stereo_aperture_spacing_mm"]-(95 if name == "overhead" else 65)) < .01
    for side in ("left", "right"):
        mount = rig["mounts"][side]
        stage = Usd.Stage.Open(str(ROOT/mount["usd"]))
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert len(meshes) == 10  # housing + 3 bracket solids + 6 screws
        assert all(p.HasAPI(UsdPhysics.CollisionAPI) for p in meshes)
        assert all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull" for p in meshes)
        assert sum("SCREW" in p.GetName() for p in meshes) == 6
        reference = np.array(mount["reference_link6_world"])
        camera = rig["cameras"][f"{side}_wrist"]
        expected = reference @ pose_matrix(camera["position"], camera["rotation_wxyz_ros"])
        np.testing.assert_allclose(expected, camera["reference_world_pose"], atol=1e-10)
    assert rig["max_geometry_roundtrip_error_m"] < 1e-7


def test_wrist_meshes_reconstruct_source_assembly_bounds():
    cfg = load()
    rig = load_camera_rig(cfg)
    manifest = json.loads((ROOT/cfg["scene"]["cad_usd"]).with_suffix(".json").read_text())
    assert {p["occurrence"] for p in manifest["parts"] if p["omission_reason"] and
            not p["omission_reason"].startswith("frozen")} == {51, 107, 122, 152, 153, 154}
    for side in ("left", "right"):
        stage = Usd.Stage.Open(str(ROOT/rig["mounts"][side]["usd"]))
        reference = np.array(rig["mounts"][side]["reference_link6_world"])
        for part in manifest["parts"]:
            if part.get("target") != side:
                continue
            points = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(part["usd_prim"])).GetPointsAttr().Get())
            world = transform_points(points, reference)
            np.testing.assert_allclose([world.min(0), world.max(0)], part["world_bounds_m"], atol=1e-7)
