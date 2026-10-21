"""Source geometry survives the closed-opening articulated visual conversion."""
import json
import numpy as np
import pytest

from piper_sim.config import ROOT
from piper_sim.kinematics import forward
from piper_sim.transforms import pose_matrix, transform_points


def test_exported_gripper_reference_and_closed_gap():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    path = ROOT/"assets/generated/cad_gripper/manifest.json"
    if not path.exists():
        pytest.skip("Run prepare_cad_gripper.py to build the source CAD visual assets")
    manifest = json.loads(path.read_text())
    for side, arm in manifest["arms"].items():
        local_points = {}
        for link, record in arm["links"].items():
            stage = Usd.Stage.Open(str(ROOT/record["asset"]))
            matrix = np.array(record["reference_link_world"])
            all_points = []
            for part in record["parts"]:
                prim = stage.GetPrimAtPath(f"/CadGripper/part_{part['occurrence']:03d}")
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                world = transform_points(points, matrix)
                actual = np.stack([world.min(0), world.max(0)])
                np.testing.assert_allclose(actual, part["source_world_bounds_m"], atol=2e-8)
                assert not prim.HasAPI(UsdPhysics.CollisionAPI)
                all_points.append(points)
            local_points[link] = np.concatenate(all_points)
        # Test the contract across the collected range without changing limits:
        # the visual opening is q7+q8, not CAD opening + q7+q8.
        base = pose_matrix([0, .3 if side == "left" else -.3, .036], [1, 0, 0, 0])
        for width in (0., .04, .07, .095):
            t = forward(ROOT/"assets/generated/piper/piper.urdf",
                        {"joint7": width/2, "joint8": width/2}, base)
            p7 = transform_points(local_points["link7"], np.linalg.inv(t["link6"]) @ t["link7"])
            p8 = transform_points(local_points["link8"], np.linalg.inv(t["link6"]) @ t["link8"])
            gap = p8[:, 1].min()-p7[:, 1].max()
            assert gap == pytest.approx(width, abs=2e-8)
        assert arm["fixed_base_width_m"] == pytest.approx(.164003, abs=1e-5)


def test_visual_replacement_preserves_collision_visibility():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    from piper_sim.appearance import apply_cad_gripper_visuals
    manifest_path = ROOT/"assets/generated/cad_gripper/manifest.json"
    if not manifest_path.exists():
        pytest.skip("Generated CAD gripper assets not installed")
    stage = Usd.Stage.CreateInMemory()
    for side in ("left", "right"):
        for link in ("link6", "link7", "link8", "gripper_base"):
            UsdGeom.Xform.Define(stage, f"/World/{side}/{link}/visuals")
            UsdGeom.Xform.Define(stage, f"/World/{side}/{link}/collisions")
    apply_cad_gripper_visuals(stage)
    for side in ("left", "right"):
        for link in ("link7", "link8", "gripper_base"):
            old = stage.GetPrimAtPath(f"/World/{side}/{link}/visuals")
            assert UsdGeom.Imageable(old).GetVisibilityAttr().Get() == "invisible"
            collision = stage.GetPrimAtPath(f"/World/{side}/{link}/collisions")
            assert UsdGeom.Imageable(collision).GetVisibilityAttr().Get() == "inherited"
        for link in ("link6", "link7", "link8"):
            assert stage.GetPrimAtPath(f"/World/{side}/{link}/CadVisual").HasAuthoredReferences()
