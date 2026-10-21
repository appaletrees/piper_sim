import pytest


def test_appearance_changes_material_bindings_only():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    Gf = pytest.importorskip("pxr.Gf")
    from piper_sim.appearance import apply_workcell_materials
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/World/left/link2")
    body.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdGeom.Xform.Define(stage, "/World/left/link2/visuals")
    collision = UsdGeom.Cube.Define(stage, "/World/left/link2/collisions")
    collision.CreateSizeAttr(.123)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    camera = UsdGeom.Mesh.Define(stage, "/World/left/link6/CameraRig/part_103_REALSENSE_D435")
    camera.CreatePointsAttr([Gf.Vec3f(0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    camera.CreateFaceVertexCountsAttr([3])
    camera.CreateFaceVertexIndicesAttr([0, 1, 2])
    guide = UsdGeom.Mesh.Define(stage, "/World/Workcell/part_125_COMPOUND")
    guide.CreatePurposeAttr("guide")
    table = UsdGeom.Mesh.Define(stage, "/World/Workcell/part_000_TABLE_DEMOA")
    before = {str(p.GetPath()): {a.GetName(): a.Get() for a in p.GetAttributes()}
              for p in stage.Traverse()}
    report = apply_workcell_materials(stage)
    for path, attributes in before.items():
        for name, value in attributes.items():
            assert stage.GetPrimAtPath(path).GetAttribute(name).Get() == value
    assert str(UsdShade.MaterialBindingAPI(table).ComputeBoundMaterial()[0].GetPath()).endswith("/table_white")
    assert str(UsdShade.MaterialBindingAPI(camera).ComputeBoundMaterial()[0].GetPath()).endswith("/camera_silver")
    assert not UsdShade.MaterialBindingAPI(collision).GetDirectBinding().GetMaterial()
    assert str(guide.GetPath()) not in report["bindings"]
    assert report["bindings"]["/World/left/link2/visuals"] == "piper_gray"
