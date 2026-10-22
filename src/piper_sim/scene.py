"""Workcell geometry; no robot, controller or recording dependencies."""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from .config import ROOT


def spawn_workcell(cfg, *, spawn_object=True):
    # Local geometry avoids a network dependency on the standard Nucleus ground USD.
    ground = sim_utils.CuboidCfg(
        size=(6., 6., .05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.16, .18, .21)))
    ground.func("/World/Ground", ground, translation=(0, 0, -.055))
    light = sim_utils.DomeLightCfg(intensity=2000., color=(.9, .9, .9))
    light.func("/World/Light", light)
    geometry = cfg["scene"].get("geometry", "primitives")
    if geometry == "cad":
        asset = ROOT / cfg["scene"]["cad_usd"]
        if not asset.is_file():
            raise FileNotFoundError(f"{asset}; run scripts/convert_workcell_cad.py first")
        workcell = sim_utils.UsdFileCfg(usd_path=str(asset))
        workcell.func("/World/Workcell", workcell)
    elif geometry != "primitives":
        raise ValueError(f"Unknown workcell geometry: {geometry}")
    for name in ("table", "mast") if geometry == "primitives" else ():
        part = cfg["scene"][name]
        shape = sim_utils.CuboidCfg(
            size=tuple(part["size"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(.3, .35, .4) if name == "table" else (.5, .5, .5)))
        shape.func(f"/World/{name}", shape, translation=tuple(part["position"]))
    if not spawn_object:
        return None
    part = cfg["scene"]["object"]
    return RigidObject(RigidObjectCfg(
        prim_path="/World/Object",
        spawn=sim_utils.CuboidCfg(
            size=tuple(part["size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .15, .08))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(part["position"]))))
