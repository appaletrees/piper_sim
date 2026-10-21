"""Camera rig factory, replaceable without modifying the control loop."""
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import convert_camera_frame_orientation_convention
import torch
from .config import ROOT
from .transforms import load_camera_rig


class WorkcellCamera(Camera):
    """Read the same live Fabric transforms that the renderer uses on Sim 5.1."""

    def _initialize_impl(self):
        super()._initialize_impl()
        # A mounted sensor retains its local pose when the articulation resets.
        self._view._non_root_link = True

    def _update_poses(self, env_ids):
        # Sim 5.1's Fabric view reinterprets indices as uint32; Lab uses int64.
        view_ids = env_ids.to(dtype=torch.int32) if self._use_fabric else env_ids
        poses, orientations = self._view.get_world_poses(
            view_ids, usd=not self._use_fabric)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
            orientations, origin="opengl", target="world")

    def __init__(self, cfg, *, use_fabric):
        self._use_fabric = use_fabric
        super().__init__(cfg)
        if use_fabric:
            import usdrt
            from isaacsim.core.utils.stage import get_current_stage
            # XFormPrim's first Fabric read calls SetWorldXformFromUsd(). After
            # physics starts, USD parent poses are stale; that first read would
            # move the camera inside its housing. Seed this cache before reset,
            # while USD and physics share the same authored reference pose.
            stage = get_current_stage(fabric=True)
            usdrt.Rt.Xformable(stage.GetPrimAtPath(cfg.prim_path)).SetWorldXformFromUsd()


def spawn_camera_mounts(cfg):
    """Geometry follows link6 even when image rendering is disabled."""
    rig = load_camera_rig(cfg)
    for mount in rig["mounts"].values():
        asset = ROOT / mount["usd"]
        if not asset.is_file():
            raise FileNotFoundError(asset)
        spawn = sim_utils.UsdFileCfg(usd_path=str(asset))
        spawn.func(mount["parent_prim"]+"/CameraRig", spawn)


def spawn_cameras(cfg):
    cam = cfg["cameras"]
    rig = load_camera_rig(cfg)
    common = dict(
        update_period=cfg["runtime"]["physics_dt"]*cfg["runtime"]["decimation"],
        update_latest_camera_pose=True,
        height=cam["height"], width=cam["width"], data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=rig["intrinsics"]["focal_length_mm"],
            horizontal_aperture=rig["intrinsics"]["horizontal_aperture_mm"],
            clipping_range=(.001, 10.)))
    use_fabric = cfg["runtime"].get("use_fabric", True)
    cameras = {}
    for name, mount in rig["cameras"].items():
        cameras[name] = WorkcellCamera(CameraCfg(
            prim_path=mount["prim_path"],
            offset=CameraCfg.OffsetCfg(
                pos=tuple(mount["position"]),
                rot=tuple(mount["rotation_wxyz_ros"]), convention="ros"), **common),
                use_fabric=use_fabric)
    return cameras
