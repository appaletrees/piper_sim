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

    def _update_intrinsic_matrices(self, env_ids):
        # Output images are sampled at these rays, independent of the engine's
        # square-pixel raw render product. Record the output calibration.
        k = torch.tensor(self._rgb["intrinsic_matrix"], device=self._device)
        self._data.intrinsic_matrices[env_ids] = k

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        # Sim 5.1/Lab 2.3 ignores independent fy on this render path. A covering
        # pinhole followed by a calibrated ray resample avoids clipped borders
        # and keeps the public horizontal AND vertical FOV in the final RGB.
        pixels = self._data.output["rgb"][env_ids]
        height, width = pixels.shape[1:3]
        if self._projection_grid is None:
            k = self._rgb["intrinsic_matrix"]
            focal = self._rgb["render_focal_px"]
            # RealSense/OpenCV K uses integer pixel centres: [0,0] is the
            # first pixel centre. The raw renderer uses edge coordinates;
            # grid_sample's normalization below performs that conversion.
            y, x = torch.meshgrid(torch.arange(height, device=self._device),
                                  torch.arange(width, device=self._device), indexing="ij")
            u = (x-k[0][2])*focal/k[0][0] + width/2
            v = (y-k[1][2])*focal/k[1][1] + height/2
            self._projection_grid = torch.stack((2*u/width-1, 2*v/height-1), dim=-1)[None]
        sampled = torch.nn.functional.grid_sample(
            pixels.permute(0, 3, 1, 2).float(),
            self._projection_grid.expand(len(env_ids), -1, -1, -1),
            mode="bilinear", padding_mode="border", align_corners=False)
        self._data.output["rgb"][env_ids] = sampled.permute(0, 2, 3, 1).round().clamp(0, 255).to(pixels.dtype)

    def __init__(self, cfg, *, use_fabric, rgb):
        if cfg.data_types != ["rgb"]:
            raise ValueError("RGB ray resampling supports RGB only; depth needs a separate adapter")
        self._use_fabric = use_fabric
        self._rgb = rgb
        self._projection_grid = None
        super().__init__(cfg)
        from pxr import Sdf
        from isaacsim.core.utils.stage import get_current_stage
        prim = get_current_stage().GetPrimAtPath(cfg.prim_path)
        # Use Kit 107.3.3's native OpenCV projection attributes for the covering
        # render. Both focal axes match here; the output ray grid supplies K.
        prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("pinholeOpenCV")
        values = {"fthetaWidth": float(cfg.width), "fthetaHeight": float(cfg.height),
                  "fthetaCx": cfg.width/2., "fthetaCy": cfg.height/2.,
                  "openCVFx": rgb["render_focal_px"], "openCVFy": rgb["render_focal_px"],
                  "fthetaMaxFov": 180.}
        values.update(dict.fromkeys(("fthetaPolyA", "fthetaPolyB", "fthetaPolyC", "fthetaPolyD",
                                     "fthetaPolyE", "fthetaPolyF", "p0", "p1", "s0", "s1", "s2", "s3"), 0.))
        for key, value in values.items():
            prim.CreateAttribute(key, Sdf.ValueTypeNames.Float).Set(value)
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
    rig = load_camera_rig(cfg)
    period = cfg["runtime"]["physics_dt"]*cfg["runtime"]["decimation"]
    common = dict(
        update_period=period,
        update_latest_camera_pose=True,
        data_types=["rgb"])
    use_fabric = cfg["runtime"].get("use_fabric", True)
    cameras = {}
    for name, mount in rig["cameras"].items():
        rgb = mount["rgb"]
        if not any(abs(1/period-fps) < 1e-5 for fps in rgb["native_fps"]):
            raise ValueError(f"{name}: capture rate is not a supported native RGB frame rate")
        width, height = rgb["resolution"]
        horizontal, vertical = rgb["render_aperture_mm"]
        cameras[name] = WorkcellCamera(CameraCfg(
            prim_path=mount["prim_path"],
            width=width, height=height,
            spawn=sim_utils.PinholeCameraCfg(
                # USD lens lengths are tenths of a stage unit (cm on metre stage).
                focal_length=rgb["focal_length_mm"]*.1,
                horizontal_aperture=horizontal*.1, vertical_aperture=vertical*.1,
                # Nominal optical origin lies behind the CAD front glass. Clip
                # that opaque CAD surface, without moving camera or housing.
                clipping_range=(.01, 10.), f_stop=0.),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(mount["position"]),
                rot=tuple(mount["rotation_wxyz_ros"]), convention="ros"), **common),
                use_fabric=use_fabric, rgb=rgb)
    return cameras
