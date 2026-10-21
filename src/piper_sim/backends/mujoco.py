"""Independent MuJoCo runtime implementing the shared SI observation contract."""
from pathlib import Path
import time

import mujoco
import numpy as np

from ..contracts import ARMS, JOINT_NAMES, LOWER, UPPER, Observation, vector
from ..mujoco_asset import resolve_model


class MujocoBackend:
    def __init__(self, cfg, cameras=False, viewer=False):
        self.cfg = cfg
        self.dt = cfg["runtime"]["physics_dt"]
        self.decimation = cfg["runtime"]["decimation"]
        self.viewer = None
        self.renderers = {}
        self._images = None
        self._closed = False
        path, self.manifest = resolve_model(cfg)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.substeps = self.manifest["substeps"]
        self.rig = self.manifest["camera_rig"]
        self.enable_cameras = cameras
        self.indices = {}
        for side in ARMS:
            names = [f"{side}/{name}" for name in (*JOINT_NAMES, "joint7", "joint8")]
            self.indices[side] = {
                "qpos": [self.model.joint(n).qposadr[0] for n in names],
                "dof": [self.model.joint(n).dofadr[0] for n in names],
                "ctrl": [self.model.actuator(n).id for n in names]}
        self.reset()
        try:
            if viewer:
                from mujoco import viewer as mj_viewer
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
                self.viewer.cam.lookat[:] = [.25, 0, .4]
                self.viewer.cam.distance = 2.3
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -28
        except BaseException:
            self.close()
            raise

    @property
    def sim_time(self):
        return float(self.data.time)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        home = np.array(self.cfg["robot"]["home"]*2)
        self.set_observation_state(home)

    def set_observation_state(self, position):
        """Exact state placement for deterministic sim audits, within model limits."""
        q = vector(position)
        if np.any(q < LOWER) or np.any(q > UPPER):
            raise ValueError("State exceeds the shared robot model limits")
        for i, side in enumerate(ARMS):
            a = q[7*i:7*i+7]
            values = [*a[:6], a[6]/2, a[6]/2]
            ids = self.indices[side]
            self.data.qpos[ids["qpos"]] = values
            self.data.qvel[ids["dof"]] = 0
            self.data.ctrl[ids["ctrl"]] = values
        mujoco.mj_forward(self.model, self.data)
        self._images = None

    def apply(self, command):
        q = vector(command.position)
        if np.any(q < LOWER) or np.any(q > UPPER):
            raise ValueError("Simulation target exceeds joint bounds")
        for i, side in enumerate(ARMS):
            a = q[7*i:7*i+7]
            self.data.ctrl[self.indices[side]["ctrl"]] = [*a[:6], a[6]/2, a[6]/2]
        return command

    def step(self):
        mujoco.mj_step(self.model, self.data, nstep=self.decimation*self.substeps)
        # mj_step leaves position-dependent buffers at the last integration
        # evaluation. Refresh so RGB, poses and qpos describe the same state.
        mujoco.mj_forward(self.model, self.data)
        self._images = None
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite MuJoCo physics state")
        if self.viewer:
            self.viewer.sync()

    def is_running(self):
        return not self._closed and (self.viewer is None or self.viewer.is_running())

    def _render(self, name, width, height):
        key = (width, height)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self.renderers[key]
        renderer.update_scene(self.data, camera=name)
        return renderer.render().copy()

    def read(self):
        q = []
        for side in ARMS:
            p = self.data.qpos[self.indices[side]["qpos"]]
            q.extend([*p[:6], p[6]+p[7]])
        if self.enable_cameras and self._images is None:
            self._images = {name: self._render(name, *mount["rgb"]["resolution"])
                            for name, mount in self.rig["cameras"].items()}
        poses = {}
        if self.enable_cameras:
            for name, mount in self.rig["cameras"].items():
                cid = self.model.camera(name).id
                rotation = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
                poses[name] = {"position_world": self.data.cam_xpos[cid].tolist(),
                               "quaternion_wxyz_ros": quat.tolist(),
                               "intrinsics": mount["rgb"]["intrinsic_matrix"]}
        images = {k: v.copy() for k, v in (self._images or {}).items()}
        obj = self.data.body("object")
        return Observation(q, time.monotonic(), self.sim_time, images,
            {name: self.sim_time for name in images},
            {"image_timestamp_clock": "simulation", "cameras": poses, "physics_time": self.sim_time,
             "engine": "mujoco", "engine_version": mujoco.__version__,
             "object": {"position_world": obj.xpos.tolist(), "quaternion_wxyz": obj.xquat.tolist()}})

    def save_overview(self, path):
        from PIL import Image
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.25, 0, .4]
        camera.distance = 2.3
        camera.azimuth, camera.elevation = 135, -28
        Image.fromarray(self._render(camera, 1280, 720)).save(path)

    def close(self):
        self._closed = True
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
