"""Independent MuJoCo runtime implementing the shared SI observation contract."""
from pathlib import Path
import time

import mujoco
import numpy as np

from ..contracts import ARMS, JOINT_NAMES, LOWER, UPPER, Observation, vector
from ..mujoco_asset import resolve_model
from ..sensor_processing import SensorImageProcessor

_PHYSICS_WARNINGS = np.array([int(x) for x in (mujoco.mjtWarning.mjWARN_INERTIA,
    mujoco.mjtWarning.mjWARN_CONTACTFULL, mujoco.mjtWarning.mjWARN_CNSTRFULL,
    mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)])


class MujocoBackend:
    def __init__(self, cfg, cameras=False, viewer=False, model_xml_transform=None):
        self.cfg = cfg
        self.dt = cfg["runtime"]["physics_dt"]
        self.decimation = cfg["runtime"]["decimation"]
        self.sensor_processor = SensorImageProcessor(cfg['runtime'].get('sensor_postprocess_workers', 1))
        self.viewer = None
        self.renderers = {}
        self._images = None
        self._closed = False
        self.task = None
        if cfg.get("task_scene"):
            from ..tasks import create_task, engine_adapter
            self.task = create_task(cfg["task_scene"], cfg)
            self.task_adapter = engine_adapter(self.task, "mujoco")
        path, self.manifest = resolve_model(cfg)
        if self.task or model_xml_transform is not None:
            xml = self.task_adapter.compose_model(path, self.task) if self.task else Path(path).read_text()
            if model_xml_transform is not None:
                xml = model_xml_transform(xml)
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        # Task-local numerical resolution does not require rebuilding CAD assets.
        # The public physics tick and control/recording rate stay unchanged.
        self.substeps = (self.task.config.get("contact", {}).get("mujoco_substeps", self.manifest["substeps"])
                         if self.task else self.manifest["substeps"])
        if type(self.substeps) is not int or self.substeps < 1:
            raise ValueError("mujoco_substeps must be a positive integer")
        self.model.opt.timestep = self.dt/self.substeps
        self.finger_contact = None
        if cfg.get('mujoco', {}).get('finger_contact') is not None:
            from ..mujoco_contact import MujocoFingerContact
            self.finger_contact = MujocoFingerContact(self.model, cfg['mujoco']['finger_contact'])
        self.rig = self.manifest["camera_rig"]
        self.enable_cameras = cameras
        self.randomizer = None
        if self.task and getattr(self.task, 'domain_profile', None):
            from ..randomization.mujoco import MujocoRandomizer
            self.randomizer = MujocoRandomizer(self.model, self.task)
        self.indices = {}
        for side in ARMS:
            names = [f"{side}/{name}" for name in (*JOINT_NAMES, "joint7", "joint8")]
            self.indices[side] = {
                "qpos": [self.model.joint(n).qposadr[0] for n in names],
                "dof": [self.model.joint(n).dofadr[0] for n in names],
                "ctrl": [self.model.actuator(f"{side}/{n}").id for n in (*JOINT_NAMES, "gripper")]}
        self.reference_controller = None
        controller_config = cfg.get('mujoco', {}).get('controller')
        if controller_config is not None:
            from ..mujoco_controller import MujocoReferenceController
            projection = np.zeros((14, self.model.nv))
            actuator_ids = []
            for i, side in enumerate(ARMS):
                ids = self.indices[side]
                for j, dof in enumerate(ids['dof'][:6]): projection[7*i+j, dof] = 1.
                projection[7*i+6, ids['dof'][6:]] = 1.
                actuator_ids.extend(ids['ctrl'])
            self.reference_controller = MujocoReferenceController(self.model, actuator_ids, projection,
                self.decimation*self.substeps, controller_config)
        self.tcp_observer = None
        observe_tcp = cfg.get('mujoco', {}).get('tcp_motion_evidence', False)
        if type(observe_tcp) is not bool:
            raise ValueError('tcp_motion_evidence must be boolean')
        if observe_tcp:
            from ..mujoco_tcp import MujocoTcpObserver
            self.tcp_observer = MujocoTcpObserver(self.model, self.decimation*self.substeps)
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

    def reset(self, seed=None, randomize=None):
        mujoco.mj_resetData(self.model, self.data)
        if self.task:
            self.task.reset(seed, randomize)
            if self.randomizer:
                self.randomizer.apply(self.task.domain_parameters)
            self.task_adapter.reset_task(self.model, self.data, self.task)
        if self.finger_contact:
            layout = self.task.layout if self.task else {}
            self.finger_contact.apply(layout.get('seed', 0 if seed is None else seed),
                                      layout.get('randomized', bool(randomize)))
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
            self.data.ctrl[ids["ctrl"]] = a
        if self.reference_controller:
            self.reference_controller.reset(q)
        if self.tcp_observer:
            self.tcp_observer.reset()
        mujoco.mj_forward(self.model, self.data)
        self._images = None

    def apply(self, command):
        q = vector(command.position)
        if np.any(q < LOWER) or np.any(q > UPPER):
            raise ValueError("Simulation target exceeds joint bounds")
        for i, side in enumerate(ARMS):
            a = q[7*i:7*i+7]
            self.data.ctrl[self.indices[side]["ctrl"]] = a
        return command

    def control_targets(self):
        """Native Control panel targets, including one full-width slider per arm."""
        return np.concatenate([self.data.ctrl[self.indices[side]["ctrl"]] for side in ARMS]).copy()

    def step(self):
        if self.finger_contact:
            self.finger_contact.assert_unchanged()
        expected_time = float(self.data.time)+self.dt*self.decimation
        warnings_before = self.data.warning.number[_PHYSICS_WARNINGS].copy()
        if self.tcp_observer:
            self.tcp_observer.begin(self.data)
        if self.reference_controller:
            self.reference_controller.advance(self.data,
                observer=self.tcp_observer.sample if self.tcp_observer else None)
        elif self.tcp_observer:
            for _ in range(self.decimation*self.substeps):
                mujoco.mj_step(self.model, self.data)
                self.tcp_observer.sample(self.data)
        else:
            mujoco.mj_step(self.model, self.data, nstep=self.decimation*self.substeps)
        if self.tcp_observer:
            self.tcp_observer.finish()
        # MuJoCo can auto-reset on huge acceleration, leaving finite qpos/qvel.
        # Reject that discontinuity before a recorder sees a restarted episode.
        if not np.isclose(self.data.time, expected_time, atol=1e-7, rtol=0):
            raise RuntimeError("MuJoCo physics reset after instability; episode time is discontinuous")
        # mj_step leaves position-dependent buffers at the last integration
        # evaluation. Refresh so RGB, poses and qpos describe the same state.
        mujoco.mj_forward(self.model, self.data)
        if not np.array_equal(self.data.warning.number[_PHYSICS_WARNINGS], warnings_before):
            raise RuntimeError('MuJoCo physics reset/warning after instability or capacity failure; episode rejected')
        self._images = None
        if self.task:
            self.task.metrics(self.task_adapter.task_poses(self.data, self.task), self.dt*self.decimation)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite MuJoCo physics state")
        if self.viewer:
            self.viewer.sync()

    def is_running(self):
        return not self._closed and (self.viewer is None or self.viewer.is_running())

    def physics_metadata(self):
        result = {'finger_contact': self.finger_contact.metadata()} if self.finger_contact else {}
        if self.reference_controller:
            result['controller'] = self.reference_controller.metadata()
        if self.tcp_observer:
            result['tcp_motion'] = self.tcp_observer.metadata()
        return result

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
            images = {name: self._render(name, *mount["rgb"]["resolution"])
                      for name, mount in self.rig["cameras"].items()}
            if self.randomizer:
                images = self.sensor_processor(images, self.task.domain_parameters,
                                               round(self.sim_time/(self.dt*self.decimation)))
            # Publish only a complete processed snapshot. A failed camera must
            # never leave raw, unprocessed pixels cached for the next read.
            self._images = images
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
        if self.task:
            objects = self.task_adapter.task_poses(self.data, self.task)
            extras = {"objects": objects, "task": {"name": self.task.config["name"],
                      "seed": self.task.layout["seed"], "metrics": self.task.metrics(objects)}}
        else:
            obj = self.data.body("object")
            extras = {"object": {"position_world": obj.xpos.tolist(), "quaternion_wxyz": obj.xquat.tolist()}}
        return Observation(q, time.monotonic(), self.sim_time, images,
            {name: self.sim_time for name in images},
            {"image_timestamp_clock": "simulation", "cameras": poses, "physics_time": self.sim_time,
             "engine": "mujoco", "engine_version": mujoco.__version__,
             **({'controller_interval': self.reference_controller.telemetry()} if self.reference_controller else {}),
             **({'tcp_motion_interval': self.tcp_observer.telemetry()} if self.tcp_observer else {}),
             **extras})

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
        self.sensor_processor.close()
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
