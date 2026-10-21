"""Isaac Lab 2.3 backend. Import only AFTER AppLauncher has started Kit."""
import time
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from ..robot_asset import piper_config
from ..scene import spawn_workcell
from ..sensors import spawn_camera_mounts, spawn_cameras
from ..contracts import ARMS, JOINT_NAMES, LOWER, UPPER, Observation, vector


class IsaacBackend:
    def __init__(self, cfg, device="cuda:0", cameras=False):
        self.cfg = cfg
        runtime = cfg["runtime"]
        self.dt = runtime["physics_dt"]
        self.decimation = runtime["decimation"]
        self.sim_time = 0.
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device, render_interval=self.decimation,
            use_fabric=runtime.get("use_fabric", True)))
        if not runtime.get("use_fabric", True):
            # Sim 5.1's PhysicsContext only applies use_fabric when True.
            # Explicitly disable its inherited default to enable USD pose updates.
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.enable_fabric(False)
        self.sim.set_camera_view([1.8, 1.7, 1.5], [.25, 0., .35])
        self.object = spawn_workcell(cfg)
        self.arms = {side: Articulation(piper_config(cfg, side)) for side in ARMS}
        spawn_camera_mounts(cfg)
        self.cameras = spawn_cameras(cfg) if cameras else {}
        self.sim.reset()
        self.indices = {}
        for side, arm in self.arms.items():
            ids, names = arm.find_joints([*JOINT_NAMES, "joint7", "joint8"], preserve_order=True)
            if names != [*JOINT_NAMES, "joint7", "joint8"]:
                raise RuntimeError(f"Imported joint mapping mismatch: {names}")
            self.indices[side] = ids
        self.reset()
        # Initialize buffers and renderer before recording or connecting hardware.
        for _ in range(8):
            self.step()

    def reset(self):
        """Reset physical state and drive targets together before starting an episode."""
        for arm in self.arms.values():
            positions = arm.data.default_joint_pos.clone()
            velocities = arm.data.default_joint_vel.clone()
            arm.write_joint_state_to_sim(positions, velocities)
            arm.set_joint_position_target(positions)
            arm.reset()
        self.object.write_root_state_to_sim(self.object.data.default_root_state.clone())
        self.object.reset()
        for camera in self.cameras.values():
            camera.reset()
        self.sim_time = 0.

    def apply(self, command):
        q = vector(command.position)
        if (q < LOWER).any() or (q > UPPER).any():
            raise ValueError("Simulation target exceeds joint bounds")
        for i, (side, arm) in enumerate(self.arms.items()):
            a = q[7*i:7*i+7]
            target = torch.tensor([[*a[:6], a[6]/2, a[6]/2]],
                                  device=self.sim.device, dtype=torch.float32)
            arm.set_joint_position_target(target, joint_ids=self.indices[side])
        return command

    def step(self):
        for _ in range(self.decimation):
            for arm in self.arms.values():
                arm.write_data_to_sim()
            # One explicit physics tick per loop. Kit's render=True path advances
            # through app.update(), whose render interval can add multiple ticks.
            self.sim.step(render=False)
            for arm in self.arms.values():
                arm.update(self.dt)
            self.object.update(self.dt)
            self.sim_time += self.dt
        # Flush live physics transforms to Fabric and render once per control tick.
        self.sim.render()
        for camera in self.cameras.values():
            camera.update(self.dt*self.decimation)

    def read(self):
        q = []
        for side, arm in self.arms.items():
            p = arm.data.joint_pos[0, self.indices[side]].detach().cpu().numpy()
            q.extend([*p[:6], p[6]+p[7]])
        images = {name: camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().copy()
                  for name, camera in self.cameras.items()}
        camera_poses = {
            name: {"position_world": camera.data.pos_w[0].cpu().tolist(),
                   "quaternion_wxyz_ros": camera.data.quat_w_ros[0].cpu().tolist(),
                   "intrinsics": camera.data.intrinsic_matrices[0].cpu().tolist()}
            for name, camera in self.cameras.items()}
        return Observation(q, time.monotonic(), self.sim_time, images,
                           {name: self.sim_time for name in images},
                           {"image_timestamp_clock": "simulation", "cameras": camera_poses,
                            "physics_time": self.sim.current_time,
                            "object": {"position_world": self.object.data.root_pos_w[0].cpu().tolist(),
                                       "quaternion_wxyz": self.object.data.root_quat_w[0].cpu().tolist()}})

    def close(self):
        # Lab 2.3.0's STOP callback renders indefinitely while the timeline is stopped.
        # Suppress that callback during our owned, explicit application shutdown.
        self.sim._disable_app_control_on_stop_handle = True

    def save_overview(self, path):
        """Capture the overview without changing any recorded sensor viewpoint."""
        import asyncio
        from pathlib import Path
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("An active viewport is required for an overview capture")
        previous = viewport.updates_enabled
        viewport.updates_enabled = True
        try:
            self.sim.render()
            capture = capture_viewport_to_file(viewport, str(path))
            future = asyncio.ensure_future(capture.wait_for_result())
            deadline = time.monotonic()+30
            while not future.done():
                if time.monotonic() > deadline:
                    future.cancel()
                    raise TimeoutError("Overview capture did not complete in 30 seconds")
                self.sim.render()
            future.result()
            if not path.is_file():
                raise RuntimeError(f"Viewport capture did not create {path}")
        finally:
            viewport.updates_enabled = previous
