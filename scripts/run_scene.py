#!/usr/bin/env python3
"""One-way leader/follower modes. No CAN connection in sim, replay or mock mode."""
import argparse
from pathlib import Path
import time
import numpy as np
from piper_sim.config import load
from piper_sim.contracts import Command, LOWER, UPPER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["isaac", "mujoco"], default="isaac")
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=["sim", "real-to-sim", "sim-to-real", "replay"],
                        default="sim")
    parser.add_argument("--steps", type=int, default=0, help="0 runs until window closes")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--task", default="dual Piper workcell demonstration")
    parser.add_argument("--task-scene", type=Path, help="Independent task YAML, e.g. configs/tasks/connector.yaml")
    parser.add_argument("--seed", type=int, help="Reproducible task object placement")
    parser.add_argument("--no-randomization", action="store_true", help="Use the task's nominal layout")
    parser.add_argument("--trajectory", type=Path, help="HDF5 episode to replay as simulation targets")
    parser.add_argument("--demo", action="store_true", help="Small joint motion in pure simulation")
    parser.add_argument("--allow-actuation", action="store_true")
    parser.add_argument("--command-port", type=int, help="Loopback HTTP /target and /state port")
    parser.add_argument("--overview-image", type=Path, help="Save the final overview viewport as PNG")
    # Probe only the backend; let the complete parser handle --help afterwards.
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--backend", choices=["isaac", "mujoco"], default="isaac")
    selected, _ = selection.parse_known_args()
    if selected.backend == "isaac":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--enable_cameras", action="store_true")
        parser.add_argument("--interactive", action="store_true",
                            help="MuJoCo native Control panel drives the simulation")
    args = parser.parse_args()
    interactive = getattr(args, "interactive", False)
    if interactive and (args.headless or args.mode != "sim" or args.demo or args.command_port or args.trajectory):
        parser.error("--interactive requires a MuJoCo GUI in sim mode without demo, trajectory or command-port")
    if args.mode == "sim-to-real" and not args.allow_actuation:
        parser.error("sim-to-real requires --allow-actuation and physically enabled, aligned arms")
    if args.mode == "sim-to-real" and not args.command_port:
        parser.error("sim-to-real requires a live --command-port stream")
    if args.command_port and args.mode not in ("sim", "sim-to-real"):
        parser.error("command ingress is supported in sim or sim-to-real")
    if args.demo and args.mode != "sim":
        parser.error("--demo is available only in pure simulation")
    if args.mode == "replay" and not args.trajectory:
        parser.error("--trajectory is required for replay")
    cfg = load(args.config)
    if args.task_scene:
        import yaml
        cfg["task_scene"] = yaml.safe_load(args.task_scene.read_text())
        if args.seed is not None:
            cfg["task_scene"]["seed"] = args.seed
        if args.no_randomization:
            cfg["task_scene"]["randomization"]["enabled"] = False
    elif args.seed is not None or args.no_randomization:
        parser.error("--seed / --no-randomization require --task-scene")
    app = AppLauncher(args).app if args.backend == "isaac" else None
    hardware = writer = sim = server = None
    complete = False
    try:
        from piper_sim.recording import EpisodeWriter
        if args.backend == "isaac":
            from piper_sim.backends.isaac import IsaacBackend
            sim = IsaacBackend(cfg, args.device, args.enable_cameras)
            is_running = app.is_running
        else:
            from piper_sim.backends.mujoco import MujocoBackend
            sim = MujocoBackend(cfg, cameras=args.enable_cameras, viewer=not args.headless)
            is_running = sim.is_running
        if args.command_port:
            from piper_sim.control import CommandServer
            server = CommandServer(args.command_port)
            server.publish(sim.read())
        from piper_sim.safety import CommandGuard
        input_guard = CommandGuard()
        last_input_sequence = -1
        initial = np.array(cfg["robot"]["home"]*2)
        trajectory = None
        if args.trajectory:
            import h5py
            with h5py.File(args.trajectory, "r") as f:
                trajectory = f["action/position"][:]
            if args.steps == 0:
                args.steps = len(trajectory)
        if args.mode in ("real-to-sim", "sim-to-real"):
            from piper_sim.backends.piper import PiperBackend
            hardware = PiperBackend(cfg["hardware"], allow_actuation=args.allow_actuation)
            deadline = time.monotonic() + 5
            while True:
                try:
                    initial = hardware.read().position
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            # Align virtual targets to physical state before allowing output.
            for index in range(90):
                sim.apply(Command(initial, time.monotonic(), index))
                sim.step()
        if args.record:
            from piper_sim.transforms import load_camera_rig
            writer = EpisodeWriter(args.record, {
                "task": args.task, "mode": args.mode, "config": cfg,
                "task_scene": sim.task.metadata() if sim.task else None,
                "demo": args.demo,
                "camera_rig": sim.rig if args.backend == "mujoco" else load_camera_rig(cfg),
                "fps": 1/(sim.dt*sim.decimation),
                "source": "isaaclab" if args.backend == "isaac" else "mujoco",
                "backend": args.backend,
                "image_timestamp_clock": "simulation",
                "state_timestamp_clock": "host_monotonic",
                "state_source": "hardware" if args.mode == "sim-to-real" else "simulation",
                "image_source": "simulation",
                "action_semantics": "absolute joint position targets and gripper width"})
        index = 0
        while is_running() and (args.steps == 0 or index < args.steps):
            start = time.monotonic()
            before = sim.read()
            if server:
                server.publish(before)
            target = initial.copy()
            if args.mode == "real-to-sim":
                target = hardware.read().position
            elif args.mode == "sim-to-real":
                target = before.position
            elif args.mode == "replay":
                target = trajectory[index % len(trajectory)]
            elif interactive:
                target = sim.control_targets()
            elif args.demo:
                amplitudes = np.array([.12, .04, -.05, .08, .06, .08, .01])
                frequencies = np.array([1., .7, .9, .8, .6, 1.1, .8])
                motion = amplitudes*np.sin(index*sim.dt*sim.decimation*frequencies)
                target[:7] += motion
                target[7:] += motion*np.array([-1., 1., 1., 1., 1., 1., 1.])
                target = np.clip(target, LOWER, UPPER)
            if server:
                incoming = server.sample()
                if incoming is None and args.mode == "sim-to-real":
                    # Wait in simulation; no actuator command before the first aligned input.
                    sim.step()
                    time.sleep(.01)
                    continue
                if incoming and incoming.sequence != last_input_sequence:
                    accepted = input_guard.accept(incoming, before.position, time.monotonic())
                    initial = accepted.position.copy()
                    last_input_sequence = incoming.sequence
                target = initial.copy()
            action = Command(target, time.monotonic(), index)
            if args.mode == "sim-to-real":
                sim.apply(action)
                physical_before = hardware.read()
                physical_before.images = before.images
                physical_before.image_timestamps = before.image_timestamps
                physical_before.sim_time = before.sim_time
                physical_before.extras.update(before.extras)
                if not hardware.armed:
                    hardware.arm()
                # Measured simulation motion is the hardware leader.
                action = hardware.apply(Command(before.position, time.monotonic(), index))
                before = physical_before
            else:
                sim.apply(action)
            sim.step()
            after = sim.read()
            if args.mode == "sim-to-real":
                physical_after = hardware.read()
                physical_after.images = after.images
                physical_after.image_timestamps = after.image_timestamps
                physical_after.sim_time = after.sim_time
                physical_after.extras.update(after.extras)
                after = physical_after
            if writer:
                writer.append(before, action, after)
            index += 1
            if hardware or server or (args.backend == "mujoco" and not args.headless):
                time.sleep(max(0., sim.dt*sim.decimation - (time.monotonic()-start)))
        if args.overview_image:
            sim.save_overview(args.overview_image)
        complete = True
        if args.record:
            print(f"Recorded {index} transitions to {args.record}")
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if hardware:
            hardware.close()
        if writer:
            writer.close(complete=complete)
        if server:
            server.close()
        if sim:
            sim.close()
        elif args.backend == "isaac":
            # Construction can fail after SimulationContext registers its STOP handler.
            from isaaclab.sim import SimulationContext
            context = SimulationContext.instance()
            if context is not None:
                context._disable_app_control_on_stop_handle = True
        # Images are read synchronously and our HDF5 writer is already closed.
        # No asynchronous Replicator writer needs draining here.
        if app:
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
