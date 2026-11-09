#!/usr/bin/env python3
"""Isolated charger free-fall/contact diagnostic in the existing workcell.

The object is placed once at a declared initial height, then left to engine
physics. No teacher, grasp assistance, live pose resets or training recording.
These engine differences are not a calibration against real hardware.
"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['mujoco', 'isaac'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    selected, _ = parser.parse_known_args()
    if selected.backend == 'isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.backend == 'isaac' and not args.headless: parser.error('Use --headless')
    args.output.mkdir(parents=True, exist_ok=False)
    app = backend = None
    report = {'status': 'running', 'backend': args.backend, 'training_ready': False,
              'scope': __doc__, 'object': 'charger', 'samples': []}
    try:
        if args.backend == 'isaac': app = AppLauncher(args).app
        import numpy as np
        import yaml
        from piper_sim.config import ROOT, load
        cfg = load('configs/clean_table_workcell.yaml')
        cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
        cfg['task_scene'].pop('domain_randomization', None)
        if args.backend == 'isaac':
            source = ROOT/cfg['robot']['urdf']
            isolated = args.output/'asset_cache'/source.name
            isolated.parent.mkdir(parents=True)
            shutil.copy2(source, isolated)
            if (source.parent/'usd').exists(): shutil.copytree(source.parent/'usd', isolated.parent/'usd')
            cfg['robot']['urdf'] = str(isolated.resolve())
            from piper_sim.backends.isaac import IsaacBackend
            backend = IsaacBackend(cfg, args.device, cameras=False)
        else:
            import mujoco
            from piper_sim.backends.mujoco import MujocoBackend
            backend = MujocoBackend(cfg, cameras=False)
        backend.reset(0, False)
        initial = [.55, 0., .1455]
        if args.backend == 'isaac':
            obj = backend.task_objects['charger']
            state = obj.data.default_root_state.clone()
            state[0, :3] = state.new_tensor(initial)
            state[0, 3:7] = state.new_tensor([1., 0., 0., 0.])
            state[0, 7:] = 0
            obj.write_root_state_to_sim(state)
            obj.reset()
            backend.sim.forward()
            mass = float(obj.root_physx_view.get_masses().reshape(-1)[0])
            native_dt = backend.dt
        else:
            joint = backend.model.joint('task/charger/free')
            address, dof = joint.qposadr[0], joint.dofadr[0]
            backend.data.qpos[address:address+7] = [*initial, 1., 0., 0., 0.]
            backend.data.qvel[dof:dof+6] = 0
            mujoco.mj_forward(backend.model, backend.data)
            mass = float(backend.model.body_mass[backend.model.body('task/charger').id])
            native_dt = float(backend.model.opt.timestep)
        report.update(initial_position_m=initial, object_mass_kg=mass, physics_dt_s=native_dt,
                      control_dt_s=backend.dt*backend.decimation)
        for step in range(91):
            obs = backend.read()
            pose = obs.extras['objects']['charger']
            if args.backend == 'isaac':
                velocity = obj.data.root_lin_vel_w[0].cpu().tolist()
            else:
                velocity = backend.data.qvel[dof:dof+3].tolist()
            row = {'time': backend.sim_time, **pose, 'linear_velocity_m_s': velocity}
            if not np.isfinite(row['position']+row['quaternion_wxyz']+velocity).all():
                raise ValueError('Nonfinite state')
            report['samples'].append(row)
            if step < 90: backend.step()
        t = np.array([s['time'] for s in report['samples']])
        z = np.array([s['position'][2] for s in report['samples']])
        v = np.array([s['linear_velocity_m_s'] for s in report['samples']])
        np.testing.assert_allclose(np.diff(t), report['control_dt_s'], atol=1e-7, rtol=0)
        # Four pre-contact points; coefficients are z = a*t² + b*t + c.
        fit = np.polyfit(t[:4], z[:4], 2)
        report.update(status='diagnostic_completed', free_fall_fitted_acceleration_m_s2=float(2*fit[0]),
            final_root_z_m=float(z[-1]), minimum_root_z_m=float(z.min()),
            final_10_samples_max_linear_speed_m_s=float(np.linalg.norm(v[-10:], axis=1).max()),
            collision_note='Object root z is not penetration depth. Different collision shapes, solvers, '
                           'contact offsets and substeps can produce different resting poses.',
            control_note='Default backend position drives hold robot home; no expert bias-feedforward or contact tuning.')
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if backend is not None: backend.close()
        if app is not None: app.close()


if __name__ == '__main__':
    main()
