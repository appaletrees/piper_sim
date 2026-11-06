#!/usr/bin/env python3
"""Check live property reset, RGB determinism and recording in either engine."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['mujoco', 'isaac'], default='mujoco')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-scene',default='configs/tasks/clean_table_rigid.yaml')
    selected, _ = parser.parse_known_args()
    if selected.backend == 'isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app if selected.backend == 'isaac' else None
    backend = None
    try:
        from piper_sim.config import ROOT, load
        from piper_sim.simulation import SimulationSession
        from PIL import Image
        cfg = load('configs/clean_table_workcell.yaml')
        cfg['task_scene'] = yaml.safe_load((ROOT/args.task_scene).read_text())
        cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
        if args.backend == 'mujoco':
            os.environ.setdefault('MUJOCO_GL', 'egl')
            from piper_sim.backends.mujoco import MujocoBackend
            backend = MujocoBackend(cfg, cameras=True)
            def props():
                m = backend.model
                return [m.body_mass.copy(), m.body_inertia.copy(), m.geom_friction.copy(), m.mat_rgba.copy()]
        else:
            from piper_sim.backends.isaac import IsaacBackend
            backend = IsaacBackend(cfg, args.device, cameras=True)
            def props():
                return [a.cpu().numpy().copy() for obj in backend.task_objects.values()
                        for a in (obj.root_physx_view.get_masses(), obj.root_physx_view.get_inertias(),
                                  obj.root_physx_view.get_material_properties())]
        session = SimulationSession(backend, record_dir=args.output)
        backend.reset(0, False)
        nominal = props()
        parameters = []
        for seed in (42, 43, 42):
            session.reset(seed, True)
            if seed == 43:
                assert any(not np.array_equal(a, b) for a, b in zip(first, props()))
            elif not parameters:
                first = props()
                assert any(not np.array_equal(a, b) for a, b in zip(nominal, first))
            else:
                for a, b in zip(first, props()): np.testing.assert_array_equal(a, b)
            parameters.append(copy.deepcopy(backend.task.domain_parameters))
            for _ in range(30): session.step()
            a, b = session.read(), session.read()
            for name in a.images:
                np.testing.assert_array_equal(a.images[name], b.images[name])
                Image.fromarray(a.images[name]).save(args.output/f'{seed}_{name}.png')
        assert parameters[0] == parameters[2]
        session.start_recording('rgb_smoke', 'domain-randomization recording smoke; no policy')
        for _ in range(5): session.step()
        recording = session.stop_recording(False)
        session.reset(0, False)
        for a, b in zip(nominal, props()): np.testing.assert_array_equal(a, b)
        report = {'passed': True, 'backend': args.backend, 'seeds': [42, 43, 42],
                  'domain_parameters': parameters[:2], 'recording': recording,
                  'scope': 'live properties, absolute reset, deterministic repeated reads, RGB/recording; no teacher validation'}
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'passed': True, 'output': str(args.output)}), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if backend: backend.close()
        if app:
            # Constructor failures still leave a context with Lab 2.3's STOP
            # callback. Do not let shutdown hide the original exception forever.
            from isaaclab.sim import SimulationContext
            context=SimulationContext.instance()
            if context:context._disable_app_control_on_stop_handle=True
            app.close(wait_for_replicator=False)


if __name__ == '__main__': main()
