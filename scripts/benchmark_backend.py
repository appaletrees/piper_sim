#!/usr/bin/env python3
"""Headless native-resolution engine/recording benchmark; no teacher acceptance.

Small joint motions exercise moving wrist cameras. No physical sorting task is
claimed. Timings include readbacks and all recorded transitions, not GUI rate.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['mujoco', 'isaac'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--storage', choices=['shared', 'duplicated'], default='shared')
    parser.add_argument('--compression', choices=['gzip', 'lzf'], default='gzip')
    parser.add_argument('--uncached-reads', action='store_true',
                        help='Isaac diagnostic: recompute CPU snapshots on every read')
    parser.add_argument('--compare-cache', action='store_true',
                        help='Isaac: run both read modes in alternating order for each repeat')
    selected, _ = parser.parse_known_args()
    if selected.backend == 'isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.frames < 150 or args.repeats < 1:
        parser.error('Use at least 150 frames and one repeat')
    if args.backend == 'isaac' and (not args.headless or (args.rgb and not args.enable_cameras)):
        parser.error('Use --headless and, for RGB, --enable_cameras')
    if (args.uncached_reads or args.compare_cache) and args.backend != 'isaac':
        parser.error('Cache options are Isaac diagnostics')
    if args.uncached_reads and args.compare_cache:
        parser.error('Choose --uncached-reads or --compare-cache')
    args.output.mkdir(parents=True, exist_ok=False)
    from piper_sim.profiling import PipelineProfiler, save_profile
    profiler = PipelineProfiler()
    app = backend = session = None
    report = {'status': 'running', 'training_ready': False, 'backend': args.backend,
              'frames_per_repeat': args.frames, 'recording': {'storage': args.storage, 'compression': args.compression},
              'uncached_reads': args.uncached_reads,
              'compare_cache': args.compare_cache,
              'scope': 'Headless moving-camera and storage diagnostic, not sorting or qualified episode throughput.',
              'repeats': []}
    try:
        if args.backend == 'isaac':
            with profiler.stage('app_startup'):
                app = AppLauncher(args).app
        import h5py
        import numpy as np
        import yaml
        from piper_sim.config import ROOT, load
        from piper_sim.simulation import SimulationSession
        from piper_sim.recording import EpisodeWriter
        from piper_sim.collection_audit import audit_streams, load_acceptance_policy
        from piper_sim import randomization
        cfg = load('configs/clean_table_workcell.yaml')
        cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
        cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
        if args.backend == 'isaac':
            # Keep importer cache writes inside this benchmark. The source
            # URDF already contains absolute mesh paths; its bytes are intact.
            urdf = ROOT/cfg['robot']['urdf']
            isolated = args.output/'asset_cache'/urdf.name
            isolated.parent.mkdir(parents=True)
            shutil.copy2(urdf, isolated)
            if (urdf.parent/'usd').exists():
                shutil.copytree(urdf.parent/'usd', isolated.parent/'usd')
            cfg['robot']['urdf'] = str(isolated.resolve())
            from piper_sim.backends.isaac import IsaacBackend
            constructor = lambda: IsaacBackend(cfg, args.device, cameras=args.rgb)
        else:
            os.environ.setdefault('MUJOCO_GL', 'egl')
            from piper_sim.backends.mujoco import MujocoBackend
            constructor = lambda: MujocoBackend(cfg, cameras=args.rgb)
        with profiler.stage('scene_load'):
            backend = constructor()
        uncached = [args.uncached_reads]
        if args.uncached_reads or args.compare_cache:
            read = backend.read
            def uncached_read():
                if uncached[0]:
                    backend.camera_cache.clear()
                return read()
            backend.read = uncached_read
        for attribute, label in (('step', 'backend_step'), ('read', 'observation_read'), ('apply', 'control_apply')):
            profiler.wrap(backend, attribute, label)
        if args.backend == 'isaac':
            profiler.wrap(backend.sim, 'step', 'physics_tick')
            profiler.wrap(backend.sim, 'render', 'render')
        else:
            import mujoco
            profiler.wrap(mujoco, 'mj_step', 'physics_integration')
            profiler.wrap(mujoco, 'mj_forward', 'physics_forward')
            profiler.wrap(backend, '_render', 'render')
        profiler.wrap(randomization, 'sensor_images', 'sensor_postprocess')
        profiler.wrap(EpisodeWriter, 'append', 'recording_append')
        profiler.wrap(EpisodeWriter, 'close', 'recording_close')
        session = SimulationSession(backend, record_dir=args.output)
        settings, _ = load_acceptance_policy()
        dt = backend.dt*backend.decimation
        runs = [(repeat, mode) for repeat in range(args.repeats) for mode in
                (([False, True] if repeat % 2 == 0 else [True, False])
                 if args.compare_cache else [args.uncached_reads])]
        for repeat, mode in runs:
            uncached[0] = mode
            with profiler.stage('reset_and_warmup'):
                session.reset(0, True)
                for _ in range(30):
                    session.step()
            suffix = ('_uncached' if uncached[0] else '_cached') if args.compare_cache else ''
            session.start_recording(f'benchmark_{repeat:03d}'+suffix, 'runtime benchmark, not task success',
                                    image_storage=args.storage, image_compression=args.compression)
            start_sim = backend.sim_time
            started = time.perf_counter()
            for frame in range(args.frames):
                command = np.array(cfg['robot']['home']*2, dtype=float)
                phase = 2*np.pi*(frame+1)*dt/4
                for offset in (0, 7):
                    command[offset] = .05*np.sin(phase)
                    command[offset+5] = .10*np.sin(phase)
                    command[offset+6] = .04+.005*np.sin(phase)
                session.step(command)
            collection_s = time.perf_counter()-started
            close_started = time.perf_counter()
            recording = session.stop_recording(False)
            close_s = time.perf_counter()-close_started
            assert abs(backend.sim_time-start_sim-args.frames*dt) < 1e-6
            with profiler.stage('same_tick_read_check'):
                a, b = session.read(), session.read()
                for name in a.images:
                    np.testing.assert_array_equal(a.images[name], b.images[name])
                np.testing.assert_array_equal(a.position, b.position)
                if args.backend == 'isaac':
                    backend.camera_cache.clear()
                    fresh = session.read()
                    for name in a.images:
                        np.testing.assert_array_equal(a.images[name], fresh.images[name])
                    assert a.extras['cameras'] == fresh.extras['cameras']
                    for pixels in fresh.images.values():
                        pixels[:] = 0
                    owned = session.read()
                    for name in a.images:
                        np.testing.assert_array_equal(a.images[name], owned.images[name])
                    # The static alignment workflow forces a fresh camera
                    # capture without advancing physics. It must invalidate.
                    old_key = backend.camera_cache.key
                    for camera in backend.cameras.values():
                        # Lab's force_recompute only eagerly reads buffers
                        # already due; one sensor period makes this due while
                        # the physics clock and recorded data remain unchanged.
                        camera.update(dt, force_recompute=True)
                    session.read()
                    if backend.cameras:
                        assert backend.camera_cache.key != old_key
            with profiler.stage('stream_audit'):
                with h5py.File(recording['path'], 'r') as source:
                    audit = audit_streams(source, json.loads(source.attrs['metadata']), settings, require_rgb=args.rgb)
            if not audit['passed']:
                raise RuntimeError('Stream diagnostic failed: '+str(audit['violations']))
            result = {'repeat': repeat, 'uncached_reads': uncached[0],
                'collection_wall_s': collection_s, 'close_wall_s': close_s,
                'wall_transitions_per_s': args.frames/(collection_s+close_s), 'simulation_seconds': args.frames*dt,
                'bytes': Path(recording['path']).stat().st_size, 'stream_audit': audit,
                'same_tick_read_identical': True}
            report['repeats'].append(result)
            print(json.dumps({k: result[k] for k in ('repeat', 'uncached_reads', 'collection_wall_s', 'wall_transitions_per_s', 'bytes')}), flush=True)
        report['status'] = 'diagnostic_passed'
    except BaseException as error:
        import traceback
        report.update(status='failed', error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        save_profile(args.output/'profile.json', profiler.close())
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if session:
            session.close(complete=report['status'] == 'diagnostic_passed')
        elif backend:
            backend.close()
        if app:
            from isaaclab.sim import SimulationContext
            context = SimulationContext.instance()
            if context:
                context._disable_app_control_on_stop_handle = True
            app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
