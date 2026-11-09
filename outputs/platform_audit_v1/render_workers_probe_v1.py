#!/usr/bin/env python3
"""Measure bounded, per-camera OpenGL owners against sequential rendering.

Each worker creates, uses and closes its own renderer and MjData on one thread.
Physics is paused at recorded poses; no trajectories or training data are made.
The main thread waits for all cameras before changing shared read-only model/data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import copy
import json
from pathlib import Path
import threading
import time

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.profiling import PipelineProfiler


class CameraOwner:
    def __init__(self, model, name, width, height):
        self.model, self.name = model, name
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='render-probe-'+name)
        self.renderer = None
        try:
            self.pool.submit(self._init, width, height).result()
        except BaseException:
            self.pool.shutdown(wait=True)
            raise

    def _init(self, width, height):
        self.owner = threading.get_ident()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)

    def _render(self, source):
        assert threading.get_ident() == self.owner
        begin = time.perf_counter()
        mujoco.mj_copyData(self.data, self.model, source)
        self.renderer.update_scene(self.data, camera=self.name)
        pixels = self.renderer.render().copy()
        cid = self.model.camera(self.name).id
        np.testing.assert_array_equal(self.data.cam_xpos[cid], source.cam_xpos[cid])
        np.testing.assert_array_equal(self.data.cam_xmat[cid], source.cam_xmat[cid])
        return pixels, time.perf_counter()-begin

    def submit(self, source):
        return self.pool.submit(self._render, source)

    def _close(self):
        assert threading.get_ident() == self.owner
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def close(self):
        try:
            self.pool.submit(self._close).result()
        finally:
            self.pool.shutdown(wait=True)
        assert not any(thread.is_alive() for thread in self.pool._threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 2450, 4385, 8770])
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    if args.repeats < 1 or min(args.frames) < 0: parser.error('Invalid count')
    args.output.mkdir(parents=True)
    report = dict(status='running', training_ready=False, scope=__doc__, script_sha256=sha256(__file__))
    backend = profiler = None
    owners = {}
    try:
        before = sha256(args.source)
        with h5py.File(args.source, 'r') as source:
            if max(args.frames) >= source.attrs['num_transitions']: raise ValueError('Frame outside source')
            meta = json.loads(source.attrs['metadata']); cfg = copy.deepcopy(meta['config'])
            cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
            backend = MujocoBackend(cfg, cameras=False)
            layout = meta['task_scene']['layout']
            backend.reset(layout['seed'], layout['randomized'])
            names = list(backend.rig['cameras'])
            if len(names) != 3: raise ValueError('Probe expects three cameras')
            profiler = PipelineProfiler(sample_interval_s=1.)
            for name in names:
                w, h = backend.rig['cameras'][name]['rgb']['resolution']
                if (w, h) != (640, 480): raise ValueError('Probe expects 640x480')
                # Serial construction avoids a race in EGL display initialization.
                owners[name] = CameraOwner(backend.model, name, w, h)
            stages = {mode: [] for mode in ('baseline', 'worker_serial', 'worker_parallel')}
            comparisons = {mode: [] for mode in stages}
            def render(mode):
                begin = time.perf_counter()
                if mode == 'baseline':
                    pixels = {name: backend._render(name, 640, 480) for name in names}
                    return pixels, dict(total_s=time.perf_counter()-begin)
                if mode == 'worker_serial':
                    rows = {name: owner.submit(backend.data).result() for name, owner in owners.items()}
                else:
                    pending = {}
                    try:
                        for name, owner in owners.items(): pending[name] = owner.submit(backend.data)
                        rows = {name: future.result() for name, future in pending.items()}
                    finally:
                        wait(pending.values())
                return {name: row[0] for name, row in rows.items()}, dict(total_s=time.perf_counter()-begin,
                    camera_wall_s={name: row[1] for name, row in rows.items()})

            for frame in args.frames:
                extra = json.loads(source['observation/extras_json'][frame])
                for side, values in extra['teacher']['physics']['arms'].items():
                    backend.data.qpos[backend.indices[side]['qpos']] = values['joint_position']
                for name, pose in extra['objects'].items():
                    adr = backend.model.joint('task/'+name+'/free').qposadr[0]
                    backend.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
                mujoco.mj_forward(backend.model, backend.data)
                kind = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.empty(mujoco.mj_stateSize(backend.model, kind))
                mujoco.mj_getState(backend.model, backend.data, state, kind)
                for mode in stages:
                    for _ in range(2): render(mode)
                reference, _ = render('baseline')
                for repeat in range(args.repeats):
                    order = list(stages); order = order[repeat%3:]+order[:repeat%3]
                    for mode in order:
                        pixels, timing = render(mode)
                        stages[mode].append(dict(frame=frame, repeat=repeat, **timing))
                        comparison = {}
                        for name in names:
                            delta = np.abs(pixels[name].astype(np.int16)-reference[name])
                            comparison[name] = dict(different_channel_values=int(np.count_nonzero(delta)),
                                max_abs=int(delta.max()), sum_abs=int(delta.sum()))
                        comparisons[mode].append(dict(frame=frame, repeat=repeat, cameras=comparison))
                after = np.empty_like(state)
                mujoco.mj_getState(backend.model, backend.data, after, kind)
                np.testing.assert_array_equal(state, after)
            median = {mode: float(np.median([row['total_s'] for row in rows])) for mode, rows in stages.items()}
            assert sha256(args.source) == before
            for owner in owners.values(): owner.close()
            owners.clear()
            resources = profiler.close(); profiler = None
            report.update(status='render_worker_probe_completed', source=str(args.source.resolve()), source_sha256=before,
                source_unchanged=True, integration_state_unchanged=True, camera_transforms_equal=True,
                worker_threads_joined=True, source_pose_frames=args.frames, repeats_per_frame=args.repeats,
                config=cfg, cameras=names, stages=stages, comparisons=comparisons, median_seconds=median,
                component_speedup={mode: median['baseline']/seconds for mode, seconds in median.items()},
                resources=resources['resources'], mean_cpu_percent_one_core=resources['mean_cpu_percent_one_core'],
                mujoco_version=mujoco.__version__, input_hashes={str(p):sha256(p) for p in [
                    Path(mujoco.__file__).parent/'renderer.py', Path('src/piper_sim/backends/mujoco.py'),
                    Path('src/piper_sim/mujoco_asset.py')]},
                limitations=['Render-only warm component timing; no full task or pipeline claim.',
                    'One thread/context/data per camera; three contexts consume extra resources.',
                    'No physical integration while rendering the same recorded poses.',
                    'Pixel differences are measured, not hidden by a new acceptance threshold.'])
    except BaseException as error:
        report.update(status='failed_probe', error=repr(error))
        raise
    finally:
        for owner in owners.values(): owner.close()
        if profiler: profiler.close()
        if backend: backend.close()
        atomic_json(args.output/'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'median_seconds', 'component_speedup')}, indent=2))


if __name__ == '__main__':
    main()
