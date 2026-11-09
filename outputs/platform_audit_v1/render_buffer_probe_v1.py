#!/usr/bin/env python3
"""Compare equal-resolution rendering with an exactly sized framebuffer; split GPU wait diagnostics.

Restores recorded poses for rendering only. Does not replay dynamics or produce
training episodes. All candidates keep the original model, optics, 4x MSAA and
shadows; compare their pixels against repeated single-viewport rendering.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.profiling import PipelineProfiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 2450, 4385, 8770])
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    if args.repeats < 1 or min(args.frames) < 0: parser.error('Invalid frame/repeat count')
    args.output.mkdir(parents=True)
    report = dict(status='running', training_ready=False, scope=__doc__,
                  source=str(args.source.resolve()), script_sha256=sha256(__file__))
    backend = profiler = fitted = None
    try:
        source_hash = sha256(args.source)
        with h5py.File(args.source, 'r') as source:
            meta = json.loads(source.attrs['metadata'])
            if max(args.frames) >= source.attrs['num_transitions']: raise ValueError('Frame outside source')
            cfg = copy.deepcopy(meta['config'])
            cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
            backend = MujocoBackend(cfg, cameras=False)
            layout = meta['task_scene']['layout']
            backend.reset(layout['seed'], layout['randomized'])
            names = list(backend.rig['cameras'])
            resolutions = [mount['rgb']['resolution'] for mount in backend.rig['cameras'].values()]
            if len(names) != 3 or any(tuple(r) != (640, 480) for r in resolutions):
                raise ValueError('This bounded probe expects three 640x480 cameras')
            width, height = resolutions[0]
            # Use the existing production context, without resizing buffers.
            backend._render(names[0], width, height)
            renderer = backend.renderers[(width, height)]
            context = renderer._mjr_context
            if context.offWidth < len(names)*width or context.offHeight < height:
                raise ValueError('Existing framebuffer cannot hold three horizontal viewports')
            rect = mujoco.MjrRect(0, 0, width, height)
            wide = mujoco.MjrRect(0, 0, width*len(names), height)
            fitted = mujoco.Renderer(backend.model, width=width, height=height)
            mujoco.mjr_resizeOffscreen(width, height, fitted._mjr_context)
            assert (fitted._mjr_context.offWidth, fitted._mjr_context.offHeight) == (width, height)
            assert fitted._mjr_context.offSamples == context.offSamples
            assert fitted._mjr_context.shadowSize == context.shadowSize
            stages = {mode: [] for mode in ('baseline', 'fit_buffer', 'finish_diagnostic')}
            comparisons = {mode: [] for mode in stages}
            profiler = PipelineProfiler(sample_interval_s=1.)
            def render(mode):
                t0 = time.perf_counter()
                if mode in ('baseline', 'fit_buffer'):
                    pixels = {}
                    for name in names:
                        if mode == 'baseline': pixels[name] = backend._render(name, width, height)
                        else:
                            fitted.update_scene(backend.data, camera=name)
                            pixels[name] = fitted.render().copy()
                    return pixels, dict(total_s=time.perf_counter()-t0)
                pixels = {}; update_s = draw_s = read_s = copy_s = finish_s = 0.
                renderer._gl_context.make_current()
                for i, name in enumerate(names):
                    start = time.perf_counter(); renderer.update_scene(backend.data, camera=name)
                    update_s += time.perf_counter()-start
                    viewport = rect if mode == 'finish_diagnostic' else mujoco.MjrRect(i*width, 0, width, height)
                    start = time.perf_counter(); mujoco.mjr_render(viewport, renderer.scene, context)
                    draw_s += time.perf_counter()-start
                    start = time.perf_counter(); mujoco.mjr_finish()
                    finish_s += time.perf_counter()-start
                    if mode == 'finish_diagnostic':
                        out = np.empty((height, width, 3), np.uint8)
                        start = time.perf_counter(); mujoco.mjr_readPixels(out, None, rect, context)
                        read_s += time.perf_counter()-start
                        start = time.perf_counter(); pixels[name] = out[::-1].copy()
                        copy_s += time.perf_counter()-start
                if mode == 'grouped_read':
                    out = np.empty((height, width*len(names), 3), np.uint8)
                    start = time.perf_counter(); mujoco.mjr_readPixels(out, None, wide, context)
                    read_s += time.perf_counter()-start
                    start = time.perf_counter()
                    pixels = {name: out[::-1, i*width:(i+1)*width].copy() for i, name in enumerate(names)}
                    copy_s += time.perf_counter()-start
                return pixels, dict(total_s=time.perf_counter()-t0, scene_update_s=update_s,
                                    render_api_s=draw_s, read_api_s=read_s, copy_s=copy_s, finish_api_s=finish_s)

            for frame in args.frames:
                extra = json.loads(source['observation/extras_json'][frame])
                for side, values in extra['teacher']['physics']['arms'].items():
                    backend.data.qpos[backend.indices[side]['qpos']] = values['joint_position']
                for name, pose in extra['objects'].items():
                    adr = backend.model.joint('task/'+name+'/free').qposadr[0]
                    backend.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
                mujoco.mj_forward(backend.model, backend.data)
                state_type = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.empty(mujoco.mj_stateSize(backend.model, state_type))
                mujoco.mj_getState(backend.model, backend.data, state, state_type)
                for mode in stages:
                    for _ in range(2): render(mode)
                reference, _ = render('baseline')
                for repeat in range(args.repeats):
                    order = list(stages); order = order[repeat%3:]+order[:repeat%3]
                    for mode in order:
                        pixels, timing = render(mode)
                        stages[mode].append(dict(frame=frame, repeat=repeat, **timing))
                        difference = {}
                        for name in names:
                            delta = np.abs(pixels[name].astype(np.int16)-reference[name])
                            difference[name] = dict(different_channel_values=int(np.count_nonzero(delta)),
                                max_abs=int(delta.max()), sum_abs=int(delta.sum()))
                        comparisons[mode].append(dict(frame=frame, repeat=repeat, cameras=difference))
                after = np.empty_like(state)
                mujoco.mj_getState(backend.model, backend.data, after, state_type)
                np.testing.assert_array_equal(state, after)
            resource_report = profiler.close(); profiler = None
            medians = {mode: {key: float(np.median([r[key] for r in rows])) for key in rows[0]
                             if key.endswith('_s')} for mode, rows in stages.items()}
            assert sha256(args.source) == source_hash
            report.update(status='render_probe_completed', source_sha256=source_hash, source_unchanged=True,
                integration_state_unchanged=True, config=cfg, source_pose_frames=args.frames,
                repeats_per_frame=args.repeats, cameras=names, resolution=[width, height],
                framebuffer=[context.offWidth, context.offHeight], msaa_samples=context.offSamples,
                shadow_size=context.shadowSize, fitted_framebuffer=[fitted._mjr_context.offWidth, fitted._mjr_context.offHeight],
                stages=stages, comparisons=comparisons,
                median_seconds=medians,
                component_speedup={mode: medians['baseline']['total_s']/v['total_s'] for mode, v in medians.items()},
                resources=resource_report['resources'], mujoco_version=mujoco.__version__,
                input_hashes={str(p):sha256(p) for p in [Path(mujoco.__file__).parent/'renderer.py',
                    Path('src/piper_sim/backends/mujoco.py'), Path('src/piper_sim/mujoco_asset.py')]},
                limitations=['Render-only warm component timing, not task or pipeline throughput.',
                    'Recorded poses omit dynamics solver history; no physical replay claim.',
                    'API wall-time includes driver submission and synchronization; it is not GPU timer-query attribution.',
                    'Pixel differences are measured without inventing an acceptance threshold.'])
    except BaseException as error:
        report.update(status='failed_probe', error=repr(error))
        raise
    finally:
        if profiler: profiler.close()
        if fitted: fitted.close()
        if backend: backend.close()
        atomic_json(args.output/'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'median_seconds', 'component_speedup')}, indent=2))


if __name__ == '__main__':
    main()
