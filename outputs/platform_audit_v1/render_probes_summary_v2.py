"""Bind render-only experiments, rejected hypotheses, resources and limits."""
import json
from pathlib import Path
import shutil

import numpy as np
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT

base = Path(__file__).resolve().parent
output = base/'render_probes_summary_v2.json'
assert not output.exists()
read = lambda path: json.loads(Path(path).read_text())
def copy_verified(source, target):
    if target.exists():
        assert sha256(source) == sha256(target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha256(source) == sha256(target)

probes = {}
for key, raw, script in [
    ('paths', '/dev/shm/piper_render_paths_v1', ROOT/'scripts/benchmark_render_paths.py'),
    ('workers', '/dev/shm/piper_render_workers_v2', ROOT/'scripts/benchmark_render_workers.py'),
    ('buffer', '/dev/shm/piper_render_buffer_v1', base/'render_buffer_probe_v1.py')]:
    raw = Path(raw)
    report = read(raw/'report.json')
    assert report['script_sha256'] == sha256(script)
    assert report['source_unchanged'] and report['integration_state_unchanged']
    name = f'render_{key}_probe_'+('v2' if key == 'workers' else 'v1')
    copy_verified(raw/'report.json', base/(name+'.json'))
    copy_verified(raw.with_suffix('.log'), base/(name+'.log'))
    probes[key] = report
raw = Path('/dev/shm/piper_render_isolated_v1')
sequence = read(raw/'runs.json')
assert sequence['status'] == 'all_six_runs_completed'
assert sequence['order'] == ['baseline', 'fit_buffer', 'workers', 'workers', 'fit_buffer', 'baseline']
assert sequence['script_sha256'] == sha256(base/'render_isolated_probe_v1.py')
evidence = base/'render_isolated_v1'
reports = []
for row in sequence['runs']:
    assert row['returncode'] == 0 and sha256(row['log']) == row['log_sha256']
    path = raw/f"{row['index']}_{row['mode']}"/'report.json'
    report = read(path)
    assert report['status'] == 'isolated_render_probe_passed'
    assert report['source_unchanged'] and report['integration_state_unchanged'] and report['worker_threads_joined']
    assert report['msaa_samples'] == 4 and report['shadow_size'] == probes['paths']['shadow_size'] == 8192
    assert len(report['samples']) == 80 and report['frames'] == [0, 2450, 4385, 8770]
    assert all(sha256(p) == digest for p, digest in report['input_hashes'].items())
    assert sha256(report['pixels_path']) == report['pixels_sha256']
    copy_verified(path, evidence/f"{row['index']}_{row['mode']}.json")
    copy_verified(Path(row['log']), evidence/Path(row['log']).name)
    reports.append(report)
copy_verified(raw/'runs.json', evidence/'runs.json')
copy_verified(raw.with_suffix('.log'), evidence/'runner.log')
median = {mode: float(np.median([sample['seconds'] for report in reports if report['mode'] == mode
                                for sample in report['samples']]))
          for mode in ('baseline', 'fit_buffer', 'workers')}
baseline_pixels = np.load(reports[0]['pixels_path'])
pixel_comparison = []
for index, report in enumerate(reports):
    with np.load(report['pixels_path']) as current:
        assert set(current.files) == set(baseline_pixels.files)
        different = error_sum = maximum = channels = 0
        for name in current.files:
            delta = np.abs(current[name].astype(np.int16)-baseline_pixels[name])
            different += int(np.count_nonzero(delta)); error_sum += int(delta.sum())
            maximum = max(maximum, int(delta.max())); channels += delta.size
        pixel_comparison.append(dict(index=index, mode=report['mode'], images=len(current.files),
            different_channel_values=different, total_channel_values=channels, max_abs=maximum,
            mean_absolute_error_0_to_255=error_sum/channels))
baseline_pixels.close()
source = reports[0]['source']
digest = sha256(source)
assert all(report['source_sha256'] == digest for report in reports)
assert all(report['source_sha256'] == digest for report in probes.values())
frozen = read(base/'sensor_workers_rgb_inputs_v1.json')
assert all(sha256(p) == value for p, value in frozen['files'].items())
profile_path = base/'sensor_workers_rgb_evidence_v1/profile.json'
profile = read(profile_path)
wall = profile['wall_seconds']; rendering = profile['stages']['render_and_readback']['inclusive_s']
paths = [p for p in base.glob('render_*probe*') if p.is_file()]
paths += [p for p in evidence.iterdir() if p.is_file()]
paths += [Path(__file__)]
hashes = {str(p.relative_to(base)):sha256(p) for p in paths}
report = dict(status='render_hypotheses_measured_not_adopted', training_ready=False,
    production_renderer_changed=False, source=source, source_sha256=digest,
    production_frozen_inputs_unchanged=len(frozen['files']),
    scope='Render-only diagnostics; no new full episode, dynamics replay, training acceptance or pipeline speedup.',
    modes=['baseline', 'fit_buffer', 'workers'], isolated_order=sequence['order'],
    isolated_samples_per_mode=160, isolated_pose_frames=reports[0]['frames'],
    isolated_run_median_seconds=[dict(index=i, mode=row['mode'], seconds=row['median_seconds'])
                                 for i, row in enumerate(reports)],
    pooled_median_seconds=median, pooled_component_ratios={k:median['baseline']/v for k,v in median.items()},
    isolated_pixel_comparison_to_first_baseline=pixel_comparison,
    isolated_resources=[dict(index=i, mode=row['mode'], resources=row['resources']) for i,row in enumerate(reports)],
    exploratory_component_ratios={k:row['component_speedup'] for k,row in probes.items()},
    finish_diagnostic_median_seconds=probes['buffer']['median_seconds']['finish_diagnostic'],
    hypotheses=[
        {'name':'grouped viewports/readback', 'decision':'not adopted',
         'reason':'Only 1.057x exploratory component timing; observed max channel difference 32/255 versus repeated baseline max 1. No visual-acceptance threshold invented.'},
        {'name':'one thread/context/data per camera', 'decision':'not adopted',
         'reason':'Both mixed and isolated measurements show slowdown, while retaining three GPU contexts. The initial unavailable Python mj_copyData call failed; fixed to supported copy.copy and failure preserved.'},
        {'name':'640x480 offscreen allocation', 'decision':'not adopted',
         'reason':'Exploratory mixed-context result suggested 1.114x but isolated forward/reverse runs do not reproduce meaningful speedup. More context-switching in an exploratory benchmark is not production evidence.'}],
    timing_interpretation='After explicit GPU finish, median actual pixel-read API time is about 0.54 ms per triplet. Much of ordinary read API wall time therefore contains waiting for preceding draw work. Finish perturbs scheduling and is diagnostic only; it is not a GPU timer query or proposed production call.',
    baseline_pipeline=dict(profile=str(profile_path), profile_sha256=sha256(profile_path), wall_seconds=wall,
        rendering_seconds=rendering, rendering_fraction=rendering/wall,
        ideal_zero_render_cost_speedup_with_other_costs_fixed=wall/(wall-rendering),
        interpretation='Analytical upper bound only: even zero rendering cost cannot alone double this serial full pipeline while other costs remain fixed.'),
    references=[
        'https://mujoco.readthedocs.io/en/3.3.7/programming/visualization.html',
        'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/src/render/render_gl2.c',
        'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/src/render/render_gl3.c'],
    evidence_hashes=hashes,
    storage=dict(evidence_bytes=sum(p.stat().st_size for p in paths),
        root_free_bytes=shutil.disk_usage(base).free, tmpfs_free_bytes=shutil.disk_usage(raw).free,
        note='Small reports/logs/scripts persisted; reference snapshots in tmpfs remain volatile. Existing raw episodes preserved.'),
    limitations=['Four static recorded poses and two isolated repetitions per mode do not estimate full-task or layout-population performance.',
                 'Warm context timing excludes setup, sensor processing, physics, recording, audit and export.',
                 'No frame rate, image resolution, shadow size, MSAA, model dimensions or acceptance thresholds changed.',
                 'Exact raw RGB across repeated or independent GPU renders is not assumed.',
                 'Current full-task candidate still has per-target finger material switching and uncalibrated real optics/dynamics.'],
    next_priorities=['Return to fixed finger material with current motion/planning changes; retain all failures.',
                     'Study safe cross-stage or environment-level concurrency after current-version reliability and durable storage are established.'])
atomic_json(output, report)
print(json.dumps({key:report[key] for key in ('status','pooled_median_seconds','pooled_component_ratios','baseline_pipeline')},indent=2))
