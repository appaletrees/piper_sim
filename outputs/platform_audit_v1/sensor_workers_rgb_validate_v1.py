"""Validate the frozen full-RGB regression and every strict-reader input row."""
from contextlib import closing
import copy
import json
from pathlib import Path
import resource
import shutil
import time

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.dataset import sequence_windows
from piper_sim.lerobot_export import _inspect
from piper_sim.training_acceptance import require_self_contained

base = Path(__file__).resolve().parent
plan_path = base/'sensor_workers_rgb_plan_v1.json'
plan = json.loads(plan_path.read_text())
root = Path(plan['raw_output'])
source = root/'seed_1340830695/episode.h5'
state = Path(plan['comparison_rgb_source'])
inputs_path = base/'sensor_workers_rgb_inputs_v1.json'
inputs = json.loads(inputs_path.read_text())
policy = inputs['options']['acceptance_policy']
output = base/'sensor_workers_rgb_validation_v1.json'
evidence = base/'sensor_workers_rgb_evidence_v1'
if output.exists(): raise FileExistsError(output)
result = {'status': 'running', 'training_ready': None, 'durable_release': False,
          'script_sha256': sha256(__file__), 'plan_sha256': sha256(plan_path),
          'scope': __doc__, 'raw_storage': plan['storage'], 'comparison_rgb_source': str(state)}
started = time.perf_counter()
try:
    assert sha256(inputs_path) == plan['inputs_sha256']
    assert sha256(state) == plan['comparison_rgb_sha256']
    prior_summary = base/'tcp_clearance_rgb_summary_v1.json'
    assert sha256(prior_summary) == plan['comparison_summary_sha256']
    prior = json.loads(prior_summary.read_text())
    fk_path = base/'tcp_clearance_rgb_camera_fk_v1.json'
    assert sha256(fk_path) == prior['evidence_hashes'][fk_path.name]
    assert all(row['numerical_fk_check_passed'] and row['samples'] == 8772
               for row in json.loads(fk_path.read_text())['results'])
    changed = [p for p, d in inputs['files'].items() if not Path(p).is_file() or sha256(p) != d]
    assert not changed, changed
    profile = json.loads((root/'profile.json').read_text())
    assert profile['returncode'] == 0
    audit = json.loads(source.with_name('audit.json').read_text())
    rollout = json.loads(source.with_name('report.json').read_text())
    assert audit['passed'] and audit['training_ready'] and rollout['training_ready']
    assert audit['acceptance_policy']['sha256'] == sha256(policy)
    digest = sha256(source)
    assert audit['source_sha256'] == digest
    state_audit = json.loads(state.with_name('audit.json').read_text())
    for section, key in [('streams', 'trajectory_sha256'), ('task', 'objects_sha256')]:
        assert audit['checks'][section][key] == state_audit['checks'][section][key]
    evidence.mkdir(exist_ok=False)
    for original in (root/'profile.json', root/'collection.log', source.with_name('report.json'), source.with_name('audit.json')):
        target = evidence/original.name
        shutil.copyfile(original, target)
        assert sha256(original) == sha256(target)
    source_hashes = {str(source): digest, str(source.with_name('audit.json')): sha256(source.with_name('audit.json')),
                     str(state): plan['comparison_rgb_sha256']}
    result.update(source=str(source), source_sha256=digest, inputs_sha256=sha256(inputs_path),
        unchanged_frozen_files=len(inputs['files']), acceptance_policy=audit['acceptance_policy'],
        closed_rgb_audit_passed=True, matching_state_trajectory_and_object_hashes=True)
    fields = {'state': 'observation/state', 'action': 'action/position',
              'next_state': 'next_observation/state', 'timestamp': 'observation/timestamp',
              'sim_timestamp': 'observation/sim_time', 'next_sim_timestamp': 'next_observation/sim_time',
              'action_timestamp': 'action/timestamp', 'action_sequence': 'action/sequence'}
    expected_keys = set(fields) | {'images', 'image_timestamps', 'task'}
    image_count, max_window_bytes = 0, 0
    with h5py.File(source, 'r') as f, h5py.File(state, 'r') as s:
        require_self_contained(f)
        require_self_contained(s)
        assert f.attrs['complete'] and f.attrs['training_ready']
        meta = json.loads(f.attrs['metadata']); n = int(f.attrs['num_transitions'])
        assert n == int(s.attrs['num_transitions']) == 8771
        current_cfg, old_cfg = copy.deepcopy(meta['config']), json.loads(s.attrs['metadata'])['config']
        assert current_cfg['runtime'].pop('sensor_postprocess_workers') == 3
        assert old_cfg['runtime'].get('sensor_postprocess_workers', 1) == 1
        current_cfg.pop('_config_path', None); old_cfg.pop('_config_path', None)
        assert current_cfg == old_cfg
        cameras = sorted(f['observation/images'])
        assert cameras == ['left_wrist', 'overhead', 'right_wrist'] and meta['fps'] == 30.
        exact_fields = ('observation/state', 'next_observation/state', 'action/position',
                        'action/sequence', 'observation/sim_time', 'next_observation/sim_time')
        for field in exact_fields: np.testing.assert_array_equal(f[field][:], s[field][:])
        exact_extra_fields = ('cameras', 'image_timestamp_clock', 'physics_time', 'engine', 'engine_version', 'controller_interval',
                              'tcp_motion_interval', 'objects', 'task', 'teacher')
        for index in range(n+1):
            prefix, row = ('observation', index) if index < n else ('next_observation', n-1)
            rgb_extra = json.loads(f[prefix+'/extras_json'][row])
            state_extra = json.loads(s[prefix+'/extras_json'][row])
            for field in exact_extra_fields:
                assert rgb_extra[field] == state_extra[field], (index, field)
        raw_differences = {camera: np.zeros((n+1, 3), dtype=np.uint64) for camera in cameras}
        def compare_raw(camera, index, pixels, previous):
            difference = np.abs(pixels.astype(np.int16)-previous)
            raw_differences[camera][index] = (np.count_nonzero(difference), difference.max(), difference.sum())
        def check_window(window, offset, horizon):
            global image_count, max_window_bytes
            assert set(window) == expected_keys and window['task'] == meta['task']
            assert sorted(window['images']) == sorted(window['image_timestamps']) == cameras
            selection = slice(offset, offset+horizon)
            size = 0
            for name, field in fields.items():
                np.testing.assert_array_equal(window[name], f[field][selection]); size += window[name].nbytes
            for camera in cameras:
                pixels, stamps = window['images'][camera], window['image_timestamps'][camera]
                assert pixels.shape == (horizon, 480, 640, 3) and pixels.dtype == np.uint8
                np.testing.assert_array_equal(pixels, f['observation/images/'+camera][selection])
                if horizon == 1:
                    compare_raw(camera, offset, pixels[0], s['observation/images/'+camera][offset])
                np.testing.assert_array_equal(stamps, f['observation/image_timestamps/'+camera][selection])
                np.testing.assert_allclose(stamps, window['sim_timestamp'], atol=1e-6, rtol=0)
                size += pixels.nbytes+stamps.nbytes; image_count += horizon
            max_window_bytes = max(size, max_window_bytes)
        count = 0
        for index, window in enumerate(sequence_windows(source, horizon=1, policy=policy)):
            check_window(window, index, 1); count += 1
        assert count == n
        windows = [0, n//2, n-16]
        for offset in windows:
            with closing(sequence_windows(source, horizon=16, stride=max(1, offset), policy=policy)) as iterator:
                window = next(iterator)
                if offset: window = next(iterator)
                check_window(window, offset, 16)
        times = f['observation/sim_time'][:]
        host = f['observation/timestamp'][:]
        host_span = float(f['next_observation/timestamp'][-1]-host[0])
        clock = {'sim_interval_range_s': [float(np.diff(times).min()), float(np.diff(times).max())],
                 'host_interval_median_s': float(np.median(np.diff(host))),
                 'host_interval_max_s': float(np.diff(host).max()), 'host_observation_span_s': host_span,
                 'host_span_note': 'Recorded host wall-clock span, not simulation duration or complete pipeline timing.'}
        for camera in cameras:
            compare_raw(camera, n, f['next_observation/images/'+camera][-1], s['next_observation/images/'+camera][-1])
            np.testing.assert_allclose(f['next_observation/image_timestamps/'+camera][:],
                                       f['next_observation/sim_time'][:], atol=1e-6, rtol=0)
    inspected = _inspect(source, require_quality=True, policy=policy)
    assert inspected['audited_sha256'] == digest
    assert all(sha256(p) == d for p, d in source_hashes.items())
    assert all(sha256(p) == d for p, d in inputs['files'].items())
    export = source.parent/'lerobot_v2'
    export_hashes = {str(p.relative_to(export)): sha256(p) for p in sorted(export.rglob('*')) if p.is_file()}
    assert export_hashes and (export/'meta/info.json').is_file()
    atomic_json(evidence/'export_checksums.json', export_hashes)
    trace_path = base/'sensor_workers_rgb_raw_comparison_v1.npz'
    assert not trace_path.exists()
    np.savez_compressed(trace_path, **raw_differences)
    result.update(raw_pixel_comparison={camera: {
        'different_channel_values': int(values[:, 0].sum()), 'max_absolute_error': int(values[:, 1].max()),
        'mean_absolute_error_0_to_255': float(values[:, 2].sum()/((n+1)*480*640*3)),
        'identical': not bool(values[:, 0].any()), 'unique_frames_compared': n+1}
        for camera, values in raw_differences.items()},
        raw_pixel_comparison_scope='Diagnostic cross-run renderer comparison, not a new acceptance threshold. '
            'Same-input serial/parallel processing equivalence was checked separately on actual EGL frames and exhaustive uint8 levels.',
        raw_pixel_trace_path=str(trace_path), raw_pixel_trace_sha256=sha256(trace_path),
        raw_pixel_trace_columns=['different_channel_values', 'max_absolute_error', 'sum_absolute_error'])
    result.update(status='passed_full_rgb_reader_regression', frames=n, unique_frames_per_camera=n+1,
        image_shape=[480, 640, 3], fps=30., simulated_duration_s=n/30,
        all_public_rows_compared=n, raw_rgb_images_compared=n*3,
        total_images_compared_including_windows=image_count, window_starts=windows, horizon=16,
        reader_public_keys=sorted(expected_keys), reader_fields_compared=fields,
        exact_state_rgb_fields=list(exact_fields), task=meta['task'], clocks=clock,
        exact_physics_rows=n+1, exact_physics_extra_fields=list(exact_extra_fields),
        max_returned_window_bytes=max_window_bytes, source_bytes=source.stat().st_size,
        export_bytes=sum(p.stat().st_size for p in export.rglob('*') if p.is_file()),
        raw_transitions_per_decimal_GB=n/(source.stat().st_size/1e9),
        strict_export_inspection_passed=True, all_sources_unchanged=True,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pipeline_profile=profile, reader_validation_seconds=time.perf_counter()-started,
        limitations=plan['remaining_limitations']+[
            'Known regression on volatile tmpfs; no durable release or unseen-layout reliability claim.',
            'All non-PPO tests and EGL comparisons finished before this acquisition. The older serial baseline had early concurrent tests; timing remains a shared-host comparison.',
            'Reader checks compare against indexed HDF5 pixels; physical camera calibration and pixel-level real-world alignment are separate.',
            'Official LeRobot consumer and decoded-video checks are separate. All camera poses equal the retained serial baseline whose complete FK audit passed.'])
except BaseException as error:
    result.update(status='failed_validation', error=type(error).__name__+': '+str(error))
    raise
finally:
    result['analysis_elapsed_s'] = time.perf_counter()-started
    atomic_json(output, result)
if result['status'].startswith('passed'):
    print(json.dumps({k: result[k] for k in ('status','frames','raw_rgb_images_compared','source_bytes',
        'export_bytes','matching_state_trajectory_and_object_hashes','all_sources_unchanged')}, indent=2), flush=True)
