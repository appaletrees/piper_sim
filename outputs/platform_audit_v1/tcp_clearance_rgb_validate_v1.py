"""Validate the frozen full-RGB regression and every strict-reader input row."""
from contextlib import closing
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
plan_path = base/'tcp_clearance_rgb_plan_v1.json'
plan = json.loads(plan_path.read_text())
root = Path(plan['raw_output'])
source = root/'seed_1340830695/episode.h5'
state = Path(plan['comparison_state_source'])
inputs_path = base/'tcp_clearance_rgb_inputs_v1.json'
inputs = json.loads(inputs_path.read_text())
policy = inputs['options']['acceptance_policy']
output = base/'tcp_clearance_rgb_validation_v1.json'
evidence = Path(plan['persistent_evidence'])
if output.exists(): raise FileExistsError(output)
result = {'status': 'running', 'training_ready': None, 'durable_release': False,
          'script_sha256': sha256(__file__), 'plan_sha256': sha256(plan_path),
          'scope': __doc__, 'raw_storage': plan['storage']}
started = time.perf_counter()
try:
    assert sha256(inputs_path) == plan['inputs_sha256']
    assert sha256(state) == plan['comparison_state_sha256']
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
                     str(state): plan['comparison_state_sha256']}
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
        assert f.attrs['complete'] and f.attrs['training_ready']
        meta = json.loads(f.attrs['metadata']); n = int(f.attrs['num_transitions'])
        assert n == int(s.attrs['num_transitions']) == 8771
        cameras = sorted(f['observation/images'])
        assert cameras == ['left_wrist', 'overhead', 'right_wrist'] and meta['fps'] == 30.
        exact_fields = ('observation/state', 'next_observation/state', 'action/position',
                        'action/sequence', 'observation/sim_time', 'next_observation/sim_time')
        for field in exact_fields: np.testing.assert_array_equal(f[field][:], s[field][:])
        exact_extra_fields = ('physics_time', 'engine', 'engine_version', 'controller_interval',
                              'tcp_motion_interval', 'objects', 'task', 'teacher')
        for index in range(n+1):
            prefix, row = ('observation', index) if index < n else ('next_observation', n-1)
            rgb_extra = json.loads(f[prefix+'/extras_json'][row])
            state_extra = json.loads(s[prefix+'/extras_json'][row])
            for field in exact_extra_fields:
                assert rgb_extra[field] == state_extra[field], (index, field)
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
            'The non-PPO test suite ran concurrently during early acquisition; stage times are diagnostic shared-host measurements.',
            'Reader checks compare against indexed HDF5 pixels; physical camera calibration and pixel-level real-world alignment are separate.',
            'Official LeRobot consumer, camera FK, and full decoded-video diagnostics are separate reports.'])
except BaseException as error:
    result.update(status='failed_validation', error=type(error).__name__+': '+str(error))
    raise
finally:
    result['analysis_elapsed_s'] = time.perf_counter()-started
    atomic_json(output, result)
if result['status'].startswith('passed'):
    print(json.dumps({k: result[k] for k in ('status','frames','raw_rgb_images_compared','source_bytes',
        'export_bytes','matching_state_trajectory_and_object_hashes','all_sources_unchanged')}, indent=2), flush=True)
