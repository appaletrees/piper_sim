"""Read-only full-episode consumer validation; no rollout or training export."""
from contextlib import closing
import json
from pathlib import Path
import resource
import time

import h5py
import numpy as np

from piper_sim.collection_audit import audit_episode
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.dataset import sequence_windows
from piper_sim.lerobot_export import _inspect
from piper_sim.training_acceptance import require_self_contained

out = ROOT/'outputs/platform_audit_v1'
plan_path = out/'training_reader_validation_plan_v1.json'
result_path = out/'training_reader_validation_v1.json'
if plan_path.exists() or result_path.exists():
    raise FileExistsError('Preserve previous evidence; use a new version for another run')
source = out/'servo_rgb_seed0_v1/seed_0/episode.h5'
state = out/'audit_binding_state_copy_v1/episode.h5'
sources = {source: 'c1d8939a732cf6f545e61c8a189d6427048abe2b89f23bea11ca7e52c4178b83',
           state: '35ce5560a81c6907b7be0b490972ad2fc023b9200302802b2fe8ba7fc12f4269'}
frozen = ['src/piper_sim/dataset.py', 'src/piper_sim/training_acceptance.py',
          'src/piper_sim/collection_audit.py', 'src/piper_sim/collection_jobs.py',
          'src/piper_sim/lerobot_export.py', 'configs/collection/quality_policy.yaml',
          'tests/test_training_reader.py']
fingerprints = {p: sha256(ROOT/p) for p in frozen}
sidecars = {}
for path, expected in sources.items():
    assert sha256(path) == expected
    with h5py.File(path, 'r') as f:
        sidecar = path.with_name(f.attrs['dataset_audit_sidecar'])
        sidecars[sidecar] = sha256(sidecar)
plan = {'scope': __doc__, 'training_ready': False,
        'source_checksums': {str(p): digest for p, digest in sources.items()},
        'sidecar_checksums': {str(p): digest for p, digest in sidecars.items()},
        'files': fingerprints, 'analysis_sha256': sha256(__file__),
        'checks': ['every row / three RGB images compared exactly with indexed HDF5 values',
                   'all returned SI states, applied actions, sequences and clock fields compared exactly',
                   'exact key whitelist excludes privileged extras',
                   'full 16-frame windows at start, middle and end',
                   'strict LeRobot inspection and full state-only re-audit',
                   'all input HDF5, sidecars and code/policy fingerprints unchanged afterward']}
atomic_json(plan_path, plan)
start = time.perf_counter()
fields = {'state': 'observation/state', 'action': 'action/position',
          'next_state': 'next_observation/state', 'timestamp': 'observation/timestamp',
          'sim_timestamp': 'observation/sim_time', 'next_sim_timestamp': 'next_observation/sim_time',
          'action_timestamp': 'action/timestamp', 'action_sequence': 'action/sequence'}
keys = set(fields) | {'images', 'image_timestamps', 'task'}
max_window_bytes = 0
images = 0
with h5py.File(source, 'r') as f:
    require_self_contained(f)
    n = int(f.attrs['num_transitions'])
    cameras = sorted(f['observation/images'])
    assert n == 7106 and len(cameras) == 3
    task = json.loads(f.attrs['metadata'])['task']
    def check_window(window, offset, horizon):
        global max_window_bytes, images
        assert set(window) == keys and window['task'] == task
        assert sorted(window['images']) == cameras
        assert sorted(window['image_timestamps']) == cameras
        selection = slice(offset, offset+horizon)
        byte_count = 0
        for key, field in fields.items():
            np.testing.assert_array_equal(window[key], f[field][selection])
            byte_count += window[key].nbytes
        for camera in cameras:
            pixels = window['images'][camera]
            assert pixels.dtype == np.uint8 and pixels.shape == (horizon,480,640,3)
            np.testing.assert_array_equal(pixels, f['observation/images/'+camera][selection])
            stamps = window['image_timestamps'][camera]
            np.testing.assert_array_equal(stamps, f['observation/image_timestamps/'+camera][selection])
            np.testing.assert_allclose(stamps, window['sim_timestamp'], rtol=0, atol=1e-6)
            byte_count += pixels.nbytes + stamps.nbytes
            images += horizon
        max_window_bytes = max(max_window_bytes, byte_count)
    count = 0
    with closing(sequence_windows(source, horizon=1)) as windows:
        for count, window in enumerate(windows, start=1):
            check_window(window, count-1, 1)
            if count % 1000 == 0: print(f'checked {count}/{n} transitions', flush=True)
    assert count == n
    full_row_images = images
    # A 16-frame start/middle/end test keeps memory bounded. Stride only changes
    # window starts; every one of the 16 original frames is compared at each start.
    window_starts = []
    with closing(sequence_windows(source, horizon=16, stride=(n-16)//2)) as windows:
        for index, window in enumerate(windows):
            offset = index*((n-16)//2)
            check_window(window, offset, 16)
            window_starts.append(offset)
    assert window_starts == [0, 3545, 7090]
    def stats(values):
        return {'min': float(np.min(values)), 'median': float(np.median(values)),
                'max': float(np.max(values))}
    clocks = {'host_dt_seconds': stats(np.diff(f['observation/timestamp'][:])),
              'sim_dt_seconds': stats(np.diff(f['observation/sim_time'][:])),
              'transition_dt_seconds': stats(f['next_observation/sim_time'][:]-f['observation/sim_time'][:])}
    sim_duration = float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
reader_seconds = time.perf_counter()-start
inspection = _inspect(source, require_quality=True)
assert inspection['length'] == n and inspection['audited_sha256'] == sources[source]
state_start = time.perf_counter()
state_audit = audit_episode(state, require_rgb=False)
assert state_audit['passed'] and not state_audit['training_ready']
atomic_json(out/'training_reader_state_audit_v1.json', state_audit)
state_seconds = time.perf_counter()-state_start
for path, expected in {**sources, **sidecars}.items(): assert sha256(path) == expected
assert fingerprints == {p: sha256(ROOT/p) for p in frozen}
result = {'training_ready': False, 'scope': __doc__, 'plan_sha256': sha256(plan_path),
          'analysis_sha256': sha256(__file__), 'source': str(source),
          'source_sha256': sources[source], 'source_training_ready': True,
          'frames_checked': count, 'camera_names': cameras, 'full_row_rgb_images_checked': full_row_images,
          'rgb_shape': [480,640,3], 'multi_frame_window_starts': window_starts,
          'multi_frame_horizon': 16, 'total_rgb_images_compared_including_windows': images,
          'fields_compared_exactly': fields, 'keys': sorted(keys), 'task': task,
          'simulation_duration_seconds': sim_duration, 'clocks': clocks,
          'max_returned_window_array_bytes': max_window_bytes,
          'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'reader_validation_seconds': reader_seconds, 'strict_export_inspection': 'passed',
          'state_only_reaudit': {'passed': state_audit['passed'], 'training_ready': state_audit['training_ready'],
                                 'seconds': state_seconds},
          'all_sources_sidecars_and_code_unchanged': True,
          'limitations': ['Existing RGB recording; no new simulation or export produced.',
                         'This verifies reader fidelity, not real-device calibration or TCP candidate RGB.',
                         'Wall time includes duplicated independent reads and is not production throughput.']}
atomic_json(result_path, result)
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
