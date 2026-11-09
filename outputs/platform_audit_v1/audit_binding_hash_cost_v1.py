"""Measure raw-file hash cost on an existing RGB file; not collection throughput."""
import json
from pathlib import Path
import statistics
import time

import h5py

from piper_sim.collection_audit import _descriptor_sha256
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT

base = ROOT/'outputs/platform_audit_v1'
output = base/'audit_binding_hash_cost_v1.json'
if output.exists(): raise FileExistsError(output)
source = base/'servo_rgb_seed0_v1/seed_0/episode.h5'
expected = json.loads(source.with_name('audit.json').read_text())['source_sha256']
samples = []
with h5py.File(source, 'r') as f:
    descriptor = f.id.get_vfd_handle()
    for _ in range(3):
        started = time.perf_counter()
        digest = _descriptor_sha256(descriptor)
        seconds = time.perf_counter()-started
        if digest != expected: raise ValueError('Historical source changed')
        samples.append(seconds)
result = {'scope': __doc__, 'source': str(source), 'source_sha256': expected,
    'source_bytes': source.stat().st_size, 'hash_wall_seconds': samples,
    'median_hash_wall_seconds': statistics.median(samples), 'three_pass_wall_seconds': sum(samples),
    'new_audit_passes': 2, 'new_pre_publication_verification_passes': 1,
    'existing_post_publication_hash_retained': True,
    'cache': 'Existing local file just used in complete validation; OS page cache not dropped.',
    'limitations': 'Raw SHA-256 reading only; no decoding, task/stream audit, metadata fsync, export or simulation. Not end-to-end regression or throughput measurement.',
    'training_ready': False, 'implementation_sha256': sha256(ROOT/'src/piper_sim/collection_audit.py'),
    'analysis_sha256': sha256(__file__)}
atomic_json(output, result)
print(json.dumps(result))
