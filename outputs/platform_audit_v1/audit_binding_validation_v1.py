"""Real closed-recording validation; no original acceptance attributes are edited."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py

from piper_sim.collection_audit import audit_episode, save_audit
from piper_sim.collection_jobs import atomic_json, sha256, validate_completed
from piper_sim.config import ROOT
from piper_sim.lerobot_export import _inspect
from piper_sim.recording_storage import compress_episode_json

base = ROOT/'outputs/platform_audit_v1'
output = base/'audit_binding_validation_v1.json'
copy_root = base/'audit_binding_state_copy_v1'
if output.exists() or copy_root.exists():
    raise FileExistsError('Preserve previous validation results')
original = base/'baseline_seed0/episode.h5'
rgb = base/'servo_rgb_seed0_v1/seed_0/episode.h5'
old_rgb_audit = json.loads(rgb.with_name('audit.json').read_text())
if shutil.disk_usage(base).free < 5*1024**3+original.stat().st_size:
    raise RuntimeError('Keep the existing 5 GiB reserve, including a worst-case state copy')
copy_root.mkdir()
plan = {'source': str(original), 'rgb_source': str(rgb), 'rgb_source_sha256': old_rgb_audit['source_sha256'],
        'scope': 'Existing complete recordings, one lossless derived state copy, no new rollout or throughput claim.',
        'auditor_sha256': sha256(ROOT/'src/piper_sim/collection_audit.py'),
        'policy_sha256': sha256(ROOT/'configs/collection/quality_policy.yaml'),
        'training_ready': False}
atomic_json(base/'audit_binding_validation_plan_v1.json', plan)

def public_hashes(path):
    result = {}
    with h5py.File(path, 'r') as f:
        names = []
        for prefix in ('observation', 'action', 'next_observation'):
            f[prefix].visititems(lambda _, item: names.append(item.name) if isinstance(item, h5py.Dataset) else None)
        for name in names:
            ds = f[name]; digest = hashlib.sha256()
            for start in range(0, len(ds), 128):
                values = ds[start:start+128]
                if ds.dtype.kind == 'O':
                    for raw in values:
                        digest.update(len(raw).to_bytes(8, 'little')); digest.update(raw)
                else: digest.update(values.tobytes())
            result[name] = {'shape': list(ds.shape), 'dtype': str(ds.dtype), 'sha256': digest.hexdigest()}
    return result

copy_path = copy_root/'episode.h5'
derivation = compress_episode_json(original, copy_path)
atomic_json(copy_root/'derivation.json', derivation)
before = public_hashes(original)
started = time.perf_counter()
state = audit_episode(copy_path, require_rgb=False)
atomic_json(copy_root/'read_only_audit.json', state)
audit_seconds = time.perf_counter()-started
if state['passed']:
    save_audit(copy_path, state, copy_root/'audit.json')
    validate_completed({'accepted': False, 'diagnostic_passed': True,
                        'path': str(copy_path), 'audit': str(copy_root/'audit.json')})
after = public_hashes(copy_path)
assert before == after and sha256(original) == derivation['source_sha256']
started = time.perf_counter()
rgb_audit = audit_episode(rgb)
rgb_seconds = time.perf_counter()-started
atomic_json(base/'audit_binding_existing_rgb_v1.json', rgb_audit)
assert rgb_audit['source_sha256'] == old_rgb_audit['source_sha256']
original_rgb_eligibility = _inspect(rgb, require_quality=True)['audited_sha256']
result = {'scope': plan['scope'], 'training_ready': False, 'analysis_sha256': sha256(__file__),
          'auditor_sha256': plan['auditor_sha256'], 'policy_sha256': plan['policy_sha256'],
          'state_derivation': derivation, 'state_audit_passed': state['passed'],
          'state_audit_violations': state['violations'], 'state_training_ready': state['training_ready'],
          'state_source_hash_before_publication': state.get('audited_source_sha256'),
          'state_source_hash_after_publication': state['source_sha256'],
          'state_publication_and_diagnostic_resume_verified': state['passed'],
          'public_values_match_original': before == after, 'verified_public_datasets': after,
          'state_audit_wall_s': audit_seconds,
          'existing_rgb_audit_passed': rgb_audit['passed'], 'existing_rgb_violations': rgb_audit['violations'],
          'existing_rgb_frames': rgb_audit.get('checks', {}).get('streams', {}).get('frames'),
          'existing_rgb_checked_images': sum(x['checked_images'] for x in rgb_audit.get('checks', {}).get('streams', {}).get('cameras', {}).values()),
          'existing_rgb_audit_wall_s': rgb_seconds, 'existing_rgb_source_unchanged': True,
          'existing_rgb_export_inspection_sha256': original_rgb_eligibility,
          'wall_time_scope': 'Validation wall time under shared host load, not a collection throughput comparison.',
          'new_simulation_episodes': 0, 'new_rgb_exports': 0}
atomic_json(output, result)
print(json.dumps({k: result[k] for k in ('state_audit_passed', 'public_values_match_original',
    'existing_rgb_audit_passed', 'existing_rgb_frames', 'existing_rgb_checked_images', 'new_simulation_episodes')}))
