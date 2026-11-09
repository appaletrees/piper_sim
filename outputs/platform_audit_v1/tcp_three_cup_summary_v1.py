"""Preserve the predeclared three-cup TCP regression, including any failure."""
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT

base = ROOT/'outputs/platform_audit_v1'
root = base/'tcp_three_cup_state_v1'
output = base/'tcp_three_cup_summary_v1.json'
if output.exists():
    raise FileExistsError(output)
plan = json.loads((base/'tcp_three_cup_plan_v1.json').read_text())
identity = json.loads((root/'run.json').read_text())
prior_identity_path = base/'tcp_states_v1/run.json'
assert sha256(prior_identity_path) == plan['preceding_cohort_manifest_sha256']
prior_identity = json.loads(prior_identity_path.read_text())
assert identity['files'] == prior_identity['files']
changed = [p for p, digest in identity['files'].items()
           if not Path(p).is_file() or sha256(p) != digest]
if changed:
    raise ValueError('Frozen inputs changed: '+str(changed))
batch = json.loads((root/'summary.json').read_text())
assert len(batch['episodes']) == 1
item = batch['episodes'][0]
assert item['seed'] == plan['seed']
source = Path(item['path'])
report = json.loads((source.parent/'report.json').read_text())
audit = json.loads((source.parent/'audit.json').read_text())
native = json.loads((base/'tcp_native_seed13_v1.json').read_text())
tcp = json.loads((base/'tcp_interval_seed13_v1.json').read_text())
digest = sha256(source)
for evidence in (audit, native, tcp):
    assert evidence['source_sha256'] == digest
assert audit['acceptance_policy']['sha256'] == plan['policy_sha256']
assert tcp['acceptance_policy']['sha256'] == plan['policy_sha256']
assert not audit['training_ready'] and not item['accepted']
with h5py.File(source, 'r') as f:
    assert f.attrs['complete']
    metadata = json.loads(f.attrs['metadata'])
    layout = metadata['task_scene']['layout']
    assert layout['content']['cup_count'] == plan['expected_cup_count']
    frames = len(f['action/position'])
    duration = float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
previous = next(row for row in json.loads((base/'packing_order_summary_v1.json').read_text())['runs']
                if row['seed'] == plan['seed'])
assert Path(previous['source']) == ROOT/plan['previous_source']
assert sha256(previous['source']) == previous['source_sha256']
initial_plan = json.loads((base/'tcp_states_plan_v1.json').read_text())
result = {
    'training_ready': False, 'seed': item['seed'], 'role': plan['role'],
    'source': str(source), 'source_sha256': digest, 'source_bytes': source.stat().st_size,
    'plan_sha256': sha256(base/'tcp_three_cup_plan_v1.json'),
    'analysis_sha256': sha256(__file__),
    'unchanged_frozen_files': len(identity['files']), 'changed_frozen_files': changed,
    'same_input_hashes_as_preceding_cohort': True,
    'original_policy_unchanged': sha256(ROOT/'configs/collection/quality_policy.yaml') == initial_plan['old_policy_sha256'],
    'cup_count': layout['content']['cup_count'], 'summary': report['summary'], 'error': item['error'],
    'audit_passed': audit['passed'], 'violations': audit['violations'],
    'acceptance_policy': audit['acceptance_policy'], 'frames': frames, 'duration_s': duration,
    'previous_duration_s': previous['duration_s'], 'duration_change_s': duration-previous['duration_s'],
    'tcp_intervals': tcp, 'native_integrity': native['integrity_status'],
    'native_violations': native['violations'], 'phase_wall_seconds': item['phase_wall_seconds'],
    'limitations': [
        'Known development regression, not an unseen success-rate estimate.',
        'Original 300 s duration and all previous quality limits retained.',
        'Native integration-boundary summaries, not continuous-time hardware certification.',
        'States only, no RGB training acceptance, LeRobot export, or qualified-throughput claim.',
    ],
}
atomic_json(output, result)
print(json.dumps({k: result[k] for k in ('seed', 'cup_count', 'summary', 'audit_passed', 'violations', 'frames', 'duration_s', 'duration_change_s')}))
