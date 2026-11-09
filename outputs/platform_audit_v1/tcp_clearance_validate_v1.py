"""Bind every regression result to retained sources and unchanged frozen inputs."""
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.dataset import sequence_windows

base = Path(__file__).resolve().parent
output = base/'tcp_clearance_validation_v1.json'
if output.exists(): raise FileExistsError(output)
identity = json.loads((base/'tcp_clearance_inputs_v1.json').read_text())
summary = json.loads((base/'tcp_clearance_summary_v1.json').read_text())
plan = json.loads((base/'tcp_clearance_plan_v1.json').read_text())
assert summary['state_audit_passed'] == 3 and summary['accepted_rgb_episodes'] == 0
changed = [p for p, d in identity['files'].items() if not Path(p).is_file() or sha256(p) != d]
assert not changed
rows = []
for run in summary['runs']:
    source = Path(run['source']); digest = sha256(source)
    assert digest == run['source_sha256']
    motion_path = base/f'tcp_clearance_motion_seed{run["seed"]}_v1.json'
    motion = json.loads(motion_path.read_text())
    assert motion['passed'] and Path(motion['source']) == source
    assert motion['acceptance_policy'] == summary['acceptance_policy']
    assert run['tcp']['passed'] and run['tcp']['source_sha256'] == digest
    assert run['native_integrity'] == 'consistent_recorded_evidence'
    assert run['native_controller_intervals'] == 'consistent_interval_summaries'
    try:
        next(sequence_windows(source, policy=plan['acceptance_policy']))
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError('A state-only trial entered the strict training reader')
    with h5py.File(source, 'r') as f:
        assert f.attrs['complete'] and not f.attrs['training_ready']
        assert not list(f.get('observation/images', {}))
        meta = json.loads(f.attrs['metadata'])
        assert meta['teacher']['grasp_approach_clearance']['margin_m'] == .001
    geometry = run['geometry']
    assert geometry['samples'] == run['frames']+1
    assert geometry['native_arm_samples'] == 2*(run['frames']+1)
    assert not any(k in geometry['category_peaks'] for k in ('same_arm', 'inter_arm'))
    assert sha256(source) == digest
    rows.append({'seed': run['seed'], 'source_sha256': digest,
        'full_motion_report_sha256': sha256(motion_path), 'independent_motion_passed': motion['passed'],
        'independent_gripper_acceleration_m_s2': motion['metrics']['measured_gripper_acceleration'],
        'independent_gripper_jerk_m_s3': motion['metrics']['measured_gripper_jerk'],
        'strict_training_reader_rejection': rejected, 'source_unchanged': True})
old = json.loads((base/'tcp_holdout_summary_v2.json').read_text())
assert all(sha256(run['source']) == run['source_sha256'] for run in old['runs'])
test = json.loads((base/'tcp_clearance_tests_v1.json').read_text())
assert all(sha256(path) == digest for path, digest in test['files'].items())
gap_source = base/'tcp_grasp_distance_query_diagnostic_v1.json'
gap = json.loads((base/'tcp_grasp_separating_axis_check_v1.json').read_text())
vertices = list(map(np.asarray, json.loads(gap_source.read_text())['world_convex_vertices']))
axis = np.asarray(gap['axis']); assert np.isclose(np.linalg.norm(axis), 1.)
ap, bp = [v@axis for v in vertices]
assert np.isclose(max(ap.min()-bp.max(), bp.min()-ap.max()), gap['gap_m'], rtol=0, atol=1e-12)
assert gap['gap_m'] > .001
result = {'status': 'verified_known_state_regressions', 'training_ready': False,
    'script_sha256': sha256(__file__), 'summary_sha256': sha256(base/'tcp_clearance_summary_v1.json'),
    'unchanged_frozen_files': len(identity['files']), 'changed_files': changed,
    'runs': rows, 'prior_six_sources_preserved': True, 'tests': test['result'],
    'raw_total_bytes': sum(run['source_bytes'] for run in summary['runs']),
    'transitions': sum(run['frames'] for run in summary['runs']),
    'geometry_poses': sum(run['geometry']['samples'] for run in summary['runs']),
    'native_arm_poses': sum(run['geometry']['native_arm_samples'] for run in summary['runs']),
    'native_integration_steps': sum(run['tcp']['integration_steps'] for run in summary['runs']),
    'distance_diagnostic_source_sha256': sha256(gap_source),
    'independent_separating_axis_gap_m': gap['gap_m'], 'accepted_rgb_episodes': 0,
    'limitations': plan['limitations'], 'raw_storage': plan['raw_storage']}
atomic_json(output, result)
print(json.dumps(result, indent=2))
