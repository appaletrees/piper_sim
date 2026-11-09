"""Summarize frozen storage regression and predeclared unseen state episodes."""
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT


def compare_recordings(old, new):
    clocks = {'observation/timestamp', 'next_observation/timestamp', 'action/timestamp'}
    result = {}
    with h5py.File(old, 'r') as before, h5py.File(new, 'r') as after:
        names = []
        before.visititems(lambda n, x: names.append(n) if isinstance(x, h5py.Dataset) else None)
        for name in names:
            a, b = before[name], after[name]
            same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
            equal = same_shape and same_dtype
            if equal:
                for start in range(0, len(a), 128):
                    x, y = a[start:start+128], b[start:start+128]
                    equal &= bool(np.array_equal(x, y) if x.dtype.kind == 'O' else x.tobytes() == y.tobytes())
            result[name] = {'shape_equal': same_shape, 'dtype_equal': same_dtype, 'values_equal': bool(equal),
                'clock_role': 'host_monotonic: expected to differ between physical reruns' if name in clocks else 'deterministic simulated data'}
    return {'source': str(old), 'source_sha256': sha256(old), 'destination_sha256': sha256(new),
        'datasets': result, 'all_simulated_fields_byte_equal': all(v['values_equal'] for k, v in result.items() if k not in clocks),
        'source_bytes': old.stat().st_size, 'output_bytes': new.stat().st_size,
        'storage_ratio': new.stat().st_size/old.stat().st_size}


def main():
    base = ROOT/'outputs/platform_audit_v1'; root = base/'json_compact_states_v1'
    output = base/'json_compact_summary_v1.json'
    if output.exists(): raise FileExistsError(output)
    plan = json.loads((base/'json_compact_states_plan_v1.json').read_text())
    identity = json.loads((root/'run.json').read_text()); batch = json.loads((root/'summary.json').read_text())
    changed = [name for name, digest in identity['files'].items() if not Path(name).is_file() or sha256(name) != digest]
    if changed: raise ValueError('Frozen input changed: '+str(changed))
    assert [r['seed'] for r in batch['episodes']] == plan['seeds']
    old_profile = yaml.safe_load((ROOT/'configs/collection/clean_table_packing_order.yaml').read_text())
    new_profile = yaml.safe_load((ROOT/plan['collection_profile']).read_text())
    assert new_profile['dataset'].pop('extras_storage') == 'compressed'
    assert new_profile == old_profile
    rows = []
    for item in batch['episodes']:
        source = Path(item['path']); folder = source.parent
        report = json.loads((folder/'report.json').read_text()); audit = json.loads((folder/'audit.json').read_text())
        native = json.loads((base/f'json_compact_native_seed{item["seed"]}_v1.json').read_text())
        digest = sha256(source)
        assert audit['source_sha256'] == native['source_sha256'] == digest
        assert not item['accepted'] and not item['training_ready'] and not audit['training_ready']
        with h5py.File(source, 'r') as f:
            assert f.attrs['complete'] and f.attrs['extras_storage'] == 'compressed_utf8_buckets_v1'
            frames = len(f['observation/sim_time'])
            duration = float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
        rows.append({'seed': item['seed'], 'role': plan['roles'][str(item['seed'])], 'split': item['split'],
            'source': str(source), 'source_sha256': digest, 'frames': frames, 'duration_s': duration,
            'bytes': source.stat().st_size, 'summary': report['summary'], 'error': item['error'],
            'audit_passed': audit['passed'], 'violations': audit['violations'], 'training_ready': False,
            'acceptance_policy': audit['acceptance_policy'], 'native_integrity': native['integrity_status'],
            'native_violations': native['violations'],
            'native_interval_integrity': native.get('controller_intervals', {}).get('integrity_status'),
            'phase_wall_seconds': item['phase_wall_seconds']})
    old = ROOT/'outputs/platform_audit_v1/packing_order_states_v1/worker_01/episode_000000_seed_15619/episode.h5'
    new = Path(next(r['source'] for r in rows if r['seed'] == 15619))
    comparison = compare_recordings(old, new)
    geometry = {}
    for seed in plan['seeds'][1:]:
        path = base/f'json_compact_geometry_seed{seed}_v1.json'
        evidence = json.loads(path.read_text())
        row = next(r for r in rows if r['seed'] == seed)
        assert Path(evidence['source']) == Path(row['source'])
        assert evidence['samples'] == row['frames']+1
        assert not evidence['changed_or_missing_provenance_files']
        assert evidence['equal_jaw_reconstruction_arm_samples'] == 0
        assert sha256(row['source']) == row['source_sha256']
        geometry[str(seed)] = {k: evidence[k] for k in ('samples', 'native_arm_samples',
            'equal_jaw_reconstruction_arm_samples', 'changed_or_missing_provenance_files',
            'category_peaks', 'tcp', 'limitations')}
        geometry[str(seed)].update(report_sha256=sha256(path), source_sha256=row['source_sha256'])
    result = {'status': 'complete_frozen_state_cohort', 'training_ready': False, 'analysis_sha256': sha256(__file__),
        'plan_sha256': sha256(base/'json_compact_states_plan_v1.json'), 'runs': rows,
        'unchanged_input_files': len(identity['files']), 'changed_input_files': changed,
        'profile_only_change': {'dataset.extras_storage': 'compressed'}, 'known_regression': comparison,
        'unseen_diagnostic_passed': sum(r['audit_passed'] for r in rows if r['seed'] != 15619),
        'unseen_attempted': 2, 'accepted_rgb_episodes': 0,
        'unseen_geometry': geometry,
        'benchmarks': {kind: json.loads((base/f'json_writer_{kind}_benchmark_v1.json').read_text()) for kind in ('state', 'rgb')},
        'limitations': ['Two previously untested state-only seeds do not estimate population success or RGB training acceptance.',
            'Original production policy unchanged; all successes and failures are retained.',
            'Short-window write measurements do not measure qualified task throughput; no pipeline speedup claim.',
            'Host timestamps correctly differ between new runs; deterministic simulated data are compared separately.',
            'Telemetry compression does not solve the storage required for complete three-camera RGB recordings.']}
    atomic_json(output, result)
    print(json.dumps({'runs': [{k: r[k] for k in ('seed', 'frames', 'duration_s', 'audit_passed', 'error')} for r in rows],
        'known_regression_simulated_fields_equal': comparison['all_simulated_fields_byte_equal'],
        'known_regression_storage_ratio': comparison['storage_ratio']}))


if __name__ == '__main__': main()
