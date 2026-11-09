"""Summarize frozen bounded-path regressions without granting RGB acceptance."""
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT


def main():
    out = ROOT/'outputs/platform_audit_v1'
    root = out/'bounded_loaded_states_v1'
    destination = out/'bounded_loaded_summary_v1.json'
    if destination.exists(): raise FileExistsError(destination)
    manifest = json.loads((root/'summary.json').read_text())
    identity = json.loads((root/'run.json').read_text())
    changed = [name for name, digest in identity['files'].items()
               if not Path(name).is_file() or sha256(name) != digest]
    if changed: raise ValueError('Inputs changed during validation: '+str(changed))
    rows = []
    for item in manifest['episodes']:
        folder = Path(item['path']).parent
        audit = json.loads((folder/'audit.json').read_text())
        report = json.loads((folder/'report.json').read_text())
        native = json.loads((out/f'bounded_loaded_native_seed{item["seed"]}_v1.json').read_text())
        assert audit['source_sha256'] == sha256(folder/'episode.h5')
        assert audit['passed'] and not audit['training_ready']
        assert not native['violations']
        rows.append({'seed': item['seed'], 'source': str(folder/'episode.h5'),
            'source_sha256': audit['source_sha256'], 'summary': report['summary'],
            'audit_passed': audit['passed'], 'training_ready': audit['training_ready'],
            'acceptance_policy': audit['acceptance_policy'],
            'frames': audit['checks']['streams']['frames'],
            'duration_s': audit['checks']['streams']['duration_s'],
            'trajectory_sha256': audit['checks']['streams']['trajectory_sha256'],
            'objects_sha256': audit['checks']['task']['objects_sha256'],
            'motion': audit['checks']['task']['motion'],
            'bounded_path_events': [e for e in report['events'] if e['phase'] == 'bounded_path_selected'],
            'native_integrity': native['integrity_status'],
            'phase_wall_seconds': item['phase_wall_seconds']})
    assert len({r['acceptance_policy']['sha256'] for r in rows}) == 1
    old = out/'servo_full_states_v1/seed_0'
    old_audit = json.loads((old/'audit.json').read_text())
    new = next(r for r in rows if r['seed'] == 0)
    assert old_audit['source_sha256'] == sha256(old/'episode.h5')
    matches = {}
    for section, key in [('streams', 'trajectory_sha256'), ('task', 'objects_sha256'),
                         ('streams', 'frames'), ('streams', 'duration_s')]:
        matches[key] = new[key] == old_audit['checks'][section][key]
    snapshots = 0
    with h5py.File(old/'episode.h5', 'r') as a, h5py.File(new['source'], 'r') as b:
        for prefix in ('observation', 'next_observation'):
            x, y = a[prefix+'/extras_json'], b[prefix+'/extras_json']
            assert len(x) == len(y)
            for first in range(0, len(x), 128):
                for p, q in zip(x[first:first+128], y[first:first+128], strict=True):
                    assert json.loads(p)['teacher']['physics'] == json.loads(q)['teacher']['physics']
                    snapshots += 1
    assert all(matches.values())
    geometry = json.loads((out/'bounded_loaded_geometry_seed9137_v1.json').read_text())
    result = {'status': 'full_state_regressions_passed', 'training_ready': False,
        'analysis_sha256': sha256(__file__), 'seed_role': 'development regressions, not held-out evaluation',
        'unchanged_frozen_input_files': len(identity['files']), 'changed_input_files': changed,
        'runs': rows, 'seed0_matches_previous': matches,
        'seed0_equal_float64_physics_snapshots': snapshots,
        'seed0_previous_source_sha256': old_audit['source_sha256'],
        'seed9137_geometry': {k: geometry[k] for k in ('samples', 'native_arm_samples',
            'equal_jaw_reconstruction_arm_samples', 'changed_or_missing_provenance_files', 'category_peaks', 'tcp')},
        'limitations': [
            'No RGB in these two runs, no LeRobot export, no training acceptance.',
            'Only two engineering seeds; placement failures 13 and 15619 remain unresolved.',
            'Fidelity errors in teacher events are online sampled planning evidence, not measured hardware accuracy.',
            'Native interval summaries are consistency-checked, not an independently stored full substep trace.',
            'State-run wall times under shared load are not a qualified throughput benchmark.',
            'No quality threshold, geometry, contact profile or nominal scene dimension changed.',
        ]}
    atomic_json(destination, result)
    print(json.dumps({'runs': [{k: r[k] for k in ('seed', 'summary', 'frames', 'duration_s', 'audit_passed')} for r in rows],
                      'seed0_matches_previous': matches, 'equal_native_snapshots': snapshots}, indent=2))


if __name__ == '__main__': main()
