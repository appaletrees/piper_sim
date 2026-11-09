"""Summarize all frozen placement trials, including the rejected episode."""
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT


def main():
    out = ROOT/'outputs/platform_audit_v1'
    root = out/'placement_search_states_v1'
    destination = out/'placement_search_summary_v1.json'
    if destination.exists(): raise FileExistsError(destination)
    manifest = json.loads((root/'summary.json').read_text())
    identity = json.loads((root/'run.json').read_text())
    changed = [name for name, digest in identity['files'].items()
               if not Path(name).is_file() or sha256(name) != digest]
    if changed: raise ValueError('Inputs changed during validation: '+str(changed))
    rows = []
    for item in manifest['episodes']:
        source = Path(item['path'])
        audit = json.loads((source.parent/'audit.json').read_text())
        report = json.loads((source.parent/'report.json').read_text())
        native = json.loads((out/f'placement_search_native_seed{item["seed"]}_v1.json').read_text())
        assert audit['source_sha256'] == native['source_sha256'] == sha256(source)
        assert not audit['training_ready'] and not native['violations']
        with h5py.File(source, 'r') as f:
            assert f.attrs['complete']
            frames = len(f['observation/sim_time'])
            duration = float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
        rows.append({'seed': item['seed'], 'source': str(source), 'source_sha256': audit['source_sha256'],
            'summary': report['summary'], 'error': item['error'], 'violations': audit['violations'],
            'audit_passed': audit['passed'], 'training_ready': False,
            'acceptance_policy': audit.get('acceptance_policy'), 'frames': frames, 'duration_s': duration,
            'placement_events': [e for e in report['events'] if e['phase']=='placement_search_selected'],
            'native_integrity': native['integrity_status'],
            'native_snapshot_tcp_speed_peak_m_s': {side: arm['peaks_all_observation_snapshots']['tcp_linear_speed_m_s']['absolute_max']
                                                 for side, arm in native['arms'].items()},
            'phase_wall_seconds': item['phase_wall_seconds']})
    # Unsuccessful episodes fail closed before the auditor loads policy fields.
    assert len({r['acceptance_policy']['sha256'] for r in rows if r['acceptance_policy']}) == 1
    old = json.loads((out/'bounded_loaded_summary_v1.json').read_text())
    prior = next(r for r in old['runs'] if r['seed']==0)
    current = next(r for r in rows if r['seed']==0)
    geometry = json.loads((out/'placement_search_geometry_seed13_v1.json').read_text())
    stability = json.loads((out/'placement_search_stability15619_v1.json').read_text())
    previews = {name: json.loads((out/name/'preview.json').read_text()) for name in
                ('placement_preview_original13_v1', 'placement_preview_chop13_v1', 'placement_preview_dish15619_v1')}
    result = {'status': 'two_state_regressions_passed_one_rejected', 'training_ready': False,
        'analysis_sha256': sha256(__file__), 'seed_role': 'development regressions, not held-out evaluation',
        'unchanged_frozen_input_files': len(identity['files']), 'changed_input_files': changed,
        'runs': rows, 'diagnostic_passed': sum(r['audit_passed'] for r in rows), 'accepted_rgb_episodes': 0,
        'seed0_duration_increase_s': current['duration_s']-prior['duration_s'],
        'seed15619_stability': stability,
        'seed13_geometry': {k: geometry[k] for k in ('samples','native_arm_samples','equal_jaw_reconstruction_arm_samples',
            'changed_or_missing_provenance_files','category_peaks','tcp')}, 'previews': previews,
        'limitations': [
            'No RGB in these three runs, no LeRobot export and no training acceptance.',
            'Seed 15619 remains rejected despite all ten objects being contained: chopstick_b keeps rolling.',
            'The original independent stability thresholds and final-settle timeout were not relaxed.',
            'First-contact estimates sample visible payload vertices against collision support triangles; this can miss middle-face support.',
            'Opening sweep and movement checks are sampled geometry checks, not continuous-time dynamic stability proofs.',
            'State-run wall times under shared load are not a qualified throughput benchmark.',
            'The support-query broad-phase benchmark is a separate prototype and does not measure end-to-end speed.',
            'The 0.25 m/s Cartesian pacing setting is not a global measured TCP velocity limit.',
            'Early isolated v1/v2/v3 diagnostics did not record all expert source hashes and are not frozen-source full-task evidence.',
            'No geometry, contact profile, nominal dimension, quality threshold or default collection profile changed.',
        ]}
    atomic_json(destination, result)
    print(json.dumps({k: result[k] for k in ('status','unchanged_frozen_input_files','diagnostic_passed','accepted_rgb_episodes','seed0_duration_increase_s')}))


if __name__=='__main__': main()
