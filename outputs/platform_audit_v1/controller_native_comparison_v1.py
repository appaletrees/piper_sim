"""Reproduce the read-only, seed-0 full-task native-controller comparison.

Run with PYTHONPATH=src .venv-mujoco/bin/python <this file> from the repository.
The quantiles below describe per-control-interval maxima, not raw substep samples.
No result here grants training acceptance or identifies a real-device limit.
"""
import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT

ROOT = Path(ROOT)
OUT = ROOT/'outputs/platform_audit_v1'
SOURCES = {
    'original': OUT/'qualified_pair_v9/workers_1/seed_0',
    'observed': OUT/'observed_baseline_seed0_v1',
    'candidate': OUT/'servo_full_states_v1/seed_0',
}


def run():
    destination = OUT/'controller_native_comparison_v1.json'
    if destination.exists():
        raise ValueError('Retain the existing comparison; choose a new experiment version')
    spec = importlib.util.spec_from_file_location('native_audit', ROOT/'scripts/audit_recorded_dynamics.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    evidence = {}; records = {}; metadata = {}; audits = {}
    for label, folder in SOURCES.items():
        path = folder/'episode.h5'
        a = json.loads((folder/'audit.json').read_text())
        assert a['passed'] and a['source_sha256'] == sha256(path), label
        audits[label] = a
        with h5py.File(path, 'r') as f:
            assert f.attrs['complete']
            metadata[label] = json.loads(f.attrs['metadata'])
            records[label] = {p: [json.loads(x) for x in f[p+'/extras_json'][:]]
                              for p in ('observation', 'next_observation')}
        evidence[label] = {'path': str(path), 'sha256': a['source_sha256'],
            'frames': a['checks']['streams']['frames'],
            'duration_s': a['checks']['streams']['duration_s'],
            'audit_passed': a['passed'], 'training_ready': a['training_ready'],
            'metrics_30hz': a['checks']['task']['motion']['metrics']}
    policy = {a['acceptance_policy']['sha256'] for a in audits.values()}
    assert len(policy) == 1
    # Equal geometry, cameras, task randomization and teacher settings are required.
    original_config = copy.deepcopy(metadata['original']['config'])
    original_config.pop('_config_path')
    def resolved_domain_path(value):
        value = copy.deepcopy(value)
        value['domain_randomization'] = str((ROOT/value['domain_randomization']).resolve())
        return value
    original_config['task_scene'] = resolved_domain_path(original_config['task_scene'])
    for label in ('observed', 'candidate'):
        config = copy.deepcopy(metadata[label]['config'])
        config.pop('_config_path')
        config['task_scene'] = resolved_domain_path(config['task_scene'])
        config.get('mujoco', {}).pop('controller', None)
        if not config.get('mujoco'): config.pop('mujoco', None)
        assert config == original_config, label
        for key in ('camera_rig', 'task_scene', 'teacher', 'collection_initial_target', 'fps'):
            a, b = [copy.deepcopy(metadata[x][key]) for x in (label, 'original')]
            if key == 'task_scene':
                a['config'] = resolved_domain_path(a['config'])
                b['config'] = resolved_domain_path(b['config'])
            assert a == b, (label, key)
    unchanged = {}
    for section, fields in {'streams': ('trajectory_sha256', 'frames', 'duration_s'),
                            'task': ('objects_sha256',)}.items():
        for field in fields:
            unchanged[field] = audits['original']['checks'][section][field] == audits['observed']['checks'][section][field]
    for field in ('physics',):
        unchanged['all_float64_'+field+'_snapshots'] = all(
            a['teacher'][field] == b['teacher'][field]
            for prefix in records['original']
            for a, b in zip(records['original'][prefix], records['observed'][prefix], strict=True))
    assert all(unchanged.values()), unchanged
    groups = {'arm': [*range(6), *range(7, 13)], 'gripper': [6, 13]}
    for label in ('observed', 'candidate'):
        native = audit.analyze(SOURCES[label]/'episode.h5')
        assert not native['violations'] and native['integrity_status'] == 'consistent_recorded_evidence'
        evidence[label]['native_audit'] = native
        rows = records[label]['next_observation']
        phase = np.array([r['teacher']['phase'] for r in rows])
        measures = {key: np.array([r['controller_interval'][key] for r in rows])
                    for key in ('native_acceleration_abs_max', 'native_jerk_abs_max')}
        measures['native_velocity_abs_max'] = np.array([
            np.maximum(np.abs(r['controller_interval']['native_velocity_min']),
                       np.abs(r['controller_interval']['native_velocity_max'])) for r in rows])
        summaries = {}
        for name, matrix in measures.items():
            summaries[name] = {}
            for group, columns in groups.items():
                peaks = matrix[:, columns].max(axis=1)
                summaries[name][group] = {}
                for tag in ('ALL', *sorted(set(phase))):
                    selected = np.ones(len(rows), bool) if tag == 'ALL' else phase == tag
                    ix = np.flatnonzero(selected)[np.argmax(peaks[selected])]
                    summaries[name][group][tag] = {
                        'interval_count': int(selected.sum()),
                        'interval_peak_quantiles_p50_p95_p99_max': np.quantile(peaks[selected], [.5, .95, .99, 1]).tolist(),
                        'peak_transition': int(ix), 'peak_object': rows[ix]['teacher']['object'],
                        'peak_phase': rows[ix]['teacher']['phase'],
                        'peak_interval_end_sim_time': rows[ix]['controller_interval']['end_sim_time']}
        evidence[label]['native_interval_distributions'] = summaries
    result = {'status': 'comparison_completed', 'seed': 0, 'training_ready': False,
        'analysis_sha256': sha256(__file__), 'native_auditor_sha256': sha256(spec.origin),
        'acceptance_policy_sha256': next(iter(policy)), 'observer_unchanged': unchanged,
        'runs': evidence, 'limitations': [
            'One development seed; no distribution-wide or hardware validity claim.',
            'Observer matches the original full native snapshots and trajectory, not merely task success.',
            'Candidate has different closed-loop actions, durations and contact timing; it is not row-aligned replay.',
            'Quantiles are over control-interval maxima, not all 960 Hz samples.',
            'Native interval summaries are consistency-checked; full substep traces are not independently recomputed.',
            'State-only runs are not training-ready RGB demonstrations.',
            'No physical threshold, actuator gain, object state or source recording is changed by this analysis.',
        ]}
    atomic_json(destination, result)
    print(json.dumps({'status': result['status'], 'observer_unchanged': unchanged,
        'native_overall': {label: {metric: {g: x['ALL']['interval_peak_quantiles_p50_p95_p99_max']
            for g,x in values.items()} for metric,values in evidence[label]['native_interval_distributions'].items()}
            for label in ('observed', 'candidate')}}, indent=2))


if __name__ == '__main__':
    run()
