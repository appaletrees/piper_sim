"""Preserve every frozen TCP holdout outcome and independently inspect states."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import time

import h5py

from piper_sim.collection_audit import load_acceptance_policy
from piper_sim.collection_jobs import atomic_json, sha256, validate_completed
from piper_sim.config import ROOT
from piper_sim.tcp_audit import audit_tcp_motion

base = ROOT/'outputs/platform_audit_v1'
plan_path = base/'tcp_holdout_plan_v1.json'
plan = json.loads(plan_path.read_text())
root = Path(plan['raw_output'])
result_path = base/'tcp_holdout_summary_v1.json'
evidence_dir = Path(plan['persistent_evidence'])
if result_path.exists() or evidence_dir.exists():
    raise FileExistsError('Preserve preceding evidence')
identity_path = base/'tcp_holdout_inputs_v1.json'
identity = json.loads(identity_path.read_text())
assert sha256(identity_path) == plan['inputs_sha256']
assert identity == json.loads((root/'run.json').read_text())
invocation = json.loads((root/'invocation_001.json').read_text())
assert invocation['status'] == 'completed'
batch = json.loads((root/'summary.json').read_text())
assert [r['seed'] for r in batch['episodes']] == plan['seeds']
def changed_files():
    return [p for p, digest in identity['files'].items() if not Path(p).is_file() or sha256(p) != digest]
assert not changed_files()
assert sha256(ROOT/'configs/collection/quality_policy.yaml') == plan['original_policy_sha256']
settings, policy = load_acceptance_policy(plan['acceptance_policy'])
spec = importlib.util.spec_from_file_location('native_diagnostics', ROOT/'scripts/audit_recorded_dynamics.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
cadence_spec = importlib.util.spec_from_file_location('recorded_cadence', ROOT/'scripts/compare_collection_motion.py')
cadence_module = importlib.util.module_from_spec(cadence_spec)
cadence_spec.loader.exec_module(cadence_module)
evidence_dir.mkdir()
# The durable evidence is small; raw state HDF5 files remain on tmpfs and are
# explicitly not a permanent release. Preserve all rejected/unverified rows too.
copied = {}
for source in sorted(root.rglob('*')):
    if source.is_file() and source.suffix in ('.json', '.jsonl', '.log'):
        target = evidence_dir/source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha256(source) == sha256(target)
        copied[str(target.relative_to(evidence_dir))] = sha256(target)
rows = []
for item in batch['episodes']:
    started = time.perf_counter()
    source = Path(item['path'])
    row = {k: item.get(k) for k in ('seed','split','category','error','violations','returncode','inventory','phase_wall_seconds')}
    row.update(training_ready=False, source=str(source), raw_storage='volatile_tmpfs_pending_archival')
    assert not item['accepted'] and not item['training_ready']
    report = json.loads(Path(item['report']).read_text()) if Path(item['report']).is_file() else {}
    audit = json.loads(Path(item['audit']).read_text()) if Path(item['audit']).is_file() else {}
    row.update(summary=report.get('summary'), audit_passed=audit.get('passed',False),
               rollout_accepted=report.get('rollout_accepted'), quality_passed=report.get('quality',{}).get('passed'),
               quality_violations=report.get('quality',{}).get('violations'),
               last_events=report.get('events',[])[-8:], results=report.get('results',[]))
    if not source.is_file() or source.suffix != '.h5':
        row['diagnostic_error'] = 'Missing closed HDF5; raw outcome retained without further acceptance'
        rows.append(row)
        continue
    digest = sha256(source)
    row.update(source_sha256=digest, source_bytes=source.stat().st_size)
    assert audit.get('source_sha256') == digest and not audit.get('training_ready')
    assert audit.get('acceptance_policy',{}).get('sha256') == policy['sha256']
    validate_completed(item, policy=plan['acceptance_policy'])
    with h5py.File(source, 'r') as f:
        assert f.attrs['complete'] and not f.attrs.get('training_ready')
        assert not list(f.get('observation/images',{}))
        meta = json.loads(f.attrs['metadata'])
        n = len(f['action/position'])
        row.update(frames=n, duration_s=float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0]),
                   cup_count=meta['task_scene']['layout']['content']['cup_count'],
                   per_target_contact_material_switching=meta['teacher'].get('per_target_contact_material_switching'),
                   task=meta['task'])
        tcp = audit_tcp_motion(f, meta, settings['tcp_motion'])
        tcp.update(source=str(source), source_sha256=digest, training_ready=False, acceptance_policy=policy)
        atomic_json(evidence_dir/f'seed_{item["seed"]}_tcp.json',tcp)
        row['tcp'] = tcp
        phases = Counter(); objects = Counter(); tail = []
        for index in range(n):
            extra = json.loads(f['next_observation/extras_json'][index])
            teacher = extra['teacher']
            phases[teacher.get('phase')] += 1
            objects[teacher.get('object')] += 1
            if index >= n-30:
                tail.append({'transition_index':index, 'sim_time':float(f['next_observation/sim_time'][index]),
                             **{key:teacher.get(key) for key in ('phase','object','holding','grasp','robot_penetration_m','robot_collision_pair','active_assist')}})
        row.update(phase_seconds={str(k):v/meta['fps'] for k,v in phases.items()},
                   object_seconds={str(k):v/meta['fps'] for k,v in objects.items()})
        atomic_json(evidence_dir/f'seed_{item["seed"]}_terminal_state.json',
                    {'source_sha256':digest,'training_ready':False,'last_transitions':tail})
    dynamics = native.analyze(source)
    dynamics['implementation_sha256'] = sha256(ROOT/'scripts/audit_recorded_dynamics.py')
    atomic_json(evidence_dir/f'seed_{item["seed"]}_native.json',dynamics)
    assert dynamics['source_sha256'] == digest
    row.update(native_integrity=dynamics['integrity_status'], native_violations=dynamics['violations'],
               native_controller_intervals=dynamics.get('controller_intervals',{}).get('integrity_status'),
               read_only_diagnostic_seconds=time.perf_counter()-started)
    cadence = cadence_module.cadence(source)
    cadence.update(source_sha256=digest, implementation_sha256=sha256(ROOT/'scripts/compare_collection_motion.py'),
                   training_ready=False)
    atomic_json(evidence_dir/f'seed_{item["seed"]}_cadence.json',cadence)
    row['cadence'] = {key:cadence[key] for key in ('stationary_fraction','stationary_bouts_at_least_250ms',
                                                 'longest_stationary_bout_s','definition')}
    geometry_path = base/f'tcp_holdout_geometry_seed{item["seed"]}_v1.json'
    geometry = json.loads(geometry_path.read_text())
    assert Path(geometry['source']) == source
    assert geometry['samples'] == n+1
    assert geometry['native_arm_samples'] == 2*(n+1)
    assert geometry['equal_jaw_reconstruction_arm_samples'] == 0
    assert not geometry['changed_or_missing_provenance_files']
    row['geometry'] = {key:geometry[key] for key in ('samples','native_arm_samples','category_peaks','limitations')}
    row['geometry']['report_sha256'] = sha256(geometry_path)
    assert sha256(source) == digest
    rows.append(row)
    print(json.dumps({k:row[k] for k in ('seed','cup_count','frames','duration_s','audit_passed','error')},ensure_ascii=False),flush=True)
assert not changed_files()
result = {'status':'completed_frozen_state_cohort', 'training_ready':False, 'runs':rows,
          'attempted':len(rows), 'state_audit_passed':sum(r['audit_passed'] for r in rows),
          'rejected_or_unverified':sum(not r['audit_passed'] for r in rows),
          'accepted_rgb_episodes':0, 'plan_sha256':sha256(plan_path),
          'analysis_sha256':sha256(__file__), 'inputs_sha256':sha256(identity_path),
          'unchanged_frozen_files':len(identity['files']), 'changed_input_files':[],
          'acceptance_policy':policy, 'raw_output':str(root),
          'raw_storage':'volatile tmpfs; all raw records retained, durable archival still required',
          'durable_reports':str(evidence_dir), 'copied_file_checksums':copied,
          'invocation':invocation,
          'limitations':['Six predeclared state-only trials; no broad population success or RGB training claim.',
                         'TCP measurements are native integration-boundary summaries, not continuous-time hardware proof.',
                         'Existing per-target pad material switching remains in this candidate and is not calibrated physical material.',
                         'Raw state recordings are on volatile tmpfs pending disk archival; no raw recording removed.',
                         'Shared-host state-only wall time is not qualified RGB production throughput.']}
atomic_json(result_path,result)
print(json.dumps({k:result[k] for k in ('status','attempted','state_audit_passed','rejected_or_unverified','accepted_rgb_episodes','unchanged_frozen_files')},indent=2),flush=True)
