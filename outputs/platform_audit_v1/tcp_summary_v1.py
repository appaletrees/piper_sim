"""Complete paired development regressions for strict native TCP budgets."""
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json,sha256
from piper_sim.config import ROOT

base=ROOT/'outputs/platform_audit_v1';root=base/'tcp_states_v1'
output=base/'tcp_summary_v1.json'
if output.exists():raise FileExistsError(output)
plan=json.loads((base/'tcp_states_plan_v1.json').read_text())
identity=json.loads((root/'run.json').read_text());batch=json.loads((root/'summary.json').read_text())
changed=[p for p,digest in identity['files'].items() if not Path(p).is_file() or sha256(p)!=digest]
if changed:raise ValueError('Frozen inputs changed: '+str(changed))
assert [r['seed'] for r in batch['episodes']]==plan['seeds']
prior=json.loads((base/'json_compact_summary_v1.json').read_text())
rows=[]
for item in batch['episodes']:
    folder=Path(item['path']).parent;source=folder/'episode.h5'
    report=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'audit.json').read_text())
    assert audit['source_sha256']==sha256(source)
    assert not audit['training_ready'] and not item['accepted']
    # Failures are retained, even if unsuccessful_episode prevents later checks.
    native=json.loads((base/f'tcp_native_seed{item["seed"]}_v1.json').read_text())
    assert native['source_sha256']==sha256(source)
    old=next(r for r in prior['runs'] if r['seed']==item['seed'])
    assert old['source_sha256']==sha256(old['source'])
    old_native=json.loads((base/f'json_compact_native_seed{item["seed"]}_v1.json').read_text())
    with h5py.File(source,'r') as f:
        frames=len(f['observation/state']);duration=float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
    strict=json.loads((base/f'tcp_interval_seed{item["seed"]}_v1.json').read_text())
    assert strict['source_sha256']==sha256(source)
    assert strict['acceptance_policy']['sha256']==plan['strict_policy_sha256']
    rows.append({'seed':item['seed'],'role':plan['roles'][str(item['seed'])],'source':str(source),
        'source_sha256':sha256(source),'summary':report['summary'],'error':item['error'],
        'audit_passed':audit['passed'],'violations':audit['violations'],'training_ready':False,
        'acceptance_policy':audit.get('acceptance_policy'),'frames':frames,'duration_s':duration,
        'previous_duration_s':old['duration_s'],'duration_change_s':duration-old['duration_s'],
        'source_bytes':source.stat().st_size,'tcp_intervals':strict,
        'native_integrity':native['integrity_status'],'native_violations':native['violations'],
        'previous_30hz_snapshot_speed_peaks':{side:{key:arm['peaks_all_observation_snapshots'][key] for key in
            ('tcp_linear_speed_m_s','tcp_angular_speed_rad_s')} for side,arm in old_native['arms'].items()},
        'phase_wall_seconds':item['phase_wall_seconds']})
resume=json.loads((root/'invocation_002.json').read_text())
assert resume['status']=='completed' and resume['resumed'] and resume['accepted_episodes_per_hour'] is None
assert len(list(root.glob('worker_*/*/episode.h5')))==len(rows)
geometry=json.loads((base/'tcp_geometry_seed15619_v1.json').read_text())
assert not geometry['changed_or_missing_provenance_files']
assert geometry['equal_jaw_reconstruction_arm_samples']==0
assert geometry['samples']==rows[0]['frames']+1
result={'training_ready':False,'runs':rows,'unchanged_frozen_files':len(identity['files']),
        'changed_frozen_files':changed,'plan_sha256':sha256(base/'tcp_states_plan_v1.json'),
        'analysis_sha256':sha256(__file__),'passed_state_episodes':sum(r['audit_passed'] for r in rows),
        'accepted_rgb_episodes':0,'original_policy_unchanged':sha256('configs/collection/quality_policy.yaml')==plan['old_policy_sha256'],
        'observer_benchmark':json.loads((base/'tcp_observer_benchmark_v1.json').read_text()),
        'resume':{'status':resume['status'],'new_episode_files':0,'source_hashes_still_match_closed_file_audits':True,
                  'accepted_episodes_per_hour':None,'invocation_sha256':sha256(root/'invocation_002.json')},
        'seed15619_geometry':{k:geometry[k] for k in ('samples','native_arm_samples',
            'equal_jaw_reconstruction_arm_samples','category_peaks','tcp')},
        'limitations':['Known development regressions, not new held-out success estimates.',
            'Native integration-boundary speed bounds are uncalibrated simulation budgets, not continuous-time hardware limits.',
            'Original motion/contact/image/duration gates are retained; no threshold relaxation.',
            'States only; no RGB training acceptance, no LeRobot export and no qualified-throughput claim.',
            'Old speed peaks are 30 Hz snapshots; the new observer covers every native integration boundary.']}
atomic_json(output,result)
print(json.dumps({'passed':result['passed_state_episodes'],'runs':[{k:r[k] for k in ('seed','frames','duration_s','duration_change_s','audit_passed','error')} for r in rows]}))
