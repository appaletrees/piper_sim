"""All outcomes for collision-surface support; never grants training acceptance."""
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT


def main():
    out = ROOT/'outputs/platform_audit_v1';root = out/'collision_support_states_v2'
    destination = out/'collision_support_summary_v2.json'
    if destination.exists(): raise FileExistsError(destination)
    batch = json.loads((root/'summary.json').read_text())
    identity = json.loads((root/'run.json').read_text())
    changed = [n for n,d in identity['files'].items() if not Path(n).is_file() or sha256(n)!=d]
    if changed: raise ValueError('Frozen inputs changed: '+str(changed))
    assert len(batch['episodes']) == 3
    previous = json.loads((out/'placement_search_summary_v1.json').read_text())
    rows = []
    for item in batch['episodes']:
        source = Path(item['path']);folder = source.parent
        audit = json.loads((folder/'audit.json').read_text());report = json.loads((folder/'report.json').read_text())
        native = json.loads((out/f'collision_support_native_seed{item["seed"]}_v2.json').read_text())
        assert audit['source_sha256']==native['source_sha256']==sha256(source)
        assert not audit['training_ready'] and not native['violations']
        old = next(r for r in previous['runs'] if r['seed']==item['seed'])
        with h5py.File(source,'r') as f, h5py.File(old['source'],'r') as before:
            assert f.attrs['complete']
            frames = len(f['observation/sim_time'])
            duration = float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
            full_equality = {key: bool(np.array_equal(f[key][:],before[key][:])) for key in
                             ('observation/sim_time','observation/state','next_observation/state','action/position')}
            event = next((e for e in report['events'] if e['phase']=='placement_search_selected' and e['object']=='chopstick_a'),None)
            prefix = None
            if event:
                count = min(int(np.searchsorted(f['observation/sim_time'][:], event['sim_time'])),len(before['observation/sim_time']))
                equality = {key: bool(np.array_equal(f[key][:count],before[key][:count])) for key in
                            ('observation/sim_time','observation/state','action/position')}
                same_objects = True
                for start in range(0,count,128):
                    stop = min(count,start+128)
                    for a,b in zip(f['observation/extras_json'][start:stop],before['observation/extras_json'][start:stop],strict=True):
                        same_objects &= json.loads(a)['objects']==json.loads(b)['objects']
                prefix = {'transitions_before_first_slender_placement':count,'exact_dataset_equality':equality,
                          'exact_object_poses_equal':same_objects,
                          'previous_source_sha256':old['source_sha256']}
        rows.append({'seed':item['seed'],'source':str(source),'source_sha256':audit['source_sha256'],
            'summary':report['summary'],'error':item['error'],'violations':audit['violations'],
            'audit_passed':audit['passed'],'training_ready':False,'acceptance_policy':audit.get('acceptance_policy'),
            'frames':frames,'duration_s':duration,'previous_duration_s':old['duration_s'],
            'full_numeric_dataset_equality_to_previous':full_equality,
            'placement_events':[e for e in report['events'] if e['phase']=='placement_search_selected'],
            'native_integrity':native['integrity_status'],'prefix_comparison':prefix,
            'phase_wall_seconds':item['phase_wall_seconds']})
    interrupted = json.loads((out/'collision_support_states_v1/invocation_001.json').read_text())
    benchmark = json.loads((out/'support_query_benchmark_v3.json').read_text())
    result = {'status':'completed_engineering_state_regressions','training_ready':False,
        'analysis_sha256':sha256(__file__),'unchanged_frozen_input_files':len(identity['files']),
        'changed_input_files':changed,'diagnostic_passed':sum(r['audit_passed'] for r in rows),
        'accepted_rgb_episodes':0,'runs':rows,'interrupted_development_attempt':interrupted,
        'interruption_reason':'A new tolerance-expanded corner test found an AABB epsilon bound too narrow. Owned workers stopped; fixed bound and restarted into a new directory. Partial records retained, not counted as completed or accepted.',
        'query_benchmark':benchmark,
        'limitations':['Three development seeds, not independent held-out reliability evidence.',
            'State-only records cannot pass RGB training acceptance; no new LeRobot dataset exported.',
            'Collision support samples improve geometric coverage but do not certify dynamic stability.',
            'Original geometry, physics parameters and independent acceptance policy unchanged.',
            'Local query acceleration is not qualified data throughput; shared-load full-task wall times are not a matched benchmark.',
            'The isolated seed15619 snapshot trial remained unstable and is not a full-task verdict.']}
    atomic_json(destination,result)
    print(json.dumps({'diagnostic_passed':result['diagnostic_passed'],'runs':[{k:r[k] for k in ('seed','frames','duration_s','audit_passed','error')} for r in rows]}))


if __name__=='__main__':main()
