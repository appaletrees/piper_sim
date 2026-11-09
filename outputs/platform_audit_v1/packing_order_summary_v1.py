"""Summarize a frozen complete packing-order state cohort, without RGB claims."""
import json
from pathlib import Path

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json,sha256
from piper_sim.config import ROOT


def main():
    out=ROOT/'outputs/platform_audit_v1';root=out/'packing_order_states_v1'
    destination=out/'packing_order_summary_v1.json'
    if destination.exists():raise FileExistsError(destination)
    batch=json.loads((root/'summary.json').read_text());identity=json.loads((root/'run.json').read_text())
    changed=[n for n,d in identity['files'].items() if not Path(n).is_file() or sha256(n)!=d]
    if changed:raise ValueError('Frozen inputs changed: '+str(changed))
    assert len(batch['episodes'])==3
    previous=json.loads((out/'collision_support_summary_v2.json').read_text())
    rows=[]
    for item in batch['episodes']:
        source=Path(item['path']);folder=source.parent
        audit=json.loads((folder/'audit.json').read_text());report=json.loads((folder/'report.json').read_text())
        native=json.loads((out/f'packing_order_native_seed{item["seed"]}_v1.json').read_text())
        assert audit['source_sha256']==native['source_sha256']==sha256(source)
        assert not audit['training_ready'] and not native['violations']
        old=next(r for r in previous['runs'] if r['seed']==item['seed'])
        with h5py.File(source,'r') as f,h5py.File(old['source'],'r') as before:
            assert f.attrs['complete']
            metadata=json.loads(f.attrs['metadata'])
            frames=len(f['observation/sim_time']);duration=float(f['next_observation/sim_time'][-1]-f['observation/sim_time'][0])
            event=next((e for e in report['events'] if e['phase']=='stack_placed'),None)
            cup_prefix=None
            if event:
                count=int(np.searchsorted(f['observation/sim_time'][:],event['sim_time']))
                equality={key:bool(np.array_equal(f[key][:count],before[key][:count])) for key in
                          ('observation/sim_time','observation/state','action/position')}
                cup_prefix={'transitions':count,'exact_numeric_dataset_equality':equality,'previous_source_sha256':old['source_sha256']}
        rows.append({'seed':item['seed'],'source':str(source),'source_sha256':audit['source_sha256'],
            'summary':report['summary'],'error':item['error'],'audit_passed':audit['passed'],
            'training_ready':False,'violations':audit['violations'],'acceptance_policy':audit.get('acceptance_policy'),
            'frames':frames,'duration_s':duration,'previous_duration_s':old['duration_s'],
            'previous_audit_passed':old['audit_passed'],'rigid_sort_order':metadata['teacher']['rigid_sort_order'],
            'placement_events':[e for e in report['events'] if e['phase']=='placement_search_selected'],
            'motion':audit.get('checks',{}).get('task',{}).get('motion'),
            'native_integrity':native['integrity_status'], 'cup_prefix_comparison':cup_prefix,
            'native_snapshot_tcp_speed_peak_m_s':{side:arm['peaks_all_observation_snapshots']['tcp_linear_speed_m_s']['absolute_max']
                                                for side,arm in native['arms'].items()},
            'phase_wall_seconds':item['phase_wall_seconds']})
    stability=json.loads((out/'packing_order_stability15619_v1.json').read_text())
    geometry=json.loads((out/'packing_order_geometry_seed15619_v1.json').read_text())
    assert stability['source_sha256']==next(r for r in rows if r['seed']==15619)['source_sha256']
    result={'status':'completed_engineering_state_regressions','training_ready':False,'runs':rows,
        'analysis_sha256':sha256(__file__),'unchanged_frozen_input_files':len(identity['files']),
        'changed_input_files':changed,'diagnostic_passed':sum(r['audit_passed'] for r in rows),
        'accepted_rgb_episodes':0,'seed_role':'development regressions; not held-out reliability evaluation',
        'seed15619_final_motion':stability,
        'seed15619_geometry':{k:geometry[k] for k in ('samples','native_arm_samples','equal_jaw_reconstruction_arm_samples',
            'changed_or_missing_provenance_files','category_peaks','tcp')},
        'limitations':['No RGB and no LeRobot export for this cohort; no training acceptance.',
            'Only ordering changes versus collision-support profile; no geometry/contact/quality changes.',
            'The existing per-kind contact settings remain uncalibrated; this does not establish hardware fidelity.',
            'Shared-load state timings are not a matched qualified-throughput benchmark.',
            'Policy guards sampled motion; native interval summaries remain diagnostics, not hardware certification.']}
    atomic_json(destination,result)
    print(json.dumps({'diagnostic_passed':result['diagnostic_passed'],'runs':[{k:r[k] for k in ('seed','frames','duration_s','audit_passed','error')} for r in rows]}))


if __name__=='__main__':main()
