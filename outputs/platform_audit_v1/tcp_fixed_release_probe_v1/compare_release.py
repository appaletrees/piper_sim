"""Compare phase-specific recorded native extrema; no new acceptance gate."""
import json,hashlib
from pathlib import Path
import h5py,numpy as np
from piper_sim.collection_jobs import atomic_json,sha256
sources=[Path('/dev/shm/piper_tcp_fixed_contact_v1/worker_01/episode_000000_seed_1/episode.h5'),Path('/dev/shm/piper_tcp_fixed_release_probe_v1/episode_000000_seed_1/episode.h5')]
output=Path('outputs/platform_audit_v1/tcp_fixed_release_probe_v1/comparison.json')
if output.exists():raise FileExistsError(output)
rows=[];arrays=[]
for source in sources:
    digest=sha256(source);groups={};prefix=hashlib.sha256();first_release=None
    with h5py.File(source,'r') as f:
        assert f.attrs['complete'];m=json.loads(f.attrs['metadata'])
        array={k:f[k][:] for k in ('observation/state','next_observation/state','action/position','observation/sim_time','next_observation/sim_time')};arrays.append(array)
        for i,raw in enumerate(f['next_observation/extras_json']):
            e=json.loads(raw);t=e['teacher'];c=e['controller_interval']
            if first_release is None and t['object']=='plate' and t['phase']=='release':first_release=i
            if first_release is None:
                prefix.update(json.dumps({k:e[k] for k in ('objects','controller_interval')},sort_keys=True).encode())
                prefix.update(json.dumps(t['physics'],sort_keys=True).encode())
            key=str(t['object'])+'/'+t['phase']
            if key not in groups:groups[key]={'intervals':0}
            g=groups[key];g['intervals']+=1
            for metric in ('native_acceleration_abs_max','native_jerk_abs_max'):
                value=float(c[metric][13])
                if metric not in g or value>g[metric]['value']:
                    g[metric]={'value':value,'transition_index':i,'interval_start_sim_time':c['start_sim_time'],'interval_end_sim_time':c['end_sim_time']}
        assert first_release is not None
        rows.append({'source':str(source),'sha256':digest,'frames':len(f['action/position']),'first_plate_release_transition':first_release,'prefix_object_native_controller_sha256':prefix.hexdigest(),'groups':groups})
    assert sha256(source)==digest
boundary=min(r['first_plate_release_transition'] for r in rows)
exact={k:np.array_equal(arrays[0][k][:boundary],arrays[1][k][:boundary]) for k in arrays[0]}
assert all(exact.values()) and rows[0]['first_plate_release_transition']==rows[1]['first_plate_release_transition']
assert rows[0]['prefix_object_native_controller_sha256']==rows[1]['prefix_object_native_controller_sha256']
atomic_json(output,{'scope':__doc__,'training_ready':False,'source_unchanged':True,'script_sha256':sha256(__file__),'exact_prefix_transitions':boundary,'exact_public_prefix':exact,'exact_objects_native_controller_prefix':True,'runs':rows,'limitations':'Profile changes numerical friction refinement in plate and side-dish release. Result includes subsequent physical consequences; not a local single-contact replay, calibrated fidelity claim or throughput comparison.'})
print(json.dumps({'exact_prefix':boundary,'side_dish_release':[r['groups']['side_dish/release'] for r in rows],'plate_release':[r['groups']['plate/release'] for r in rows]},indent=2))
