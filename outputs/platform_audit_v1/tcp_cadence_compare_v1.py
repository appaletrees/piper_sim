"""Explain TCP retiming's duration cost using an existing matched three-cup pair."""
import importlib.util
import json
from pathlib import Path

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT

base = ROOT/'outputs/platform_audit_v1'
output = base/'tcp_cadence_seed13_v1.json'
if output.exists(): raise FileExistsError(output)
baseline = next(r for r in json.loads((base/'packing_order_summary_v1.json').read_text())['runs'] if r['seed']==13)
candidate = json.loads((base/'tcp_three_cup_summary_v1.json').read_text())
sources = [Path(row['source']) for row in (baseline,candidate)]
for path, row in zip(sources,(baseline,candidate)): assert sha256(path)==row['source_sha256']
spec = importlib.util.spec_from_file_location('cadence', ROOT/'scripts/compare_collection_motion.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
before, after = [module.cadence(path) for path in sources]
for path, row in zip(sources,(baseline,candidate)): assert sha256(path)==row['source_sha256']
phase_delta = {phase: {key:after['phases'].get(phase,{}).get(key,0)-before['phases'].get(phase,{}).get(key,0)
                       for key in ('seconds','stationary_seconds')}
               for phase in set(before['phases'])|set(after['phases'])}
result = {'training_ready':False, 'seed':13, 'baseline':before, 'candidate':after,
          'baseline_sha256':baseline['source_sha256'], 'candidate_sha256':candidate['source_sha256'],
          'simulation_duration_change_s':after['simulation_duration_s']-before['simulation_duration_s'],
          'stationary_seconds_change':sum(p['stationary_seconds'] for p in after['phases'].values())-sum(p['stationary_seconds'] for p in before['phases'].values()),
          'phase_delta':phase_delta, 'analysis_sha256':sha256(__file__),
          'cadence_implementation_sha256':sha256(ROOT/'scripts/compare_collection_motion.py'),
          'limitations':['Existing matched development recordings, not fresh holdout trials.',
                         'Stationary thresholds include intentional physical contact holds; not all stopped time is removable.',
                         'No rendering, retiming, hardware calibration or training acceptance is performed.']}
atomic_json(output,result)
print(json.dumps({k:result[k] for k in ('seed','simulation_duration_change_s','stationary_seconds_change','phase_delta')},indent=2))
