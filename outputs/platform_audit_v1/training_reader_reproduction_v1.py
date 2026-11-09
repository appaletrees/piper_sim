"""Compare default sequence and export eligibility for a real state diagnostic."""
import argparse
import inspect
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.dataset import sequence_windows
from piper_sim.lerobot_export import _inspect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
source = ROOT/'outputs/platform_audit_v1/audit_binding_state_copy_v1/episode.h5'
digest = sha256(source)
with h5py.File(source, 'r') as f:
    source_ready = bool(f.attrs.get('training_ready'))
assert not source_ready
result = {'training_ready': False, 'source': str(source), 'source_sha256': digest,
          'source_training_ready': source_ready, 'reader_sha256': sha256(ROOT/'src/piper_sim/dataset.py'),
          'analysis_sha256': sha256(__file__)}
def read_window(**options):
    generator = sequence_windows(source, horizon=16, **options)
    try:
        window = next(generator)
        return {'status': 'yielded', 'frames': len(window['state']), 'cameras': sorted(window['images']),
                'keys': sorted(window), 'has_sim_timestamp': 'sim_timestamp' in window}
    except ValueError as error:
        return {'status': 'rejected', 'reason': str(error)}
    finally: generator.close()
result['default_reader'] = read_window()
try:
    _inspect(source, require_quality=True)
    result['strict_export'] = 'allowed'
except ValueError as error:
    result['strict_export'] = 'rejected: '+str(error)
if 'require_quality' in inspect.signature(sequence_windows).parameters:
    result['explicit_diagnostic_reader'] = read_window(require_quality=False)
assert sha256(source) == digest
result['source_unchanged'] = True
atomic_json(args.output, result)
print(json.dumps(result))
