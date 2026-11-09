"""Isolated negative test: never export or alter an existing recording."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from piper_sim.collection_audit import audit_episode, save_audit
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT
from piper_sim.contracts import Command, Observation
from piper_sim.lerobot_export import _inspect
from piper_sim.recording import EpisodeWriter

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
reference_path = ROOT/'outputs/platform_audit_v1/servo_rgb_seed0_v1/seed_0/audit.json'
reference = json.loads(reference_path.read_text())
assert reference['training_ready'] and reference['passed']
source = reference_path.parent/'episode.h5'
assert sha256(source) == reference['source_sha256']
result = {'training_ready': False, 'scope': 'Publication fault injection, not a simulated training episode.',
          'reference_report': str(reference_path), 'reference_report_sha256': sha256(reference_path),
          'reference_source_sha256': reference['source_sha256'],
          'auditor_sha256': sha256(ROOT/'src/piper_sim/collection_audit.py'),
          'reproducer_sha256': sha256(__file__)}
with tempfile.TemporaryDirectory(prefix='piper-audit-binding-fault-', dir='/dev/shm') as directory:
    target = Path(directory)/'fault-only.h5'
    q = np.array([0., 0., 0., 0., 0., 0., .04]*2)
    with EpisodeWriter(target, {'fps': 30, 'task': 'FAULT INJECTION; NOT TRAINING DATA'}) as writer:
        for i in range(3):
            before = Observation(q, 100.+i, i/30, {}, {}, {})
            after = Observation(q, 100.+i+.5, (i+1)/30, {}, {}, {})
            writer.append(before, Command(q, 100.+i+.2, i), after)
        writer.close(success=True)
    initial_hash = sha256(target)
    fresh = audit_episode(target)
    result['fresh_audit'] = {k: fresh[k] for k in ('passed', 'training_ready', 'violations')}
    assert not fresh['passed']
    try:
        save_audit(target, copy.deepcopy(reference), target.with_name('audit.json'))
        result['mismatched_report_publication'] = 'allowed'
        try:
            inspected = _inspect(target, require_quality=True)
            result['strict_export_inspection'] = 'allowed'
            result['incorrectly_eligible_frames'] = inspected['length']
        except ValueError as error:
            result['strict_export_inspection'] = str(error)
    except ValueError as error:
        result['mismatched_report_publication'] = 'rejected: '+str(error)
        result['strict_export_inspection'] = 'not attempted after rejection'
    result['target_unchanged'] = sha256(target) == initial_hash
    result['temporary_fault_artifacts_removed'] = True
    result['existing_recordings_modified'] = False
atomic_json(args.output, result)
print(json.dumps(result))
