#!/usr/bin/env python3
"""Read-only diagnosis of a recording's 14D position execution contract."""
import argparse
import json
from pathlib import Path

import h5py

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.position_execution import audit_position_execution
from piper_sim.training_acceptance import require_self_contained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        require_self_contained(f)
        if not f.attrs['complete']:
            raise ValueError('Closed recording required')
        result = audit_position_execution(f, json.loads(f.attrs['metadata']))
    if sha256(args.source) != digest:
        raise ValueError('Recording changed during audit')
    result.update(source=str(args.source.resolve()), source_sha256=digest,
                  script_sha256=sha256(__file__), source_unchanged=True, training_ready=False)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
