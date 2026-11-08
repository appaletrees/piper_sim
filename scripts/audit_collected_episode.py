#!/usr/bin/env python3
"""Read-only independent recheck of a scripted HDF5 episode."""
import argparse
import json
from pathlib import Path
from piper_sim.collection_audit import audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output already exists')
    result = audit_episode(args.episode, require_rgb=not args.states_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': result['passed'], 'training_ready': result['training_ready'],
                      'violations': result['violations'], 'output': str(args.output)}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
