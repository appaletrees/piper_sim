#!/usr/bin/env python3
"""Independent recheck; read-only unless --update-acceptance is explicitly set."""
import argparse
import json
from pathlib import Path
from piper_sim.collection_audit import audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states-only', action='store_true')
    parser.add_argument('--policy', type=Path, help='Operator-selected production policy; never read from the episode')
    parser.add_argument('--update-acceptance', action='store_true',
                        help='Write the audit/acceptance attributes into this HDF5 and checksum its new contents')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output already exists')
    if args.update_acceptance and args.output.resolve().parent != args.episode.resolve().parent:
        parser.error('Updated acceptance needs a sidecar beside the HDF5')
    result = audit_episode(args.episode, require_rgb=not args.states_only, policy=args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.update_acceptance:
        from piper_sim.collection_audit import save_audit
        save_audit(args.episode, result, args.output)
    else:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': result['passed'], 'training_ready': result['training_ready'],
                      'violations': result['violations'], 'output': str(args.output)}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
