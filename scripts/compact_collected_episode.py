#!/usr/bin/env python3
"""Losslessly compact a closed RGB recording into a NEW file and re-audit it."""
import argparse
import json
from pathlib import Path
from piper_sim.recording_storage import compact_episode
from piper_sim.collection_audit import audit_episode, save_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.h5': parser.error('Output must be a new .h5 file')
    evidence = args.output.with_suffix('.audit.json')
    if args.output.exists() or evidence.exists(): parser.error('Output or sidecar already exists')
    info = compact_episode(args.episode, args.output, progress=lambda name: print('Verified '+name, flush=True))
    audit = audit_episode(args.output, require_rgb=True)
    save_audit(args.output, audit, evidence)
    print(json.dumps({**info, 'passed': audit['passed'], 'training_ready': audit['training_ready'],
                      'violations': audit['violations'], 'audit': str(evidence)}, ensure_ascii=False))
    return 0 if audit['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
