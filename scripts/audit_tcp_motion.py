#!/usr/bin/env python3
"""Read-only native TCP interval diagnostic, including failed task recordings."""
import argparse
import json
from pathlib import Path

import h5py

from piper_sim.collection_audit import load_acceptance_policy
from piper_sim.collection_jobs import atomic_json,sha256
from piper_sim.tcp_audit import audit_tcp_motion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve previous diagnostic output')
    digest=sha256(args.source)
    result={'source':str(args.source.resolve()),'source_sha256':digest,'training_ready':False,
            'passed':False,'violations':[], 'script_sha256':sha256(__file__)}
    try:
        settings,policy=load_acceptance_policy(args.policy)
        result['acceptance_policy']=policy
        with h5py.File(args.source,'r') as f:
            if not f.attrs.get('complete') or f.attrs.get('schema_version')!='piper.transitions.v1':
                raise ValueError('A closed complete recording is required')
            result.update(audit_tcp_motion(f,json.loads(f.attrs['metadata']),settings['tcp_motion']))
    except (KeyError,ValueError,TypeError,OSError,IndexError) as error:
        result['violations'].append(type(error).__name__+': '+str(error))
    if sha256(args.source)!=digest:result['violations'].append('Source changed during read-only audit')
    result['passed']=not result['violations']
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('passed','violations','training_ready')}))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
