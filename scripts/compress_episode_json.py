#!/usr/bin/env python3
"""Losslessly compress JSON in a new copy; retain source and require new audit."""
import argparse
import json
from pathlib import Path

from piper_sim.collection_jobs import atomic_json
from piper_sim.recording_storage import compress_episode_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    if args.report.exists():p.error('Preserve existing reports')
    reserved = (args.source, args.output, args.output.with_name(args.output.name+'.partial'))
    if args.report.resolve() in {path.resolve() for path in reserved}:
        p.error('Report must be separate from source, output and partial recording')
    result=compress_episode_json(args.source,args.output)
    atomic_json(args.report,result)
    print(json.dumps({k:result[k] for k in ('frames','source_bytes','output_bytes','training_ready')}))


if __name__=='__main__':main()
