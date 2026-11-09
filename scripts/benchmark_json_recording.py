#!/usr/bin/env python3
"""Compare JSON storage using identical recorded transitions; no training verdict.

Only temporary benchmark copies are removed. Input recordings remain read-only.
Includes append, flush and close; excludes source loading and fsync durability.
"""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import h5py
import numpy as np

from piper_sim.collection_audit import audit_streams,load_acceptance_policy
from piper_sim.collection_jobs import atomic_json,sha256
from piper_sim.contracts import Command,Observation
from piper_sim.recording import EpisodeWriter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=500);p.add_argument('--frames',type=int,default=300)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--rgb',action='store_true')
    args=p.parse_args();policy,policy_source=load_acceptance_policy()
    if args.output.exists() or args.start<0 or args.frames<policy['dataset']['min_frames'] or args.repeats<1:
        p.error('Use a new output and at least the original policy minimum frame count')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    digest=sha256(args.source);transitions=[];expected={}
    with h5py.File(args.source,'r') as source:
        if not source.attrs.get('complete') or args.start+args.frames>source.attrs['num_transitions']:
            p.error('Source must be closed and contain the requested range')
        metadata=json.loads(source.attrs['metadata'])
        names=list(source.get('observation/images',{})) if args.rgb else []
        if args.rgb and not names:p.error('Source has no RGB')
        for prefix in ('observation','next_observation'):
            expected[prefix]=source[prefix+'/extras_json'][args.start:args.start+args.frames]
        def observation(prefix,index):
            return Observation(source[prefix+'/state'][index],float(source[prefix+'/timestamp'][index]),
                float(source[prefix+'/sim_time'][index]),{n:source[prefix+'/images/'+n][index] for n in names},
                {n:float(source[prefix+'/image_timestamps/'+n][index]) for n in names},
                json.loads(source[prefix+'/extras_json'][index]))
        for i in range(args.start,args.start+args.frames):
            transitions.append((observation('observation',i),
                Command(source['action/position'][i],float(source['action/timestamp'][i]),int(source['action/sequence'][i])),
                observation('next_observation',i)))
    results=[];reference=None
    with tempfile.TemporaryDirectory(prefix='json-writer-benchmark-',dir=args.output.parent) as temporary:
        for repeat in range(args.repeats):
            for mode in (('vlen','compressed') if repeat%2==0 else ('compressed','vlen')):
                path=Path(temporary)/(mode+'.h5');started=time.perf_counter()
                with EpisodeWriter(path,metadata,image_compression='gzip',image_storage='shared',
                                   flush_every=1,extras_storage=mode) as writer:
                    for before,command,after in transitions:writer.append(before,command,after)
                write_seconds=time.perf_counter()-started;started=time.perf_counter()
                with h5py.File(path,'r') as f:
                    audit=audit_streams(f,metadata,policy,require_rgb=args.rgb)
                    hashes={'trajectory':audit['trajectory_sha256'],**{n:v['rgb_sha256'] for n,v in audit['cameras'].items()}}
                    if reference is None:reference=hashes
                    assert audit['passed'] and hashes==reference
                    for prefix,raw in expected.items():np.testing.assert_array_equal(f[prefix+'/extras_json'][:],raw)
                results.append({'mode':mode,'repeat':repeat,'write_seconds':write_seconds,
                    'verify_seconds':time.perf_counter()-started,'bytes':path.stat().st_size,
                    'hashes':hashes,'every_json_byte_equal_to_source':True})
                path.unlink()  # This newly created benchmark copy only.
    assert sha256(args.source)==digest
    summary={mode:{'median_write_seconds':statistics.median(r['write_seconds'] for r in results if r['mode']==mode),
                   'median_verify_seconds':statistics.median(r['verify_seconds'] for r in results if r['mode']==mode),
                   'median_bytes':statistics.median(r['bytes'] for r in results if r['mode']==mode)} for mode in ('vlen','compressed')}
    report={'scope':__doc__,'training_ready':False,'source':str(args.source.resolve()),'source_sha256':digest,
        'script_sha256':sha256(__file__),'writer_sha256':sha256(Path('src/piper_sim/recording.py')),
        'json_writer_sha256':sha256(Path('src/piper_sim/recording_json.py')),'policy':policy_source,
        'start_frame':args.start,'frames':args.frames,'rgb':args.rgb,'repeats':args.repeats,'results':results,'summary':summary,
        'compressed_storage_ratio':summary['compressed']['median_bytes']/summary['vlen']['median_bytes'],
        'compressed_write_time_ratio':summary['compressed']['median_write_seconds']/summary['vlen']['median_write_seconds'],
        'limitations':['Short fixed recorded windows; not a task success or qualified-throughput measurement.',
            'Original stream policy unchanged; full task audit and training acceptance are not run for benchmark copies.']}
    atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('summary','compressed_storage_ratio','compressed_write_time_ratio')}))


if __name__=='__main__':main()
