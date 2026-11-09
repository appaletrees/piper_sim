#!/usr/bin/env python3
"""Compare lossless writers on identical recorded RGB; not a training episode run.

Inputs are read-only. Temporary diagnostic HDF5s are removed after checking
every pixel through the stream auditor. Keep the JSON evidence, not duplicates.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import tempfile
import time

import h5py
import numpy as np

from piper_sim.collection_audit import audit_streams
from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=2400)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.frames < 2 or args.start < 0 or args.repeats < 1:
        parser.error('Use a new output; frames>=2, start>=0, repeats>=1')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transitions = []
    start = time.perf_counter()
    with h5py.File(args.source, 'r') as source:
        if not source.attrs.get('complete') or args.start+args.frames > source.attrs['num_transitions']:
            parser.error('Source must be complete and contain the requested range')
        metadata = json.loads(source.attrs['metadata'])
        settings = copy.deepcopy(metadata['teacher']['motion_and_quality'])
        # Stream-only diagnostic: no task/acceptance audit or training export.
        settings['dataset']['min_frames'] = 1
        cameras = list(source['observation/images'])
        def observation(prefix, i):
            return Observation(source[prefix+'/state'][i], float(source[prefix+'/timestamp'][i]),
                float(source[prefix+'/sim_time'][i]),
                {n: source[prefix+'/images/'+n][i] for n in cameras},
                {n: float(source[prefix+'/image_timestamps/'+n][i]) for n in cameras},
                json.loads(source[prefix+'/extras_json'][i]))
        for i in range(args.start, args.start+args.frames):
            transitions.append((observation('observation', i),
                Command(source['action/position'][i], float(source['action/timestamp'][i]), int(source['action/sequence'][i])),
                observation('next_observation', i)))
    source_read = time.perf_counter()-start
    modes = [('duplicated_gzip_flush1', 'duplicated', 'gzip', 1),
             ('shared_gzip_flush1', 'shared', 'gzip', 1),
             ('shared_gzip_flush30', 'shared', 'gzip', 30),
             ('shared_lzf_flush30', 'shared', 'lzf', 30)]
    results, reference = [], None
    with tempfile.TemporaryDirectory(prefix='writer-benchmark-', dir=args.output.parent) as temp:
        for repeat in range(args.repeats):
            # Rotate execution order to reduce systematic cache/thermal bias.
            for mode, storage, compression, flush in modes[repeat % len(modes):]+modes[:repeat % len(modes)]:
                path = Path(temp)/(mode+'.h5')
                start = time.perf_counter()
                with EpisodeWriter(path, metadata, image_compression=compression,
                                   image_storage=storage, flush_every=flush) as writer:
                    for before, action, after in transitions:
                        writer.append(before, action, after)
                elapsed = time.perf_counter()-start
                start = time.perf_counter()
                with h5py.File(path, 'r') as f:
                    audit = audit_streams(f, metadata, settings, require_rgb=True)
                    hashes = {'trajectory': audit['trajectory_sha256'],
                              **{n: v['rgb_sha256'] for n, v in audit['cameras'].items()}}
                if reference is None:
                    reference = hashes
                if not audit['passed'] or hashes != reference:
                    raise RuntimeError('Storage changed stream integrity: '+str(audit))
                size = path.stat().st_size
                result = {'mode': mode, 'repeat': repeat, 'wall_s': elapsed,
                          'wall_transitions_per_s': args.frames/elapsed,
                          'audit_s': time.perf_counter()-start, 'bytes': size,
                          'transitions_per_decimal_GB': args.frames/(size/1e9),
                          'stream_integrity_passed': True, 'hashes': hashes}
                results.append(result)
                print(json.dumps(result), flush=True)
                path.unlink()  # Only this benchmark's temporary diagnostic copy.
    summary = {}
    for mode, *_ in modes:
        selected = [r for r in results if r['mode'] == mode]
        summary[mode] = {'median_wall_s': statistics.median(r['wall_s'] for r in selected),
                         'median_bytes': statistics.median(r['bytes'] for r in selected),
                         'median_audit_s': statistics.median(r['audit_s'] for r in selected)}
    baseline = summary[modes[0][0]]
    for stats in summary.values():
        stats['writer_speedup'] = baseline['median_wall_s']/stats['median_wall_s']
        stats['storage_ratio'] = stats['median_bytes']/baseline['median_bytes']
    report = {'source': str(args.source.resolve()), 'start_frame': args.start, 'frames': args.frames,
              'source_read_seconds': source_read, 'repeats': args.repeats,
              'scope': 'Lossless writer microbenchmark only, same recorded pixels/state/action/timestamps. '
                       'Close/flush included; not fsync durability or end-to-end qualified episode throughput. '
                       'Full-stream checks use min_frames=1 only for this short diagnostic; no training approval.',
              'training_ready': False, 'results': results, 'summary': summary}
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
