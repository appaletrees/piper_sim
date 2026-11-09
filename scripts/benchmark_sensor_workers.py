#!/usr/bin/env python3
"""Isolated bounded camera-thread benchmark; no collection behavior is changed.

Recorded, already processed pixels are reused ONLY as component inputs. This is
not a sensor observation, new training data, or an end-to-end throughput result.
All comparisons require byte-identical output with the serial production code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import h5py
import numpy as np

from piper_sim import randomization
from piper_sim.collection_jobs import atomic_json, sha256


def threaded(images, parameters, tick, executor):
    def process(item):
        name, pixels = item
        return name, randomization.sensor_images({name: pixels}, parameters, tick)[name]
    return dict(executor.map(process, images.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', nargs='+', type=int, default=[0, 2450, 8770])
    p.add_argument('--repeats', type=int, default=15)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a new output')
    if args.repeats < 3: p.error('Use at least three repeats per input')
    raw_hash = sha256(args.source)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'): p.error('Supply a closed source')
        meta = json.loads(f.attrs['metadata'])
        params = meta['task_scene']['layout']['domain_randomization']
        frames = [{name: f['observation/images/'+name][i] for name in f['observation/images']}
                  for i in args.frames]
    timings = {k: [] for k in ('serial', 'workers2', 'workers3')}
    mismatches = {k: 0 for k in timings}
    setup = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as two, ThreadPoolExecutor(max_workers=3) as three:
        functions = {'serial': randomization.sensor_images,
            'workers2': lambda x, p, t: threaded(x,p,t,two),
            'workers3': lambda x, p, t: threaded(x,p,t,three)}
        for fn in functions.values(): fn(frames[0], params, args.frames[0])
        setup = time.perf_counter()-setup
        order = list(functions)
        for frame, images in zip(args.frames, frames):
            for repeat in range(args.repeats):
                results = {}
                shift = repeat % len(order)
                for name in order[shift:]+order[:shift]:
                    start = time.perf_counter()
                    results[name] = functions[name](images, params, frame+repeat)
                    timings[name].append(time.perf_counter()-start)
                for name, result in results.items():
                    mismatches[name] += sum(np.count_nonzero(result[k] != results['serial'][k]) for k in images)
    medians = {name: float(np.median(values)) for name,values in timings.items()}
    assert sha256(args.source) == raw_hash
    result = dict(scope=__doc__, training_ready=False, source=str(args.source.resolve()), source_sha256=raw_hash,
        source_unchanged=True, input_frames=args.frames, repeats_per_frame=args.repeats,
        pixels={str(i): {k: {'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()}
                         for k,v in images.items()} for i,images in zip(args.frames,frames)},
        input_parameters=params, seconds=timings, median_seconds=medians,
        different_channel_values={k:int(v) for k,v in mismatches.items()},
        component_speedup={k:medians['serial']/v for k,v in medians.items()}, warmup_and_pool_setup_seconds=setup,
        source_hashes={str(Path(file).resolve()):sha256(file) for file in (__file__, randomization.__file__)},
        versions={k:importlib.metadata.version(k) for k in ('numpy','Pillow','h5py')},
        limitations=['Warmed thread pools; host is shared. No whole-pipeline or unseen-scene speedup claim.',
                     'Each synchronous call waits for all three cameras; no image or simulation tick is skipped.'])
    atomic_json(args.output, result)
    print(json.dumps({k:result[k] for k in ('median_seconds','component_speedup','different_channel_values')},indent=2))
    return 2 if any(mismatches.values()) else 0


if __name__ == '__main__': raise SystemExit(main())
