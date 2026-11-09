#!/usr/bin/env python3
"""Isolated colour-LUT experiment; does not modify production processing.

Reads one existing recorded RGB triplet as representative pixel input. These
images already contain sensor effects: processing them again is ONLY a component
benchmark, never a new observation or training recording. All candidates must
match production output exactly. Timings exclude input I/O and equality checks.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image, ImageFilter

from piper_sim import randomization
from piper_sim.collection_jobs import atomic_json, sha256


@lru_cache(maxsize=64)
def colour_curve(gamma, gain, balance):
    curve = randomization._gamma_curve(gamma)
    scale = gain*np.asarray(balance, np.float32)
    result = curve[:, None]*scale
    result.flags.writeable = False
    return result


def candidate(images, parameters, tick, method):
    out = {}
    for name, pixels in images.items():
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('This isolated candidate supports uint8 RGB only')
        p = parameters['cameras'][name]
        if p.get('blur_sigma_px', 0):
            pixels = np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(p['blur_sigma_px'])))
        table = colour_curve(p.get('gamma', 1.), p['gain'], tuple(p['white_balance']))
        if method == 'broadcast':
            rgb = table[pixels, np.arange(3)]
        elif method == 'channels':
            rgb = np.empty(pixels.shape, dtype=np.float32)
            for channel in range(3):
                rgb[..., channel] = table[:, channel][pixels[..., channel]]
        else:
            raise ValueError('Unknown lookup method')
        if p['noise_std_uint8']:
            rng = np.random.default_rng(randomization.stream_seed(parameters['seed'], f'sensor/{name}/{tick}'))
            rgb += rng.normal(0, p['noise_std_uint8'], rgb.shape).astype(np.float32)
        out[name] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=2500)
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh output')
    if args.repeats < 3 or args.frame < 0: parser.error('Use at least three repeats and a nonnegative frame')
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'): parser.error('A closed recording is required')
        meta = json.loads(f.attrs['metadata'])
        parameters = meta['task_scene']['layout']['domain_randomization']
        images = {n: f['observation/images/'+n][args.frame] for n in f['observation/images']}
    functions = {'production': randomization.sensor_images,
        'fused_broadcast': lambda images, p, tick: candidate(images, p, tick, 'broadcast'),
        'fused_channels': lambda images, p, tick: candidate(images, p, tick, 'channels')}
    names = list(functions)
    # Warm imports and LUTs for each variant before rotating the measured order.
    for function in functions.values(): function(images, parameters, args.frame)
    timings = {name: [] for name in functions}
    mismatches = {name: 0 for name in functions}
    start = time.time()
    for repeat in range(args.repeats):
        results = {}
        for name in names[repeat % 3:]+names[:repeat % 3]:
            started = time.perf_counter()
            results[name] = functions[name](images, parameters, args.frame+repeat)
            timings[name].append(time.perf_counter()-started)
        for name, result in results.items():
            mismatches[name] += sum(int(np.count_nonzero(result[k] != results['production'][k])) for k in images)
    medians = {name: float(np.median(v)) for name, v in timings.items()}
    report = {'scope': __doc__, 'training_ready': False, 'source': str(args.source.resolve()),
        'frame': args.frame, 'repeats': args.repeats, 'started_unix': start, 'finished_unix': time.time(),
        'versions': {k: importlib.metadata.version(k) for k in ('numpy', 'Pillow', 'h5py')},
        'source_hashes': {str(Path(p).resolve()): sha256(p) for p in (__file__, randomization.__file__)},
        'inputs': {k: {'shape': list(v.shape), 'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k,v in images.items()},
        'parameters': parameters['cameras'], 'median_seconds': medians, 'seconds': timings,
        'different_channel_values': mismatches,
        'stage_speedup_vs_production': {k: medians['production']/v for k,v in medians.items()},
        'limitations': ['Shared host; CPU component experiment, no exclusive machine or end-to-end throughput claim.',
            'Already processed source pixels are used as a representative input distribution.',
            'LUT caches are warm; per-episode setup and first-use costs are not represented.',
            'Candidates are not enabled in collection; only uint8 RGB is supported by this experiment.']}
    atomic_json(args.output, report)
    print(json.dumps({k:report[k] for k in ('median_seconds','different_channel_values','stage_speedup_vs_production')}))
    return 0 if not any(mismatches.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
