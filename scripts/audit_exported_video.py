#!/usr/bin/env python3
"""Decode every exported Piper frame and compare it with its source RGB index.

Checks all frame counts and presentation timestamps. H.264 pixel errors are
measurements, not a new task or visual-quality acceptance threshold. Supports
the repository's single-episode native-resolution LeRobot v2.1 exports.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.training_acceptance import verify_training_acceptance


def inspect_video(path, frames, fps):
    count, height, width, channels = frames.shape
    if count < 1 or channels != 3 or frames.dtype != np.uint8 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Expected nonempty uint8 RGB source frames and finite positive fps')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-show_frames', '-show_entries',
        'stream=width,height,avg_frame_rate,codec_name,pix_fmt:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], text=True))
    streams = probe['streams']
    if len(streams) != 1 or (streams[0]['height'], streams[0]['width']) != (height, width):
        raise ValueError('Encoded dimensions differ from source')
    numerator, denominator = map(int, streams[0]['avg_frame_rate'].split('/'))
    if denominator <= 0 or not np.isclose(numerator/denominator, fps, atol=1e-9, rtol=0):
        raise ValueError('Encoded frame rate differs from source')
    timestamps = np.array([float(frame['best_effort_timestamp_time']) for frame in probe['frames']])
    if len(timestamps) != count or not np.allclose(timestamps, np.arange(count)/fps, atol=1e-6, rtol=0):
        raise ValueError('Encoded frame count or presentation timestamp mismatch')
    errors = np.empty((count, 3), dtype=np.float64)
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:v:0',
               '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        length = height*width*3
        for index in range(count):
            data = process.stdout.read(length)
            if len(data) != length: raise ValueError('Truncated decoded frame '+str(index))
            decoded = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            difference = decoded.astype(np.int32)-frames[index]
            errors[index] = [np.mean(abs(difference)), np.mean(difference*difference), np.max(abs(difference))]
        if process.stdout.read(1): raise ValueError('Unexpected additional decoded frame')
        stderr = process.stderr.read().decode('utf-8', errors='replace')
        if process.wait(timeout=30): raise ValueError('Video decoding failed: '+stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        process.stdout.close(); process.stderr.close()
    mse = float(errors[:, 1].mean())
    peak = int(np.argmax(errors[:, 0]))
    result = {'frames': count, 'height': height, 'width': width, 'fps': fps,
        'stream': streams[0], 'max_pts_error_s': float(np.max(abs(timestamps-np.arange(count)/fps))),
        'mean_absolute_pixel_error_0_to_255': float(errors[:, 0].mean()),
        'rmse_0_to_255': float(np.sqrt(mse)), 'psnr_db': 10*np.log10(255**2/mse) if mse else None,
        'pixels_identical': mse == 0, 'max_channel_absolute_error_0_to_255': int(errors[:, 2].max()),
        'frame_mean_absolute_error_percentiles': dict(zip(('p50', 'p95', 'p99', 'max'),
            map(float, np.percentile(errors[:, 0], [50, 95, 99, 100])))),
        'worst_frame_by_mae': peak,
        'note': 'PSNR is null only for pixel-identical output; no visual-quality threshold is inferred.'}
    return result, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    trace = args.output.with_suffix('.npz')
    if args.output.exists() or trace.exists(): p.error('Choose new output paths')
    consumer_path = Path(__file__).with_name('check_lerobot_reader.py')
    input_files = [Path(__file__), consumer_path, args.dataset/'meta/info.json',
                   args.dataset/'data/chunk-000/episode_000000.parquet']
    input_hashes = {str(path.resolve()): sha256(path) for path in input_files}
    spec = importlib.util.spec_from_file_location('consumer_contract', consumer_path)
    consumer = importlib.util.module_from_spec(spec); spec.loader.exec_module(consumer)
    info = json.loads((args.dataset/'meta/info.json').read_text())
    consumer.validate_public_schema(info)
    reports, traces = {}, {}
    with h5py.File(args.source, 'r') as f:
        audit, digest = verify_training_acceptance(f, args.source, policy=args.policy)
        numeric = consumer.check_tabular_rows(args.dataset, f, info)
        for exported, original in consumer.CAMERAS.items():
            source_frames = f['observation/images/'+original]
            feature = info['features']['observation.images.'+exported]
            consumer.need(feature['shape'] == [3, *source_frames.shape[1:3]],
                          'Declared video dimensions differ from indexed source: '+original)
            video = args.dataset/f'videos/chunk-000/observation.images.{exported}/episode_000000.mp4'
            before = sha256(video)
            reports[original], traces[original] = inspect_video(video, source_frames, info['fps'])
            stream = reports[original]['stream']
            consumer.need(stream['codec_name'] == feature['info']['video.codec']
                          and stream['pix_fmt'] == feature['info']['video.pix_fmt'],
                          'Declared video codec or pixel format differs from encoded stream: '+original)
            if sha256(video) != before: raise ValueError('Exported video changed during validation')
            reports[original].update(video=str(video.resolve()), video_sha256=before)
    if sha256(args.source) != digest: raise ValueError('Source changed during validation')
    if any(sha256(path) != digest for path, digest in input_hashes.items()):
        raise ValueError('Checker, metadata or numeric export changed during validation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace, **traces)
    result = {'status': 'all_video_frames_and_pts_checked', 'training_ready': None, 'scope': __doc__,
        'source': str(args.source.resolve()), 'source_sha256': digest, 'source_unchanged': True,
        'dataset': str(args.dataset.resolve()), 'acceptance_policy': audit['acceptance_policy'],
        'checker_and_tabular_input_hashes': input_hashes,
        'script_sha256': sha256(__file__), 'consumer_schema_script_sha256': sha256(consumer_path),
        'tabular': numeric, 'cameras': reports, 'trace_path': str(trace.resolve()), 'trace_sha256': sha256(trace),
        'trace_columns': ['mean_absolute_error_0_to_255', 'mean_squared_error_0_to_65025', 'max_absolute_error_0_to_255'],
        'limitations': ['Pixel errors measure lossy encoding against indexed source frames, not real-camera fidelity.',
                       'This report does not independently grant physical task success or certify perceptual quality.']}
    atomic_json(args.output, result)
    print(json.dumps({name: {k: r[k] for k in ('frames', 'psnr_db', 'mean_absolute_pixel_error_0_to_255')}
                      for name, r in reports.items()}, indent=2), flush=True)


if __name__ == '__main__': main()
