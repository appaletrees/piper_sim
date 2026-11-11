#!/usr/bin/env python3
"""Export a complete accepted three-camera episode as a 1x review video.

Every recorded observation is displayed once, without resizing sensor pixels.
This is a review montage, not a replacement for the HDF5 training episode.
"""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.training_acceptance import verify_training_acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing evidence is preserved')
    cameras = ('overhead', 'left_wrist', 'right_wrist')
    with h5py.File(args.source, 'r') as f:
        audit, source_hash = verify_training_acceptance(f, args.source, policy=args.policy)
        if not audit['training_ready']:
            raise ValueError('An independently accepted RGB episode is required')
        times = f['observation/sim_time'][:]
        fps = float(json.loads(f.attrs['metadata'])['fps'])
        if fps != 30. or not np.allclose(np.diff(times), 1/fps, atol=1e-6, rtol=0):
            raise ValueError('Expected intact 30 Hz clocks; no resampling is performed')
        streams = [f['observation/images/'+name] for name in cameras]
        height, width, channels = streams[0].shape[1:]
        if channels != 3 or any(s.shape != (len(times), height, width, 3) or
                                s.dtype != np.uint8 for s in streams):
            raise ValueError('Matching uint8 RGB streams are required')
        args.output.mkdir(parents=True)
        partial = args.output/'full_sensor_rgb.partial.mp4'
        process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width*3}x{height+32}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-threads', '2', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE)
        try:
            for i, time in enumerate(times):
                panel = Image.new('RGB', (width*3, height+32), (15, 20, 28))
                draw = ImageDraw.Draw(panel)
                for j, (name, stream) in enumerate(zip(cameras, streams)):
                    panel.paste(Image.fromarray(stream[i]), (j*width, 32))
                    draw.text((j*width+8, 5),
                              f'SENSOR RGB | {name} | 1x / 30 Hz | t={time:.3f}s', fill='white')
                process.stdin.write(np.asarray(panel).tobytes())
            process.stdin.close()
            if process.wait(timeout=30):
                raise RuntimeError('Video encoding failed')
        finally:
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    if sha256(args.source) != source_hash:
        raise ValueError('The source recording changed during export')
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,avg_frame_rate,duration',
        '-of', 'json', str(partial)], text=True))['streams'][0]
    if int(probe['nb_read_frames']) != len(times) or probe['avg_frame_rate'] != '30/1':
        raise ValueError('Encoded count/rate differs from the complete source')
    target = args.output/'full_sensor_rgb.mp4'
    partial.rename(target)
    atomic_json(args.output/'export.json', {
        'scope': __doc__, 'training_ready': False, 'source_training_ready': True,
        'source': str(args.source.resolve()), 'source_sha256': source_hash,
        'source_unchanged': True, 'acceptance_policy': audit['acceptance_policy'],
        'frames': len(times), 'fps': fps, 'duration_s': len(times)/fps,
        'frame_range_exclusive': [0, len(times)], 'source_pixels_resized': False,
        'cameras': list(cameras), 'video_sha256': sha256(target), 'ffprobe': probe,
        'script_sha256': sha256(__file__)})
    print(json.dumps({'video': str(target), 'frames': len(times), 'duration_s': len(times)/fps}))


if __name__ == '__main__':
    main()
