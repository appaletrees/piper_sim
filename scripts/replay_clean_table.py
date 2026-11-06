#!/usr/bin/env python3
"""Replay real clean-table states and render synchronized three-camera comparisons.

Kinematic state playback only: no physics integration, object tracking or CAN.
The 15D radian profile is explicit; do not use the degree-based connector data.
"""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from piper_sim.recorded_replay import (
    CAMERAS, PROFILE, digest, limit_report, load_rad15_episode, place_kinematic_observation,
)


def probe_video(path, frames, fps, resolution):
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,codec_name', '-of', 'json', str(path)]
    stream = json.loads(subprocess.check_output(args))['streams'][0]
    if (stream['width'], stream['height']) != resolution:
        raise ValueError(f'Video resolution mismatch: {path}')
    if abs(float(Fraction(stream['avg_frame_rate'])) - fps) > 1e-8:
        raise ValueError(f'Video rate mismatch: {path}')
    if stream.get('nb_frames', 'N/A') != 'N/A' and int(stream['nb_frames']) != frames:
        raise ValueError(f'Video/frame table length mismatch: {path}')
    return stream


class VideoReader:
    """Sequential decoding by frame index; never seek approximately by time."""
    def __init__(self, path, resolution):
        self.bytes_per_frame = resolution[0] * resolution[1] * 3
        self.shape = (resolution[1], resolution[0], 3)
        self.process = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-map', '0:v:0',
             '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-threads', '1', 'pipe:1'], stdout=subprocess.PIPE)

    def read(self):
        pixels = self.process.stdout.read(self.bytes_per_frame)
        if len(pixels) != self.bytes_per_frame:
            raise ValueError('Truncated source video before the required saved frame')
        return np.frombuffer(pixels, dtype=np.uint8).reshape(self.shape)

    def verify_end(self):
        if self.process.stdout.read(1):
            raise ValueError('Source video contains more frames than the numeric episode')
        if self.process.wait() != 0:
            raise RuntimeError('Source video decoder failed')

    def close(self):
        # A prefix replay intentionally leaves unread frames. Stop our decoder
        # before closing its pipe, avoiding misleading ffmpeg Broken pipe logs.
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdout.close()


def encoder(path, width, height, fps):
    return subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an',
         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '2',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--config', default='configs/clean_table_workcell.yaml')
    p.add_argument('--task-scene', default='configs/tasks/clean_table.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1, help='Video sampling only; all recorded states are replayed')
    p.add_argument('--max-frames', type=int, help='Optional prefix for a quick inspection')
    p.add_argument('--viewer', action='store_true', help='Also show MuJoCo at the recorded playback speed')
    args = p.parse_args()
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        p.error('stride and max-frames must be positive')
    out, root = args.output.expanduser().resolve(), args.dataset.expanduser().resolve()
    if out == root or out.is_relative_to(root):
        p.error('Output must be outside the read-only source dataset')
    if out.exists():
        p.error('Use a new output directory to preserve prior evidence')
    if not args.viewer:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    from piper_sim.backends.mujoco import MujocoBackend
    from piper_sim.config import load
    from piper_sim.contracts import FEATURE_NAMES

    episode = load_rad15_episode(root, args.episode)
    video_info = {k: probe_video(v, len(episode), episode.fps, episode.resolution) for k, v in episode.videos.items()}
    n = min(len(episode), args.max_frames or len(episode))
    cfg = load(args.config)
    cfg['task_scene'] = args.task_scene
    out.mkdir(parents=True)
    (out / 'samples').mkdir()
    report = {
        'schema_version': 1, 'profile': PROFILE, 'episode': args.episode,
        'dataset': str(root), 'status': 'running',
        'mode': 'kinematic_measured_state_replay', 'field_replayed': 'state',
        'source_frames': len(episode), 'requested_frames': n, 'source_fps': episode.fps,
        'render_stride': args.stride, 'comparison_fps': episode.fps / args.stride,
        'feature_order_si': list(FEATURE_NAMES),
        'joint_unit': 'rad', 'gripper_unit': 'm, normalized SDK stroke multiplied by 0.095',
        'auxiliary': {'meaning': 'unknown; preserved, excluded from robot and termination',
                      'state_unique': np.unique(episode.state_raw[:, 14]).tolist(),
                      'actions_unique': np.unique(episode.action_raw[:, 14]).tolist()},
        'limits': {'state': limit_report(episode.state_si), 'actions': limit_report(episode.action_si)},
        'camera_mapping': CAMERAS, 'video_info': video_info,
        'time_alignment': 'same saved frame index at 30 Hz; no temporal offset or interpolation',
        'hardware_exposure_synchronization_verified': False,
        'physics_integrated': False, 'hardware_accessed': False,
        'object_poses': 'static nominal scene; no object pose trajectories provided by source',
        'physical_alignment_verified': False,
        'calibration': 'CAD extrinsics and unverified nominal RGB stream intrinsics; no device calibration in dataset',
        'geometry_or_calibration_fitted': False,
        'source_files': {str(f): digest(f) for f in [root / 'meta/info.json', episode.parquet, *episode.videos.values()]},
        'config_sha256': digest(cfg['_config_path']), 'task_sha256': digest(args.task_scene),
        'reader_sha256': digest(Path(__file__).resolve().parents[1] / 'src/piper_sim/recorded_replay.py'),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    encoded, frames_done, max_error = 0, 0, 0.
    frames_to_save = set(np.linspace(0, n - 1, 7, dtype=int))
    w, h = episode.resolution
    header, band = 44, 28
    video_w, video_h = w * 3, header + 2 * (h + band)
    if video_w % 2 or video_h % 2:
        raise ValueError('Comparison dimensions must be even for H264')
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    backend = None
    writer = None
    observed = []
    camera_poses = []
    try:
        with ExitStack() as stack:
            readers = {}
            for key, path in episode.videos.items():
                reader = VideoReader(path, episode.resolution)
                readers[key] = reader
                stack.callback(reader.close)
            backend = MujocoBackend(cfg, cameras=True, viewer=args.viewer)
            stack.callback(backend.close)
            if any(tuple(m['rgb']['resolution']) != episode.resolution for m in backend.rig['cameras'].values()):
                raise ValueError('Simulation cameras must match source native image dimensions')
            report['camera_rig'] = backend.rig
            initial_task = backend.read().extras['objects']
            initial_time = backend.sim_time
            writer = encoder(out / 'comparison.mp4', video_w, video_h, episode.fps / args.stride)
            t0 = time.monotonic()
            for frame in range(n):
                if args.viewer and not backend.is_running():
                    break
                real = {k: reader.read() for k, reader in readers.items()}
                place_kinematic_observation(backend, episode.state_si[frame])
                # All rows are placed and checked, even with render subsampling.
                backend.enable_cameras = frame % args.stride == 0 or frame in frames_to_save
                obs = backend.read()
                max_error = max(max_error, float(np.max(np.abs(obs.position - episode.state_si[frame]))))
                observed.append(obs.position)
                if backend.sim_time != initial_time or obs.extras['objects'] != initial_task:
                    raise RuntimeError('Kinematic replay changed simulation time or task object poses')
                if backend.enable_cameras:
                    camera_poses.append({'frame': frame, 'timestamp': float(episode.timestamps[frame]),
                                         'cameras': obs.extras['cameras']})
                    panel = Image.new('RGB', (video_w, video_h), '#17202a')
                    draw = ImageDraw.Draw(panel)
                    draw.text((12, 10), f'Episode {args.episode:06d} | Frame {frame:04d} | {frame/episode.fps:06.2f} s | '
                              'Recorded STATE playback; SIM objects static; CAD / nominal RGB intrinsics', font=font, fill='white')
                    for col, (key, name) in enumerate(CAMERAS.items()):
                        x = col * w
                        draw.text((x+8, header+3), f'REAL / {key}', font=font, fill='#9fdbff')
                        panel.paste(Image.fromarray(real[key]), (x, header+band))
                        draw.text((x+8, header+band+h+3), f'SIM / {name}', font=font, fill='#ffd58a')
                        panel.paste(Image.fromarray(obs.images[name]), (x, header+2*band+h))
                    if frame % args.stride == 0:
                        writer.stdin.write(panel.tobytes())
                        encoded += 1
                    if frame in frames_to_save:
                        panel.save(out / 'samples' / f'frame_{frame:06d}.jpg', quality=94)
                        for key, name in CAMERAS.items():
                            Image.fromarray(real[key]).save(out / 'samples' / f'{frame:06d}_real_{name}.png')
                            Image.fromarray(obs.images[name]).save(out / 'samples' / f'{frame:06d}_sim_{name}.png')
                        Image.blend(Image.fromarray(real['image']), Image.fromarray(obs.images['overhead']), .5).save(
                            out / 'samples' / f'{frame:06d}_overhead_overlay.png')
                frames_done += 1
                if args.viewer:
                    backend.viewer.sync()
                    delay = t0 + frames_done / episode.fps - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                if frame % 150 == 0:
                    print(f'Replayed {frames_done}/{n}; comparison frames {encoded}; elapsed {time.monotonic()-t0:.1f}s', flush=True)
            if frames_done == len(episode):
                for reader in readers.values():
                    reader.verify_end()
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError('Comparison video encoder failed')
            writer = None
            report['comparison_video_info'] = probe_video(
                out / 'comparison.mp4', encoded, episode.fps / args.stride, (video_w, video_h))
            report['render_wall_seconds'] = time.monotonic()-t0
        np.savez_compressed(out / 'states.npz', state_raw=episode.state_raw[:frames_done],
                            actions_raw=episode.action_raw[:frames_done], state_si=episode.state_si[:frames_done],
                            actions_si=episode.action_si[:frames_done], replayed_state_si=np.asarray(observed),
                            timestamp=episode.timestamps[:frames_done], frame_index=np.arange(frames_done))
        (out / 'camera_poses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in camera_poses))
        report.update(status='complete' if frames_done == n else 'interrupted', replayed_frames=frames_done,
                      comparison_frames=encoded, max_state_replay_error_si=max_error,
                      source_video_lengths_verified_by_decode=frames_done == len(episode),
                      source_files_unchanged=all(digest(f) == sha for f, sha in report['source_files'].items()))
        if not report['source_files_unchanged'] or max_error > 1e-12:
            raise RuntimeError('Source integrity or state-placement check failed')
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'report': str(out / 'report.json'), 'frames': frames_done, 'video': str(out / 'comparison.mp4')}))
    except BaseException as exc:
        report.update(status='failed', replayed_frames=frames_done, error=str(exc))
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        raise
    finally:
        if writer:
            writer.stdin.close()
            if writer.poll() is None:
                writer.terminate()
            writer.wait()


if __name__ == '__main__':
    main()
