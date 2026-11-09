#!/usr/bin/env python3
"""Read-only split/duplicate retrieval over an explicit JSONL episode manifest.

Manifest rows contain path and split. This reports candidates, not acceptance or
semantic equivalence. Source HDF5 files and published splits are never modified.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from piper_sim.contracts import LOWER, UPPER


def fingerprint(row):
    path = Path(row['path']).resolve()
    if row['split'] not in ('train', 'validation', 'test'):
        raise ValueError('Every row needs an explicit train/validation/test split')
    with h5py.File(path, 'r') as f:
        if not f.attrs.get('complete', False): raise ValueError('Incomplete episode: '+str(path))
        meta = json.loads(f.attrs['metadata'])
        if meta.get('units') != (['rad']*6+['m'])*2:
            raise ValueError('Joint units must be radians and total gripper width in metres')
        state = f['observation/state'][:]
        action = f['action/position'][:]
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14 or len(state) < 2:
            raise ValueError('Expected at least two 14D rad/metre transitions')
        if not np.isfinite(state).all() or not np.isfinite(action).all(): raise ValueError('Nonfinite data')
        times = f['observation/sim_time'][:]
        if np.any(np.diff(times) <= 0): raise ValueError('Nonmonotonic time')
        hashed = hashlib.sha256()
        for value in (state, action, f['next_observation/state'][-1], times-times[0]):
            hashed.update(np.asarray(value, dtype='<f8').tobytes())
        phase = (times-times[0])/(times[-1]-times[0])
        curve = np.column_stack([np.interp(np.linspace(0, 1, 129), phase, state[:, c]) for c in range(14)])
        curve = (curve-LOWER)/(UPPER-LOWER)
        layout = meta['task_scene']['layout']
        names = sorted(layout['active_objects'])
        positions = np.asarray([layout['poses'][n]['position'] for n in names], float)
        quaternions = np.asarray([layout['poses'][n]['quaternion_wxyz'] for n in names], float)
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all() or np.any(np.linalg.norm(quaternions, axis=1) == 0):
            raise ValueError('Invalid layout pose')
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        # q and -q represent the same orientation.
        for q in quaternions:
            if q[np.flatnonzero(abs(q) > 1e-12)[0]] < 0: q *= -1
        canonical = np.column_stack([positions, quaternions]).round(12)
        canonical[canonical == 0] = 0.  # Canonicalize -0 after quaternion sign/rounding.
        layout_hash = hashlib.sha256(json.dumps([names, canonical.tolist()], separators=(',', ':')).encode()).hexdigest()
        images = {}
        sample_indices = np.unique(np.rint(np.linspace(0, len(state)-1, 9)).astype(int))
        for camera, frames in f.get('observation/images', {}).items():
            images[camera] = np.stack([np.asarray(Image.fromarray(frames[i]).resize((32, 24),
                Image.Resampling.BOX), dtype=np.float32)/255 for i in sample_indices])
        result = {**row, 'path': str(path), 'frames': len(state), 'seed': layout['seed'],
            'randomized': layout['randomized'], 'layout_sha256': layout_hash,
            'trajectory_sha256': hashed.hexdigest(), 'training_ready_claim': bool(f.attrs.get('training_ready', False)),
            'rgb_sample_indices': sample_indices.tolist(), 'image_cameras': sorted(images),
            'instruction': meta.get('task'), 'bytes': path.stat().st_size}
    return result, dict(curve=curve, images=images, names=names, positions=positions, quaternions=quaternions)


def compare(a, b, *, trajectory_rms=.01, image_mae=.03):
    left, x = a
    right, y = b
    rms = float(np.sqrt(np.mean((x['curve']-y['curve'])**2)))
    cameras = sorted(x['images'].keys() & y['images'].keys())
    visual = {camera: float(np.mean(abs(x['images'][camera]-y['images'][camera]))) for camera in cameras
              if x['images'][camera].shape == y['images'][camera].shape}
    exact_layout = left['layout_sha256'] == right['layout_sha256']
    exact_trajectory = left['trajectory_sha256'] == right['trajectory_sha256']
    result = {'left': left['path'], 'right': right['path'], 'cross_split': left['split'] != right['split'],
        'exact_layout': exact_layout, 'exact_trajectory': exact_trajectory,
        'normalized_joint_rms': rms, 'near_trajectory_candidate': rms <= trajectory_rms,
        'sampled_rgb_mae_0_1': visual,
        'near_visual_candidate': bool(visual) and all(v <= image_mae for v in visual.values())}
    if x['names'] == y['names']:
        result['layout_position_rms_m'] = float(np.sqrt(np.mean((x['positions']-y['positions'])**2)))
        cosine = np.minimum(1., abs(np.sum(x['quaternions']*y['quaternions'], axis=1)))
        result['layout_max_rotation_difference_deg'] = float(np.rad2deg(2*np.arccos(cosine)).max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trajectory-rms', type=float, default=.01)
    parser.add_argument('--image-mae', type=float, default=.03)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not 0 <= args.trajectory_rms <= 1 or not 0 <= args.image_mae <= 1: parser.error('Invalid retrieval thresholds')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    episodes = [fingerprint(row) for row in rows]
    pairs = [compare(a, b, trajectory_rms=args.trajectory_rms, image_mae=args.image_mae)
             for a, b in combinations(episodes, 2)]
    report = {'status': 'retrieval_diagnostic_no_acceptance', 'episodes': [r for r, _ in episodes],
        'pairs': pairs, 'exact_cross_split_pairs': [p for p in pairs if p['cross_split'] and
            (p['exact_layout'] or p['exact_trajectory'])],
        'thresholds': {'normalized_joint_rms': args.trajectory_rms, 'sampled_rgb_mae_0_1': args.image_mae},
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['Training-ready flags are source claims; this retrieval does not recertify quality/checksums.',
            '129 time-normalized joint samples and 9 low-resolution RGB samples are retrieval features only.',
            'Shared backgrounds can look similar; phase alignment can hide local differences. Review flagged pairs.',
            'No semantic visual model, crop/occlusion invariance, dynamic-time warping or large-scale validation.',
            'No detected overlap in a small manifest is not proof that a larger dataset is leakage-free.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'episodes': len(episodes), 'pairs': len(pairs),
        'exact_cross_split_pairs': len(report['exact_cross_split_pairs'])}))


if __name__ == '__main__':
    main()
