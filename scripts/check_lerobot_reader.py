#!/usr/bin/env python3
"""Read a local export with the official LeRobot v2.1 reader, without network.

Use a separate environment with lerobot==0.3.3, datasets==3.6.0 and PyAV.
This is a consumer compatibility check; it does not grant task acceptance.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import h5py
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
    from piper_sim.lerobot_alignment import CAMERA_MAPPING

    info = json.loads((args.dataset/'meta/info.json').read_text())
    if info['total_episodes'] != 1:
        raise ValueError('Supply a single-episode export for source comparison')
    fps = info['fps']
    chunk_size = 16
    dataset = LeRobotDataset('local/piper_audit', root=args.dataset,
        download_videos=False, video_backend='pyav',
        delta_timestamps={'action': [i/fps for i in range(chunk_size)]})
    camera_keys = sorted(k for k in info['features'] if k.startswith('observation.images.'))
    expected_keys = set(info['features']) | {'task', 'action_is_pad'}
    checks = []
    # Compute units independently of the exporter implementation.
    def public_units(x):
        result = np.array(x, dtype=np.float64, copy=True)
        result[..., :6] *= 180/np.pi
        result[..., 7:13] *= 180/np.pi
        result[..., [6, 13]] /= .095
        return result.astype(np.float32)

    with h5py.File(args.source, 'r') as source:
        count = int(source.attrs['num_transitions'])
        assert len(dataset) == count == info['total_frames']
        for index in sorted({0, count//2, count-2, count-1}):
            item = dataset[index]
            assert set(item) == expected_keys, (set(item)-expected_keys, expected_keys-set(item))
            assert item['observation.state'].shape == (14,)
            assert item['action'].shape == (chunk_size, 14)
            assert item['task'] == json.loads(source.attrs['metadata'])['task']
            assert int(item['frame_index']) == index
            assert abs(float(item['timestamp'])-index/fps) < 2e-5
            np.testing.assert_allclose(item['observation.state'].numpy(),
                public_units(source['observation/state'][index]), atol=1e-5, rtol=1e-6)
            indices = np.minimum(np.arange(index, index+chunk_size), count-1)
            expected_actions = public_units(np.stack([source['action/position'][i] for i in indices]))
            np.testing.assert_allclose(item['action'].numpy(), expected_actions, atol=1e-5, rtol=1e-6)
            np.testing.assert_array_equal(item['action_is_pad'].numpy(),
                np.arange(index, index+chunk_size) >= count)
            images = {}
            for key in camera_keys:
                pixels = item[key]
                assert pixels.shape == tuple(info['features'][key]['shape'])
                assert pixels.dtype == torch.float32 and torch.isfinite(pixels).all()
                assert 0 <= pixels.min() <= pixels.max() <= 1
                camera = CAMERA_MAPPING[key.rsplit('.', 1)[-1]]
                raw = source['observation/images/'+camera][index].astype(np.float32)/255
                difference = pixels.permute(1, 2, 0).numpy()-raw
                images[key] = {'shape': list(pixels.shape),
                    'lossy_video_mean_absolute_error_0_to_1': float(np.abs(difference).mean()),
                    'lossy_video_rmse_0_to_1': float(np.sqrt(np.mean(difference**2)))}
            checks.append({'index': index, 'task': item['task'], 'images': images,
                'action_chunk_shape': list(item['action'].shape),
                'padding': item['action_is_pad'].tolist(), 'public_keys': sorted(item)})
    report = {'status': 'consumer_check_passed', 'training_ready': None,
        'scope': 'Official local reader compatibility and sampled HDF5 alignment; no new quality grant.',
        'dataset': str(args.dataset.resolve()), 'source': str(args.source.resolve()),
        'versions': {k: importlib.metadata.version(k) for k in
                     ('lerobot', 'datasets', 'torch', 'torchvision', 'av')},
        'reader_codebase': CODEBASE_VERSION, 'frames': count, 'fps': fps,
        'units': 'joint degrees, gripper full width / 0.095 m',
        'checks': checks, 'privileged_fields_in_public_rows': False,
        'limitations': 'Sampled consumer rows only; H.264 is lossy. Full frame-count/stream/quality checks are separate.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': report['status'], 'frames': count, 'checks': len(checks)}))


if __name__ == '__main__':
    main()
