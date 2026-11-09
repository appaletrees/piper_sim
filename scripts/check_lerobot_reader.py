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


# Consumer-owned contract, independent of what the export claims in info.json.
CAMERAS = {'cam_external_rgb': 'overhead', 'cam_left_wrist_rgb': 'left_wrist',
           'cam_right_wrist_rgb': 'right_wrist'}
PUBLIC_NAMES = [name for side in ('left', 'right') for name in
    [*[f'{side}_arm.joint_{i}' for i in range(1, 7)], f'{side}_arm.gripper_open']]
SOURCE_NAMES = [f'{side}.{name}' for side in ('left', 'right')
                for name in [*[f'joint{i}' for i in range(1, 7)], 'gripper_width']]
SCALARS = {'timestamp': 'float32', 'source_timestamp': 'float64', 'sim_timestamp': 'float64',
           'frame_index': 'int64', 'episode_index': 'int64', 'index': 'int64', 'task_index': 'int64'}
NUMERIC_KEYS = {'observation.state', 'action', *SCALARS}
CAMERA_KEYS = {'observation.images.'+name for name in CAMERAS}
PUBLIC_KEYS = NUMERIC_KEYS | CAMERA_KEYS


def need(condition, message):
    if not condition:
        raise ValueError(message)


def public_units(x):
    """Consumer conversion, without importing the exporter implementation."""
    import numpy as np
    result = np.array(x, dtype=np.float64, copy=True)
    result[..., :6] *= 180/np.pi
    result[..., 7:13] *= 180/np.pi
    result[..., [6, 13]] /= .095
    return result.astype(np.float32)


def validate_public_schema(info):
    import math
    need(info['codebase_version'] == 'v2.1' and info['robot_type'] == 'piper', 'Unsupported consumer schema')
    need(info['total_episodes'] == 1 and info['total_tasks'] == 1 and info['total_videos'] == 3,
         'Supply a single-episode, three-camera Piper export')
    need(math.isfinite(info['fps']) and info['fps'] > 0, 'Invalid consumer fps')
    need(set(info['features']) == PUBLIC_KEYS,
         'Unexpected public features: '+str(sorted(set(info['features']) ^ PUBLIC_KEYS)))
    for key in ('observation.state', 'action'):
        field = info['features'][key]
        need(field['dtype'] == 'float32' and field['shape'] == [14] and field['names'] == PUBLIC_NAMES,
             'Invalid consumer vector schema: '+key)
    for key, dtype in SCALARS.items():
        field = info['features'][key]
        need(field['dtype'] == dtype and field['shape'] == [1] and field['names'] is None,
             'Invalid consumer scalar schema: '+key)
    for key in CAMERA_KEYS:
        field = info['features'][key]
        shape = field['shape']
        need(field['dtype'] == 'video' and len(shape) == 3 and shape[0] == 3
             and all(type(v) is int and v > 0 for v in shape)
             and field['names'] == ['channels', 'height', 'width'], 'Invalid consumer camera schema: '+key)
        video = field['info']
        need(video['video.height'] == shape[1] and video['video.width'] == shape[2]
             and video['video.channels'] == 3 and video['video.fps'] == info['fps']
             and video['video.is_depth_map'] is False and video['has_audio'] is False,
             'Inconsistent video metadata: '+key)
    # This check targets this repository's single-episode export layout.
    need(info['data_path'] == 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
         and info['video_path'] == 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
         'Unsupported consumer file layout')


def check_tabular_rows(root, source, info):
    """Validate every numeric row, including actual (not only declared) columns."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    validate_public_schema(info)
    need(bool(source.attrs.get('complete')), 'Incomplete HDF5 source')
    need(json.loads(source.attrs['features']) == SOURCE_NAMES, 'Source joint order mismatch')
    need(json.loads(source.attrs['units']) == ['rad']*6+['m']+['rad']*6+['m'], 'Source units mismatch')
    count = int(source.attrs['num_transitions'])
    need(count > 0 and count == info['total_frames'], 'Source/export frame count mismatch')
    need(float(json.loads(source.attrs['metadata'])['fps']) == info['fps'], 'Source/export fps mismatch')
    path = Path(root)/'data/chunk-000/episode_000000.parquet'
    table = pq.read_table(path)
    need(set(table.column_names) == NUMERIC_KEYS, 'Unexpected actual Parquet columns')
    need(table.num_rows == count, 'Parquet frame count mismatch')
    need(all(column.null_count == 0 for column in table.columns), 'Null numeric row')
    for key in ('observation.state', 'action'):
        dtype = table.schema.field(key).type
        need(pa.types.is_fixed_size_list(dtype) and dtype.list_size == 14 and dtype.value_type == pa.float32(),
             'Invalid Parquet vector type: '+key)
    for key, dtype in SCALARS.items():
        need(table.schema.field(key).type == pa.from_numpy_dtype(np.dtype(dtype)), 'Invalid Parquet scalar type: '+key)
    values = {key: np.asarray(table[key].to_pylist()) for key in NUMERIC_KEYS}
    need(all(np.isfinite(v).all() for v in values.values()), 'Nonfinite exported numeric data')
    differences = {}
    for key, original in (('observation.state', 'observation/state'), ('action', 'action/position')):
        expected = public_units(source[original][:])
        np.testing.assert_allclose(values[key], expected, atol=1e-5, rtol=1e-6, err_msg=key)
        differences[key] = float(np.max(np.abs(values[key]-expected)))
    expected = {'timestamp': (np.arange(count)/info['fps']).astype(np.float32),
        'source_timestamp': source['observation/timestamp'][:], 'sim_timestamp': source['observation/sim_time'][:],
        'frame_index': np.arange(count), 'index': np.arange(count),
        'episode_index': np.zeros(count, dtype=np.int64), 'task_index': np.zeros(count, dtype=np.int64)}
    for key, original in expected.items():
        np.testing.assert_array_equal(values[key], original, err_msg=key)
    return {'checked_rows': count, 'checked_columns': sorted(NUMERIC_KEYS),
            'max_vector_absolute_error': differences, 'unexpected_public_columns': []}


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

    info = json.loads((args.dataset/'meta/info.json').read_text())
    validate_public_schema(info)
    fps = info['fps']
    chunk_size = 16
    dataset = LeRobotDataset('local/piper_audit', root=args.dataset,
        download_videos=False, video_backend='pyav',
        delta_timestamps={'action': [i/fps for i in range(chunk_size)]})
    camera_keys = sorted(CAMERA_KEYS)
    expected_keys = PUBLIC_KEYS | {'task', 'action_is_pad'}
    checks = []

    with h5py.File(args.source, 'r') as source:
        tabular = check_tabular_rows(args.dataset, source, info)
        count = int(source.attrs['num_transitions'])
        need(len(dataset) == count == info['total_frames'], 'Reader frame count mismatch')
        for index in sorted({0, count//2, count-2, count-1}):
            item = dataset[index]
            need(set(item) == expected_keys, 'Unexpected reader row fields: '+str(set(item)^expected_keys))
            need(item['observation.state'].shape == (14,), 'Reader state shape mismatch')
            need(item['action'].shape == (chunk_size, 14), 'Reader action chunk shape mismatch')
            need(item['task'] == json.loads(source.attrs['metadata'])['task'], 'Reader task mismatch')
            need(int(item['frame_index']) == index, 'Reader frame index mismatch')
            need(abs(float(item['timestamp'])-index/fps) < 2e-5, 'Reader timestamp mismatch')
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
                need(pixels.shape == tuple(info['features'][key]['shape']), 'Reader image shape mismatch: '+key)
                need(pixels.dtype == torch.float32 and torch.isfinite(pixels).all(), 'Invalid reader image: '+key)
                need(0 <= pixels.min() <= pixels.max() <= 1, 'Reader image range mismatch: '+key)
                camera = CAMERAS[key.rsplit('.', 1)[-1]]
                raw = source['observation/images/'+camera][index].astype(np.float32)/255
                need(tuple(raw.shape) == tuple(pixels.permute(1, 2, 0).shape),
                     'This checker requires native source image resolution: '+key)
                difference = pixels.permute(1, 2, 0).numpy()-raw
                images[key] = {'shape': list(pixels.shape),
                    'lossy_video_mean_absolute_error_0_to_1': float(np.abs(difference).mean()),
                    'lossy_video_rmse_0_to_1': float(np.sqrt(np.mean(difference**2)))}
            checks.append({'index': index, 'task': item['task'], 'images': images,
                'action_chunk_shape': list(item['action'].shape),
                'padding': item['action_is_pad'].tolist(), 'public_keys': sorted(item)})
    report = {'status': 'consumer_check_passed', 'training_ready': None,
        'scope': 'Independent Piper public schema, all numeric rows and official local reader samples; no new quality grant.',
        'dataset': str(args.dataset.resolve()), 'source': str(args.source.resolve()),
        'versions': {k: importlib.metadata.version(k) for k in
                     ('lerobot', 'datasets', 'torch', 'torchvision', 'av')},
        'reader_codebase': CODEBASE_VERSION, 'frames': count, 'fps': fps,
        'units': 'joint degrees, gripper full width / 0.095 m',
        'tabular': tabular, 'checks': checks, 'privileged_fields_in_public_rows': False,
        'limitations': 'All numeric rows checked; reader/action-chunk/RGB checks are sampled. H.264 errors are measured, '
                       'not a visual quality threshold. Native-resolution single-episode Piper exports only. '
                       'Full video frame-count/stream/quality checks are separate.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': report['status'], 'frames': count, 'checks': len(checks)}))


if __name__ == '__main__':
    main()
