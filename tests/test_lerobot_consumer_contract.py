import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location('consumer_checker',
    Path(__file__).resolve().parents[1]/'scripts/check_lerobot_reader.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def fixture_export(tmp_path):
    count, fps = 6, 30.
    source = tmp_path/'episode.h5'
    root = tmp_path/'dataset'
    path = root/'data/chunk-000/episode_000000.parquet'
    path.parent.mkdir(parents=True)
    states = np.tile([0.,np.pi/2,-np.pi/4,0.,0.,0.,.0475]*2, (count, 1)).astype(np.float32)
    states[:, 0] = np.arange(count)*.01
    actions = states.copy(); actions[:, 0] += .02; actions[:, 6] += .001
    with h5py.File(source, 'w') as f:
        f.attrs.update(complete=True, num_transitions=count,
            features=json.dumps([f'{s}.{j}' for s in ['left','right']
                for j in ['joint1','joint2','joint3','joint4','joint5','joint6','gripper_width']]),
            units=json.dumps(['rad']*6+['m']+['rad']*6+['m']), metadata=json.dumps({'fps': fps}))
        f['observation/state'] = states
        f['action/position'] = actions
        f['observation/timestamp'] = 100.+np.arange(count)/fps
        f['observation/sim_time'] = 1.+np.arange(count)/fps
    def convert(values):
        result = values.astype(np.float64)
        for offset in [0, 7]:
            result[:, offset:offset+6] = np.rad2deg(result[:, offset:offset+6])
            result[:, offset+6] /= .095
        return result.astype(np.float32)
    columns = {'observation.state': pa.FixedSizeListArray.from_arrays(pa.array(convert(states).ravel()), 14),
        'action': pa.FixedSizeListArray.from_arrays(pa.array(convert(actions).ravel()), 14),
        'timestamp': pa.array((np.arange(count)/fps).astype(np.float32)),
        'source_timestamp': pa.array(100.+np.arange(count)/fps),
        'sim_timestamp': pa.array(1.+np.arange(count)/fps),
        'frame_index': pa.array(np.arange(count)), 'index': pa.array(np.arange(count)),
        'episode_index': pa.array(np.zeros(count, np.int64)), 'task_index': pa.array(np.zeros(count, np.int64))}
    pq.write_table(pa.table(columns), path)
    names = [name for s in ['left','right'] for name in
        [*[f'{s}_arm.joint_{i}' for i in range(1,7)], f'{s}_arm.gripper_open']]
    features = {k: {'dtype':'float32','shape':[14],'names':names.copy()} for k in ['action','observation.state']}
    for k,dtype in {'timestamp':'float32','source_timestamp':'float64','sim_timestamp':'float64',
                    'frame_index':'int64','index':'int64','episode_index':'int64','task_index':'int64'}.items():
        features[k] = {'dtype':dtype,'shape':[1],'names':None}
    for name in ['cam_external_rgb','cam_left_wrist_rgb','cam_right_wrist_rgb']:
        features['observation.images.'+name] = {'dtype':'video','shape':[3,240,320],
            'names':['channels','height','width'], 'info':{'video.height':240,'video.width':320,
                'video.channels':3,'video.fps':fps,'video.is_depth_map':False,'has_audio':False}}
    info = {'codebase_version':'v2.1','robot_type':'piper','total_episodes':1,'total_tasks':1,
        'total_videos':3,'total_frames':count,'fps':fps,'features':features,
        'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
    return source, root, path, info


def test_all_numeric_rows_checked_without_trusting_public_metadata(tmp_path):
    source, root, _, info = fixture_export(tmp_path)
    with h5py.File(source) as f:
        result = checker.check_tabular_rows(root, f, info)
    assert result['checked_rows'] == 6
    assert len(result['checked_columns']) == 9
    assert not result['unexpected_public_columns']
    assert max(result['max_vector_absolute_error'].values()) < 1e-5


@pytest.mark.parametrize('fault', ['declared_privileged', 'undeclared_privileged', 'missing_camera',
                                  'wrong_joint_order', 'wrong_dtype', 'unsafe_layout'])
def test_consumer_owned_schema_rejects_extra_fields_and_incompatible_layout(tmp_path, fault):
    source, root, path, info = fixture_export(tmp_path)
    if fault == 'declared_privileged':
        info['features']['observation.objects.pose'] = {'dtype':'float32','shape':[3],'names':None}
    elif fault == 'undeclared_privileged':
        table = pq.read_table(path)
        pq.write_table(table.append_column('teacher.target', pa.array([1]*6)), path)
    elif fault == 'missing_camera':
        info['features'].pop('observation.images.cam_right_wrist_rgb')
    elif fault == 'wrong_joint_order':
        info['features']['action']['names'].reverse()
    elif fault == 'wrong_dtype':
        info['features']['action']['dtype'] = 'float64'
    else:
        info['data_path'] = '../unrelated.parquet'
    with h5py.File(source) as f, pytest.raises(ValueError):
        checker.check_tabular_rows(root, f, info)


def test_schema_guard_remains_active_with_python_optimization(tmp_path):
    _, _, _, info = fixture_export(tmp_path)
    info['features']['teacher.object_pose'] = {'dtype':'float32','shape':[3],'names':None}
    path = tmp_path/'bad_info.json'; path.write_text(json.dumps(info))
    code = 'import runpy,json,sys; m=runpy.run_path(sys.argv[1]); m["validate_public_schema"](json.load(open(sys.argv[2])))'
    run = subprocess.run([sys.executable, '-O', '-c', code, str(spec.origin), str(path)], capture_output=True, text=True)
    assert run.returncode != 0 and 'Unexpected public features' in run.stderr


@pytest.mark.parametrize('column', ['observation.state','action','timestamp','source_timestamp',
    'sim_timestamp','frame_index','index','episode_index','task_index'])
def test_corruption_in_unsampled_middle_row_is_rejected(tmp_path, column):
    source, root, path, info = fixture_export(tmp_path)
    table = pq.read_table(path)
    values = table[column].to_pylist()
    # Existing official-reader sampling uses 0, n//2, n-2, n-1. Row 1 is outside it.
    if column in ('observation.state','action'):
        values[1][0] += 1.
    else:
        values[1] += 1
    table = table.set_column(table.column_names.index(column), column, pa.array(values, type=table[column].type))
    pq.write_table(table, path)
    with h5py.File(source) as f, pytest.raises(AssertionError):
        checker.check_tabular_rows(root, f, info)


@pytest.mark.parametrize('fault', ['units', 'order', 'partial', 'fps', 'nan', 'null'])
def test_inconsistent_source_or_invalid_values_are_rejected(tmp_path, fault):
    source, root, path, info = fixture_export(tmp_path)
    if fault in ('nan','null'):
        table = pq.read_table(path)
        values = table['sim_timestamp'].to_pylist(); values[1] = np.nan if fault == 'nan' else None
        table = table.set_column(table.column_names.index('sim_timestamp'), 'sim_timestamp', pa.array(values))
        pq.write_table(table, path)
    else:
        with h5py.File(source, 'r+') as f:
            if fault == 'units': f.attrs['units'] = json.dumps(['degree']*14)
            if fault == 'order': f.attrs['features'] = json.dumps(list(reversed(checker.SOURCE_NAMES)))
            if fault == 'partial': f.attrs['complete'] = False
            if fault == 'fps': f.attrs['metadata'] = json.dumps({'fps':10})
    with h5py.File(source) as f, pytest.raises(ValueError):
        checker.check_tabular_rows(root, f, info)
