"""Offline, fail-closed audit of scripted demonstrations. No simulator control.

Inspect the closed HDF5, not the collector's in-memory success flag. Images are
read one at a time; a full three-camera episode need not fit in RAM.
"""
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .contracts import FEATURE_NAMES, LOWER, UPPER
from .experts.quality import MotionQuality
from .config import ROOT

AUDIT_VERSION = 'piper.collection.v3'


def load_acceptance_policy(path=None):
    """Only an operator-supplied file, never a path/config from the recording."""
    import yaml
    path = Path(path) if path is not None else ROOT/'configs/collection/quality_policy.yaml'
    raw = path.read_bytes()
    try:
        settings = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError('Malformed production acceptance policy') from exc
    if not isinstance(settings, dict) or settings.get('schema_version') != 1:
        raise ValueError('Unsupported production acceptance policy')
    for name in ('motion', 'quality', 'dataset'):
        if not isinstance(settings.get(name), dict):
            raise ValueError('Missing policy section: '+name)
        for key, value in settings[name].items():
            if key == 'required_cameras':
                if (not isinstance(value, list) or len(value) != 3 or
                        set(value) != {'overhead', 'left_wrist', 'right_wrist'}):
                    raise ValueError('Production policy requires the three workcell cameras')
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not np.isfinite(value) or value < 0):
                raise ValueError('Non-finite or invalid policy limit: '+name+'/'+key)
    if 'tcp_motion' in settings:
        from .tcp_audit import validate_tcp_policy
        validate_tcp_policy(settings['tcp_motion'])
    return settings, {'source': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_streams(f, metadata, settings, require_rgb=True):
    """Check shapes, clocks, transitions and EVERY image, including next_obs."""
    limits = settings['dataset']
    n, fps = int(f.attrs['num_transitions']), float(metadata['fps'])
    _need(np.isfinite(fps) and fps > 0, 'invalid_fps')
    _need(n >= limits['min_frames'], 'insufficient_frames')
    _need(n/fps <= limits['max_duration_s'], 'episode_too_long')
    _need(tuple(json.loads(f.attrs['features'])) == FEATURE_NAMES, 'feature_order')
    _need(json.loads(f.attrs['units']) == ['rad']*6+['m']+['rad']*6+['m'], 'units')
    tol = limits['timestamp_tolerance_s']
    problems, stats = [], {}
    cameras = sorted(f.get('observation/images', {}))
    if require_rgb and set(cameras) != set(limits['required_cameras']):
        problems.append('missing_required_rgb_streams')
    arrays = {}
    for prefix in ('observation', 'next_observation'):
        _need(sorted(f.get(prefix+'/images', {})) == cameras, 'camera_set_changed')
        for field in ('state', 'sim_time', 'timestamp', 'extras_json'):
            ds = f[prefix+'/'+field]
            _need(ds.shape == ((n, 14) if field == 'state' else (n,)), 'shape/'+prefix+'/'+field)
            if field != 'extras_json':
                arrays[prefix+'/'+field] = ds[:]
                _need(np.isfinite(ds[:]).all(), 'nonfinite/'+prefix+'/'+field)
        _need(np.all(np.diff(arrays[prefix+'/timestamp']) > 0), 'nonmonotonic_host_clock')
    before, after = arrays['observation/sim_time'], arrays['next_observation/sim_time']
    _need(np.allclose(after-before, 1/fps, atol=tol, rtol=0), 'transition_duration')
    _need(np.allclose(after[:-1], before[1:], atol=tol, rtol=0), 'simulation_clock_gap')
    _need(np.array_equal(arrays['next_observation/state'][:-1], arrays['observation/state'][1:]),
          'state_transition_discontinuity')
    _need(np.all(arrays['observation/timestamp'][1:] >= arrays['next_observation/timestamp'][:-1]),
          'overlapping_host_transitions')
    action = f['action/position'][:]
    _need(action.shape == (n, 14) and np.isfinite(action).all(), 'invalid_action')
    _need(np.all(action >= LOWER-1e-6) and np.all(action <= UPPER+1e-6), 'action_out_of_bounds')
    sequence, timestamps = f['action/sequence'][:], f['action/timestamp'][:]
    _need(sequence.shape == (n,) and np.issubdtype(sequence.dtype, np.integer)
          and sequence[0] >= 0 and np.all(np.diff(sequence) == 1), 'action_sequence_gap')
    _need(timestamps.shape == (n,) and np.isfinite(timestamps).all()
          and np.all(np.diff(timestamps) > 0), 'action_clock')
    _need(np.all(timestamps >= arrays['observation/timestamp'])
          and np.all(timestamps <= arrays['next_observation/timestamp']), 'action_not_between_states')
    for name in cameras:
        b, a = (f[p+'/images/'+name] for p in ('observation', 'next_observation'))
        _need(b.ndim == 4 and b.shape[0] == n and b.shape[-1] == 3 and b.dtype == np.uint8
              and a.shape == b.shape and a.dtype == np.uint8, 'rgb_shape_or_dtype/'+name)
        width, height = metadata['camera_rig']['cameras'][name]['rgb']['resolution']
        _need(b.shape[1:] == (height, width, 3), 'camera_resolution/'+name)
        for prefix, times in (('observation', before), ('next_observation', after)):
            captured = f[prefix+'/image_timestamps/'+name][:]
            _need(captured.shape == (n,) and np.allclose(captured, times, atol=tol, rtol=0),
                  'rgb_state_desynchronized/'+name)
        bad, repeats, discontinuities, previous = 0, 0, 0, None
        digest = hashlib.sha256()
        # Include every recorded frame in integrity checks; sample pixels only
        # for exposure/contrast statistics, never skip temporal frames.
        for i in range(n):
            pixels, following = b[i], a[i]
            bh = hashlib.sha256(pixels.tobytes()).digest()
            ah = hashlib.sha256(following.tobytes()).digest()
            if previous is not None:
                discontinuities += int(bh != previous)
            repeats += int(bh == ah)
            previous = ah
            digest.update(bh); digest.update(ah)
            for frame in (pixels, following):
                luma = frame[::8, ::8].astype(float) @ [0.2126, 0.7152, 0.0722]
                clipped = max(np.mean(luma <= 1), np.mean(luma >= 254))
                bad += int(luma.std() < limits['min_image_luma_std']
                           or clipped > limits['max_image_clipped_fraction'])
        stats[name] = {'shape': list(b.shape), 'checked_images': 2*n,
                       'bad_image_fraction': bad/(2*n), 'identical_transition_fraction': repeats/n,
                       'discontinuous_transitions': discontinuities, 'rgb_sha256': digest.hexdigest()}
        if discontinuities: problems.append('rgb_transition_discontinuity/'+name)
        if bad/(2*n) > limits['max_bad_image_fraction']: problems.append('uninformative_rgb/'+name)
        if repeats/n > limits['max_identical_image_fraction']: problems.append('frozen_rgb/'+name)
    return {'passed': not problems, 'violations': problems, 'frames': n, 'fps': fps,
            'duration_s': n/fps, 'cameras': stats, 'rgb': bool(cameras),
            'trajectory_sha256': hashlib.sha256(action.tobytes()+arrays['observation/state'].tobytes()
                                                +arrays['next_observation/state'].tobytes()).hexdigest()}


def audit_task(f, metadata, settings):
    """Recompute geometric success and motion from recorded evidence."""
    from .tasks.clean_table import CleanTableTask
    from .transforms import pose_matrix
    saved = metadata['task_scene']
    cfg = copy.deepcopy(saved['config'])
    cfg['domain_randomization'] = saved.get('domain_profile')
    task = CleanTableTask(cfg, metadata['config'])
    task.reset(saved['layout']['seed'], saved['layout']['randomized'])
    _need(list(task.active_names) == saved['layout']['active_objects'], 'active_inventory_mismatch')
    for name, bounds in saved['bounds_m'].items():
        _need(np.allclose(task.bounds[name], bounds, atol=1e-8, rtol=0), 'geometry_changed/'+name)
    _need(cfg.get('variant') == 'rigid_stack_v1', 'unsupported_collection_task')
    _need(metadata['teacher']['physical_only'] and metadata['teacher']['grasp_mode'] == 'physical',
          'nonphysical_teacher')
    limits, dt = settings['dataset'], 1/metadata['fps']
    motion = MotionQuality(settings, dt)
    motion.add(float(f['observation/sim_time'][0]), metadata['collection_initial_target'],
               f['observation/state'][0], context={'phase': 'prepared'})
    cups = [n for n in task.active_names if task.specs[n]['kind'] == 'cup']
    problems, grasps, previous_poses = set(), {}, None
    object_digest = hashlib.sha256()
    first_table, first_basket, stable_seconds, peak_depth = None, None, 0., 0.
    peak_final_speed, peak_final_angular = 0., 0.
    first_extra = json.loads(f['observation/extras_json'][0])
    previous_poses = first_extra['objects']
    times, actions, measured = (f[k][:] for k in ('next_observation/sim_time', 'action/position',
                                                'next_observation/state'))
    _need(np.isfinite(times).all() and np.isfinite(actions).all() and np.isfinite(measured).all(),
          'nonfinite_motion_data')
    final_hold_frames = int(np.ceil(limits['final_stable_hold_s']/dt))
    previous_grasp = None
    for i, raw in enumerate(f['next_observation/extras_json']):
        before_extra = json.loads(f['observation/extras_json'][i])
        _need(before_extra['objects'] == previous_poses, 'object_transition_discontinuity')
        extra = json.loads(raw)
        teacher, poses = extra['teacher'], extra['objects']
        _need(extra['task']['seed'] == saved['layout']['seed'], 'episode_seed_changed')
        _need(abs(extra['physics_time']-times[i]) <= limits['timestamp_tolerance_s'], 'physics_state_clock')
        _need(set(poses) == set(task.active_names), 'frame_inventory_mismatch')
        for pose in poses.values():
            _need(np.shape(pose['position']) == (3,) and np.shape(pose['quaternion_wxyz']) == (4,)
                  and np.isfinite(pose['position']+pose['quaternion_wxyz']).all()
                  and abs(np.linalg.norm(pose['quaternion_wxyz'])-1) < 1e-4, 'invalid_object_pose')
        object_digest.update(json.dumps(poses, sort_keys=True, separators=(',', ':')).encode())
        for name in f.get('observation/images', {}):
            camera = extra['cameras'][name]
            expected = metadata['camera_rig']['cameras'][name]['rgb']['intrinsic_matrix']
            _need(np.shape(camera['intrinsics']) == (3, 3)
                  and np.allclose(camera['intrinsics'], expected, atol=1e-7, rtol=0),
                  'camera_intrinsics_changed/'+name)
            _need(np.shape(camera['position_world']) == (3,)
                  and np.isfinite(camera['position_world']).all()
                  and np.shape(camera['quaternion_wxyz_ros']) == (4,)
                  and abs(np.linalg.norm(camera['quaternion_wxyz_ros'])-1) < 1e-4,
                  'invalid_camera_pose/'+name)
        _need(teacher['grasp_mode'] == 'physical' and not teacher['active_assist'], 'assisted_frame')
        depth = float(teacher['robot_penetration_m'])
        _need(np.isfinite(depth) and depth >= 0, 'invalid_contact_depth')
        peak_depth = max(peak_depth, depth)
        if depth > settings['quality']['max_robot_penetration_m']: problems.add('robot_penetration')
        motion.add(times[i], actions[i], measured[i], holding=teacher['holding'],
                   context={'phase': teacher['phase'], 'object': teacher['object']})
        grasp = teacher['grasp']
        if grasp:
            name = grasp['object']
            _need(name in task.required and grasp['side'] in ('left', 'right'), 'invalid_grasp_object')
            g = grasps.setdefault(name, {'samples': 0, 'bilateral_samples': 0, 'max_slip_m': 0.,
                                       'max_contact_loss_s': 0., 'current_loss_s': 0.})
            _need(np.isfinite(grasp['slip_m']) and grasp['slip_m'] >= 0, 'invalid_grasp_slip')
            bilateral = set(grasp['contact_fingers']) == {'7', '8'}
            if previous_grasp != name: g['current_loss_s'] = 0.
            g['samples'] += 1; g['bilateral_samples'] += int(bilateral)
            g['max_slip_m'] = max(g['max_slip_m'], grasp['slip_m'])
            g['current_loss_s'] = 0. if bilateral else g['current_loss_s']+dt
            g['max_contact_loss_s'] = max(g['max_contact_loss_s'], g['current_loss_s'])
            previous_grasp = name
        else:
            previous_grasp = None
        metrics = task.metrics(poses, dt)
        stack = metrics['cup_stack']
        if first_table is None and task.stack_on_table_verified: first_table = float(times[i])
        basket = pose_matrix(poses['basket']['position'], poses['basket']['quaternion_wxyz'])
        p = (np.asarray(poses[cups[0]]['position'])-basket[:3, 3]) @ basket[:3, :3]
        size = np.asarray(task.specs['basket']['size_m'])
        if first_basket is None and np.all(abs(p[:2]) < size[:2]/2) and 0 <= p[2] < size[2]:
            first_basket = float(times[i])
        speeds, angular = [], []
        for name in task.required:
            a, b = previous_poses[name], poses[name]
            speeds.append(float(np.linalg.norm(np.subtract(a['position'], b['position']))/dt))
            dot = abs(float(np.dot(a['quaternion_wxyz'], b['quaternion_wxyz'])))
            angular.append(float(2*np.arccos(np.clip(dot, 0., 1.))/dt))
        stable = (metrics['summary']['success'] and grasp is None
                  and max(speeds) <= limits['final_object_speed_m_s']
                  and max(angular) <= limits['final_object_angular_speed_rad_s'])
        stable_seconds = stable_seconds+dt if stable else 0.
        if i >= len(times)-final_hold_frames:
            peak_final_speed = max(peak_final_speed, max(speeds))
            peak_final_angular = max(peak_final_angular, max(angular))
        previous_poses = poses
    for name in task.required:
        g = grasps.get(name)
        if not g:
            problems.add('missing_physical_grasp/'+name)
            continue
        g.pop('current_loss_s')
        g['bilateral_fraction'] = g['bilateral_samples']/g['samples']
        if g['max_slip_m'] > limits['max_grasp_slip_m']: problems.add('grasp_slip/'+name)
        if g['max_contact_loss_s'] > limits['max_contact_loss_s']: problems.add('grasp_contact_loss/'+name)
        if g['bilateral_fraction'] < limits['min_bilateral_contact_fraction']:
            problems.add('grasp_contact_fraction/'+name)
    sequence = first_table is not None and first_basket is not None and first_table < first_basket
    if not sequence: problems.add('cups_not_stacked_on_table_first')
    if not metrics['summary']['success']: problems.add('incomplete_final_inventory')
    if stable_seconds+1e-7 < limits['final_stable_hold_s']: problems.add('unstable_final_placement')
    motion_result = motion.report()
    problems.update('motion/'+v for v in motion_result['violations'])
    outcome = metadata['collection_outcome']
    _need(outcome.get('rollout_accepted', outcome.get('accepted', False)), 'rollout_not_accepted')
    results = outcome['object_results']
    if (len(results) != len(task.required) or {r['name'] for r in results} != set(task.required)
            or not all(r['success'] for r in results)):
        problems.add('incomplete_object_results')
    return {'passed': not problems, 'violations': sorted(problems), 'motion': motion_result,
            'objects_sha256': object_digest.hexdigest(),
            'grasps': grasps, 'max_robot_penetration_m': peak_depth,
            'first_stable_table_stack_s': first_table, 'first_base_cup_in_basket_s': first_basket,
            'final_stable_seconds': stable_seconds, 'final_metrics': metrics,
            'final_peak_object_speed_m_s': peak_final_speed,
            'final_peak_object_angular_speed_rad_s': peak_final_angular}


def audit_episode(path, *, require_rgb=True, policy=None):
    """Failures produce a rejected report; no missing field means 'pass'."""
    result = {'schema_version': AUDIT_VERSION, 'episode': str(Path(path).resolve()),
              'require_rgb': require_rgb, 'passed': False, 'training_ready': False,
              'violations': [], 'checks': {},
              'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        with h5py.File(path, 'r') as f:
            _need(f.attrs.get('complete', False) and f.attrs.get('schema_version') == 'piper.transitions.v1',
                  'incomplete_or_unknown_episode')
            _need(f.attrs.get('success', False), 'unsuccessful_episode')
            meta = json.loads(f.attrs['metadata'])
            _need(np.isfinite(meta['fps']) and meta['fps'] > 0, 'invalid_fps')
            settings, policy_evidence = load_acceptance_policy(policy)
            result['acceptance_policy'] = policy_evidence
            # Retain the declared settings as provenance, but acceptance uses
            # the separately supplied policy even if the producer relaxed its gates.
            result['declared_settings'] = meta['teacher']['motion_and_quality']
            result['settings'] = settings
            checks = [('streams', lambda: audit_streams(f, meta, settings, require_rgb)),
                      ('task', lambda: audit_task(f, meta, settings))]
            if 'tcp_motion' in settings:
                from .tcp_audit import audit_tcp_motion
                checks.append(('tcp_motion', lambda: audit_tcp_motion(f, meta, settings['tcp_motion'])))
            for name, check in checks:
                try:
                    report = check()
                    result['checks'][name] = report
                    result['violations'].extend(name+'/'+v for v in report['violations'])
                except (KeyError, ValueError, TypeError, OSError, IndexError) as exc:
                    result['violations'].append(name+'/'+type(exc).__name__+': '+str(exc))
            result['passed'] = not result['violations']
            result['training_ready'] = bool(result['passed'] and require_rgb)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        result['violations'].append(type(exc).__name__+': '+str(exc))
    return result


def save_audit(episode, report, output):
    """Only this final gate sets training acceptance on a scripted episode."""
    if Path(output).resolve().parent != Path(episode).resolve().parent:
        raise ValueError('The audit sidecar must be saved beside the episode')
    report.pop('source_sha256', None)
    with h5py.File(episode, 'r+') as f:
        f.attrs['dataset_audit'] = json.dumps(report, ensure_ascii=False, allow_nan=False)
        f.attrs['accepted'] = report['training_ready']
        f.attrs['training_ready'] = report['training_ready']
        f.attrs['dataset_audit_sidecar'] = Path(output).name
        meta = json.loads(f.attrs['metadata'])
        # Older teacher descriptors carried a permanently false placeholder.
        meta.get('teacher', {}).pop('training_ready', None)
        outcome = meta.setdefault('collection_outcome', {})
        outcome.setdefault('rollout_accepted', outcome.get('accepted', False))
        outcome.update(accepted=report['training_ready'], training_ready=report['training_ready'],
                       audit_version=AUDIT_VERSION, audit_violations=report['violations'])
        f.attrs['metadata'] = json.dumps(meta, ensure_ascii=False)
    # Detached checksum: storing a file's own checksum inside it is recursive.
    digest = hashlib.sha256()
    with open(episode, 'rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): digest.update(block)
    report['source_sha256'] = digest.hexdigest()
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
