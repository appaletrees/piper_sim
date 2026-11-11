"""Simulator-independent validation of the 14-position execution contract."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'piper.position_execution.v1'
INTERVAL_SCHEMA = 'piper.position_execution.interval.v1'


def profile_digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_profile(value):
    fields = {'schema_version', 'bias_compensation', 'gripper_kp_N_per_m',
              'gripper_kv_Ns_per_m', 'gripper_force_limit_N', 'noslip_iterations'}
    if not isinstance(value, dict) or set(value)-{'impratio'} != fields or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Position execution requires exactly schema v1 fields')
    p = copy.deepcopy(value)
    if p['bias_compensation'] not in ('robot_bias', 'none'):
        raise ValueError('Unsupported bias compensation')
    for key in ('gripper_kp_N_per_m', 'gripper_kv_Ns_per_m', 'gripper_force_limit_N'):
        if isinstance(p[key], bool) or not isinstance(p[key], (float, int)) or not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError('Positive finite execution parameter required: '+key)
    if type(p['noslip_iterations']) is not int or p['noslip_iterations'] < 0:
        raise ValueError('Nonnegative integer noslip_iterations required')
    if 'impratio' in p:
        v=p['impratio']
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 1 <= v <= 1000:
            raise ValueError('Fixed friction impedance ratio must be finite in 1..1000')
    return p


def _finite(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid finite execution evidence shape')
    return result


def audit_position_execution(f, metadata):
    """Check all action intervals against the supported backend-owned executor.

This establishes declared execution completeness, not task success, calibrated
forces, camera quality or bitwise dynamics replay. Legacy files are read-only.
"""
    issues = set()
    declaration = metadata.get('physics_model', {}).get('position_execution')
    teacher = metadata.get('teacher', {})
    profile, expected = None, {}
    reference_mode = None
    if declaration is None:
        issues.add('missing_backend_position_execution')
    else:
        if declaration.get('schema_version') != SCHEMA or declaration.get('constant_within_episode') is not True:
            raise ValueError('Unsupported position execution declaration')
        profile = validate_profile(declaration['profile'])
        if declaration['profile_sha256'] != profile_digest(profile):
            issues.add('profile_digest_mismatch')
        if metadata.get('config', {}).get('mujoco', {}).get('position_execution') != profile:
            issues.add('config_profile_mismatch')
        if 'impratio' in profile and declaration.get('solver',{}).get('impratio')!=profile['impratio']:
            issues.add('declared_solver_impedance_mismatch')
        reference_mode = metadata.get('config', {}).get('mujoco', {}).get('controller', {}).get('mode')
        reference = metadata.get('physics_model', {}).get('controller', {})
        if (reference_mode not in ('monotone_cubic_velocity', 'zero_order_observed')
                or reference.get('mode') != reference_mode):
            issues.add('reference_controller_mismatch')
        implementation = Path(__file__).with_name('mujoco_execution.py')
        if declaration.get('implementation_sha256') != hashlib.sha256(implementation.read_bytes()).hexdigest():
            issues.add('unsupported_executor_implementation')
        if teacher and teacher.get('position_execution') != declaration:
            issues.add('teacher_executor_disagreement')
        for key, shape in [('position_gain', (14,)), ('velocity_gain', (14,)), ('actuator_force_range', (14, 2))]:
            expected[key] = _finite(declaration[key], shape)
        ids = [6, 13]
        if (not np.array_equal(expected['position_gain'][ids], [profile['gripper_kp_N_per_m']]*2)
                or not np.array_equal(expected['velocity_gain'][ids], [profile['gripper_kv_Ns_per_m']]*2)
                or not np.array_equal(expected['actuator_force_range'][ids],
                    [[-profile['gripper_force_limit_N'], profile['gripper_force_limit_N']]]*2)):
            issues.add('gripper_profile_mismatch')
    if teacher.get('per_target_controller_switching') or 'per_kind_gripper_force_limits_N' in teacher:
        issues.add('teacher_selects_controller_by_target')
    if teacher.get('per_target_contact_material_switching'):
        issues.add('teacher_selects_contact_material_by_target')
    n = len(f['action/position'])
    if n <= 0 or len(f['next_observation/extras_json']) != n:
        raise ValueError('Empty or mismatched execution stream')
    segments = []
    missing_execution = missing_controller = 0
    for i, raw in enumerate(f['next_observation/extras_json']):
        extras = json.loads(raw)
        controller = extras.get('controller_interval')
        execution = extras.get('position_execution_interval')
        if controller is None:
            missing_controller += 1
        else:
            kp = _finite(controller['position_gain'], (14,))
            kv = _finite(controller['velocity_gain'], (14,))
            limits = _finite(controller['actuator_force_range'], (14, 2))
            regime = np.column_stack((kp[[6, 13]], kv[[6, 13]], limits[[6, 13]])).tolist()
            if not segments or segments[-1]['left_right_kp_kv_force_range'] != regime:
                segments.append({'first_transition': i, 'last_transition': i,
                                 'left_right_kp_kv_force_range': regime})
            else:
                segments[-1]['last_transition'] = i
            if profile is not None:
                if controller.get('mode') != reference_mode:
                    issues.add('interval_reference_mode_mismatch')
                for key, actual in [('position_gain', kp), ('velocity_gain', kv), ('actuator_force_range', limits)]:
                    if not np.array_equal(actual, expected[key]):
                        issues.add('interval_parameter_mismatch/'+key)
            action = f['action/position'][i]
            if not np.array_equal(_finite(controller['command'], (14,)).astype(action.dtype), action):
                issues.add('interval_action_mismatch')
        if execution is None:
            missing_execution += 1
            continue
        if profile is None:
            issues.add('undeclared_interval_executor')
            continue
        if (execution.get('schema_version') != INTERVAL_SCHEMA
                or execution.get('profile_sha256') != declaration['profile_sha256']):
            issues.add('interval_executor_mismatch')
        times = _finite([execution['start_sim_time'], execution['end_sim_time']], (2,))
        if not np.allclose(times, [f['observation/sim_time'][i], f['next_observation/sim_time'][i]], atol=1e-8, rtol=0):
            issues.add('interval_clock_mismatch')
        applied = _finite(execution['applied_robot_generalized_force'], (16,))
        if profile['bias_compensation'] == 'none' and np.any(applied != 0):
            issues.add('unexpected_bias_force')
        if (execution['noslip_iterations'] != profile['noslip_iterations']
                or execution.get('impratio',10.) != profile.get('impratio',10.)
                or execution['nonrobot_applied_force_abs_max'] != 0.
                or execution['external_body_force_abs_max'] != 0.):
            issues.add('undeclared_solver_or_external_force')
        physics = extras.get('teacher', {}).get('physics', {}).get('arms')
        if physics is not None:
            recorded = np.concatenate([_finite(physics[s]['applied_generalized_force'], (8,)) for s in ('left', 'right')])
            if not np.array_equal(recorded, applied):
                issues.add('recorded_applied_force_mismatch')
    if missing_execution:
        issues.add('missing_execution_intervals')
    if missing_controller:
        issues.add('missing_controller_intervals')
    return {'passed': not issues, 'violations': sorted(issues), 'checked_intervals': n,
            'missing_execution_intervals': missing_execution, 'missing_controller_intervals': missing_controller,
            'gripper_regime_segments': segments, 'position_only_execution_supported': not issues,
            'scope': 'Declared fixed controller and complete interval evidence for 14 position actions; not task, RGB or hardware acceptance.'}
