"""Independent closed-file TCP evidence checks; no simulator imports or writes."""
import json
import numpy as np


def validate_tcp_policy(config):
    keys = {'linear_speed_m_s', 'angular_speed_rad_s'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError('TCP policy requires exactly '+', '.join(sorted(keys)))
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0
           for v in config.values()):
        raise ValueError('Positive finite TCP policy limits required')
    return dict(config)


def audit_tcp_motion(f, metadata, policy):
    limits = validate_tcp_policy(policy)
    model = metadata['physics_model']['tcp_motion']
    steps, dt = model['integration_steps'], model['integration_dt']
    if (type(model['schema_version']) is not int or model['schema_version'] != 1
            or type(steps) is not int or steps < 1
            or isinstance(dt, bool) or not isinstance(dt, (float, int))
            or not np.isfinite(dt) or dt <= 0 or abs(steps*dt-1/metadata['fps']) > 1e-7
            or model['tcp_in_link6_m'] != [0., 0., .1375]):
        raise ValueError('Invalid native TCP sampling definition')
    def need(condition, message):
        if not condition: raise ValueError(message)
    def vector(value):
        value = np.asarray(value, float)
        need(value.shape == (3,) and np.isfinite(value).all(), 'Invalid native TCP boundary velocity')
        return value
    peaks = {}; violations = set(); previous = None
    count = len(f['action/position'])
    for i in range(count):
        before, after = [json.loads(f[p+'/extras_json'][i]) for p in ('observation', 'next_observation')]
        evidence = after['tcp_motion_interval']
        t0, t1 = [float(f[p+'/sim_time'][i]) for p in ('observation', 'next_observation')]
        need(isinstance(evidence, dict) and type(evidence.get('schema_version')) is int
             and evidence['schema_version'] == 1, 'Missing native TCP interval')
        need(type(evidence['integration_steps']) is int and evidence['integration_steps'] == steps
             and type(evidence['samples']) is int and evidence['samples'] == steps+1
             and np.isfinite([evidence['integration_dt'], evidence['start_sim_time'], evidence['end_sim_time']]).all()
             and abs(evidence['integration_dt']-dt) < 1e-12
             and abs(evidence['start_sim_time']-t0) < 1e-7 and abs(evidence['end_sim_time']-t1) < 1e-7
             and abs(t1-t0-steps*dt) < 1e-7, 'TCP interval samples or clocks inconsistent')
        if previous is not None:
            need(before['tcp_motion_interval'] == previous, 'TCP interval transition discontinuity')
        need(set(evidence['arms']) == {'left', 'right'}, 'TCP arm inventory mismatch')
        for side, arm in evidence['arms'].items():
            for kind, unit, limit in (('linear', 'world_m_s', limits['linear_speed_m_s']),
                                       ('angular', 'world_rad_s', limits['angular_speed_rad_s'])):
                boundary = []
                for label, extra in (('start', before), ('end', after)):
                    recorded = vector(arm[label+'_'+kind+'_velocity'])
                    physical = vector(extra['teacher']['physics']['arms'][side]['tcp_'+kind+'_velocity_'+unit])
                    need(np.allclose(recorded, physical, atol=1e-8, rtol=0), 'TCP boundary differs from native snapshot')
                    boundary.append(float(np.linalg.norm(recorded)))
                speed, step = arm[kind+'_speed_max'], arm[kind+'_peak_step']
                need(not isinstance(speed, bool) and isinstance(speed, (float, int)) and np.isfinite(speed)
                     and speed >= 0 and speed >= max(boundary)-1e-8 and type(step) is int and 0 <= step <= steps,
                     'Invalid native TCP extrema')
                if step in (0, steps):
                    need(abs(speed-boundary[int(step == steps)]) <= 1e-8, 'TCP peak boundary mismatch')
                key = side+'/'+kind
                if speed > limit*(1+1e-9): violations.add(key+'_speed_limit')
                if key not in peaks or speed > peaks[key]['speed']:
                    peaks[key] = {'speed': speed, 'transition_index': i, 'substep': step,
                        'sim_time': t0+step*dt, 'phase': after['teacher'].get('phase'),
                        'object': after['teacher'].get('object')}
        previous = evidence
    return {'passed': not violations, 'violations': sorted(violations), 'limits': limits,
        'intervals': count, 'integration_steps': count*steps, 'peaks': peaks,
        'scope': 'Recorded native substep summaries and boundary snapshot consistency; not independent full-trace reconstruction or hardware certification.'}
