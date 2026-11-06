"""Measured-motion and command quality gates, independent of task success."""
import copy
import numpy as np

ARM = np.array([*range(6), *range(7, 13)])
GRIP = np.array([6, 13])


class MotionQuality:
    def __init__(self, settings, dt):
        self.settings = copy.deepcopy(settings)
        self.dt = dt
        self.reset()

    def reset(self):
        self.samples = []

    def add(self, time, command, measured, *, holding=False, context=None):
        self.samples.append((float(time), np.array(command).copy(), np.array(measured).copy(),
                             bool(holding), context))

    def report(self):
        if len(self.samples) < 5:
            return {'passed': False, 'violations': ['insufficient motion samples'], 'frames': len(self.samples)}
        times = np.array([s[0] for s in self.samples])
        q = np.array([s[1] for s in self.samples])
        measured = np.array([s[2] for s in self.samples])
        holding = np.array([s[3] for s in self.samples])
        m, limits = self.settings['motion'], self.settings['quality']
        values, thresholds, peaks = {}, {}, {}
        for source, data in (('command', q), ('measured', measured)):
            for order, label, unit in ((1, 'velocity', 's'), (2, 'acceleration', 's2'), (3, 'jerk', 's3')):
                derivative = np.diff(data, n=order, axis=0)/self.dt**order
                for group, indices in (('arm', ARM), ('gripper', GRIP)):
                    key = f'{source}_{group}_{label}'
                    a = np.abs(derivative[:, indices])
                    row, col = np.unravel_index(np.argmax(a), a.shape)
                    values[key] = float(a[row, col])
                    peaks[key] = {'sim_time': float(times[row+order]), 'channel': int(indices[col]),
                                  'context': self.samples[row+order][4]}
                    if source == 'command':
                        thresholds[key] = m[f'{"joint" if group == "arm" else "gripper"}_{label}_{"rad" if group == "arm" else "m"}_{unit}']*1.01
                    elif group == 'arm':
                        thresholds[key] = limits[f'measured_{label}_rad_{unit}']
                    else:
                        thresholds[key] = limits[f'measured_gripper_{label}_m_{unit}']
        values['tracking_error_rad'] = float(np.max(abs(q[:, ARM]-measured[:, ARM])))
        thresholds['tracking_error_rad'] = limits['tracking_error_rad']
        # All three samples must be hold samples to exclude approach transients.
        stable_hold = holding[2:] & holding[1:-1] & holding[:-2]
        v = np.diff(measured[:, ARM], axis=0)/self.dt
        values['hold_velocity_rms_rad_s'] = float(np.sqrt(np.mean(v[1:][stable_hold]**2))) if stable_hold.any() else 0.
        thresholds['hold_velocity_rms_rad_s'] = limits['hold_velocity_rms_rad_s']
        gv = np.diff(measured[:, GRIP], axis=0)/self.dt
        values['hold_gripper_velocity_rms_m_s'] = float(np.sqrt(np.mean(gv[1:][stable_hold]**2))) if stable_hold.any() else 0.
        thresholds['hold_gripper_velocity_rms_m_s'] = limits['hold_gripper_velocity_rms_m_s']
        violations = [k for k, bound in thresholds.items() if values[k] > bound]
        if not np.isfinite(q).all() or not np.isfinite(measured).all():
            violations.append('nonfinite_state_or_command')
        if not np.allclose(np.diff(times), self.dt, atol=1e-7, rtol=0):
            violations.append('nonuniform_simulation_timestamps')
        return {'passed': not violations, 'violations': violations, 'frames': len(times),
                'metrics': values, 'thresholds': thresholds, 'peaks': peaks,
                'measurement': 'finite differences at control/recording frequency; arm SI radians, full gripper opening metres',
                'hold_samples': int(stable_hold.sum())}
