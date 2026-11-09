"""Read-only TCP velocity extrema at every native integration boundary.

Kinematics and Jacobians are evaluated on private scratch data. Neither live
state, solver buffers, controls nor forces are modified by this observer.
"""
import copy
import mujoco
import numpy as np


class MujocoTcpObserver:
    def __init__(self, model, steps, offset=(0., 0., .1375)):
        self.model = model
        self.data = mujoco.MjData(model)
        self.steps = steps
        self.dt = float(model.opt.timestep)
        self.offset = np.asarray(offset, float)
        self.bodies = {side: model.body(side+'/link6').id for side in ('left', 'right')}
        self.jp = np.empty((3, model.nv)); self.jr = np.empty_like(self.jp)
        self.reset()

    def reset(self):
        self.last = None
        self.current = None

    def begin(self, live):
        if self.model.opt.timestep != self.dt:
            raise ValueError('TCP observer timestep changed')
        self.last = None
        self.current = {'schema_version': 1, 'start_sim_time': float(live.time),
                        'integration_dt': self.dt, 'integration_steps': self.steps, 'samples': 0, 'arms': {}}
        self.sample(live)

    def sample(self, live):
        record = self.current
        index = record['samples']
        if index > self.steps or abs(live.time-(record['start_sim_time']+index*self.dt)) > 1e-7:
            raise ValueError('Noncontiguous TCP integration evidence')
        self.data.qpos[:] = live.qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        for side, body in self.bodies.items():
            point = self.data.xpos[body]+self.data.xmat[body].reshape(3, 3)@self.offset
            mujoco.mj_jac(self.model, self.data, self.jp, self.jr, point, body)
            linear, angular = self.jp@live.qvel, self.jr@live.qvel
            if not np.isfinite([linear, angular]).all():
                raise ValueError('Nonfinite native TCP velocity')
            arm = record['arms'].setdefault(side, {})
            for kind, value in (('linear', linear), ('angular', angular)):
                speed = float(np.linalg.norm(value))
                if index == 0:
                    arm['start_'+kind+'_velocity'] = value.tolist()
                arm['end_'+kind+'_velocity'] = value.tolist()
                if index == 0 or speed > arm[kind+'_speed_max']:
                    arm[kind+'_speed_max'] = speed
                    arm[kind+'_peak_step'] = index
        record['samples'] += 1
        record['end_sim_time'] = float(live.time)

    def finish(self):
        if self.current['samples'] != self.steps+1:
            raise ValueError('Incomplete native TCP evidence')
        self.last = self.current
        self.current = None

    def telemetry(self):
        return copy.deepcopy(self.last)

    def metadata(self):
        return {'schema_version': 1, 'integration_steps': self.steps, 'integration_dt': self.dt,
                'tcp_in_link6_m': self.offset.tolist(), 'hardware_calibration': 'unidentified',
                'scope': 'Jacobian velocity at initial and every post-integration native state, on scratch data; not continuous time between integrator boundaries.'}
