"""Optional internal reference controller and native observer; uncalibrated.

The public command remains an absolute joint/full-gripper endpoint accepted at
the control boundary. This controller generates bounded position references and
their velocities inside that interval. The zero-order observation mode retains
the original constant target without writing control or force during integration.
Neither mode assigns robot/object state.
"""
import copy

import mujoco
import numpy as np


def monotone_schedule(previous, target, previous_velocity, period, steps):
    a,b,va = [np.asarray(x,dtype=float) for x in (previous,target,previous_velocity)]
    if (a.ndim != 1 or not len(a) or b.shape != a.shape or va.shape != a.shape
            or not np.isfinite([a,b,va]).all() or not np.isfinite(period) or period <= 0
            or type(steps) is not int or steps < 1):
        raise ValueError('Finite matching controller targets and positive timing required')
    speed=(b-a)/period
    start=np.clip(va,np.minimum(0,3*speed),np.maximum(0,3*speed))
    s=np.arange(1,steps+1)[:,None]/steps
    c1,c2=a+period*start/3,b-period*speed/3
    q=(1-s)**3*a+3*s*(1-s)**2*c1+3*s*s*(1-s)*c2+s**3*b
    v=3*((1-s)**2*(c1-a)+2*s*(1-s)*(c2-c1)+s*s*(b-c2))/period
    # A constant polynomial is exactly constant. Avoid roundoff moving a held
    # endpoint infinitesimally beyond its actuator limit; this is not clipping.
    stationary=a==b
    q[:,stationary]=a[stationary];v[:,stationary]=0
    q[-1]=b;v[-1]=speed
    return q,v,start-va


class MujocoReferenceController:
    """A causal, opt-in reference generator with per-interval native evidence."""
    def __init__(self, model, actuator_ids, velocity_projection, steps, config):
        if (not isinstance(config,dict) or set(config) != {'schema_version','mode'}
                or type(config['schema_version']) is not int or config['schema_version'] != 1
                or config['mode'] not in ('monotone_cubic_velocity','zero_order_observed')):
            raise ValueError('Unsupported MuJoCo controller configuration')
        self.mode=config['mode']
        self.model=model
        self.ids=np.asarray(actuator_ids)
        if (self.ids.ndim != 1 or not len(self.ids) or not np.issubdtype(self.ids.dtype,np.integer)
                or len(np.unique(self.ids)) != len(self.ids) or np.any(self.ids<0)
                or np.any(self.ids>=model.nu)):
            raise ValueError('Unique valid actuator IDs required')
        self.projection=np.asarray(velocity_projection,dtype=float)
        if self.projection.shape != (len(self.ids),model.nv) or not np.isfinite(self.projection).all():
            raise ValueError('Finite actuator-coordinate velocity projection required')
        if type(steps) is not int or steps < 1:
            raise ValueError('Positive integration step count required')
        self.steps=steps; self.dt=float(model.opt.timestep)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Positive finite physics timestep required')
        if np.any(model.actuator_biasprm[self.ids,0] != 0):
            raise ValueError('Controller requires exclusive ownership of initially zero force bias')
        self._validate_servo()
        self.previous=None;self.previous_velocity=None;self.previous_acceleration=None;self.last=None

    def _validate_servo(self):
        m,ids=self.model,self.ids
        kp=m.actuator_gainprm[ids,0];bias=m.actuator_biasprm[ids]
        if (np.any(m.actuator_gaintype[ids] != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(m.actuator_biastype[ids] != mujoco.mjtBias.mjBIAS_AFFINE)
                or np.any(m.actuator_dyntype[ids] != mujoco.mjtDyn.mjDYN_NONE)
                or not np.isfinite(kp).all() or np.any(kp<=0) or not np.isfinite(bias).all()
                or not np.allclose(bias[:,1],-kp,atol=1e-12,rtol=0) or np.any(bias[:,2]>0)
                or not np.all(m.actuator_ctrllimited[ids]) or not np.all(m.actuator_forcelimited[ids])
                or not np.isfinite(m.actuator_ctrlrange[ids]).all()
                or not np.isfinite(m.actuator_forcerange[ids]).all()):
            raise ValueError('Bounded stateless affine position servos with nonnegative damping required')
        return -bias[:,2]

    def _target(self, value):
        value=np.asarray(value,dtype=float)
        limits=self.model.actuator_ctrlrange[self.ids]
        if (value.shape != self.ids.shape or not np.isfinite(value).all()
                or np.any(value<limits[:,0]) or np.any(value>limits[:,1])):
            raise ValueError('Controller endpoint outside actuator bounds')
        return value.copy()

    def reset(self,target):
        self.previous=self._target(target)
        self.previous_velocity=np.zeros_like(self.previous)
        self.previous_acceleration=None;self.last=None
        self.model.actuator_biasprm[self.ids,0]=0

    def advance(self,data, *, observer=None):
        if self.previous is None: raise RuntimeError('Reset controller before stepping')
        if self.model.opt.timestep != self.dt: raise ValueError('Controller physics timestep changed')
        target=self._target(data.ctrl[self.ids]); damping=self._validate_servo()
        if self.mode == 'zero_order_observed':
            if np.any(self.model.actuator_biasprm[self.ids,0] != 0):
                raise ValueError('Zero-order observation requires unchanged zero force bias')
            q=np.broadcast_to(target,(self.steps,len(target)))
            v=np.zeros_like(q);jump=np.zeros_like(target)
        else:
            q,v,jump=monotone_schedule(self.previous,target,self.previous_velocity,self.steps*self.dt,self.steps)
        bounds=self.model.actuator_ctrlrange[self.ids]
        if np.any(q<bounds[:,0]) or np.any(q>bounds[:,1]):
            raise ValueError('Internal target outside actuator bounds; no output clipping')
        start=float(data.time)
        warning_ids=[int(x) for x in (mujoco.mjtWarning.mjWARN_INERTIA,
            mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL,
            mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL)]
        warnings=[data.warning[i].number for i in warning_ids]
        native=[self.projection@data.qvel]
        acceleration=[];jerk=[]
        self.last=None
        for i,(position,velocity) in enumerate(zip(q,v)):
            if self.mode == 'monotone_cubic_velocity':
                data.ctrl[self.ids]=position
                # This contribution remains inside the original actuator force clamp.
                self.model.actuator_biasprm[self.ids,0]=damping*velocity
            mujoco.mj_step(self.model,data)
            if (abs(data.time-(start+(i+1)*self.dt)) > 1e-7
                    or [data.warning[j].number for j in warning_ids] != warnings
                    or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()):
                raise RuntimeError('MuJoCo controller detected physics warning, reset or nonfinite state')
            if observer is not None:
                observer(data)
            current=self.projection@data.qvel
            acc=(current-native[-1])/self.dt
            if self.previous_acceleration is not None:
                jerk.append((acc-self.previous_acceleration)/self.dt)
            acceleration.append(acc);native.append(current);self.previous_acceleration=acc
        native=np.asarray(native)
        self.last={'mode':self.mode,'start_sim_time':start,'end_sim_time':float(data.time),
            'integration_steps':self.steps,'integration_dt':self.dt,
            'previous_command':self.previous.tolist(),'command':target.tolist(),
            'position_reference_min':q.min(axis=0).tolist(),'position_reference_max':q.max(axis=0).tolist(),
            'velocity_reference_min':v.min(axis=0).tolist(),'velocity_reference_max':v.max(axis=0).tolist(),
            'last_velocity_reference':v[-1].tolist(),'limiter_start_velocity_jump':jump.tolist(),
            'native_velocity_min':native.min(axis=0).tolist(),'native_velocity_max':native.max(axis=0).tolist(),
            'native_acceleration_abs_max':np.abs(acceleration).max(axis=0).tolist(),
            'native_jerk_abs_max':np.abs(jerk).max(axis=0).tolist() if jerk else [0.]*len(target),
            'native_jerk_samples':len(jerk),
            'position_gain':self.model.actuator_gainprm[self.ids,0].tolist(),
            'velocity_gain':damping.tolist(),'actuator_force_range':self.model.actuator_forcerange[self.ids].tolist()}
        self.previous=target;self.previous_velocity=v[-1].copy()

    def telemetry(self):
        return copy.deepcopy(self.last)

    def metadata(self):
        observed=self.mode=='zero_order_observed'
        return {'schema_version':1,'mode':self.mode,'experimental':True,
            'hardware_calibration':'unidentified','integration_dt':self.dt,'integration_steps':self.steps,
            'action_semantics':('Accepted absolute position target held unchanged over the following control interval; observer performs the original steps individually without additional forwards or state/control/force writes.' if observed else
                'Accepted absolute position endpoint; internal monotone cubic position and derivative velocity references over the following control interval.'),
            'velocity_feedback':'Native actuator-coordinate velocity; public gripper coordinate is full opening.',
            'force_semantics':('Original kp*(target-position)-kv*velocity, inside existing actuator force clamp; external bias compensation is separate.' if observed else
                'kp*(reference-position)+kv*(reference_velocity-velocity), inside existing actuator force clamp; external bias compensation is separate.'),
            'continuity':('Zero-order hold has position jumps at control boundaries; zero velocity-reference and limiter fields do not imply a continuous position reference.' if observed else
                'Derivative limiting at stops/reversals can break C1 continuity; limiter jumps are recorded.'),
            'telemetry_scope':'Per-interval extrema of post-integration native velocity and its differences, not calibrated hardware or new quality limits.'}
