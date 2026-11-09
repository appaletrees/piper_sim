"""Privileged scripted sorting teacher, independent of task geometry and recording.

MuJoCo implementation: ground-truth poses -> bounded IK -> 14 position targets.
Optional contact assistance is explicitly logged and never enabled in physical
mode. The teacher never assigns live object or robot positions/velocities.
"""
import copy
import numpy as np
import mujoco

from .mujoco_ik import PiperIK, grasp_rotation, rotation_error
from .grasp_assist import ContactGraspAssist, finger_contacts
from .trajectory import smooth_trajectory, path_trajectory
from .path_fidelity import loaded_path_limits, path_pose_deviation
from .placement_search import validate_profile as placement_search_profile
from .quality import MotionQuality
from .collision import robot_penetration, carried_obstacle_penetration, carried_obstacle_clearance, robot_obstacle_clearance
from ..transforms import pose_matrix
from ..config import ROOT
from ..contracts import LOWER, UPPER


class SortFailure(RuntimeError):
    pass


class CleanTableTeacher:
    name = 'clean_table_privileged_v2'
    def __init__(self, session, *, grasp_mode='physical', frame_hook=None, settings=None):
        if grasp_mode not in ('assisted', 'physical'):
            raise ValueError('grasp_mode must be assisted or physical')
        self.session = session
        self.b = session.backend
        if self.b.task is None or self.b.task.config['name'] != 'clean_table':
            raise ValueError('This teacher requires the independent clean_table task')
        self.mode = grasp_mode
        self.assist = ContactGraspAssist(self.b) if grasp_mode == 'assisted' else None
        self.frame_hook = frame_hook
        self.q = self.b.control_targets()
        self.ik = {side: PiperIK(self.b, side) for side in ('left', 'right')}
        self.phase = 'initializing'
        self.current = None
        self.armed = None
        self.events = []
        self.results = []
        self.fps = 1/(self.b.dt*self.b.decimation)
        import yaml
        self.settings = copy.deepcopy(settings) if settings is not None else yaml.safe_load(
            (ROOT/'configs/collection/clean_table.yaml').read_text())
        self.motion = self.settings['motion']
        self.pacing = self.settings.get('pacing', {})
        self.bounded_loaded_path = loaded_path_limits(self.pacing.get('bounded_loaded_path'))
        self.placement_search = placement_search_profile(self.pacing.get('placement_search'))
        self.release_preload_m = float(self.pacing.get('release_preload_threshold_m', .003))
        if not np.isfinite(self.release_preload_m) or not 0 <= self.release_preload_m <= .095:
            raise ValueError('Release preload threshold must be finite and within the gripper stroke')
        self.quality = MotionQuality(self.settings, 1/self.fps)
        self.holding = False
        self.prepared = False
        self.collision_peak = {'depth_m': 0., 'pair': None}
        self.camera_quality = {}
        self.carried = None
        self.releasing = None
        self.carried_anchor = np.zeros(3)
        self.grasps = []
        self.configure_contact_control()

    def configure_contact_control(self):
        """Explicit simulation controller settings, not identified hardware values."""
        m = self.b.model
        m.opt.impratio = 10
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        m.opt.noslip_iterations = 0
        for i in range(m.ngeom):
            name = m.body(int(m.geom_bodyid[i])).name
            if (getattr(self.b, 'finger_contact', None) is None and
                    name in ('left/link7', 'left/link8', 'right/link7', 'right/link8')):
                m.geom_condim[i] = 6
                m.geom_friction[i] = [1., .015, .002]
                m.geom_priority[i] = 1
                m.geom_solref[i] = [self.settings['controller']['finger_contact_time_constant_s'], 1.]
                m.geom_solimp[i] = [.95, .99, .001, .5, 2.]
        for side in ('left', 'right'):
            a = m.actuator(side+'/gripper').id
            controller = self.settings['controller']
            m.actuator_gainprm[a, 0] = controller['gripper_kp_N_per_m']
            m.actuator_biasprm[a, 1] = -controller['gripper_kp_N_per_m']
            m.actuator_biasprm[a, 2] = -controller['gripper_kv_Ns_per_m']
            force = controller['gripper_force_limit_N']
            m.actuator_forcerange[a] = [-force, force]

    def metadata(self):
        value = {'name': self.name, 'grasp_mode': self.mode,
                'privileged_inputs': ['object poses', 'container poses', 'geometry', 'contacts', 'robot kinematics'],
                'physical_only': self.mode == 'physical',
                'collision_gate': 'all robot bodies, including nonadjacent same-arm contacts; model adjacent exclusions retained',
                'physics_evidence': {
                    'scope': '30 Hz sampled solver state, not substep extrema or calibrated real hardware forces',
                    'joint_order': [f'joint{i}' for i in range(1, 9)],
                    'position_units': ['rad']*6+['m', 'm'],
                    'velocity_units': ['rad/s']*6+['m/s', 'm/s'],
                    'generalized_force_units': ['N*m']*6+['N', 'N'],
                    'combined_drive': 'qfrc_actuator + qfrc_applied; constraint and passive forces are separate',
                },
                'training_note': 'Training acceptance is determined only by the final closed-file dataset audit.',
                'arm_bias_feedforward': True, **self.settings['controller'],
                'finger_friction': [1., .015, .002],
                'multiccd': True,
                'noslip_iterations': 0,
                'tcp_in_link6_m': [0, 0, .1375], 'placemat': 'left on table',
                'motion_and_quality': copy.deepcopy(self.settings),
                'trajectory': ('C2 natural joint splines; unloaded joint-limit overshoot may use a '
                    'collision-checked convex B-spline with exact endpoints and approximate interior knots; '
                    'C3 time law and sampled derivative retiming'
                    if self.pacing.get('continuous_paths') else 'septic C3 rest-to-rest per segment'),
                'grasp_frames': 'measured object SE(3) relative to nominal object grasp frame'}
        if self.bounded_loaded_path:
            value['bounded_loaded_path'] = {**self.bounded_loaded_path,
                'reference': 'original solved-IK pose polyline at the same path parameter',
                'scope': 'additional sampled approximation only; IK and tracking errors are separate',
                'calibration': 'uncalibrated simulation planning budget',
                'collision_checks': 'original actual-curve robot and carried-object gates retained'}
            value['trajectory'] += '; loaded overshoot may use the explicit pose-fidelity guarded convex profile'
        profile = getattr(self.b, 'finger_contact', None)
        value['contact_material_source'] = 'backend_episode_profile' if profile else 'legacy_teacher_parameters'
        if profile:
            value.pop('finger_friction')
            value.pop('finger_contact_time_constant_s', None)
            value['finger_contact'] = profile.metadata()
        return value

    def event(self, phase, **fields):
        self.phase = phase
        record = {'phase': phase, 'object': self.current, 'sim_time': self.b.sim_time, **fields}
        self.events.append(record)
        print(record, flush=True)

    def telemetry(self):
        """Read-only evidence captured with each state, before the writer appends it."""
        from .physics_evidence import physics_snapshot
        depth, pair = robot_penetration(self.b.model, self.b.data, self.current)
        grasp = None
        if self.carried:
            side, name, relative = self.carried
            hand, obj = self.b.data.body(side+'/link6'), self.b.data.body('task/'+name)
            anchor = hand.xmat.reshape(3, 3).T @ (
                obj.xpos + obj.xmat.reshape(3, 3) @ self.carried_anchor - hand.xpos)
            grasp = {'object': name, 'side': side,
                     'contact_fingers': sorted(finger_contacts(self.b, side, name)),
                     'slip_m': float(np.linalg.norm(anchor-relative))}
        return {'name': self.name, 'grasp_mode': self.mode, 'phase': self.phase,
                'object': self.current, 'holding': self.holding, 'grasp': grasp,
                'robot_penetration_m': depth, 'robot_collision_pair': pair,
                'active_assist': self.assist.active[1:] if self.assist and self.assist.active else None,
                'physics': physics_snapshot(self.b, self.current if self.current in self.b.task.specs else None)}

    def step(self):
        # Robot feedforward only; free task objects always obey the physics solver.
        for ids in self.b.indices.values():
            self.b.data.qfrc_applied[ids['dof']] = self.b.data.qfrc_bias[ids['dof']]
        obs = self.session.step(self.q)
        if self.carried:
            side, name, relative = self.carried
            hand, obj = self.b.data.body(side+'/link6'), self.b.data.body('task/'+name)
            measured = hand.xmat.reshape(3, 3).T@(obj.xpos+obj.xmat.reshape(3,3)@self.carried_anchor-hand.xpos)
            slip = float(np.linalg.norm(measured-relative))
            record = self.grasps[-1]
            contacts = finger_contacts(self.b, side, name)
            record['samples'] += 1
            record['bilateral_samples'] += int(len(contacts) == 2)
            record['max_slip_m'] = max(record['max_slip_m'], slip)
            record['lost_contact_s'] = 0. if len(contacts) == 2 else record['lost_contact_s']+1/self.fps
            if slip > .012 or record['lost_contact_s'] > .20:
                raise SortFailure(f'{name} lost physical grasp: slip={slip:.4f} m, fingers={contacts}')
        if obs.images:
            if set(obs.images) != {'overhead', 'left_wrist', 'right_wrist'}:
                raise SortFailure('Collection requires all three RGB streams')
            for name, pixels in obs.images.items():
                if abs(obs.image_timestamps[name]-obs.sim_time) > 1e-6:
                    raise SortFailure(f'Unsynchronized camera {name}')
                stats = self.camera_quality.setdefault(name, {'frames': 0, 'low_contrast_frames': 0})
                stats['frames'] += 1
                stats['low_contrast_frames'] += int(np.std(pixels[::8, ::8].astype(float)) < 1.)
        depth, pair = robot_penetration(self.b.model, self.b.data, self.current)
        if depth > self.collision_peak['depth_m']:
            self.collision_peak = {'depth_m': depth, 'pair': pair, 'sim_time': obs.sim_time,
                                   'phase': self.phase, 'object': self.current}
        if depth > self.settings['quality']['max_robot_penetration_m']:
            raise SortFailure(f'Robot obstacle penetration {depth*1000:.2f} mm: {pair}')
        self.quality.add(obs.sim_time, self.q, obs.position, holding=self.holding,
                         context={'phase': self.phase, 'object': self.current})
        if self.armed and self.assist and self.assist.active is None:
            side, name = self.armed
            if len(finger_contacts(self.b, side, name)) == 2:
                activation = self.assist.engage(side, name)
                self.events.append({'phase': 'assist_engaged', **activation})
                self.armed = None
        if self.frame_hook:
            self.frame_hook(self, obs)
        if not self.b.is_running():
            raise SortFailure('Viewer closed')
        return obs

    def wait(self, seconds):
        previous = self.holding
        self.holding = True
        for _ in range(round(seconds*self.fps)):
            self.step()
        self.holding = previous

    def track_grasp(self,side,name):
        """Track material at the pad contact, not the COM of a long rim grasp."""
        m,d=self.b.model,self.b.data
        hand,obj=d.body(side+'/link6'),d.body('task/'+name)
        points=[]
        for contact in d.contact:
            names=[m.body(int(m.geom_bodyid[g])).name for g in contact.geom]
            if 'task/'+name in names and any(n in (side+'/link7',side+'/link8') for n in names):
                points.append(contact.pos.copy())
        if not points:raise SortFailure('No physical grasp contact anchor')
        point=np.mean(points,axis=0)
        self.carried_anchor=obj.xmat.reshape(3,3).T@(point-obj.xpos)
        self.carried=(side,name,hand.xmat.reshape(3,3).T@(point-hand.xpos))
        self.grasps.append({'name':name,'assisted':self.mode!='physical','samples':0,
            'bilateral_samples':0,'max_slip_m':0.,'lost_contact_s':0.,'anchor':'mean initial pad contact in object frame'})

    def settle(self):
        """Bounded feedback wait; do not carry tracking oscillation into a new phase."""
        good = 0
        for _ in range(round(self.motion['settle_timeout_s']*self.fps)):
            self.wait(1/self.fps)
            errors, speeds = [], []
            for side, offset in (('left', 0), ('right', 7)):
                ids = self.b.indices[side]
                errors.extend(abs(self.b.data.qpos[ids['qpos'][:6]]-self.q[offset:offset+6]))
                speeds.extend(abs(self.b.data.qvel[ids['dof'][:6]]))
            good = good+1 if max(errors) < self.motion['settle_error_rad'] and max(speeds) < self.motion['settle_velocity_rad_s'] else 0
            if good >= round(self.motion['settle_hold_s']*self.fps):
                return
        raise SortFailure(f'Robot did not settle: error={max(errors):.4f} rad, speed={max(speeds):.4f} rad/s')

    def prepare(self, objects=None):
        """Settle and reject unreachable grasp frames BEFORE starting a recorder."""
        self.wait(1.)
        for name in objects or self.b.task.required:
            side, p, r, _ = self.reachable_grasp(name)
            q = self.solve(side, self.approach_position(name,p,r,.035), r)
            self.solve(side, p, r, q)
        self.prepared = True
        self.quality.reset()
        self.camera_quality = {}
        self.quality.add(self.b.sim_time, self.q, self.session.read().position,
                         context={'phase': 'prepared', 'object': None})
        self.event('prepared')

    def move_joints(self, side, joints, seconds=None, gripper_width=None):
        offset = 0 if side == 'left' else 7
        before = self.q.copy()
        target = before.copy();target[offset:offset+6] = joints
        if gripper_width is not None: target[offset+6] = gripper_width
        self.check_joint_path(side, before[offset:offset+6], joints,
                              gripper=(before[offset+6], target[offset+6]) if gripper_width is not None else None)
        if seconds is None: seconds = self.pacing.get('minimum_move_s', 2.)
        trajectory = smooth_trajectory(before, target, 1/self.fps, seconds=seconds, **self.motion_limits())
        for command in trajectory:
            self.q = command
            self.step()
        self.settle()

    def motion_limits(self):
        motion = self.motion.copy()
        if (self.carried and self.b.task.specs[self.carried[1]]['kind'] in ('plate','side_dish')
                and self.pacing.get('continuous_paths')):
            motion.update(self.pacing.get('rim_load_limits', {}))
            motion.update(self.pacing.get('loaded_limits_by_kind', {}).get(self.b.task.specs[self.carried[1]]['kind'], {}))
        return {label:[*[motion['joint_'+label+suffix]]*6, motion['gripper_'+label+grip_suffix]]*2
                for label,suffix,grip_suffix in [('velocity','_rad_s','_m_s'),
                    ('acceleration','_rad_s2','_m_s2'),('jerk','_rad_s3','_m_s3')]}

    def pose_path(self, side, poses, *, speed=None):
        """Plan a complete Cartesian path on scratch IK data, then execute it."""
        if not self.pacing.get('continuous_paths'):
            for p, r in poses: self.pose(side, p, r)
            return
        diagnostics = {}
        trajectory = self.plan_pose_path(side, poses, speed=speed, diagnostics=diagnostics)
        if diagnostics.get('interpolation') == 'convex':
            self.events.append({'phase': 'bounded_path_selected', 'object': self.current,
                'sim_time': self.b.sim_time, **diagnostics})
        for q in trajectory:
            self.q = q
            self.step()
        self.settle()

    def plan_pose_path(self, side, poses, *, speed=None, start=None, diagnostics=None):
        """Return a checked trajectory without advancing state or control.

        An explicit start supports candidate evaluation from a planned pregrasp.
        Collision queries retain actual other-arm, finger and object geometry.
        """
        initial = self.q.copy() if start is None else np.asarray(start, dtype=float).copy()
        offset = 0 if side == 'left' else 7
        start, rotation = self.ik[side].forward(initial[offset:offset+6])
        points = [initial]
        length = 0.
        for target, target_r in poses:
            target, target_r = np.asarray(target), np.asarray(target_r)
            axis = rotation_error(target_r, rotation)
            angle = np.linalg.norm(axis)
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            distance = np.linalg.norm(target-start)
            length += distance
            count = max(1, int(np.ceil(max(distance/.01, angle/np.deg2rad(5)))))
            for fraction in np.linspace(0, 1, count+1)[1:]:
                delta = (np.eye(3) if angle < 1e-9 else
                    np.eye(3)+np.sin(fraction*angle)/angle*skew+(1-np.cos(fraction*angle))/angle**2*(skew@skew))
                q = points[-1].copy()
                q[offset:offset+6] = self.solve(side, start+fraction*(target-start), delta@rotation,
                                                points[-1][offset:offset+6])
                points.append(q)
            start, rotation = target, target_r
        duration = 2.1875*length/(speed or self.pacing['cartesian_speed_m_s'])
        trajectory = path_trajectory(points, 1/self.fps, seconds=duration, **self.motion_limits())
        interpolation = 'natural'
        if ((self.carried is None or self.bounded_loaded_path)
                and (np.any(trajectory < LOWER) or np.any(trajectory > UPPER))
                and np.all(np.asarray(points) >= LOWER) and np.all(np.asarray(points) <= UPPER)):
            # Legal control points need not produce a legal natural cubic.
            # Loaded alternatives require an explicit, additional pose-fidelity
            # gate. Neither alternative clips targets or bypasses geometry checks.
            if self.carried is None:
                trajectory = path_trajectory(points, 1/self.fps, seconds=duration,
                                             interpolation='convex', **self.motion_limits())
            else:
                trajectory, fidelity = self.bounded_loaded_trajectory(side, points, duration)
                if diagnostics is not None:
                    diagnostics['fidelity'] = fidelity
            interpolation = 'convex'
        if np.any(trajectory < LOWER) or np.any(trajectory > UPPER):
            excess = np.maximum(LOWER-trajectory, trajectory-UPPER)
            row, channel = np.unravel_index(np.argmax(excess), excess.shape)
            raise SortFailure(f'Continuous path exceeds joint limits: channel={channel}, value={trajectory[row,channel]:.8f}')
        previous = initial[offset:offset+6]
        # Check the spline that will actually run, not just its original knots.
        for q in trajectory:
            self.check_joint_path(side, previous, q[offset:offset+6])
            previous = q[offset:offset+6]
        if diagnostics is not None:
            diagnostics.update(interpolation=interpolation, command_samples=len(trajectory))
        return trajectory

    def bounded_loaded_trajectory(self, side, points, duration):
        """Retime and bound approximation against the ORIGINAL IK pose polygon.

        Midpoint refinement stays inside the joint hull and preserves path
        parameters; it never replaces the reference with a more permissive one.
        This method does not certify collision clearance; the caller checks the
        entire returned curve through the normal robot/payload gates.
        """
        limits = self.bounded_loaded_path
        offset = 0 if side == 'left' else 7
        reference = None
        attempts = []
        controls = np.asarray(points, float)
        for refinement in range(limits['max_refinements']+1):
            sampling = {}
            path = path_trajectory(controls, 1/self.fps, seconds=duration, interpolation='convex',
                                   sampling=sampling, **self.motion_limits())
            if reference is None:
                reference = {'points': sampling['points'][:, offset:offset+6], 'knots': sampling['knots']}
                controls = sampling['points']
            error = path_pose_deviation(path[:, offset:offset+6], {**reference, 'samples': sampling['samples']},
                                        self.ik[side].forward)
            attempts.append({'refinement': refinement, 'control_points': len(controls),
                             'command_samples': len(path), **error})
            if (error['position_deviation_m'] <= limits['max_position_deviation_m']
                    and error['rotation_deviation_rad'] <= limits['max_rotation_deviation_rad']):
                return path, {'limits': dict(limits), 'attempts': attempts,
                              'reference_ik_knots': len(reference['points'])}
            refined = np.empty((2*len(controls)-1, controls.shape[1]))
            refined[::2] = controls
            refined[1::2] = (controls[:-1]+controls[1:])/2
            controls = refined
        raise SortFailure(f'Bounded loaded path exceeds pose fidelity: {attempts[-1]}')

    def check_joint_path(self, side, start, end, gripper=None, *, transit_clearance=0.):
        """Check sampled interpolation on scratch state before executing a segment.

        This is a rejection gate, not a general collision-free motion planner.
        It checks self, robot-obstacle, inter-arm and carried-object/scene contacts.
        These are sampled geometric predictions; live physics is still audited.
        """
        ik = self.ik[side]
        carried = self.carried_objects()
        transforms=[]
        for name in carried:
            hand = self.b.data.body(side+'/link6')
            obj = self.b.data.body('task/'+name)
            rel_p = hand.xmat.reshape(3, 3).T@(obj.xpos-hand.xpos)
            rel_r = hand.xmat.reshape(3, 3).T@obj.xmat.reshape(3, 3)
            obj_q = self.b.model.joint('task/'+name+'/free').qposadr[0]
            transforms.append((obj_q,rel_p,rel_r))
        spacing = .01 if transit_clearance else .08
        steps = max(2, int(np.ceil(np.max(abs(np.asarray(end)-start))/spacing)))
        if transit_clearance and gripper is not None:
            steps = max(steps, int(np.ceil(abs(gripper[1]-gripper[0])/.001)))
        containers = ({spec['target'] for spec in self.b.task.specs.values() if spec.get('target')}
                      if transit_clearance else ())
        for t in np.linspace(0, 1, steps+1):
            ik.data.qpos[:] = self.b.data.qpos
            ik.data.qpos[ik.qids] = (1-t)*np.asarray(start)+t*np.asarray(end)
            if gripper is not None:
                ik.data.qpos[self.b.indices[side]['qpos'][6:]] = ((1-t)*gripper[0]+t*gripper[1])/2
            mujoco.mj_kinematics(self.b.model, ik.data)
            for obj_q,rel_p,rel_r in transforms:
                hand = ik.data.body(side+'/link6')
                r = hand.xmat.reshape(3, 3)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (r@rel_r).reshape(-1))
                ik.data.qpos[obj_q:obj_q+7] = [*(hand.xpos+r@rel_p), *quat]
                mujoco.mj_kinematics(self.b.model, ik.data)
            mujoco.mj_collision(self.b.model, ik.data)
            depth, pair = robot_penetration(self.b.model, ik.data, self.current)
            if depth > self.settings['quality']['max_robot_penetration_m']:
                raise SortFailure(f'Planned path obstructed ({depth*1000:.2f} mm): {pair}')
            if transit_clearance:
                distance, pair = robot_obstacle_clearance(self.b.model, ik.data, side, containers, transit_clearance)
                if distance < transit_clearance:
                    raise SortFailure(f'Planned arm/container clearance {distance*1000:.2f} mm '
                                      f'< {transit_clearance*1000:.2f} mm: {pair}')
            if self.pacing.get('continuous_paths') and carried:
                depth, pair = carried_obstacle_penetration(self.b.model, ik.data, carried)
                if depth > self.settings['quality']['max_robot_penetration_m']:
                    raise SortFailure(f'Planned payload path obstructed ({depth*1000:.2f} mm): {pair}')
                if self.phase == 'transfer':
                    margins = self.pacing.get('payload_transport_clearance_m_by_kind', {})
                    margin = max(margins.get(self.b.task.specs[name]['kind'], 0.) for name in carried)
                    if margin > 0:
                        distance, pair = carried_obstacle_clearance(self.b.model, ik.data, carried, margin)
                        if distance < margin:
                            raise SortFailure(f'Planned payload clearance {distance*1000:.2f} mm '
                                              f'< {margin*1000:.2f} mm: {pair}')

    def carried_objects(self):
        return [self.carried[1]] if self.carried else []

    def transit_to_pregrasp(self, side, joints, width):
        """Plan the whole unloaded container departure before executing a detour."""
        if self.carried is not None:
            raise SortFailure('Pregrasp transit requires an unloaded arm')
        off = 0 if side == 'left' else 7
        margin = self.pacing.get('unloaded_transit_clearance_m', 0.)
        start = self.q[off:off+6].copy()
        aperture = (self.q[off+6], width)
        try:
            self.check_joint_path(side, start, joints, gripper=aperture, transit_clearance=margin)
        except SortFailure as direct_error:
            tcp, rotation = self.ik[side].forward(start)
            routes = [('vertical_rise', rise, [(tcp+[0, 0, rise], rotation)])
                      for rise in self.pacing.get('unloaded_transit_rises_m', [.02, .04, .06])]
            # Retain the existing central waypoint, but validate both legs before
            # moving to it. Near-start contact is checked by the original gate;
            # the positive margin applies to the subsequent free-space transit.
            central = [.27, .30 if side == 'left' else -.30, .27]
            routes.append(('central_waypoint', None, [(central, grasp_rotation(30, 0))]))
            chosen = None
            for kind, rise, poses in routes:
                try:
                    if kind == 'central_waypoint':
                        target = self.q.copy()
                        target[off:off+6] = self.solve(side, *poses[0])
                        self.check_joint_path(side, start, target[off:off+6])
                        trajectory = smooth_trajectory(self.q, target, 1/self.fps,
                            seconds=self.pacing.get('minimum_move_s', 2.), **self.motion_limits())
                    else:
                        trajectory = self.plan_pose_path(side, poses)
                    self.check_joint_path(side, trajectory[-1, off:off+6], joints,
                                          gripper=aperture, transit_clearance=margin)
                    chosen = (kind, rise, trajectory)
                    break
                except SortFailure:
                    continue
            if chosen is None:
                raise SortFailure(f'No checked pregrasp transit: {direct_error}') from direct_error
            kind, rise, trajectory = chosen
            self.events.append({'phase': 'transit_detour_selected', 'object': self.current,
                'sim_time': self.b.sim_time, 'route': kind, 'vertical_rise_m': rise,
                'container_margin_m': margin, 'direct_rejection': str(direct_error)})
            for command in trajectory:
                self.q = command
                self.step()
            self.settle()
        self.move_joints(side, joints, gripper_width=width)

    def solve(self, side, position, rotation, seed=None):
        offset = 0 if side == 'left' else 7
        reference = self.b.task.config['reference']['joint14_rad_fraction'][offset:offset+6]
        yaw = np.arctan2(position[1]-(.3 if side == 'left' else -.3), position[0])
        seeds = [self.q[offset:offset+6] if seed is None else seed, reference]
        seeds += [[yaw, 1.8, -1., 0, -.8, roll] for roll in (-1.57, 1.57, 0)]
        best = None
        for s in seeds:
            q, ep, er = self.ik[side].solve(position, rotation, s)
            if best is None or ep+.1*er < best[1]+.1*best[2]:
                best = (q, ep, er)
            if ep < .003 and er < .035:
                return q
        raise SortFailure(f'{side} IK failed at {np.round(position,4).tolist()}: '
                          f'{best[1]*1000:.1f} mm, {np.rad2deg(best[2]):.1f} deg')

    def pose(self, side, position, rotation, seconds=None, gripper_width=None):
        self.move_joints(side, self.solve(side, np.asarray(position), rotation), seconds, gripper_width)

    def grip(self, side, width):
        index = 6 if side == 'left' else 13
        if self.pacing and width == 0. and self.current:
            # Broad rigid items do not require an impossible zero-aperture
            # target. Keep a force preload, still verify real bilateral contact.
            spec = self.b.task.specs[self.current]
            if spec['kind'] == 'cup': width = max(0., spec['top_diameter_m']-.013)
            elif spec['kind'] == 'charger': width = max(0., min(spec['size_m'][:2])-.015)
        limits = {key:self.motion['gripper_'+key+suffix] for key,suffix in
                  [('velocity','_m_s'),('acceleration','_m_s2'),('jerk','_m_s3')]}
        if width < self.q[index]:
            # Keep gentle capture dynamics even when free-space opening and
            # arm transit are faster. The contact acceleration gate is unchanged.
            for key, bound in self.pacing.get('closing_limits', {}).items():
                limits[key] = min(limits[key], bound)
        trajectory = smooth_trajectory([self.q[index]], [width], 1/self.fps,
            seconds=self.pacing.get('minimum_grip_s', 1.), **limits)
        for command in trajectory:
            self.q[index] = command[0]
            self.step()
        self.wait(self.pacing.get('grip_hold_s', .5))

    def object_pose(self, name):
        # Same privileged dictionary exposed through the external read API.
        return copy.deepcopy(self.session.read().extras['objects'][name])

    def tcp(self, side):
        ik = self.ik[side]
        return ik.forward(self.b.data.qpos[ik.qids])

    def plan_grasp(self, name):
        pose = self.object_pose(name)
        actual = pose_matrix(pose['position'], pose['quaternion_wxyz'])
        # Existing nominal grasp offsets were specified in world axes. Convert
        # once through the nominal object frame, then follow the measured SE(3).
        a = np.deg2rad(self.b.task.specs[name].get('yaw_deg', 0))
        nominal_r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        p = np.zeros(3)
        kind = self.b.task.specs[name]['kind']
        side = 'left' if kind in ('paper', 'towel') else 'right'
        if kind == 'paper':
            p += [0, 0, .009];r = grasp_rotation(0, 20);width = .095
        elif name == 'towel':
            p += [0, 0, .014];r = grasp_rotation(0, -8);width = .095
        elif name == 'spoon':
            p += [-.042, .009, .011];r = grasp_rotation(10, -12);width = .04
        elif name == 'bowl':
            p += [0, -.045, .043];r = grasp_rotation(10, 0);width = .045
        elif name == 'plate':
            p += [-.0687, -.0518, .014];r = grasp_rotation(45, 37, 90);width = .025
        elif name == 'side_dish':
            p += [-.040, -.049, .022];r = grasp_rotation(30, 0);width = .050
        else:
            p += [0, 0, .005];r = grasp_rotation(0, 0);width = .012
        delta_r = actual[:3, :3]@nominal_r.T
        return side, actual[:3, 3]+delta_r@p, delta_r@r, width

    def approach_position(self,name,p,r,distance):
        # The broad charger can be approached vertically. An oblique approach
        # can pass too close to the base even when its final grasp is reachable.
        return p+np.array([0,0,distance]) if self.b.task.specs[name]['kind'] in ('charger','cup') else p-r[:,2]*distance

    def contact_approach(self, side, p, r, name):
        """Use one continuous physical descent; assistance can stop at contact.

        No live object/robot state is repositioned. The legacy/assisted path
        retains bounded rest-to-rest segments to permit early engagement.
        """
        if self.assist:
            self.armed = (side, name)
        elif self.pacing.get('continuous_paths'):
            self.pose_path(side, [(p, r)], speed=self.pacing['contact_speed_m_s'])
            return
        for remaining in np.linspace(.030, 0., 11):
            self.pose(side, self.approach_position(name,p,r,remaining), r, seconds=.35)
            if self.assist and self.assist.active is not None:
                return

    def grasp_candidates(self, name, position, rotation, *, check_approach):
        # Small local tilt alternatives preserve the intended object grasp
        # region while avoiding the wrist joint limits of a single fixed frame.
        heights = self.pacing.get('grasp_height_offsets_m', [0.]) if check_approach else [0.]
        for height in heights:
            p = position+np.array([0., 0., height])
            for tilt in (0, 5, -5, 10, -10, 15, -15):
                a = np.deg2rad(tilt)
                ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                yield p, rotation@ry, {'height_offset_m': float(height), 'local_tilt_deg': tilt}

    def reachable_grasp(self, name, *, check_approach=False):
        side, original_p, r, width = self.plan_grasp(name)
        failures = []
        for p, candidate, details in self.grasp_candidates(name, original_p, r, check_approach=check_approach):
            try:
                pregrasp = self.solve(side, self.approach_position(name,p,candidate,.035), candidate)
                q = pregrasp
                for remaining in np.linspace(.030,0.,11):
                    q = self.solve(side,self.approach_position(name,p,candidate,remaining),candidate,q)
                q = self.solve(side, p+[0, 0, .035], candidate, q)
                if self.b.task.specs[name]['kind']=='plate':
                    self.solve(side,p+[0,0,.15],candidate,q)
                elif self.b.task.specs[name]['kind']=='cup':
                    self.solve(side,p+[0,0,.10],candidate,q)
                if check_approach:
                    off = 0 if side == 'left' else 7
                    self.check_joint_path(side, self.q[off:off+6], pregrasp)
                    start = self.q.copy()
                    start[off:off+6] = pregrasp
                    self.plan_pose_path(side, [(p, candidate)], start=start,
                                        speed=self.pacing['contact_speed_m_s'])
                    self.events.append({'phase': 'grasp_candidate_selected', 'object': name,
                        'sim_time': self.b.sim_time, **details, 'rejected_candidates': len(failures)})
                return side, p, candidate, width
            except SortFailure as exc:
                failures.append(str(exc))
        raise SortFailure(f'No reachable grasp for {name}: {failures[-1]}')

    def placement(self, side, name, target):
        """Search upright/tilted placement frames using the measured grasp offset."""
        if self.placement_search and self.b.task.specs[name]['kind'] in self.placement_search['kinds']:
            from .placement_search import search
            return search(self, side, name, target, self.placement_search)
        tcp, hand_r = self.tcp(side)
        obj = self.b.data.body('task/'+name)
        offset = hand_r.T@(obj.xpos-tcp)
        relative_r = hand_r.T@obj.xmat.reshape(3, 3)
        bin_body = self.b.data.body('task/'+target)
        bin_r = bin_body.xmat.reshape(3, 3)
        bin_p = bin_body.xpos.copy()
        size = np.array(self.b.task.specs[target]['size_m'])
        wall = self.b.task.specs[target]['wall_m']
        rim = bin_p[2]+size[2]
        # Reserve room for the fingers as well as the utensil itself. The old
        # +/-65 mm slots left too little clearance to open beside the rim.
        lateral = .025 if name == 'chopstick_a' else -.025 if name == 'chopstick_b' else 0.
        slot = np.array(self.b.task.config.get('drop_slots_m',{}).get(name,[0.,lateral])+[0.])
        goal_p = bin_p+bin_r@slot
        points = self.b.task.surface_points[name]
        choices = []
        for pitch in (0, 15, -15, 30, -30, 45, -45, 60, -60):
            for yaw in ((0, 15, -15, 30, -30, 45, -45, 90, -90, 180)
                        if name in ('towel','chopstick_a','chopstick_b','spoon') else (0, 90, 180, -90)):
                a, t = np.deg2rad([yaw, pitch])
                rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
                ry = np.array([[np.cos(t),0,np.sin(t)], [0,1,0], [-np.sin(t),0,np.cos(t)]])
                object_r = bin_r@rz@ry
                desired_r = object_r@relative_r.T
                rotated = points@object_r.T
                local = rotated@bin_r+slot
                if np.any(abs(local[:,:2]).max(0)>size[:2]/2-wall+.001):
                    continue
                # Entire item clears the rim before entering the basket footprint.
                high_obj = goal_p+np.array([0,0,rim-bin_p[2]+.035-rotated[:,2].min()])
                low_obj = high_obj.copy();low_obj[2] -= .045
                high = high_obj-desired_r@offset
                low = low_obj-desired_r@offset
                try:
                    q_high = self.solve(side, high, desired_r)
                    q_low = self.solve(side, low, desired_r, q_high)
                    self.check_joint_path(side, self.q[(0 if side=='left' else 7):][:6], q_high)
                    self.check_joint_path(side, q_high, q_low)
                except SortFailure:
                    continue
                choices.append((abs(pitch)+.1*abs(yaw), high, low, desired_r, q_high, q_low))
            if choices:
                break
        if not choices:
            # Slender utensils can roll in the jaw. Do not force their local
            # "up" axis to world +Z: search reachable hand frames directly.
            for tilt in (35,45,55,65):
                for yaw in (-20,-10,0,15,30):
                    desired_r = grasp_rotation(tilt,yaw)
                    object_r = desired_r@relative_r
                    local = points@object_r.T@bin_r+slot
                    if np.any(abs(local[:,:2]).max(0)>size[:2]/2-wall):
                        continue
                    rotated = points@object_r.T
                    high_obj = goal_p+np.array([0,0,rim-bin_p[2]+.035-rotated[:,2].min()])
                    low_obj = high_obj.copy();low_obj[2]-=.045
                    high,low = high_obj-desired_r@offset,low_obj-desired_r@offset
                    try:
                        q_high = self.solve(side,high,desired_r)
                        q_low = self.solve(side,low,desired_r,q_high)
                        self.check_joint_path(side, self.q[(0 if side=='left' else 7):][:6], q_high)
                        self.check_joint_path(side, q_high, q_low)
                    except SortFailure:
                        continue
                    choices.append((100+tilt,high,low,desired_r,q_high,q_low))
            if not choices:
                raise SortFailure(f'No reachable container placement for {name}')
        return min(choices, key=lambda c:c[0])

    def sort_one(self, name):
        self.current = name
        side, p, r, width = self.reachable_grasp(name)
        initial_z = self.object_pose(name)['position'][2]
        target = self.b.task.specs[name]['target']
        yaw = np.rad2deg(np.arctan2(p[1]-(.3 if side == 'left' else -.3), p[0]))
        raised_r = grasp_rotation(30, yaw)
        self.event('approach', side=side, target=target)
        if self.pacing.get('continuous_paths'):
            # Opening/closing to the pregrasp aperture overlaps free-space reach.
            q_above = self.solve(side, p+[0,0,.15], raised_r)
            self.transit_to_pregrasp(side, q_above, width)
        else:
            self.grip(side, width)
            self.pose(side, p+[0,0,.15], raised_r)
        # Objects may shift while approaching a stack; refresh from live truth.
        side, p, r, _ = self.reachable_grasp(name, check_approach=bool(
            self.pacing.get('continuous_paths') and self.mode == 'physical'))
        self.pose(side, self.approach_position(name,p,r,.035), r)
        self.event('grasp')
        self.contact_approach(side, p, r, name)
        self.grip(side, 0.)
        if self.assist and self.assist.active is None:
            raise SortFailure(f'No finger contact with {name}; no assisted grasp was made')
        contacts = finger_contacts(self.b, side, name)
        if len(contacts) != 2:
            raise SortFailure(f'{name} not pinched by both fingers: {contacts}')
        self.track_grasp(side,name)
        self.wait(self.pacing.get('grasp_hold_s', .25))
        self.event('lift', contact_fingers=contacts)
        rim_grasp=self.b.task.specs[name]['kind']=='plate'
        # A short normal lift clears the support before reorienting. Joint-space
        # interpolation between these two semantic goals avoids an unreachable
        # intermediate Cartesian orientation near a wrist limit.
        self.pose(side, p+[0,0,.035], r)
        self.pose(side,p+[0,0,.15],r if rim_grasp else raised_r)
        lifted = self.object_pose(name)['position'][2]-initial_z
        if lifted < .06:
            raise SortFailure(f'{name} did not lift: {lifted:.3f} m')
        self.event('transfer', lift_m=lifted)
        if not rim_grasp and (not self.pacing.get('direct_transfer') or self.b.task.specs[name]['kind']=='side_dish'):
            self.pose(side, [.35,.12,.28] if side == 'left' else [.38,-.20,.32],
                      grasp_rotation(40,-25) if side == 'left' else grasp_rotation(45,15))
        try:
            _, high, low, place_r, q_high, q_low = self.placement(side, name, target)
        except SortFailure:
            if not self.pacing.get('direct_transfer') or rim_grasp: raise
            self.pose(side, [.35,.12,.28] if side == 'left' else [.38,-.20,.32],
                      grasp_rotation(40,-25) if side == 'left' else grasp_rotation(45,15))
            _, high, low, place_r, q_high, q_low = self.placement(side, name, target)
        self.move_joints(side, q_high, None if self.pacing else 2.5)
        self.event('lower')
        self.move_joints(side, q_low, None if self.pacing else 1.5)
        self.event('open_gripper', object_pose=self.object_pose(name))
        self.releasing = (side, name)
        self.carried = None
        self.armed = None
        # Slender utensils need only a small release gap. Full opening beside
        # the basket wall unnecessarily sweeps the outer finger into its rim.
        release_width = .04 if name.startswith('chopstick') else min(.095, width+.020)
        # A rim can hold the jaws away from a closed target. Opening directly
        # toward the full gap then crosses contact loss midway through a fast
        # trajectory. First remove preload with a short rest-to-rest move near
        # the measured width, so contact loss occurs at low command speed.
        # This applies to physical contact as well as explicit assisted demos.
        ids = self.b.indices[side]['qpos']
        measured_width = float(self.b.data.qpos[ids[6:]].sum())
        index = 6 if side == 'left' else 13
        if self.assist or measured_width-self.q[index] > self.release_preload_m:
            self.event('unload_gripper', measured_width_m=measured_width,
                       command_width_m=float(self.q[index]))
            self.grip(side, min(release_width, measured_width+.002))
        if self.assist:
            for _ in range(3):
                measured_width = float(self.b.data.qpos[ids[6:]].sum())
                if abs(self.q[index]-measured_width) <= .003:
                    break
                self.grip(side, float(np.clip(measured_width+.001, 0., release_width)))
            else:
                raise SortFailure('Cannot unload gripper tracking error before releasing assistance')
            self.assist.release()
        self.event('release', object_pose=self.object_pose(name))
        self.grip(side, release_width)
        self.releasing = None
        self.wait(self.pacing.get('release_hold_s', .5))
        # Preserve a vertical retreat, avoiding a bowed joint-space path.
        # Continuous mode passes through its IK knots without intermediate stops.
        offset = 0 if side == 'left' else 7
        retreat_start, _ = self.ik[side].forward(self.q[offset:offset+6])
        if self.pacing.get('continuous_paths'):
            self.pose_path(side, [(high, place_r)])
        else:
            for t in np.linspace(0, 1, 5)[1:]:
                self.pose(side, (1-t)*retreat_start+t*high, place_r, seconds=.4)
        self.wait(self.pacing.get('placement_hold_s', 2.))
        metric = self.session.read().extras['task']['metrics'][name]
        result = {'name': name, 'target': target, 'grasp_mode': self.mode,
                  'lift_m': lifted, **metric}
        self.results.append(result)
        self.event('placed', success=result['success'])
        if not self.pacing.get('direct_next_object'):
            self.pose(side, [.27,.30,.24] if side == 'left' else [.27,-.30,.24], grasp_rotation(30,0))
        if not metric['success']:
            raise SortFailure(f'{name} was released but is not fully contained in {target}')

    def run(self, objects=None):
        try:
            if not self.prepared:
                self.prepare(objects)
            if objects is None:
                # Each object travels directly from the setting to its bin.
                for name in (*[n for n in self.b.task.active_names if self.b.task.specs[n]['kind'] == 'paper'],
                             'towel','spoon','bowl','plate','side_dish','chopstick_a','chopstick_b'):
                    self.sort_one(name)
            else:
                for name in objects:
                    self.sort_one(name)
            self.wait(2.)
            summary = self.session.read().extras['task']['metrics']['summary']
            self.event('finished', **summary)
            if objects is None and not summary['success']:
                raise SortFailure('Final active-inventory containment check failed')
            return summary
        except BaseException:
            self.armed=None
            if self.assist:self.assist.release()
            raise
