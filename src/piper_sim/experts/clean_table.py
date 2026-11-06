"""Privileged scripted sorting teacher, independent of task geometry and recording.

MuJoCo implementation: ground-truth poses -> bounded IK -> 14 position targets.
Optional contact assistance is explicitly logged and never enabled in physical
mode. The teacher never assigns live object or robot positions/velocities.
"""
import copy
import numpy as np
import mujoco

from .mujoco_ik import PiperIK, grasp_rotation
from .grasp_assist import ContactGraspAssist, finger_contacts
from .trajectory import smooth_trajectory
from .quality import MotionQuality
from .collision import robot_penetration
from ..transforms import pose_matrix
from ..config import ROOT


class SortFailure(RuntimeError):
    pass


class CleanTableTeacher:
    def __init__(self, session, *, grasp_mode='assisted', frame_hook=None, settings=None):
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
        self.quality = MotionQuality(self.settings, 1/self.fps)
        self.holding = False
        self.prepared = False
        self.collision_peak = {'depth_m': 0., 'pair': None}
        self.camera_quality = {}
        self.configure_contact_control()

    def configure_contact_control(self):
        """Explicit simulation controller settings, not identified hardware values."""
        m = self.b.model
        m.opt.impratio = 10
        for i in range(m.ngeom):
            name = m.body(int(m.geom_bodyid[i])).name
            if name in ('left/link7', 'left/link8', 'right/link7', 'right/link8'):
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
        return {'name': 'clean_table_privileged_v2', 'grasp_mode': self.mode,
                'privileged_inputs': ['object poses', 'container poses', 'geometry', 'contacts', 'robot kinematics'],
                'physical_only': self.mode == 'physical', 'training_ready': False,
                'training_note': 'Scripted demo; assisted mode is not a physical grasp-success benchmark.',
                'arm_bias_feedforward': True, **self.settings['controller'],
                'finger_friction': [1., .015, .002],
                'tcp_in_link6_m': [0, 0, .1375], 'placemat': 'left on table',
                'motion_and_quality': copy.deepcopy(self.settings),
                'trajectory': 'septic C3 rest-to-rest; per-joint velocity/acceleration/jerk duration limits',
                'grasp_frames': 'measured object SE(3) relative to nominal object grasp frame'}

    def event(self, phase, **fields):
        self.phase = phase
        record = {'phase': phase, 'object': self.current, 'sim_time': self.b.sim_time, **fields}
        self.events.append(record)
        print(record, flush=True)

    def step(self):
        # Robot feedforward only; free task objects always obey the physics solver.
        for ids in self.b.indices.values():
            self.b.data.qfrc_applied[ids['dof']] = self.b.data.qfrc_bias[ids['dof']]
        obs = self.session.step(self.q)
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
            if finger_contacts(self.b, side, name):
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
        for name in objects or self.b.task.config['success']['required']:
            side, p, r, _ = self.reachable_grasp(name)
            q = self.solve(side, p-r[:, 2]*.035, r)
            self.solve(side, p, r, q)
        self.prepared = True
        self.quality.reset()
        self.camera_quality = {}
        self.quality.add(self.b.sim_time, self.q, self.session.read().position,
                         context={'phase': 'prepared', 'object': None})
        self.event('prepared')

    def move_joints(self, side, joints, seconds=2.):
        offset = 0 if side == 'left' else 7
        before = self.q.copy()
        target = before.copy();target[offset:offset+6] = joints
        self.check_joint_path(side, before[offset:offset+6], joints)
        trajectory = smooth_trajectory(before, target, 1/self.fps, seconds=seconds,
            velocity=[*[self.motion['joint_velocity_rad_s']]*6, self.motion['gripper_velocity_m_s']]*2,
            acceleration=[*[self.motion['joint_acceleration_rad_s2']]*6, self.motion['gripper_acceleration_m_s2']]*2,
            jerk=[*[self.motion['joint_jerk_rad_s3']]*6, self.motion['gripper_jerk_m_s3']]*2)
        for command in trajectory:
            self.q = command
            self.step()
        self.settle()

    def check_joint_path(self, side, start, end):
        """Check sampled interpolation on scratch state before executing a segment.

        This is a rejection gate, not a general collision-free motion planner.
        It checks robot-obstacle and inter-arm contacts; carried-object sweeps
        remain covered only by live physics/outcome checks.
        """
        ik = self.ik[side]
        carried = self.assist.active[2] if self.assist and self.assist.active else None
        if carried:
            hand = self.b.data.body(side+'/link6')
            obj = self.b.data.body('task/'+carried)
            rel_p = hand.xmat.reshape(3, 3).T@(obj.xpos-hand.xpos)
            rel_r = hand.xmat.reshape(3, 3).T@obj.xmat.reshape(3, 3)
            obj_q = self.b.model.joint('task/'+carried+'/free').qposadr[0]
        steps = max(2, int(np.ceil(np.max(abs(np.asarray(end)-start))/.08)))
        for t in np.linspace(0, 1, steps+1):
            ik.data.qpos[:] = self.b.data.qpos
            ik.data.qpos[ik.qids] = (1-t)*np.asarray(start)+t*np.asarray(end)
            mujoco.mj_kinematics(self.b.model, ik.data)
            if carried:
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

    def pose(self, side, position, rotation, seconds=2.):
        self.move_joints(side, self.solve(side, np.asarray(position), rotation), seconds)

    def grip(self, side, width):
        index = 6 if side == 'left' else 13
        trajectory = smooth_trajectory([self.q[index]], [width], 1/self.fps, seconds=1.,
            velocity=self.motion['gripper_velocity_m_s'], acceleration=self.motion['gripper_acceleration_m_s2'],
            jerk=self.motion['gripper_jerk_m_s3'])
        for command in trajectory:
            self.q[index] = command[0]
            self.step()
        self.wait(.5)

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
        side = 'left' if name in ('paper', 'towel') else 'right'
        if name == 'paper':
            p += [0, 0, .009];r = grasp_rotation(0, 20);width = .095
        elif name == 'towel':
            p += [0, 0, .014];r = grasp_rotation(0, -8);width = .095
        elif name == 'spoon':
            p += [-.042, .009, .011];r = grasp_rotation(10, -12);width = .04
        elif name == 'bowl':
            p += [0, -.045, .043];r = grasp_rotation(10, 0);width = .045
        elif name == 'plate':
            p += [0, -.089, .018];r = grasp_rotation(0, 0);width = .025
        elif name == 'side_dish':
            p += [-.063, -.025, .018];r = grasp_rotation(25, 60);width = .035
        else:
            p += [0, 0, .005];r = grasp_rotation(0, 0);width = .012
        delta_r = actual[:3, :3]@nominal_r.T
        return side, actual[:3, 3]+delta_r@p, delta_r@r, width

    def contact_approach(self, side, p, r, name):
        """Short smooth descent segments stop at the first contact-assisted grasp.

        Each segment ends at rest, so stopping introduces no target derivative
        discontinuity. No live object/robot state is repositioned.
        """
        if self.assist:
            self.armed = (side, name)
        for remaining in np.linspace(.030, 0., 11):
            self.pose(side, p-r[:, 2]*remaining, r, seconds=.35)
            if self.assist and self.assist.active is not None:
                return

    def reachable_grasp(self, name):
        side, p, r, width = self.plan_grasp(name)
        failures = []
        # Small local tilt alternatives preserve the intended object grasp
        # region while avoiding the wrist joint limits of a single fixed frame.
        for tilt in (0, 5, -5, 10, -10, 15, -15):
            a = np.deg2rad(tilt)
            ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            candidate = r@ry
            try:
                q = self.solve(side, p-candidate[:, 2]*.035, candidate)
                q = self.solve(side, p, candidate, q)
                self.solve(side, p+[0, 0, .035], candidate, q)
                return side, p, candidate, width
            except SortFailure as exc:
                failures.append(str(exc))
        raise SortFailure(f'No reachable grasp for {name}: {failures[-1]}')

    def staging_position(self):
        pose = self.object_pose('placemat')
        t = pose_matrix(pose['position'], pose['quaternion_wxyz'])
        p = t[:3, 3]+t[:3, :3]@np.array([.015, -.25, 0.])
        p[2] = self.b.task.table_top
        return p

    def placement(self, side, name, target):
        """Search upright/tilted placement frames using the measured grasp offset."""
        tcp, hand_r = self.tcp(side)
        obj = self.b.data.body('task/'+name)
        offset = hand_r.T@(obj.xpos-tcp)
        relative_r = hand_r.T@obj.xmat.reshape(3, 3)
        staging = target == 'staging_table'
        if staging:
            bin_r = np.eye(3)
            bin_p = self.staging_position()
            size, wall, rim = np.array([.20,.20,.10]), 0., self.b.task.table_top
        else:
            bin_body = self.b.data.body('task/'+target)
            bin_r = bin_body.xmat.reshape(3, 3)
            bin_p = bin_body.xpos.copy()
            size = np.array(self.b.task.specs[target]['size_m'])
            wall = self.b.task.specs[target]['wall_m']
            rim = bin_p[2]+size[2]
        # Reserve room for the fingers as well as the utensil itself. The old
        # +/-65 mm slots left too little clearance to open beside the rim.
        lateral = .025 if name == 'chopstick_a' else -.025 if name == 'chopstick_b' else 0.
        goal_p = bin_p+bin_r@np.array([0.,lateral,0.])
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
                local = rotated@bin_r+np.array([0,lateral,0])
                if np.any(abs(local[:,:2]).max(0)>size[:2]/2-wall+.001):
                    continue
                # Entire item clears the rim before entering the basket footprint.
                high_obj = goal_p+np.array([0,0,rim-bin_p[2]+.035-rotated[:,2].min()])
                low_obj = high_obj.copy();low_obj[2] -= .020 if staging else .045
                high = high_obj-desired_r@offset
                low = low_obj-desired_r@offset
                try:
                    q_high = self.solve(side, high, desired_r)
                    q_low = self.solve(side, low, desired_r, q_high)
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
                    local = points@object_r.T@bin_r+np.array([0,lateral,0])
                    if np.any(abs(local[:,:2]).max(0)>size[:2]/2-wall):
                        continue
                    rotated = points@object_r.T
                    high_obj = goal_p+np.array([0,0,rim-bin_p[2]+.035-rotated[:,2].min()])
                    low_obj = high_obj.copy();low_obj[2]-=.020 if staging else .045
                    high,low = high_obj-desired_r@offset,low_obj-desired_r@offset
                    try:
                        q_high = self.solve(side,high,desired_r)
                        q_low = self.solve(side,low,desired_r,q_high)
                    except SortFailure:
                        continue
                    choices.append((100+tilt,high,low,desired_r,q_high,q_low))
            if not choices:
                raise SortFailure(f'No reachable container placement for {name}')
        return min(choices, key=lambda c:c[0])

    def sort_one(self, name, staging=False):
        self.current = name
        side, p, r, width = self.reachable_grasp(name)
        initial_z = self.object_pose(name)['position'][2]
        target = 'staging_table' if staging else self.b.task.specs[name]['target']
        yaw = np.rad2deg(np.arctan2(p[1]-(.3 if side == 'left' else -.3), p[0]))
        raised_r = grasp_rotation(30, yaw)
        self.event('approach', side=side, target=target)
        self.grip(side, width)
        self.pose(side, p+[0,0,.15], raised_r)
        # Objects may shift while approaching a stack; refresh from live truth.
        side, p, r, _ = self.reachable_grasp(name)
        self.pose(side, p-r[:,2]*.035, r)
        self.event('grasp')
        self.contact_approach(side, p, r, name)
        self.grip(side, 0.)
        if self.assist and self.assist.active is None:
            raise SortFailure(f'No finger contact with {name}; no assisted grasp was made')
        contacts = finger_contacts(self.b, side, name)
        self.event('lift', contact_fingers=contacts)
        self.pose(side, p+[0,0,.035], r)
        self.pose(side, p+[0,0,.15], raised_r)
        lifted = self.object_pose(name)['position'][2]-initial_z
        if lifted < .06:
            raise SortFailure(f'{name} did not lift: {lifted:.3f} m')
        self.event('transfer', lift_m=lifted)
        self.pose(side, [.35,.12,.28] if side == 'left' else [.38,-.20,.32],
                  grasp_rotation(40,-25) if side == 'left' else grasp_rotation(45,15))
        _, high, low, place_r, q_high, q_low = self.placement(side, name, target)
        self.move_joints(side, q_high, 2.5)
        self.event('lower')
        self.move_joints(side, q_low, 1.5)
        self.event('open_gripper', object_pose=self.object_pose(name))
        self.armed = None
        # Slender utensils need only a small release gap. Full opening beside
        # the basket wall unnecessarily sweeps the outer finger into its rim.
        release_width = .04 if name.startswith('chopstick') else min(.095, width+.020)
        # Unload the finger servo BEFORE removing the constraint. A full-open
        # target while a rim blocks a finger stores a large tracking error; on
        # release that error launches the jaw. First open only 2 mm beyond the
        # measured aperture, then remove assistance and finish the smooth opening.
        if self.assist:
            ids = self.b.indices[side]['qpos']
            measured_width = float(self.b.data.qpos[ids[6:]].sum())
            self.event('unload_gripper', measured_width_m=measured_width)
            self.grip(side, min(release_width, measured_width+.002))
            index = 6 if side == 'left' else 13
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
        self.wait(.5)
        # Approximate a vertical Cartesian retreat with short rest-to-rest
        # segments, avoiding the bowed TCP path of one long joint interpolation.
        offset = 0 if side == 'left' else 7
        retreat_start, _ = self.ik[side].forward(self.q[offset:offset+6])
        for t in np.linspace(0, 1, 5)[1:]:
            self.pose(side, (1-t)*retreat_start+t*high, place_r, seconds=.4)
        self.wait(2.)
        if staging:
            pos = np.array(self.object_pose(name)['position'])
            metric = {'success': bool(np.linalg.norm(pos[:2]-self.staging_position()[:2])<.03
                                      and abs(pos[2]-self.b.task.table_top)<.01)}
        else:
            metric = self.session.read().extras['task']['metrics'][name]
        result = {'name': name, 'target': target, 'grasp_mode': self.mode,
                  'lift_m': lifted, **metric}
        self.results.append(result)
        self.event('placed', success=result['success'])
        self.pose(side, [.27,.30,.24] if side == 'left' else [.27,-.30,.24], grasp_rotation(30,0))
        if not metric['success']:
            raise SortFailure(f'{name} was released but is not fully contained in {target}')

    def run(self, objects=None):
        try:
            if not self.prepared:
                self.prepare(objects)
            if objects is None:
                # Put the large plate on the basket floor before restoring the
                # bowl. Both staging moves are real recorded arm trajectories.
                for name in ('paper','towel','spoon'):
                    self.sort_one(name)
                self.sort_one('bowl',staging=True)
                for name in ('plate','side_dish','chopstick_a','chopstick_b','bowl'):
                    self.sort_one(name)
            else:
                for name in objects:
                    self.sort_one(name)
            self.wait(2.)
            summary = self.session.read().extras['task']['metrics']['summary']
            self.event('finished', **summary)
            if objects is None and not summary['success']:
                raise SortFailure('Final eight-object containment check failed')
            return summary
        except BaseException:
            self.armed=None
            if self.assist:self.assist.release()
            raise
