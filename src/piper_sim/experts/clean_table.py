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


class SortFailure(RuntimeError):
    pass


class CleanTableTeacher:
    def __init__(self, session, *, grasp_mode='assisted', frame_hook=None):
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
                m.geom_solref[i] = [.0042, 1.]
                m.geom_solimp[i] = [.999, .9999, .001, .5, 2.]
        for side in ('left', 'right'):
            a = m.actuator(side+'/gripper').id
            m.actuator_gainprm[a, 0] = 10000
            m.actuator_biasprm[a, 1] = -10000
            m.actuator_forcerange[a] = [-25, 25]

    def metadata(self):
        return {'name': 'clean_table_privileged_v1', 'grasp_mode': self.mode,
                'privileged_inputs': ['object poses', 'container poses', 'geometry', 'contacts', 'robot kinematics'],
                'physical_only': self.mode == 'physical', 'training_ready': False,
                'training_note': 'Scripted demo; assisted mode is not a physical grasp-success benchmark.',
                'arm_bias_feedforward': True, 'gripper_kp_N_per_m': 10000,
                'gripper_force_limit_N': 25, 'finger_friction': [1., .015, .002],
                'tcp_in_link6_m': [0, 0, .1375], 'placemat': 'left on table'}

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
        for _ in range(round(seconds*self.fps)):
            self.step()

    def move_joints(self, side, joints, seconds=2.):
        offset = 0 if side == 'left' else 7
        before = self.q.copy()
        target = before.copy();target[offset:offset+6] = joints
        steps = max(round(seconds*self.fps), int(np.max(abs(target-before))/.65*self.fps), 1)
        for i in range(steps):
            t = (i+1)/steps
            blend = t**3*(10-15*t+6*t*t)
            self.q = before+(target-before)*blend
            self.step()
        self.wait(.25)

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
        start = self.q[index]
        for i in range(round(self.fps)):
            self.q[index] = start+(width-start)*(i+1)/round(self.fps)
            self.step()
        self.wait(.5)

    def object_pose(self, name):
        # Same privileged dictionary exposed through the external read API.
        return copy.deepcopy(self.session.read().extras['objects'][name])

    def tcp(self, side):
        ik = self.ik[side]
        return ik.forward(self.b.data.qpos[ik.qids])

    def plan_grasp(self, name):
        p = np.array(self.object_pose(name)['position'])
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
        return side, p, r, width

    def placement(self, side, name, target):
        """Search upright/tilted placement frames using the measured grasp offset."""
        tcp, hand_r = self.tcp(side)
        obj = self.b.data.body('task/'+name)
        offset = hand_r.T@(obj.xpos-tcp)
        relative_r = hand_r.T@obj.xmat.reshape(3, 3)
        staging = target == 'staging_table'
        if staging:
            bin_r = np.eye(3)
            bin_p = np.array([.34,-.25,self.b.task.table_top])
            size, wall, rim = np.array([.20,.20,.10]), 0., self.b.task.table_top
        else:
            bin_body = self.b.data.body('task/'+target)
            bin_r = bin_body.xmat.reshape(3, 3)
            bin_p = bin_body.xpos.copy()
            size = np.array(self.b.task.specs[target]['size_m'])
            wall = self.b.task.specs[target]['wall_m']
            rim = bin_p[2]+size[2]
        lateral = .065 if name == 'chopstick_a' else -.065 if name == 'chopstick_b' else 0.
        goal_p = bin_p+bin_r@np.array([0.,lateral,0.])
        points = self.b.task.surface_points[name]
        choices = []
        for pitch in (0, 15, -15, 30, -30, 45, -45, 60, -60):
            for yaw in ((0, 180) if name in ('towel','chopstick_a','chopstick_b','spoon') else (0, 90, 180, -90)):
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
        side, p, r, width = self.plan_grasp(name)
        initial_z = self.object_pose(name)['position'][2]
        target = 'staging_table' if staging else self.b.task.specs[name]['target']
        yaw = np.rad2deg(np.arctan2(p[1]-(.3 if side == 'left' else -.3), p[0]))
        raised_r = grasp_rotation(30, yaw)
        self.event('approach', side=side, target=target)
        self.grip(side, width)
        self.pose(side, p+[0,0,.15], raised_r)
        self.pose(side, p-r[:,2]*.035, r)
        self.event('grasp')
        if self.assist:
            self.armed = (side, name)
        self.pose(side, p, r)
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
        self.grip(side, .095)
        # Remove assistance after the pads clear the item. Disabling a weld
        # while a commanded-closed jaw is loaded can launch a thin utensil.
        if self.assist:
            self.assist.release()
        self.event('release', object_pose=self.object_pose(name))
        self.wait(.5)
        self.pose(side, high, place_r)
        self.wait(2.)
        if staging:
            pos = np.array(self.object_pose(name)['position'])
            metric = {'success': bool(np.linalg.norm(pos[:2]-[.34,-.25])<.03
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
            self.wait(1.)
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
