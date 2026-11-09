"""Rigid clearing with physical cup nesting, then whole-stack transport.

All live motion goes through the existing position-control interface. No welds,
object state assignments or scripted attachments are used by this teacher.
"""
import numpy as np
from .clean_table import CleanTableTeacher, SortFailure
from .mujoco_ik import grasp_rotation
from .grasp_assist import finger_contacts
from ..transforms import pose_matrix


class RigidTableTeacher(CleanTableTeacher):
    name = 'rigid_table_privileged_v1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.mode!='physical': raise ValueError('Rigid stacking requires physical grasps')
        self.stack_carried=False
        self.rim_release_noslip_iterations = self.pacing.get('rim_release_noslip_iterations', 0)
        if type(self.rim_release_noslip_iterations) is not int or self.rim_release_noslip_iterations < 0:
            raise ValueError('Rim release noslip iterations must be a nonnegative integer')

    def metadata(self):
        value = {**super().metadata(),'name':'rigid_table_privileged_v1',
                'task_sequence':'nest cups on table, transport entire stack, clear remaining rigid props',
                'per_kind_gripper_force_limits_N':{'pen':.35,'chopstick':.30,'plate':30.,'side_dish':15.,'other':5.},
                'plate_pad_rolling_friction_m':.008,
                'plate_gripper_kp_N_per_m':10000.,
                'noslip_iterations_loaded_rim_grasp':10,
                'noslip_iterations_rim_release':self.rim_release_noslip_iterations,
                'placemat':'removed with all flexible objects','attachments':False}
        value['per_target_contact_material_switching'] = getattr(self.b, 'finger_contact', None) is None
        if not value['per_target_contact_material_switching']:
            value.pop('plate_pad_rolling_friction_m')
        return value

    @property
    def cups(self):
        return [n for n in self.b.task.active_names if self.b.task.specs[n]['kind']=='cup']

    def plan_grasp(self,name):
        spec=self.b.task.specs[name];kind=spec['kind']
        if kind not in ('cup','pen','charger','chopstick'): return super().plan_grasp(name)
        pose=self.object_pose(name);t=pose_matrix(pose['position'],pose['quaternion_wxyz'])
        if kind=='chopstick':
            # A stick may roll onto another face when its neighbour is picked.
            # Follow its long axis, not the arbitrary roll of its local frame.
            axis=t[:3,0]
            yaw=(np.rad2deg(np.arctan2(axis[1],axis[0]))+90)%180-90
            center=t[:3,3]+t[:3,:3]@np.array([0,0,spec['size_m'][2]/2])
            r = grasp_rotation(0,yaw-spec.get('yaw_deg',0))
            # Fit the local cross-section plus approach clearance. Excessive
            # opening can make the outside pad strike a neighbouring stick.
            cross_width = abs(r[:, 1]@t[:3, 1:3])@np.asarray(spec['size_m'][1:])
            width = float(cross_width+self.pacing.get('slender_pregrasp_clearance_m', .004))
            return 'right',center+[0,0,.002],r,width
        if kind=='cup':
            # Axisymmetric cup: choose jaw yaw for arm reach, independent of print.
            p=t[:3,3]+t[:3,:3]@np.array([0,0,.080])
            side='right' if name=='cup_00' else 'left'
            return side,p,grasp_rotation(45,40 if side=='right' else -40),.095
        if kind=='pen':offset=[-.025,0,.008];width=.025
        else:offset=[0,0,.022];width=.095
        # Align with opposed flat faces, not diagonally across the corners.
        return 'right',t[:3,3]+t[:3,:3]@offset,t[:3,:3]@grasp_rotation(20 if kind=='charger' else 0,0),width

    def grasp_candidates(self, name, position, rotation, *, check_approach):
        spec = self.b.task.specs[name]
        if spec['kind'] != 'chopstick':
            yield from super().grasp_candidates(name, position, rotation, check_approach=check_approach)
            return
        # Tilting a pad beside a 6 mm stick can put its tip on the table before
        # its face reaches the stick. Keep the pad upright and vary the grasp
        # along the actual long axis, inside the central half of the object.
        pose = self.object_pose(name)
        axis = pose_matrix(pose['position'], pose['quaternion_wxyz'])[:3, 0]
        extent = min(.04, spec['size_m'][0]/4)
        for fraction in (0., -.5, .5, -1., 1.):
            offset = fraction*extent
            yield position+offset*axis, rotation, {'height_offset_m': 0., 'local_tilt_deg': 0.,
                                                   'long_axis_offset_m': offset}

    def sort_one(self,name):
        aid=self.b.model.actuator('right/gripper').id
        kind=self.b.task.specs[name]['kind']
        force={'pen':.35,'chopstick':.30,'plate':30.,'side_dish':15.}.get(kind,self.settings['controller']['gripper_force_limit_N'])
        self.b.model.actuator_forcerange[aid]=[-force,force]
        kp=10000. if kind in ('plate','side_dish') else self.settings['controller']['gripper_kp_N_per_m']
        self.b.model.actuator_gainprm[aid,0]=kp
        self.b.model.actuator_biasprm[aid,1]=-kp
        for g in range(self.b.model.ngeom):
            if (getattr(self.b, 'finger_contact', None) is None and
                    self.b.model.body(self.b.model.geom_bodyid[g]).name in ('right/link7','right/link8')):
                self.b.model.geom_friction[g,2]=.008 if kind in ('plate','side_dish') else .002
                self.b.model.geom_solref[g,0]=.0042 if kind=='side_dish' else self.settings['controller']['finger_contact_time_constant_s']
                self.b.model.geom_solimp[g,:2]=[.99,.999] if kind=='side_dish' else [.95,.99]
        return super().sort_one(name)

    def step(self):
        # Refine friction constraints for a loaded thin rim. The physical
        # contact model still decides whether it can stick or slips/falls.
        iterations = (10 if self.carried and
            self.b.task.specs[self.carried[1]]['kind'] in ('plate','side_dish') else 0)
        # Tracking may stop before the pads physically separate. A configured
        # release keeps the solver resolution through unloading/opening, without
        # retaining a grasp constraint, contact force or object pose.
        if self.releasing and self.b.task.specs[self.releasing[1]]['kind'] in ('plate', 'side_dish'):
            iterations = max(iterations, self.rim_release_noslip_iterations)
        self.b.model.opt.noslip_iterations = iterations
        obs=super().step()
        if self.stack_carried:
            metric=self.b.task.stack_metrics(obs.extras['objects'])
            if not metric['nested']:
                raise SortFailure('Cup stack separated/tilted during transport: '+str(metric))
        return obs

    def carried_objects(self):
        return self.cups if self.stack_carried else super().carried_objects()

    def wait_for_final_stability(self):
        """Keep recording real physics until the final inventory stays still.

        Preserve the existing two-second minimum. Extend only when needed, with
        a bounded timeout; the independent closed-file auditor remains decisive.
        No state reset, threshold change, frame trimming or timestamp rewriting.
        """
        limits = self.settings['dataset']
        minimum = 2.
        timeout = self.pacing.get('final_settle_timeout_s', 5.)
        if not np.isfinite(timeout) or timeout < max(minimum, limits['final_stable_hold_s']):
            raise ValueError('Final settle timeout is too short or nonfinite')
        dt = 1/self.fps
        previous = self.session.read().extras['objects']
        stable_frames = 0
        required_frames = int(np.ceil(limits['final_stable_hold_s']/dt))
        minimum_frames = int(np.ceil(minimum/dt))
        was_holding = self.holding
        self.holding = True
        try:
            for frame in range(1, int(np.ceil(timeout/dt))+1):
                observation = self.step()
                poses = observation.extras['objects']
                speeds, angular = [], []
                for name in self.b.task.required:
                    before, after = previous[name], poses[name]
                    speeds.append(np.linalg.norm(np.subtract(after['position'], before['position']))/dt)
                    dot = abs(float(np.dot(before['quaternion_wxyz'], after['quaternion_wxyz'])))
                    angular.append(2*np.arccos(np.clip(dot, 0., 1.))/dt)
                stable = (observation.extras['task']['metrics']['summary']['success'] and self.carried is None
                    and max(speeds) <= limits['final_object_speed_m_s']
                    and max(angular) <= limits['final_object_angular_speed_rad_s'])
                stable_frames = stable_frames+1 if stable else 0
                previous = poses
                if frame >= minimum_frames and stable_frames >= required_frames:
                    return
                if frame == minimum_frames:
                    self.event('final_settle_extension', stable_seconds=stable_frames*dt, timeout_s=timeout)
            raise SortFailure(f'Final objects did not remain stable for {limits["final_stable_hold_s"]} s '
                              f'within {timeout} s; last stable interval={stable_frames*dt:.3f} s')
        finally:
            self.holding = was_holding

    def straight(self,side,position,rotation,spacing=.025):
        """Keep nesting/retreat Cartesian, with continuous motion when configured."""
        if self.pacing.get('continuous_paths'):
            self.pose_path(side, [(position, rotation)],
                speed=self.pacing['contact_speed_m_s'] if spacing <= .010 else None)
            return
        start,_=self.tcp(side);target=np.asarray(position)
        count=max(1,int(np.ceil(np.linalg.norm(target-start)/spacing)))
        for fraction in np.linspace(0,1,count+1)[1:]:
            self.pose(side,start+(target-start)*fraction,rotation,seconds=.35)

    def pick_cup(self,name):
        self.current=name
        side,p,r,width=self.reachable_grasp(name)
        self.event('approach',side=side,target='cup_stack')
        if self.pacing.get('continuous_paths'):
            self.pose(side,p+[0,0,.10],r,gripper_width=width)
        else:
            self.grip(side,width)
            self.pose(side,p+[0,0,.10],r)
        side,p,r,width=self.reachable_grasp(name)
        self.straight(side,p,r)
        self.event('grasp')
        self.grip(side,0.)
        if len(finger_contacts(self.b,side,name))!=2:
            raise SortFailure(f'{name}: missing bilateral cup contact')
        self.track_grasp(side,name)
        self.wait(self.pacing.get('grasp_hold_s', .25))
        initial=self.object_pose(name)['position'][2]
        self.event('lift',contact_fingers=finger_contacts(self.b,side,name))
        self.straight(side,p+[0,0,.10],r)
        if self.object_pose(name)['position'][2]-initial<.07:raise SortFailure('Cup did not lift')
        return side,r

    def cup_target(self,side,rotation,object_position):
        tcp,hand_r=self.tcp(side)
        obj=self.b.data.body('task/'+self.current)
        offset=hand_r.T@(obj.xpos-tcp)
        return np.asarray(object_position)-rotation@offset

    def stack_cups(self):
        base=self.cups[0]
        for upper in self.cups[1:]:
            side,r=self.pick_cup(upper)
            lower=self.cups[self.cups.index(upper)-1]
            p=np.array(self.object_pose(lower)['position'])
            # Hold the upper cup above the lower rim, align axes, then descend.
            high=self.cup_target(side,r,p+[0,0,.115])
            current,_=self.tcp(side)
            route=[([current[0],current[1],max(high[2],current[2],.22)],r),
                   ([high[0],high[1],max(high[2],current[2],.22)],r),(high,r)]
            if self.pacing.get('continuous_paths'):self.pose_path(side,route)
            else:
                for target,rot in route:self.straight(side,target,rot)
            self.event('nest',base=lower)
            low=self.cup_target(side,r,p+[0,0,.026])
            self.straight(side,low,r,spacing=.010)
            self.carried=None
            self.event('release_for_nesting',base=lower)
            self.grip(side,.095)
            self.straight(side,high+[0,0,.04],r)
            self.wait(1.)
            names=self.cups[:self.cups.index(upper)+1]
            check=self.b.task.stack_metrics(self.session.read().extras['objects'],names)
            self.event('stack_checked',**check)
            if not check['nested']:raise SortFailure('Cups did not nest: '+str(check))
        self.wait(.8)
        if not self.b.task.stack_on_table_verified:
            raise SortFailure('No stable complete stack on table')
        self.event('stack_ready',cups=self.cups)
        self.pose('left',[.27,.32,.26],grasp_rotation(30,0))
        side,r=self.pick_cup(base)
        self.stack_carried=True
        self.current=base
        binpose=self.object_pose('basket');t=pose_matrix(binpose['position'],binpose['quaternion_wxyz'])
        goal=t[:3,3]+t[:3,:3]@np.array([-.105,.085,.035])
        start_obj=np.array(self.object_pose(base)['position'])
        transit_z=max(start_obj[2],t[2,3]+self.b.task.specs['basket']['size_m'][2]+.035)
        high_obj=np.array([goal[0],goal[1],transit_z])
        self.event('transfer_stack',target='basket')
        self.straight(side,self.cup_target(side,r,[start_obj[0],start_obj[1],transit_z]),r)
        # Change only world yaw; the nested cups remain upright throughout.
        route=[]
        for fraction in np.linspace(0,1,21)[1:]:
            angle=np.deg2rad(-40*fraction)
            rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            place_r=rz@r
            pos=(1-fraction)*np.array([start_obj[0],start_obj[1],transit_z])+fraction*high_obj
            target=self.cup_target(side,place_r,pos)
            if self.pacing.get('continuous_paths'):route.append((target,place_r))
            else:self.straight(side,target,place_r)
        if route:self.pose_path(side,route)
        r=place_r
        high=self.cup_target(side,r,high_obj)
        low=self.cup_target(side,r,goal)
        self.straight(side,low,r,spacing=.02)
        self.carried=None;self.stack_carried=False
        self.event('release_stack',target='basket')
        self.grip(side,.095)
        self.straight(side,high,r)
        self.wait(1.5)
        metrics=self.session.read().extras['task']['metrics']
        for n in self.cups:
            self.results.append({'name':n,'target':'basket','grasp_mode':'physical',**metrics[n]})
        if not all(metrics[n]['success'] for n in self.cups) or not metrics['cup_stack']['nested']:
            raise SortFailure('Stack not fully nested and contained after release')
        self.event('stack_placed',success=True)
        if not self.pacing.get('direct_next_object'):
            self.pose(side,[.27,-.30,.30],grasp_rotation(30,0))

    def run(self,objects=None):
        try:
            if not self.prepared:self.prepare(None if objects==['cup_stack'] else objects)
            if objects is None or objects==['cup_stack']:
                self.stack_cups()
            if objects is None:
                for n in ('pen','charger','spoon','bowl','plate','side_dish','chopstick_a','chopstick_b'):
                    if n in self.b.task.active_names:self.sort_one(n)
            elif objects!=['cup_stack']:
                for n in objects:self.sort_one(n)
            if objects is None:
                self.wait_for_final_stability()
            else:
                self.wait(2.)
            summary=self.session.read().extras['task']['metrics']['summary']
            self.event('finished',**summary)
            if objects is None and not summary['success']:raise SortFailure('Rigid clearing final check failed')
            return summary
        finally:
            self.carried=None;self.stack_carried=False
