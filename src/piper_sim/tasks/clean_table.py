"""Independent table-clearing scene, reproducible placement and sorting metrics."""
import copy
import json
import math

import numpy as np

from ..config import ROOT
from ..transforms import pose_matrix
from .clean_table_meshes import build_parts


def yaw_quaternion(degrees):
    angle=math.radians(degrees)/2
    return [math.cos(angle),0.,0.,math.sin(angle)]


class CleanTableTask:
    def __init__(self, config, workcell):
        self.config=copy.deepcopy(config)
        c=self.config
        if c.get('schema_version')!=1 or c.get('name')!='clean_table':
            raise ValueError('Unsupported clean-table configuration')
        self.specs=c['objects'];self.names=tuple(self.specs)
        for name,spec in self.specs.items():
            dimensions=spec.get('size_m',[spec.get('diameter_m',0),spec.get('height_m',0)])
            if not all(np.isfinite(dimensions)) or min(dimensions)<=0 or spec['mass_kg']<=0:
                raise ValueError(f'Invalid dimensions/mass for {name}')
            if spec.get('target') and spec['target'] not in self.specs:
                raise ValueError(f'Unknown target for {name}')
            if spec.get('support') and spec['support'] not in self.specs:
                raise ValueError(f'Unknown support for {name}')
        if not set(c['success']['required']).issubset(self.names):
            raise ValueError('Unknown required sorting object')
        self.parts={n:build_parts(s) for n,s in self.specs.items()}
        self.bounds={};self.surface_points={}
        for n,parts in self.parts.items():
            p=np.concatenate([p['points'] for p in parts if not p.get('hidden')])
            self.bounds[n]=np.array([p.min(0),p.max(0)])
            self.surface_points[n]=np.unique(p,axis=0)
        self.material_path=ROOT/c['materials']
        materials=json.loads(self.material_path.read_text())
        self.materials=materials['materials']
        self.domain_profile=None
        if c.get('domain_randomization'):
            from ..randomization import load_profile
            self.domain_profile=load_profile(c['domain_randomization'])
        for spec in self.materials.values():
            if 'texture' in spec and not (self.material_path.parent/spec['texture']).is_file():
                raise FileNotFoundError('Run scripts/prepare_clean_table_textures.py')
        table=workcell['scene']['table']
        self.table_top=table['position'][2]+table['size'][2]/2
        self.table_bounds=np.array(table['position'][:2])+np.array([[-1],[1]])*np.array(table['size'][:2])/2
        self.original_table_bounds=self.table_bounds.copy()
        # The video also shows a contiguous rear tabletop. It may support the
        # baskets without resizing/repositioning the authoritative workcell CAD.
        for s in c.get('environment',{}).values():
            if not s.get('supports_placement'): continue
            size=np.array(s['size_m']);pos=np.array(s['position_m'])
            bounds=pos[:2]+np.array([[-1],[1]])*size[:2]/2
            if (abs(pos[2]+size[2]/2-self.table_top)>1e-6
                    or abs(bounds[0,0]-self.table_bounds[1,0])>1e-6
                    or bounds[0,1]>self.table_bounds[0,1]+1e-6
                    or bounds[1,1]<self.table_bounds[1,1]-1e-6):
                raise ValueError('Placement extension must adjoin the rear edge at the same height and cover its width')
            self.table_bounds[1,0]=bounds[1,0]
        self.containers=tuple(n for n,s in self.specs.items() if s['kind'] in ('basket','trash_bin'))
        self.hold={};self.layout=None
        self.reset()

    def corners(self,name,pose):
        lo,hi=self.bounds[name]
        points=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        t=pose_matrix(pose['position'],pose['quaternion_wxyz'])
        return points@t[:3,:3].T+t[:3,3]

    def _valid_layout(self,poses):
        clearance=self.config['randomization']['table_clearance_m']
        bounds={n:self.corners(n,p)[:,:2] for n,p in poses.items()}
        for p in bounds.values():
            if np.any(p.min(0)<self.table_bounds[0]+clearance) or np.any(p.max(0)>self.table_bounds[1]-clearance):
                return False
            if p[:,0].min()<.15: return False  # keep clear of original bases/mast
        # Intended place-setting stacks may overlap. Containers must not intersect
        # the setting or each other, including the wider visible top lips.
        for i,a in enumerate(self.containers):
            for b in (*self.containers[i+1:],*(n for n in self.names if n not in self.containers)):
                lo_a,hi_a=bounds[a].min(0),bounds[a].max(0)
                lo_b,hi_b=bounds[b].min(0),bounds[b].max(0)
                if np.all(hi_a+clearance>lo_b) and np.all(hi_b+clearance>lo_a): return False
        return True

    def reset(self,seed=None,randomize=None):
        seed=self.config['seed'] if seed is None else int(seed)
        if seed<0: raise ValueError('Seed must be nonnegative')
        r=copy.deepcopy(self.config['randomization'])
        if self.domain_profile:
            r.update(self.domain_profile.get('layout',{}))
        randomize=r['enabled'] if randomize is None else bool(randomize)
        rng=np.random.default_rng(seed)
        origin=np.array(self.specs['placemat']['position_xy_m'])
        heights={}
        def height(n,trail=()):
            if n in trail: raise ValueError('Cyclic object supports')
            s=self.specs[n]
            if n not in heights:
                heights[n]=height(s['support'],(*trail,n))+s['support_height_m']+.0005 if s.get('support') else self.table_top+.0005
            return heights[n]
        for n in self.names: height(n)
        for _ in range(500):
            poses={}
            delta=rng.uniform(-np.array(r['setting_xy_m']),r['setting_xy_m']) if randomize else np.zeros(2)
            yaw=rng.uniform(-r['setting_yaw_deg'],r['setting_yaw_deg']) if randomize else 0.
            a=math.radians(yaw);rotation=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
            for n,s in self.specs.items():
                xy=np.array(s['position_xy_m']);angle=s.get('yaw_deg',0)
                if n in self.containers:
                    if randomize:
                        xy+=rng.uniform(-np.array(r['container_xy_m']),r['container_xy_m'])
                        angle+=rng.uniform(-r['container_yaw_deg'],r['container_yaw_deg'])
                else:
                    xy=origin+rotation@(xy-origin)+delta;angle+=yaw
                poses[n]={'position':[*xy,heights[n]],'quaternion_wxyz':yaw_quaternion(angle)}
            if self._valid_layout(poses): break
        else: raise ValueError('Cannot fit clean-table objects on the supporting tables; check dimensions/ranges')
        self.layout={'seed':seed,'randomized':randomize,'poses':poses}
        self.domain_parameters=None
        if self.domain_profile and randomize:
            from ..randomization import sample
            self.domain_parameters=sample(self.domain_profile,seed,self.names,self.materials)
        self.layout['domain_randomization']=copy.deepcopy(self.domain_parameters)
        self.hold={n:0. for n in self.names if self.specs[n].get('target')}
        return copy.deepcopy(self.layout)

    def metrics(self,poses,dt=0.):
        """Full visible-mesh containment in a moving bin frame, with dwell time.

        A centre over a bin, a wrong bin, or an object still above the rim is not
        success. This geometric metric does not assert an autonomous policy.
        """
        result={};tol=self.config['success']['containment_tolerance_m']
        for n in self.hold:
            target=self.specs[n]['target'];s=self.specs[target]
            t=pose_matrix(poses[target]['position'],poses[target]['quaternion_wxyz'])
            obj=pose_matrix(poses[n]['position'],poses[n]['quaternion_wxyz'])
            world=self.surface_points[n]@obj[:3,:3].T+obj[:3,3]
            points=(world-t[:3,3])@t[:3,:3]
            size=np.array(s['size_m']);wall=s['wall_m']
            inside=bool(np.all(abs(points[:,:2])<=size[:2]/2-wall+tol)
                        and points[:,2].min()>=wall-tol and points[:,2].max()<=size[2]+tol)
            self.hold[n]=self.hold[n]+max(0.,dt) if inside else 0.
            result[n]={'target':target,'contained':inside,'hold_seconds':self.hold[n],
                       'success':inside and self.hold[n]>=self.config['success']['hold_seconds']}
        required=self.config['success']['required']
        result['summary']={'success':all(result[n]['success'] for n in required),
                           'sorted_count':sum(result[n]['success'] for n in required),
                           'required_count':len(required)}
        return result

    def metadata(self):
        return {'name':self.config['name'],'instruction':self.config['instruction'],
                'config':copy.deepcopy(self.config),'layout':copy.deepcopy(self.layout),
                'bounds_m':{n:b.tolist() for n,b in self.bounds.items()},
                'placement_bounds_xy_m':self.table_bounds.tolist(),
                'original_table_bounds_xy_m':self.original_table_bounds.tolist(),
                'materials':copy.deepcopy(self.materials),
                'domain_profile':copy.deepcopy(self.domain_profile),
                'geometry_authority':'original workcell CAD; task object sizes estimated from video',
                'approximations':['rigid towel, crumpled paper and placemat; no cloth deformation',
                    'basket perforations are visual; collisions use separate solid walls and floor',
                    'mass, friction and reflectance estimated; not identified hardware parameters',
                    'sorting metrics exclude folding the rigid placemat into the basket']}
