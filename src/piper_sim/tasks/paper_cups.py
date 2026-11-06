"""Independent randomized paper-cup scene; no robot or engine imports."""
import copy
import json
import math

import numpy as np

from ..config import ROOT
from ..transforms import pose_matrix
from .paper_cup_meshes import build_cup


class PaperCupsTask:
    def __init__(self,config,workcell):
        self.config=copy.deepcopy(config);c=self.config
        if c.get('schema_version')!=1 or c.get('name')!='paper_cups':
            raise ValueError('Unsupported paper-cup scene')
        s=c['cup'];count=s['count']
        if type(count) is not int or not 1<=count<=12:
            raise ValueError('Cup count must be an integer in [1,12]')
        keys=('height_m','top_diameter_m','bottom_diameter_m','wall_m','rim_radius_m','floor_height_m','mass_kg')
        if any(not math.isfinite(s[k]) or s[k]<=0 for k in keys):
            raise ValueError('Cup dimensions and mass must be finite and positive')
        if not (.008<s['bottom_diameter_m']<s['top_diameter_m']<.2 and
                .02<s['height_m']<.3 and s['wall_m']<s['bottom_diameter_m']/8 and
                .001<s['floor_height_m']<s['height_m']/4 and
                s['rim_radius_m']<min(s['height_m'],s['top_diameter_m'])/10):
            raise ValueError('Invalid cup taper, wall, rim or recessed bottom')
        self.names=tuple(f'cup_{i:02d}' for i in range(count))
        self.specs={n:copy.deepcopy(s) for n in self.names}
        template=build_cup(s)
        self.parts={n:template for n in self.names}  # Immutable, identical SI geometry
        points=np.concatenate([p['points'] for p in template if not p['hidden']])
        self.bounds={n:np.array([points.min(0),points.max(0)]) for n in self.names}
        self.radius=s['top_diameter_m']/2
        table=workcell['scene']['table']
        self.table_top=table['position'][2]+table['size'][2]/2
        self.table_bounds=np.array(table['position'][:2])+np.array([[-1],[1]])*np.array(table['size'][:2])/2
        self.material_path=ROOT/c['materials']
        self.materials=json.loads(self.material_path.read_text())['materials']
        for m in self.materials.values():
            if 'texture' in m and not (self.material_path.parent/m['texture']).is_file():
                raise FileNotFoundError(self.material_path.parent/m['texture'])
        r=c['randomization'];self.center_bounds=np.asarray(r['center_bounds_xy_m'],float)
        if self.center_bounds.shape!=(2,2) or not np.isfinite(self.center_bounds).all() or np.any(self.center_bounds[0]>=self.center_bounds[1]):
            raise ValueError('Invalid placement bounds')
        yaw=np.asarray(r['yaw_degrees'],float)
        if yaw.shape!=(2,) or not np.isfinite(yaw).all() or yaw[0]>yaw[1]:raise ValueError('Invalid yaw range')
        if any(not math.isfinite(r[k]) or r[k]<0 for k in ('min_gap_m','table_clearance_m')):
            raise ValueError('Invalid cup/table clearance')
        if type(r['max_attempts']) is not int or r['max_attempts']<1:raise ValueError('Invalid sampling attempts')
        # Never sample over the robot bases, mast, or beyond the table edge.
        clearance=self.radius+r['table_clearance_m']
        if (np.any(self.center_bounds[0]<self.table_bounds[0]+clearance) or
                np.any(self.center_bounds[1]>self.table_bounds[1]-clearance) or self.center_bounds[0,0]<.16+self.radius):
            raise ValueError('Cup region must fit the table and stay clear of robot bases/mast')
        if not math.isfinite(c['contact']['friction']) or c['contact']['friction']<0:raise ValueError('Invalid friction')
        self.reset()

    def reset(self,seed=None,randomize=None):
        seed=self.config['seed'] if seed is None else seed
        if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or seed<0:raise ValueError('Seed must be a nonnegative integer')
        r=self.config['randomization'];randomize=r['enabled'] if randomize is None else bool(randomize)
        rng=np.random.default_rng(seed);xy=[]
        gap=2*self.radius+r['min_gap_m']
        if randomize:
            for _ in range(r['max_attempts']):
                candidate=rng.uniform(*self.center_bounds)
                if all(np.linalg.norm(candidate-p)>=gap for p in xy):xy.append(candidate)
                if len(xy)==len(self.names):break
        else:
            # Centred lattice for any configured count, independent of the seed.
            shape=np.floor(np.diff(self.center_bounds,axis=0)[0]/gap).astype(int)+1
            choices=[(nx,ny) for nx in range(1,shape[0]+1) for ny in range(1,shape[1]+1) if nx*ny>=len(self.names)]
            if choices:
                nx,ny=min(choices,key=lambda a:(a[0]*a[1],abs(a[0]-a[1])))
                center=self.center_bounds.mean(0)
                xy=[center+[(i-(nx-1)/2)*gap,(j-(ny-1)/2)*gap] for i in range(nx) for j in range(ny)][:len(self.names)]
        if len(xy)!=len(self.names):raise ValueError('Cannot place all cups without overlap; reduce count/gap or enlarge region')
        poses={}
        for n,p in zip(self.names,xy):
            yaw=math.radians(float(rng.uniform(*r['yaw_degrees'])) if randomize else 180.)/2
            poses[n]={'position':[*map(float,p),self.table_top+.0003],
                      'quaternion_wxyz':[math.cos(yaw),0.,0.,math.sin(yaw)]}
        self.layout={'seed':int(seed),'randomized':randomize,'poses':poses}
        return copy.deepcopy(self.layout)

    def metrics(self,poses,dt=0.):
        result={}
        for n in self.names:
            p=poses[n];t=pose_matrix(p['position'],p['quaternion_wxyz'])
            tilt=math.degrees(math.acos(np.clip(t[2,2],-1,1)))
            xy=np.array(p['position'][:2]);z=p['position'][2]
            on_table=bool(np.all(xy>=self.table_bounds[0]+self.radius) and np.all(xy<=self.table_bounds[1]-self.radius) and abs(z-self.table_top)<.004)
            result[n]={'upright':tilt<12.,'tilt_degrees':tilt,'on_table':on_table,
                       'lift_height_m':float(z-self.table_top)}
        # No target task was requested, so no invented success criterion.
        result['summary']={'cup_count':len(self.names),'upright_count':sum(v['upright'] for v in result.values()),
                           'on_table_count':sum(v['on_table'] for v in result.values()),'success':False,
                           'success_defined':False}
        return result

    def metadata(self):
        return {'name':self.config['name'],'instruction':self.config['instruction'],
                'config':copy.deepcopy(self.config),'layout':copy.deepcopy(self.layout),
                'bounds_m':{n:b.tolist() for n,b in self.bounds.items()},'materials':copy.deepcopy(self.materials),
                'geometry_authority':'CAD workcell unchanged; paper-cup dimensions estimated from user photographs',
                'approximations':['rigid empty paper cups; no crushing, liquid or deformation',
                                 'sector contacts approximate thin curved paper walls and rolled rims',
                                 'mass, friction, appearance and room dimensions estimated',
                                 'generated print texture follows supplied photographs; not scanned albedo']}
