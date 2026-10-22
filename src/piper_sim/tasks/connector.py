"""Robot/engine-independent connector geometry, seeded layouts and insertion metrics."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT
from ..transforms import pose_matrix


def quat_z(angle):
    return [math.cos(angle/2), 0., 0., math.sin(angle/2)]


def prism(xy, low, high):
    xy = np.asarray(xy, dtype=float)
    # Counterclockwise polygon produces an outward-facing closed convex mesh.
    if np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)) < 0:
        xy = xy[::-1]
    n = len(xy)
    vertices = np.concatenate([np.column_stack([xy, np.full(n,z)]) for z in (low,high)])
    faces = [(0,i+1,i) for i in range(1,n-1)]+[(n,n+i,n+i+1) for i in range(1,n-1)]
    for i in range(n):
        j=(i+1)%n
        faces.extend([(i,j,n+j),(i,n+j,n+i)])
    return vertices, np.asarray(faces)


def rounded_rectangle(width, depth, radius):
    if not 0 < radius < min(width,depth)/2:
        raise ValueError('Invalid housing corner radius')
    points=[]
    for (x,y), start in zip(((1,1),(-1,1),(-1,-1),(1,-1)), (0,90,180,270)):
        for a in np.deg2rad(np.linspace(start,start+90,7)):
            points.append([x*(width/2-radius)+radius*np.cos(a), y*(depth/2-radius)+radius*np.sin(a)])
    return points


def part(name, xy, zmin, zmax, color, *, position=(0,0), yaw=0):
    points, faces=prism(xy,zmin,zmax)
    c,s=math.cos(yaw),math.sin(yaw)
    points[:,:2]=points[:,:2]@np.array([[c,s],[-s,c]])+position
    return {'name':name,'points':points.tolist(),'faces':faces.tolist(),'color':list(color)}


def box_part(name,size,position,color,yaw=0):
    x,y,z=np.array(size)/2
    return part(name,[[-x,-y],[x,-y],[x,y],[-x,y]],position[2]-z,position[2]+z,color,
                position=position[:2],yaw=yaw)


def plug_parts(name, cfg):
    p=cfg['plugs'][name]
    if name=='black_charger':
        w,d,h=p['housing_size_m']
        result=[part('housing',rounded_rectangle(w,d,p['corner_radius_m']),0,h,[.022,.025,.032,1])]
        # Slightly inset end plate follows the photographed black housing.
        result.append(part('face',rounded_rectangle(w-.001,d-.001,p['corner_radius_m']),-.0002,0,[.055,.06,.07,1]))
        length=p['pin_size_m'][2]
        for i,x in enumerate((-.5*p['pin_spacing_m'],.5*p['pin_spacing_m'])):
            result.append(box_part(f'pin_{i}',p['pin_size_m'],[x,0,-length/2],[.72,.72,.67,1]))
        return result
    if name!='yellow_plug':
        raise ValueError(f'Unknown connector {name}')
    angle=np.linspace(0,2*np.pi,48,endpoint=False)
    circle=np.column_stack([np.cos(angle),np.sin(angle)])
    result=[part('rim',circle*p['face_radius_m'],0,p['rim_height_m'],[.96,.77,.015,1]),
            part('face',circle*(p['face_radius_m']-.002),-.0002,0,[.84,.83,.77,1]),
            part('handle',circle*p['handle_radius_m'],p['rim_height_m'],p['housing_height_m'],[.96,.77,.015,1])]
    for i,x in enumerate((-.5*p['pin_spacing_m'],.5*p['pin_spacing_m'])):
        size=[p['pin_thickness_m'],p['pin_width_m'],p['pin_length_m']]
        result.append(box_part(f'pin_{i}',size,[x,0,-size[2]/2],[.72,.72,.67,1],
                               math.radians(p['angled_pin_deg'])*(1 if i==0 else -1)))
    length=p['pin_length_m']+p['earth_extra_length_m']
    result.append(box_part('pin_earth',[p['pin_thickness_m'],p['pin_width_m'],length],
                           [0,p['earth_offset_m'],-length/2],[.72,.72,.67,1]))
    return result


class ConnectorTask:
    """Owns only task assets/layout/metrics; independent of robot and simulator."""
    def __init__(self, config, workcell):
        if isinstance(config,(str,Path)):
            path=Path(config)
            if not path.is_absolute(): path=ROOT/path
            self.config=yaml.safe_load(path.read_text())
            self.config_path=str(path.resolve())
        else:
            self.config=copy.deepcopy(config)
            self.config_path=None
        c=self.config
        if c.get('schema_version')!=1 or c.get('name')!='charger_socket':
            raise ValueError('Unsupported task configuration')
        self.names=tuple(c['objects'])
        if not self.names or len(set(self.names))!=len(self.names) or any(n not in ('black_charger','yellow_plug') for n in self.names):
            raise ValueError('objects must contain unique supported connectors')
        self.asset_path=ROOT/c['asset_manifest']
        self.assets=json.loads(self.asset_path.read_text())
        source=ROOT/self.assets['source']
        if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()!=self.assets['source_sha256']:
            raise ValueError('Source board changed; rerun scripts/prepare_connector.py')
        for name,expected in self.assets['mesh_sha256'].items():
            if hashlib.sha256((self.asset_path.parent/'meshes'/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Connector asset changed: {name}; rerun prepare_connector.py')
        table=workcell['scene']['table']
        self.table_top=float(table['position'][2]+table['size'][2]/2)
        self.table_xy_bounds=np.array(table['position'][:2])+np.array([[-1], [1]])*np.array(table['size'][:2])/2
        self.parts={name:plug_parts(name,c) for name in self.names}
        self.hold={name:0. for name in self.names}
        self.layout=None
        self.reset(c['seed'])

    def _valid_layout(self,poses):
        clearance=self.config['randomization']['clearance_m']
        footprints=[]
        for name,pose in poses.items():
            if name=='board':
                lo,hi=np.array(self.assets['bounds_m'])
                corners=np.array([[x,y,0] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
            else:
                points=np.concatenate([p['points'] for p in self.parts[name]])
                lo,hi=points.min(0),points.max(0)
                corners=np.array([[x,y,0] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
            t=pose_matrix(pose['position'],pose['quaternion_wxyz'])
            xy=(corners@t[:3,:3].T+t[:3,3])[:,:2]
            lo,hi=xy.min(0),xy.max(0)
            if (lo<self.table_xy_bounds[0]+clearance).any() or (hi>self.table_xy_bounds[1]-clearance).any():
                return False
            # The task footprint stays in front of the robot bases/mast.
            if lo[0]<.16: return False
            for a,b in footprints:
                if np.all(hi+clearance>a) and np.all(b+clearance>lo): return False
            footprints.append((lo,hi))
        return True

    def reset(self,seed=None,randomize=None):
        seed=self.config['seed'] if seed is None else int(seed)
        if seed<0: raise ValueError('Seed must be nonnegative')
        rng=np.random.default_rng(seed)
        randomize=self.config['randomization']['enabled'] if randomize is None else bool(randomize)
        r=self.config['randomization']; nominal=self.config['nominal']
        for _ in range(1000):
            poses={}
            for name in ('board',*self.names):
                xy=rng.uniform(*np.array(r[name+'_xy']).T) if randomize else np.array(nominal[name+'_xy'])
                angle=math.radians(rng.uniform(*r['board_yaw_deg' if name=='board' else 'plug_yaw_deg']) if randomize
                                   else nominal['board_yaw_deg' if name=='board' else 'plug_yaw_deg'])
                if name=='board': z=self.table_top; q=quat_z(angle)
                else:
                    points=np.concatenate([p['points'] for p in self.parts[name]])
                    z=self.table_top+points[:,2].max()+.0005
                    # Start pins-up like the photographs; users/policies must turn the part over.
                    q=[0,math.cos(angle/2),math.sin(angle/2),0]
                poses[name]={'position':[*xy,float(z)],'quaternion_wxyz':q}
            if self._valid_layout(poses): break
        else: raise ValueError('No separated on-table layout found; widen ranges or reduce clearance')
        self.layout={'seed':seed,'randomized':randomize,'poses':poses}
        self.hold={name:0. for name in self.names}
        return copy.deepcopy(self.layout)

    def target_pose(self,name,board_pose=None):
        b=board_pose or self.layout['poses']['board']
        t=pose_matrix(b['position'],b['quaternion_wxyz'])
        xy=self.assets['targets'][name]['center_xy_m']
        local=np.array([*xy,self.assets['top_z_m']+.0002])
        return {'position':(t[:3,:3]@local+t[:3,3]).tolist(),'quaternion_wxyz':b['quaternion_wxyz'][:]}

    def metrics(self,poses,dt=0):
        """Geometric seating test; never welds/snaps a plug or claims electrical contact."""
        board=pose_matrix(poses['board']['position'],poses['board']['quaternion_wxyz'])
        result={}
        limits=self.config['success']
        for name in self.names:
            plug=pose_matrix(poses[name]['position'],poses[name]['quaternion_wxyz'])
            relative=np.linalg.inv(board)@plug
            target=np.array(self.assets['targets'][name]['center_xy_m'])
            lateral=float(np.linalg.norm(relative[:2,3]-target))
            tilt=float(np.arccos(np.clip(relative[2,2],-1,1)))
            # Symmetric 2-pin plug also accepts a 180-degree turn; 3-pin does not.
            yaw=math.atan2(relative[1,0],relative[0,0])
            yaw_error=abs(math.atan2(math.sin(yaw),math.cos(yaw)))
            if name=='black_charger': yaw_error=min(yaw_error,abs(math.pi-yaw_error))
            gap=float(relative[2,3]-self.assets['top_z_m']-.0002)
            holes=[self.assets['holes'][i] for i in self.assets['targets'][name]['hole_ids']]
            pin_fits=[]
            for p in self.parts[name]:
                if not p['name'].startswith('pin_'): continue
                points=np.asarray(p['points'])@relative[:3,:3].T+relative[:3,3]
                # Every pin must fit a different original rectangular opening.
                candidates=[]
                for h in holes:
                    lo,hi=np.array(h['bounds_xy_m'])
                    if np.all(points[:,:2]>=lo-1e-5) and np.all(points[:,:2]<=hi+1e-5):
                        candidates.append(h['id'])
                pin_fits.append(candidates[0] if len(candidates)==1 else None)
            fits=all(p is not None for p in pin_fits) and len(set(pin_fits))==len(pin_fits)
            near=(lateral<=limits['lateral_tolerance_m'] and tilt<=math.radians(limits['tilt_tolerance_deg'])
                  and yaw_error<=math.radians(limits['tilt_tolerance_deg']) and -.0006<=gap<=limits['seated_gap_m'] and fits)
            self.hold[name]=self.hold[name]+dt if near else 0.
            result[name]={'lateral_error_m':lateral,'tilt_error_rad':tilt,'yaw_error_rad':yaw_error,
                          'face_gap_m':gap,'seated':bool(near),'hold_seconds':self.hold[name],
                          'all_pins_in_matching_holes':bool(fits),
                          'success':bool(near and self.hold[name]>=limits['hold_seconds'])}
        return result

    def metadata(self):
        return {'name':self.config['name'],'config':self.config,'layout':self.layout,
                'board_source_sha256':self.assets['source_sha256'],
                'object_pose_frame':'world; metres; quaternion wxyz',
                'dimension_status':{'board':'source CAD, mm to m', 'plugs':'photo estimates, configurable'},
                'targets':{n:self.target_pose(n) for n in self.names}}
