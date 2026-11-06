"""Engine-neutral, metre-scale tableware meshes with separate convex contacts.

Visual meshes retain open bowls and basket perforations. Collision pieces never
use a single convex hull across a container opening. Cloth is a rigid proxy.
"""
import numpy as np


def mesh(name, points, faces, material, *, collision=False, hidden=False, uv=None):
    result = dict(name=name, points=np.asarray(points, float).tolist(),
                  faces=np.asarray(faces, int).tolist(), material=material,
                  collision=collision, hidden=hidden)
    if uv is not None: result['uv'] = np.asarray(uv, float).tolist()
    return result


def box(name, size, center, material, **kwargs):
    # Independent face vertices preserve sharp normals and a UV seam at each edge.
    corners = np.array([[x,y,z] for z in (-1,1) for y in (-1,1) for x in (-1,1)])
    corners = corners*np.array(size)/2+center
    quads = [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
    points = np.concatenate([corners[list(q)] for q in quads])
    faces = [[4*i+a,4*i+b,4*i+c] for i in range(6) for a,b,c in ((0,1,2),(0,2,3))]
    result=mesh(name, points, faces, material, uv=np.tile([[0,0],[1,0],[1,1],[0,1]],(6,1)), **kwargs)
    result['box']={'size':list(size),'center':list(center)}
    return result


def merge(name, parts, material):
    vertices, faces, offset = [], [], 0
    for p in parts:
        vertices.extend(p['points']); faces.extend((np.array(p['faces'])+offset).tolist())
        offset += len(p['points'])
    return mesh(name, vertices, faces, material)


def transform(part, *, scale=(1,1,1), offset=(0,0,0)):
    part['points'] = (np.asarray(part['points'])*scale+offset).tolist()
    return part


def convex_profile_parts(profile):
    """Ear-clip a radial section, then merge adjacent triangles if still convex.

    A hull of an entire rim-to-floor section bridges the concave inside of a
    dish. Splitting only by azimuth does not fix that radial overfill.
    """
    p=np.asarray(profile,float)
    def cross(a,b,c):
        u=b-a;v=c-a
        return u[0]*v[1]-u[1]*v[0]
    def area(ids):
        a=p[ids];return abs(np.sum(a[:,0]*np.roll(a[:,1],-1)-a[:,1]*np.roll(a[:,0],-1)))/2
    remaining=list(range(len(p)));triangles=[]
    while len(remaining)>3:
        for k,b in enumerate(remaining):
            a,c=remaining[k-1],remaining[(k+1)%len(remaining)]
            if cross(p[a],p[b],p[c])<=1e-12:continue
            if any(all(cross(p[u],p[v],p[j])>=-1e-12 for u,v in ((a,b),(b,c),(c,a)))
                   for j in remaining if j not in (a,b,c)):continue
            triangles.append([a,b,c]);remaining.pop(k);break
        else:raise ValueError('Vessel profile must be a simple counterclockwise polygon')
    triangles.append(remaining)
    def hull(ids):
        ids=sorted(set(ids),key=lambda i:tuple(p[i]));lower=[];upper=[]
        for chain,order in ((lower,ids),(upper,ids[::-1])):
            for i in order:
                while len(chain)>1 and cross(p[chain[-2]],p[chain[-1]],p[i])<=1e-12:chain.pop()
                chain.append(i)
        return lower[:-1]+upper[:-1]
    while True:
        merged=False
        for i,a in enumerate(triangles):
            for j in range(i+1,len(triangles)):
                b=triangles[j]
                if len(set(a)&set(b))<2:continue
                h=hull(a+b)
                if abs(area(h)-area(a)-area(b))<1e-10:
                    triangles[i]=h;triangles.pop(j);merged=True;break
            if merged:break
        if not merged:return triangles


def vessel(name, profile, material='porcelain', *, scale=(1,1,1), exponent=2, segments=64):
    """Closed radial section, ordered from outer bottom through rim to inner floor.

    The profile uses normalized radii for both round and rounded-rectangle dishes.
    Convex radial pieces in each angular sector leave the cavity and rim open.
    """
    profile = np.asarray(profile)
    angle = np.arange(segments)*2*np.pi/segments
    xy = np.column_stack([np.sign(np.cos(angle))*np.abs(np.cos(angle))**(2/exponent),
                          np.sign(np.sin(angle))*np.abs(np.sin(angle))**(2/exponent)])
    points = np.concatenate([np.column_stack([xy*r, np.full(segments,z)]) for r,z in profile])*scale
    faces = []
    for i in range(len(profile)):
        ni = (i+1)%len(profile)
        for j in range(segments):
            nj = (j+1)%segments
            faces.extend([(i*segments+j,i*segments+nj,ni*segments+nj),
                          (i*segments+j,ni*segments+nj,ni*segments+j)])
    parts = [mesh(name,points,faces,material)]
    # Small angular sectors bound the convex approximation error at the inner wall.
    for radial,indices in enumerate(convex_profile_parts(profile)):
        for k, ids in enumerate(np.array_split(np.arange(segments),16)):
            ids = np.r_[ids, (ids[-1]+1)%segments]
            count = len(ids);rings=len(indices)
            p = points.reshape(len(profile),segments,3)[indices][:,ids,:].reshape(-1,3)
            f = []
            for i in range(rings):
                ni = (i+1)%rings
                for j in range(count-1):
                    f.extend([(i*count+j,i*count+j+1,ni*count+j+1),
                              (i*count+j,ni*count+j+1,ni*count+j)])
            for j in (0,count-1):
                f.extend([(j,i*count+j,(i+1)*count+j) for i in range(1,rings-1)])
            parts.append(mesh(f'{name}_contact_{radial}_{k}',p,f,material,collision=True,hidden=True))
    return parts


def beam(start,end,width,material):
    start,end=np.array(start),np.array(end)
    delta=end-start;length=np.linalg.norm(delta);axis=delta/length
    tangent=np.cross(axis,[0,0,1])
    if np.linalg.norm(tangent)<1e-6:tangent=np.cross(axis,[0,1,0])
    tangent/=np.linalg.norm(tangent);normal=np.cross(axis,tangent)
    p=box('rib',[length,width,width],[0,0,0],material)
    p['points']=(np.array(p['points'])@np.array([axis,tangent,normal])+(start+end)/2).tolist()
    return p


def container(size, wall, material):
    """Rounded, flared diamond lattice with handholds and an open contact cavity."""
    x,y,h=size
    # Arc-length resampling keeps perforation pitch constant around rounded corners.
    a=np.linspace(0,2*np.pi,1025)
    outline=np.column_stack([np.sign(np.cos(a))*abs(np.cos(a))**.25*x/2,
                             np.sign(np.sin(a))*abs(np.sin(a))**.25*y/2])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(outline,axis=0),axis=1))]
    cols=round(arc[-1]/(.011 if material=='basket' else .008))
    def point(u,z):
        s=(u/cols%1)*arc[-1]
        xy=np.array([np.interp(s,arc,outline[:,i]) for i in range(2)])
        return [*(xy*(.91+.09*z/h)),z]
    floor=vessel('bottom',[(.001,0),(1,0),(1,1),(.001,1)],material,
                 scale=[x*.455,y*.455,wall],exponent=8)[0]
    visual=[floor]
    rows=8
    zs=np.linspace(wall,h-.020,rows+1)
    for row in range(rows):
        for col in range(cols):
            u=col+(row%2)*.5
            for direction in (-.5,.5):
                visual.append(beam(point(u,zs[row]),point(u+direction,zs[row+1]),
                                   .0020 if material=='basket' else .0012,material))
    # Continuous rounded rolled lips at the floor and the top edge.
    for z,r in ((wall,.915),(h-.020,.98),(h-.004,1)):
        visual.append(vessel('rim',[(r-.018,z),(r,z),(r,z+.004),(r-.018,z+.004)],material,
                             scale=[x/2,y/2,1],exponent=8,segments=128)[0])
    for col in range(cols):
        p=point(col,h)
        # Broad handholds on both ends, with a narrow one on each long side.
        handhold=(abs(p[0])>x*.46 and abs(p[1])<y*.21) or (abs(p[1])>y*.46 and abs(p[0])<x*.15)
        if not handhold:
            vertices=[]
            for z in (h-.020,h-.004):
                a,b=np.array(point(col-.51,z)),np.array(point(col+.51,z))
                ai,bi=a.copy(),b.copy()
                ai[:2]*=1-wall/min(x,y)*2;bi[:2]*=1-wall/min(x,y)*2
                vertices.extend([a,b,bi,ai])
            faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
            for k in range(4):
                j=(k+1)%4;faces.extend([(k,j,j+4),(k,j+4,k+4)])
            visual.append(mesh('handle_band',vertices,faces,material))
    contacts=[box('floor_contact',[x,y,wall],[0,0,wall/2],material,collision=True,hidden=True)]
    for axis,span,cross in ((0,y,x),(1,x,y)):
        for sign in (-1,1):
            center=[0.,0.,h/2];center[axis]=sign*(cross-wall)/2
            s=[x,wall,h] if axis==1 else [wall,y,h]
            contacts.append(box(f'wall_{axis}_{sign+1}',s,center,material,collision=True,hidden=True))
    return [merge('perforated_shell',visual,material),*contacts]


def fabric(size, *, paper=False):
    x,y,h = size
    n = 25 if not paper else 19
    u,v = np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n))
    if paper:
        radius = np.maximum(abs(u),abs(v))
        # Broad intersecting creases and irregular soft edges, instead of a
        # high-frequency saw-toothed outline. Still a closed rigid paper proxy.
        px = x/2*u*(1+.06*np.sin(4*v+2*u))
        py = y/2*v*(1+.06*np.sin(5*u-v))
        z = (.40*(1-radius**2) + .09*abs(np.sin(5*u+3*v))
             + .08*abs(np.sin(6*v-2*u)) + .05*np.sin(4*u-5*v))
    else:
        px = x/2*u*(1+.035*np.sin(4*v))
        py = y/2*v*(1+.04*np.sin(5*u))
        z = .28*np.sin(4*u+2*v)+.22*np.sin(7*v-2*u)+.20*np.exp(-8*(u+.25)**2)
    z = .001+(h-.002)*(z-z.min())/(z.max()-z.min())
    top = np.column_stack([px.ravel(),py.ravel(),z.ravel()])
    points = np.concatenate([top,top-[0,0,.001]])
    faces=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;b=a+1;c=a+n;d=c+1
            faces.extend([(a,b,d),(a,d,c),(a+n*n,d+n*n,b+n*n),(a+n*n,c+n*n,d+n*n)])
    edge = list(range(n))+[j*n+n-1 for j in range(1,n)]+list(range(n*n-2,n*(n-1)-1,-1))+[j*n for j in range(n-2,0,-1)]
    for a,b in zip(edge,edge[1:]+edge[:1]):
        faces.extend([(a,b+n*n,b),(a,a+n*n,b+n*n)])
    uv = np.tile(np.column_stack([(u.ravel()+1)/2,(v.ravel()+1)/2]),(2,1))
    p=mesh('folds',points,faces,'paper' if paper else 'terry',uv=uv)
    # Contact geometry follows the folds, including their outermost points.
    # An inset box lets a turned-over scrap sink visually through the bin floor
    # and leaves no raised fold for a parallel gripper to pinch. These are still
    # rigid convex patches, not a cloth/deformation model.
    parts=[p]
    if paper:
        parts.append(mesh('fabric_contact',points,faces,p['material'],collision=True,hidden=True))
    else:
        for j in range(0,n-1,4):
            for i in range(0,n-1,4):
                ids=np.array([r*n+c for r in range(j,min(j+4,n-1)+1)
                              for c in range(i,min(i+4,n-1)+1)])
                ids=np.r_[ids,ids+n*n]
                remap={old:new for new,old in enumerate(ids)}
                f=[[remap[v] for v in face] for face in faces if all(v in remap for v in face)]
                parts.append(mesh(f'fold_contact_{j}_{i}',points[ids],f,p['material'],
                                  collision=True,hidden=True))
    if not paper:
        seams=[]
        for a,b in zip(edge,edge[1:]+edge[:1]):
            mid=(top[a]+top[b])/2; length=np.linalg.norm(top[b]-top[a])
            s=[length,.001,.001] if abs(top[b,0]-top[a,0])>abs(top[b,1]-top[a,1]) else [.001,length,.001]
            seams.append(box('stitch',s,mid+[0,0,.0003],'cotton_hem'))
        parts.append(merge('stitched_hem',seams,'cotton_hem'))
    return parts


def build_parts(spec):
    kind=spec['kind']
    if kind in ('cup','pen','charger'):
        from .rigid_table_meshes import build_prop
        return build_prop(spec)
    if kind in ('basket','trash_bin'):
        parts=container(spec['size_m'],spec['wall_m'],'basket' if kind=='basket' else 'bin')
        if kind=='trash_bin':
            # Existing scraps visible in the source bin: decorative, not task items.
            for i,(px,py) in enumerate(((-.025,.015),(.018,-.020))):
                p=fabric([.050,.055,.022],paper=True)[0]
                p['name']=f'existing_scrap_{i}';transform(p,offset=[px,py,.004]);parts.append(p)
        return parts
    if kind in ('towel','paper'): return fabric(spec['size_m'],paper=kind=='paper')
    if kind=='placemat':
        x,y,h=spec['size_m']
        p=box('woven_surface',[x,y,h],[0,0,h/2],'woven',collision=True)
        p['uv']=(np.asarray(p['uv'])*2).tolist()
        trim=[box('trim',[x,.004,.0006],[0,s*(y/2-.002),h+.0003],'mat_edge') for s in (-1,1)]
        trim += [box('trim',[.004,y,.0006],[s*(x/2-.002),0,h+.0003],'mat_edge') for s in (-1,1)]
        return [p,merge('bound_edge',trim,'mat_edge')]
    if kind in ('plate','bowl','side_dish'):
        if kind=='bowl':
            profile=[(.01,0),(.48,0),(.54,.06),(.58,.15),(.80,.54),(.98,.93),(1,.98),(.99,1),
                     (.94,1),(.91,.93),(.72,.51),(.48,.13),(.01,.13)]
            scale=[spec['diameter_m']/2]*2+[spec['height_m']]
        elif kind=='plate':
            profile=[(.01,0),(.55,0),(.58,.16),(.82,.56),(.99,.82),(1,.95),(.98,1),
                     (.86,.87),(.67,.45),(.53,.36),(.01,.36)]
            scale=[spec['diameter_m']/2]*2+[spec['height_m']]
        else:
            profile=[(.01,0),(.75,0),(.83,.15),(.98,.82),(1,.96),(.97,1),(.91,.87),(.73,.30),(.01,.30)]
            x,y,h=spec['size_m'];scale=[x/2,y/2,h]
        return vessel('glazed_shell',profile,scale=scale,exponent=8 if kind=='side_dish' else 2)
    if kind=='chopstick':
        x,y,z=spec['size_m']
        # Tapered square stick, dark lacquer with a gold band near the thick end.
        p=box('lacquer',[x,y,z],[0,0,z/2],'chopstick')
        points=np.asarray(p['points']);ratio=.38+.62*(x/2-points[:,0])/x
        points[:,1]*=ratio;points[:,2]=(points[:,2]-z/2)*ratio+z/2;p['points']=points.tolist()
        band=box('gold_band',[.016,y+.0002,z+.0002],[-x*.32,0,z/2],'gold')
        contact=box('stick_contact',[x,y,z],[0,0,z/2],'chopstick',collision=True,hidden=True)
        return [p,band,contact]
    if kind=='spoon':
        x,y,z=spec['size_m']
        bowl=vessel('spoon_cup',[(.01,0),(.5,.05),(.87,.45),(1,.85),(.98,1),(.88,.9),(.72,.5),(.40,.27),(.01,.25)],scale=[x*.21,y/2,z],segments=32)
        for p in bowl: transform(p,offset=[x*.29,0,0])
        # A thin porcelain handle with rounded perimeter, continuous with the cup.
        handle=vessel('spoon_handle',[(.01,0),(.93,0),(1,.35),(.98,.85),(.9,1),(.01,1)],
                      scale=[x*.33,y*.14,z*.27],exponent=4,segments=32)
        for p in handle: transform(p,offset=[-x*.17,0,z*.8])
        return bowl+handle
    raise ValueError(f'Unsupported clean-table object kind {kind!r}')
