"""SI paper-cup geometry: UV print, rolled lip, recessed bottom, open contacts."""
import numpy as np


def mesh(name, points, faces, material, **extras):
    return dict(name=name, points=np.asarray(points, float).tolist(),
                faces=np.asarray(faces, int).tolist(), material=material,
                collision=False, hidden=False, **extras)


def lathe(name, profile, material, *, closed=False, segments=128, uv=False):
    """Revolve a radial profile; duplicate the seam for an unmirrored UV wrap."""
    p=np.asarray(profile,float); angles=np.linspace(-np.pi,np.pi,segments+1)
    points=np.concatenate([np.c_[r*np.cos(angles),r*np.sin(angles),np.full(len(angles),z)] for r,z in p])
    n=segments+1; faces=[]
    for j in range(len(p) if closed else len(p)-1):
        k=(j+1)%len(p)
        for i in range(segments):
            faces.extend([(j*n+i,j*n+i+1,k*n+i+1),(j*n+i,k*n+i+1,k*n+i)])
    result=mesh(name,points,faces,material)
    if uv:
        # PNG top is v=1; bottom is v=0 in USD/MuJoCo texture coordinates.
        # Label centre u=.5 faces local +X.
        result['uv']=[[u,(z-p[:,1].min())/np.ptp(p[:,1])] for _,z in p for u in np.linspace(0,1,n)]
    return result


def sector(name, profile, a, b):
    p=np.asarray(profile)
    points=np.concatenate([np.c_[p[:,0]*np.cos(t),p[:,0]*np.sin(t),p[:,1]] for t in (a,b)])
    n=len(profile); faces=[]
    for j in range(1,n-1):faces.extend([(0,j+1,j),(n,n+j,n+j+1)])
    for j in range(n):
        k=(j+1)%n;faces.extend([(j,k,n+k),(j,n+k,n+j)])
    result=mesh(name,points,faces,'paper');result.update(collision=True,hidden=True)
    return result


def build_cup(spec):
    h=spec['height_m'];rb=spec['bottom_diameter_m']/2
    lip=spec['rim_radius_m'];rt=spec['top_diameter_m']/2-lip
    wall=spec['wall_m'];floor=spec['floor_height_m']
    side_h=h-lip
    def radius(z):return rb+(rt-rb)*z/side_h
    parts=[
        lathe('printed_wall',[(radius(.006),.006),(rt,side_h)],'print',uv=True),
        lathe('white_foot',[(rb,0),(radius(.006),.006)],'paper'),
        # Reverse profile to face into the cavity.
        lathe('inner_wall',[(rt-wall,side_h),(radius(floor)-wall,floor)],'lining'),
        lathe('inner_floor',[(radius(floor)-wall,floor),(0.,floor)],'lining'),
        lathe('underside',[(0.,floor-.001),(radius(floor)-wall,floor-.001)],'paper'),
        lathe('recess_wall',[(radius(floor)-wall,floor-.001),(rb-wall,0),(rb,0)],'paper'),
    ]
    theta=np.linspace(0,2*np.pi,17)[:-1]
    parts.append(lathe('rolled_lip',np.c_[rt+lip*np.cos(theta),side_h+lip*np.sin(theta)],'rim',closed=True))
    # Small visible glued side seam on the back, with no baked-in shadow.
    seam=lathe('glued_seam',[(rb+.00008,0),(rt+.00008,side_h)],'paper',segments=128)
    # A narrow angular strip at the UV join instead of a second entire shell.
    pts=np.asarray(seam['points']).reshape(2,129,3)[:, :2].reshape(-1,3)
    parts.append(mesh('glued_seam',pts,[(0,1,3),(0,3,2)],'seam'))
    # Separate convex sectors do not span the mouth. The lip has its own small
    # convex contacts; it is never treated as a filled disc.
    count=32
    angles=np.linspace(0,2*np.pi,count+1)
    body_profile=[(rb-wall,0),(rb,0),(rt,side_h),(rt-wall,side_h)]
    rim_profile=np.c_[rt+lip*np.cos(theta),side_h+lip*np.sin(theta)]
    for i,(a,b) in enumerate(zip(angles,angles[1:])):
        parts.append(sector(f'wall_contact_{i}',body_profile,a,b))
        parts.append(sector(f'rim_contact_{i}',rim_profile,a,b))
    bottom=lathe('floor_contact',[(0.,floor-.001),(radius(floor)-wall,floor-.001),
                                 (radius(floor)-wall,floor),(0.,floor)],'paper',closed=True,segments=32)
    bottom.update(collision=True,hidden=True,cylinder={'radius':radius(floor)-wall,'height':.001,
                                                       'center':[0.,0.,floor-.0005]})
    parts.append(bottom)
    return parts
