"""Photo-inspired rigid props, in metres; shared by both simulation engines."""
import numpy as np
from .clean_table_meshes import box, mesh, vessel
from .paper_cup_meshes import build_cup, lathe


def build_prop(spec):
    kind=spec['kind']
    if kind=='cup':
        parts=build_cup(spec)
        for p in parts: p['material']='cup_'+p['material']
        # Separate an untextured, recolourable paper shell from its printed
        # badge. Tinting the original navy bitmap hides palette changes.
        shell=next(p for p in parts if p['name']=='printed_wall')
        shell['material']='cup_body'
        h=spec['height_m']-spec['rim_radius_m'];rb=spec['bottom_diameter_m']/2
        rt=spec['top_diameter_m']/2-spec['rim_radius_m']
        u=np.linspace(.396,.604,33);v=np.array([1-.615,1-.431])
        points=[];uv=[]
        for vv in v:
            z=.006+vv*(h-.006);radius=rb+(rt-rb)*z/h+.00008
            for uu in u:
                a=-np.pi+2*np.pi*uu
                points.append([radius*np.cos(a),radius*np.sin(a),z]);uv.append([uu,vv])
        faces=[]
        for i in range(len(u)-1):faces.extend([(i,i+1,len(u)+i+1),(i,len(u)+i+1,len(u)+i)])
        parts.append(mesh('printed_badge',points,faces,'cup_print',uv=uv))
        parts.append(lathe('bottom_print_line',[(rb+(rt-rb)*.015/h+.00006,.015),
                                               (rb+(rt-rb)*.01525/h+.00006,.01525)],'cup_paper'))
        return parts
    if kind=='charger':
        x,y,z=spec['size_m']
        shell=vessel('rounded_shell',[(.01,0),(.88,0),(.98,.12),(1,.28),(1,.72),(.98,.90),(.88,1),(.01,1)],
                     'charger_white',scale=[x/2,y/2,z],exponent=10)[0]
        # Folded-pin travel configuration from the supplied photograph.
        parts=[shell,box('body_contact',[x*.96,y*.96,z],[0,0,z/2],'charger_white',collision=True,hidden=True),
               box('folding_plug_panel',[x*.37,y*.27,.001],[-x*.25,y*.30,z*.97],'charger_panel'),
               box('usb_c_recess',[.002,.009,.003],[x*.5,0,z*.5],'rubber'),
               box('case_seam',[x*.98,.0005,z*.83],[0,-y*.35,z*.5],'charger_panel')]
        # Surface wear is geometry/material only, not baked lighting.
        for i in range(4):
            parts.append(box(f'worn_label_{i}',[.004,.003,.00012],[.003+i*.004,-.003-i*.002,z*.996],'label_wear'))
        return parts
    if kind=='pen':
        length,diameter=spec['size_m'][:2];r=diameter/2
        parts=[]
        # The long axis is local X, resting on the black rubber grip.
        def segment(name,profile,material):
            p=lathe(name,profile,material,closed=True,segments=48)
            v=np.asarray(p['points']);p['points']=np.column_stack([v[:,2]-length/2,v[:,0],v[:,1]+r]).tolist()
            parts.append(p)
        segment('metal_tip',[(0.,0),(.003,.018),(.003,.023),(0.,.023)],'pen_metal')
        segment('rubber_grip',[(0.,.018),(r,.018),(r,.065),(0.,.065)],'rubber')
        segment('clear_barrel',[(0.,.065),(r*.82,.065),(r*.82,.112),(0.,.112)],'pen_barrel')
        segment('black_cap',[(0.,.112),(r,.112),(r,.142),(r*.7,.146),(0.,.146)],'rubber')
        segment('button',[(0.,.142),(r*.5,.142),(r*.5,length),(0.,length)],'pen_metal')
        parts.append(box('pocket_clip',[.031,.002,.002],[.047,0,diameter+.001],'rubber'))
        parts.append(box('barrel_label',[.019,.0002,.006],[.026,-r*.83,r],'pen_label'))
        for i in range(16):
            parts.append(box(f'barcode_{i}',[.0003+(i%3)*.00012,.00015,.004],[.017+i*.001,-r*.85,r],'rubber'))
        # Conservative finite-width contact, wholly within the visible body.
        parts.append(box('grip_contact',[length-.023,diameter*.85,diameter],
                         [.008,0,r],'rubber',collision=True,hidden=True))
        return parts
    raise ValueError(kind)
