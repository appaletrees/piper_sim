"""Opt-in room lighting/surface preset; never changes CAD sizes or camera poses."""
import xml.etree.ElementTree as ET


def configure_mujoco(root,task,prefix):
    c=task.config['appearance'];world=root.find('worldbody');visual=root.find('visual')
    visual.find('quality').set('shadowsize','8192')
    visual.find('map').set('shadowscale','.35')
    head=visual.find('headlight');head.attrib.update(ambient='.45 .45 .45',diffuse='.18 .18 .18',specular='.05 .05 .05')
    for light in list(world.findall('light')):world.remove(light)
    # Distributed ceiling illumination. Only the key casts a shadow; broad fill
    # reduces the excessive black silhouettes of the default workcell lighting.
    lights=[([.3,-.25,1.2],[0,.15,-1],[.28,.27,.25],True),
            ([.8,.9,1.6],[-.2,-.4,-1],[.28,.29,.31],False),
            ([-.8,0,1.2],[.8,0,-1],[.24,.24,.24],False)]
    directional_profile=c.get('lighting_profile')=='reference_lab_v2'
    if directional_profile:
        head.attrib.update(ambient='.30 .30 .30',diffuse='.14 .14 .14',specular='.03 .03 .03')
        lights=[([.2,-.65,1.62],[.05,.15,-1],[.34,.33,.31],True),
                ([.2,2.10,1.0],[.1,-1,-.5],[.27,.29,.32],False),
                ([-.8,-.6,1.2],[.8,.3,-1],[.16,.16,.16],False)]
    for i,(pos,direction,color,shadow) in enumerate(lights):
        ET.SubElement(world,'light',name=f'{prefix}/ceiling_{i}',pos=' '.join(map(str,pos)),
                      dir=' '.join(map(str,direction)),directional='false',
                      diffuse=' '.join(map(str,color)),specular='.12 .12 .12',
                      ambient=('.08 .08 .08' if i==0 else '.05 .05 .05') if directional_profile else
                              ('.15 .15 .15' if i==0 else '.04 .04 .04'),
                      attenuation='1 0 0',cutoff='60' if directional_profile and shadow else '85',castshadow=str(shadow).lower())
    # Native MuJoCo sums ambient/diffuse terms without the RTX tone mapping.
    # Keep this task-local so a white tableware scene need not clip its texture.
    scale=float(c.get('mujoco_light_scale',1.0))
    for element in [head,*world.findall('light')]:
        for attr in ('ambient','diffuse','specular'):
            element.set(attr,' '.join(str(float(v)*scale) for v in element.get(attr,'0 0 0').split()))
    ground=world.find("geom[@name='ground']")
    if ground is not None:
        ground.set('pos',f"0 0 {c['ground_height_m']}")
        ground.set('rgba','1 1 1 1');ground.set('material',prefix+'/floor')
    table=world.find("geom[@name='cad_static_0']")
    if table is not None:
        table.set('rgba','1 1 1 1');table.set('material',prefix+'/'+c['tabletop_material'])
    sky=root.find("asset/texture[@type='skybox']")
    if sky is not None:
        sky.set('rgb1','.69 .72 .74');sky.set('rgb2','.82 .83 .82')


def configure_isaac(stage,root,task):
    import numpy as np
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
    c=task.config['appearance']
    dome=UsdLux.DomeLight(stage.GetPrimAtPath('/World/Light'))
    if dome:
        dome.CreateIntensityAttr(c['dome_intensity']);dome.CreateColorAttr(Gf.Vec3f(1.,1.,1.))
    # RectLight local -Z points down for the default orientation.
    lights=[
        ([.1,-.6,1.8],[1.2,.5],c['rect_light_intensity'],[1.,.98,.95]),
        ([.8,.8,1.6],[.8,.8],c['rect_light_intensity']*.55,[.94,.97,1.])]
    directional_profile=c.get('lighting_profile')=='reference_lab_v2'
    if directional_profile:
        lights=[([.2,-.65,1.63],[1.2,.28],c['rect_light_intensity'],[1.,.97,.92]),
                ([.2,2.12,1.0],[1.5,1.0],c['rect_light_intensity']*.70,[.89,.95,1.])]
    for i,(position,size,intensity,color) in enumerate(lights):
        light=UsdLux.RectLight.Define(stage,root+f'/Lighting/Ceiling_{i}')
        light.CreateWidthAttr(size[0]);light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(intensity);light.CreateColorAttr(Gf.Vec3f(*color))
        xf=UsdGeom.Xformable(light);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        if directional_profile and i==1:
            xf.AddRotateXOp().Set(-63.435)
    ground=stage.GetPrimAtPath('/World/Ground')
    if ground:
        for op in UsdGeom.Xformable(ground).GetOrderedXformOps():
            if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0,0,c['ground_height_m']-.025))
        mat=UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/floor'))
        UsdShade.MaterialBindingAPI.Apply(ground).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    workcell=stage.GetPrimAtPath('/World/Workcell')
    if workcell:
        mat=UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/'+c['tabletop_material']))
        for prim in Usd.PrimRange(workcell):
            if prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith('part_000_'):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                axes=np.argsort(np.ptp(points,axis=0))[-2:]
                uv=(points[:,axes]-points[:,axes].min(0))/np.ptp(points[:,axes],axis=0)
                UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.vertex).Set(uv.tolist())
