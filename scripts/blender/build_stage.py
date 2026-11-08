#!/usr/bin/env python3
"""Blender-authored lab context and articulated hardware surface details.

Photo-inspired appearance, not a survey of the room. The original CAD and robot
meshes remain authoritative. New parts are visual only; all attachment transforms
come from the project's reference pose. Run prepare_inputs.py first.
"""
import hashlib
import json
import math
from pathlib import Path
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'assets/blender/lab_stage'
PREVIEW = ROOT/'outputs/lab_visual_quality'
OUT.mkdir(parents=True, exist_ok=True); PREVIEW.mkdir(parents=True, exist_ok=True)
source = json.loads((ROOT/'outputs/blender_clean_table/inputs.json').read_text())
rig = json.loads((ROOT/'assets/generated/workcell/camera_rig.json').read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'; scene.unit_settings.scale_length = 1.
materials = {}; blmats = {}; authored = []; parent = 'world'


def linear(rgb):
    a = np.array(rgb); return np.where(a <= .04045, a/12.92, ((a+.055)/1.055)**2.4)


def material(name, color, roughness=.5, metallic=0., family=None):
    name = 'lab_'+name
    spec = dict(color=color, roughness=roughness, metallic=metallic)
    mat = bpy.data.materials.new(name); mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value=(*linear(color),1)
    bsdf.inputs['Roughness'].default_value=roughness; bsdf.inputs['Metallic'].default_value=metallic
    if family:
        for channel,field in [('albedo','detail_texture'),('roughness','roughness_texture'),('normal','normal_texture')]:
            file='../clean_table/textures/'+family+'_'+channel+'.png'; spec[field]=file
            image=bpy.data.images.load(str(OUT/file),check_existing=True)
            image.colorspace_settings.name='sRGB' if channel=='albedo' else 'Non-Color'
            tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image
            if channel=='albedo':
                mix=mat.node_tree.nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY'
                mix.inputs[0].default_value=1;mix.inputs[2].default_value=(*linear(color),1)
                mat.node_tree.links.new(tex.outputs['Color'],mix.inputs[1])
                mat.node_tree.links.new(mix.outputs[0],bsdf.inputs['Base Color'])
            elif channel=='normal':
                n=mat.node_tree.nodes.new('ShaderNodeNormalMap');n.inputs['Strength'].default_value=.35
                mat.node_tree.links.new(tex.outputs['Color'],n.inputs['Color'])
                mat.node_tree.links.new(n.outputs[0],bsdf.inputs['Normal']);spec['normal_strength']=.35
            else:
                m=mat.node_tree.nodes.new('ShaderNodeMath');m.operation='MULTIPLY';m.inputs[1].default_value=2*roughness
                mat.node_tree.links.new(tex.outputs['Color'],m.inputs[0]);mat.node_tree.links.new(m.outputs[0],bsdf.inputs['Roughness'])
    materials[name]=spec;blmats[name]=mat
    return name


black=material('anodized_black',[.055,.062,.066],.36,.25,'metal')
rubber=material('cable',[.027,.03,.032],.7,0,'rubber')
silver=material('brushed_aluminum',[.64,.66,.68],.31,.72,'metal')
white=material('warm_paint',[.82,.82,.79],.62,0,'plastic')
ivory=material('cabinet',[.71,.72,.70],.42,0,'plastic')
dark=material('cabinet_gap',[.14,.155,.16],.65)
blue=material('blue_partition',[.25,.49,.59],.27,.10)
frost=material('frosted_band',[.53,.68,.71],.65)
glass=material('optical_cover',[.026,.033,.042],.12,.15)
lens=material('lens_coating',[.055,.105,.16],.10,.32)
ir=material('ir_projector',[.095,.14,.065],.16,.28)
label=material('label_ink',[.85,.87,.87],.48)
red=material('red_ink',[.61,.055,.06],.4)
pot=material('ceramic_pot',[.73,.72,.67],.26,0,'porcelain')
soil=material('soil',[.15,.115,.08],.95)
leaf=[material('leaf_'+str(i),c,.48) for i,c in enumerate([[.18,.30,.09],[.25,.39,.13],[.12,.24,.065],[.33,.43,.15]])]
stem=material('stem',[.25,.30,.10],.66)
screen=material('screen',[.10,.14,.16],.2)
tile=[material('floor_'+str(i),[.64+v,.65+v,.64+v],.57,0,'table') for i,v in enumerate([-.008,0,.008,.012])]
grout=material('grout',[.47,.48,.46],.83)
appearance=json.loads((ROOT/'configs/appearance.json').read_text())
cad_materials={}
for key,s in appearance['materials'].items():
    # Existing USD appearance colors are linear, while the portable task catalog
    # is sRGB. Native MuJoCo previously received those dark linear values directly.
    c=np.array(s['diffuse_color']);c=np.where(c<=.0031308,c*12.92,1.055*c**(1/2.4)-.055)
    cad_materials[key]=material('cad_'+key,c.tolist(),s['roughness'],s['metallic'])
material_bindings={};usd_bindings={}
part_materials={i:k for k,ids in appearance['cad_parts'].items() for i in ids}
for p in source['environment']:
    if p['name'].startswith(('left/','right/')) and '/mesh' in p['name']:
        link=p['name'].split('/')[1]
        if link in appearance['robot_links']:
            key=cad_materials[appearance['robot_links'][link]]
            material_bindings[p['name']]=key
            usd_bindings['/World/'+'/'.join(p['name'].split('/')[:2])+'/visuals']=key
    elif p['name'].startswith('cad_') and p['name']!='cad_static_0':
        index=int(p['name'].split('_')[-1]);key=cad_materials[part_materials.get(index,appearance['remaining_cad_parts'])]
        material_bindings[p['name']]=key


def select(obj):
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj


def finish(obj,name,mat,bevel=0):
    obj.name=name;obj['attachment']=parent;obj['material_key']=mat
    obj.data.materials.append(blmats[mat]);select(obj)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if bevel:
        mod=obj.modifiers.new('Manufactured edge','BEVEL');mod.width=bevel;mod.segments=3
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-9)
    bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-10)
    bm.normal_update();bm.to_mesh(obj.data);bm.free();obj.data.update()
    for p in obj.data.polygons:p.use_smooth=True
    if obj.type=='MESH':
        mod=obj.modifiers.new('Panel normals','WEIGHTED_NORMAL');mod.keep_sharp=True;mod.weight=50
        bpy.ops.object.modifier_apply(modifier=mod.name)
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(island_margin=.02);bpy.ops.object.mode_set(mode='OBJECT')
    authored.append(obj);return obj


def box(name,position,size,mat,bevel=.001):
    bpy.ops.mesh.primitive_cube_add(size=1,location=position)
    o=bpy.context.object;o.scale=size;return finish(o,name,mat,bevel)


def cylinder(name,position,radius,depth,mat,rotation=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=radius,depth=depth,location=position)
    o=bpy.context.object
    if rotation:o.rotation_euler=rotation
    return finish(o,name,mat,.0002 if radius<.01 else .001)


def tube(name,points,radius,mat):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.resolution_u=16
    spline=curve.splines.new('BEZIER');spline.bezier_points.add(len(points)-1)
    for bp,co in zip(spline.bezier_points,points):
        bp.co=co;bp.handle_left_type='AUTO';bp.handle_right_type='AUTO'
    curve.bevel_depth=radius;curve.bevel_resolution=3
    o=bpy.data.objects.new(name,curve);scene.collection.objects.link(o);select(o)
    bpy.ops.object.convert(target='MESH');return finish(bpy.context.object,name,mat)


def text(name,body,position,width,mat,rotation=(0,0,0)):
    curve=bpy.data.curves.new(name,'FONT');curve.body=body;curve.align_x='CENTER';curve.align_y='CENTER'
    curve.size=.01;curve.extrude=.000015;curve.resolution_u=5
    o=bpy.data.objects.new(name,curve);scene.collection.objects.link(o);o.location=position
    select(o);bpy.context.view_layer.update();factor=width/max(o.dimensions.x,1e-5)
    o.scale=(factor,)*3;o.rotation_euler=rotation;bpy.ops.object.convert(target='MESH')
    return finish(bpy.context.object,name,mat)


# Room context beyond the task table. The cabinet/counter surfaces are not
# placement targets: this visual package does not create physical obstacles.
box('room_return_wall',[-2.3,0,.51],[.04,4.5,2.5],white,.001)
box('room_opposite_wall',[0,-2.25,.51],[4.6,.04,2.5],white,.001)
box('room_return_skirt',[-2.27,0,-.65],[.025,4.46,.12],ivory)
box('room_opposite_skirt',[0,-2.22,-.65],[4.56,.025,.12],ivory)
for y in [-1.8,-.9,0,.9,1.8]:
    box('wall_panel_seam_'+str(y),[-2.277,y,.55],[.001,.003,2.3],ivory,.0001)
box('floor_grout',[.2,0,-.744],[5,4.5,.004],grout,0)
for i in range(8):
    for j in range(7):
        box(f'floor_tile_{i}_{j}',[-2.2+i*.62,-2.1+j*.62,-.7408],[.617,.617,.002],tile[(i*3+j)%4],.0005)
for y in [-1.44,-.72,0,.72,1.44]:
    box(f'partition_glass_{y}',[2.345,y,.67],[.014,.704,1.88],blue,.001)
    box(f'partition_frost_{y}',[2.336,y,.42],[.001,.704,.095],frost,.0001)
    box(f'partition_frost_upper_{y}',[2.336,y,.57],[.001,.704,.025],frost,.0001)
for y in np.arange(-1.8,1.81,.72):
    box('partition_mullion_'+str(y),[2.325,float(y),.67],[.035,.026,1.94],silver,.0015)
for z in [-.30,1.64]:box('partition_rail_'+str(z),[2.325,0,z],[.035,3.62,.03],silver,.0015)
box('wall_skirt',[2.34,0,-.65],[.035,4.46,.12],ivory)
box('side_skirt',[0,2.215,-.65],[4.6,.035,.12],ivory)
# Window and structural reveals give the fill light a visible source.
window=material('window_daylight',[.76,.83,.85],.15,.06)
for x in [-.8,.1,1.0]:
    box('window_glass_'+str(x),[x,2.21,.76],[.87,.012,1.3],window)
    for h in np.linspace(.25,1.3,12):
        box(f'blind_{x}_{h}',[x,2.18,float(h)],[.87,.012,.022],ivory,.0005)
for x in [-1.25,-.35,.55,1.45]:box('window_frame_'+str(x),[x,2.165,.76],[.035,.05,1.36],white)
for z in [.08,1.44]:box('window_sill_'+str(z),[.1,2.15,z],[2.74,.10,.035],white)
# Counter behind the workbench; understated panels, recessed doors and handles.
box('cabinet_body',[2.03,0,-.365],[.5,3.12,.7],ivory,.006)
box('counter_top',[1.99,0,.003],[.62,3.22,.035],white,.006)
box('cabinet_toe_kick',[1.80,0,-.68],[.05,3.10,.08],dark)
for i in range(6):
    y=-1.30+i*.52
    box('cabinet_door_'+str(i),[1.769,y,-.33],[.02,.509,.57],white,.003)
    tube('cabinet_handle_'+str(i),[[1.752,y+.18,-.27],[1.72,y+.18,-.27],
         [1.72,y+.18,-.12],[1.752,y+.18,-.12]],.005,silver)
# Laptop with hinge, inset keyboard and dark, non-luminous screen.
box('laptop_base',[1.94,-.8,.035],[.27,.34,.016],silver,.007)
box('keyboard',[1.91,-.8,.044],[.15,.29,.001],black,.003)
for i in range(5):
    for j in range(12):box(f'key_{i}_{j}',[1.85+i*.021,-.93+j*.023,.045],[.016,.017,.0015],dark,.001)
lid=box('laptop_lid',[2.06,-.8,.155],[.015,.34,.24],black,.006)
box('laptop_display',[2.051,-.8,.155],[.001,.314,.205],screen,.003)
for i in range(5):box('display_line_'+str(i),[2.050,-.83+i*.025,.17],[.0003,.006,.09],dark,.0001)
# Plant: individual curved blades and stems, avoiding identical cone foliage.
cylinder('plant_pot',[1.94,.93,.12],.10,.20,pot)
cylinder('soil',[1.94,.93,.222],.091,.005,soil)
rng=np.random.default_rng(81)
for i in range(27):
    angle=i*2.39996;start=np.array([1.94,.93,.22]);height=float(rng.uniform(.17,.47))
    end=start+np.array([math.cos(angle)*.18,math.sin(angle)*.18,height])
    tube('plant_stem_'+str(i),[start,start+[0,0,height*.6],end],.0018,stem)
    direction=np.array([math.cos(angle),math.sin(angle),.2]);cross=np.array([-math.sin(angle),math.cos(angle),0])
    vertices=[];faces=[]
    length=float(rng.uniform(.12,.20));width=float(rng.uniform(.028,.045))
    for j,t in enumerate(np.linspace(0,1,18)):
        center=end+direction*length*t+np.array([0,0,.06*math.sin(t*math.pi)-.06*t*t])
        for k in np.linspace(-1,1,7):vertices.append(center+cross*k*width*math.sin(math.pi*t)+[0,0,-k*k*.007*math.sin(math.pi*t)])
        if j:
            for k in range(6):a=(j-1)*7+k;faces.append((a,a+1,a+8,a+7))
    me=bpy.data.meshes.new('leaf');me.from_pydata(vertices,[],faces);me.update()
    o=bpy.data.objects.new('leaf',me);scene.collection.objects.link(o);finish(o,'plant_leaf_'+str(i),leaf[i%4])
# Surface-mounted electrical fittings; no invented sockets on the task table.
for y in [-1.2,.25]:
    box('wall_outlet_'+str(y),[2.31,y,.16],[.014,.085,.085],white,.006)
    for dy in [-.014,.014]:box(f'outlet_hole_{y}_{dy}',[2.302,y+dy,.17],[.001,.005,.016],dark,.0005)
# Slim ceiling fixture bodies; actual engine light definitions remain separate.
for y in [-.65,.85]:
    box('ceiling_fixture_'+str(y),[.3,y,1.67],[1.3,.32,.028],white,.005)
    box('ceiling_diffuser_'+str(y),[.3,y,1.653],[1.26,.28,.002],label,.001)

# Camera front covers use the already extracted CAD front frame and apertures.
# Housing and optical transforms are not moved; thickness is visual only.
for camera_name,c in rig['cameras'].items():
    parent='world' if camera_name=='overhead' else camera_name.split('_')[0]+'/link6'
    front=Matrix(c['front_reference_world_pose'])
    if parent!='world':
        # Use the immutable CAD local optical frame, not the slightly deflected
        # body pose from the settled preview snapshot.
        rotation=Quaternion(c['rotation_wxyz_ros']).to_matrix()
        front=rotation.to_4x4()
        front.translation=Vector(c['position'])-rotation@Vector(np.array(c['rgb_from_front_center_optical_mm'])*.001)
    start=len(authored)
    width=.115 if camera_name=='overhead' else .082
    cover=box(camera_name+'_front_cover',[0,0,.00018],[width,.020,.0004],glass,.009)
    # Bevel clamps to thickness: use a 2-D capsule outline instead.
    bpy.data.objects.remove(cover,do_unlink=True);authored.pop()
    v=[]
    for center,startangle in [(width/2-.009,-math.pi/2),(-width/2+.009,math.pi/2)]:
        for a in np.linspace(startangle,startangle+math.pi,25):v.append([center+.009*math.cos(a),.009*math.sin(a),.00025])
    me=bpy.data.meshes.new('optical_cover');me.from_pydata(v,[],[list(range(len(v)))]);me.update()
    o=bpy.data.objects.new('optical_cover',me);scene.collection.objects.link(o)
    finish(o,camera_name+'_front_cover',glass)
    # D435i RGB is at -32.5 mm; the stereo pair is -17.5/+32.5 mm
    # (50 mm baseline). D455 stereo pair is -47.5/+47.5 mm (95 mm).
    positions=[-.0475,.0115,.0475] if camera_name=='overhead' else [-.0325,-.0175,.0325]
    for i,x in enumerate(positions):
        cylinder(camera_name+'_aperture_'+str(i),[x,0,.0005],.0058,.0003,black)
        cylinder(camera_name+'_glass_'+str(i),[x,0,.00070],.0048,.00015,lens)
        cylinder(camera_name+'_pupil_'+str(i),[x,0,.00081],.0027,.00010,glass)
    cylinder(camera_name+'_projector',[-.0115 if camera_name=='overhead' else .006,0,.00055],.0042,.0004,ir)
    for o in authored[start:]:o.matrix_world=front@o.matrix_world

# Printed link panels aligned to the flat faces of the original URDF meshes.
# Geometric text remains crisp in both engines and introduces no external fonts.
for side in ['left','right']:
    for link,center,size,angle,caption in [('link2',[.147,-.0295,0],[.13,.019,.00015],-.18,'AGILE  X'),
                                          ('link3',[-.028,-.104,0],[.063,.014,.00015],-math.pi/2,'PiPER')]:
        parent=side+'/'+link
        for sign in [-1,1]:
            z=sign*(.0207 if link=='link2' else .02015)
            o=box(side+'_'+link+'_panel_'+str(sign),[center[0],center[1],z],size,black,.00006)
            o.rotation_euler[2]=angle
            text(side+'_'+link+'_wordmark_'+str(sign),caption,[center[0],center[1],z+sign*.00017],
                 size[0]*.72,label,(0 if sign==1 else math.pi,0,angle+(math.pi if link=='link2' else 0)))
    # Short fixed cable runs within a link. No cable is stretched across joints.
    parent=side+'/link6'
    tube(side+'_wrist_cable',[[-.077,-.075,.03],[-.10,-.085,.006],[-.098,-.045,-.016],[-.04,-.035,-.008]],.0022,rubber)
    parent=side+'/link2'
    tube(side+'_link_cable',[[.025,.012,.029],[.08,.009,.029],[.17,-.005,.029],[.233,-.006,.047]],.0025,rubber)
    for x in [.07,.20]:
        tube(side+'_cable_tie_'+str(x),[[x,-.05,.021],[x,-.01,.025],[x,.003,0],[x,-.01,-.025],[x,-.05,-.021]],.001,black)

# Consolidate by parent/material, keeping contact-free geometry small at runtime.
parts=[];arrays={};groups={}
for o in authored:groups.setdefault((o['attachment'],o['material_key']),[]).append(o)
for index,((attachment,mat),objects) in enumerate(groups.items()):
    points=[];normals=[];uv=[]
    for o in objects:
        me=o.data;me.calc_loop_triangles();matrix=o.matrix_world
        loops=[loop for t in me.loop_triangles for loop in t.loops]
        rot=matrix.to_3x3().inverted().transposed()
        for loop in loops:
            points.append(matrix@me.vertices[me.loops[loop].vertex_index].co)
            normals.append((rot@me.corner_normals[loop].vector).normalized())
            uv.append(me.uv_layers.active.data[loop].uv[:])
    packed=np.concatenate([points,normals,uv],axis=1).astype(np.float32)
    unique,inverse=np.unique(packed,axis=0,return_inverse=True);key=f'part_{index:03d}'
    for field,data in [('points',unique[:,:3]),('normals',unique[:,3:6]),('uv',unique[:,6:]),('faces',inverse.reshape(-1,3))]:arrays[key+'_'+field]=data
    parts.append(dict(key=key,name=key+'_'+mat,parent=attachment,material=mat))
np.savez_compressed(OUT/'meshes.npz',**arrays)
source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT/'assets/mujoco/workcell.xml',ROOT/'assets/mujoco/meshes/cad_static_124.obj',
                        ROOT/'assets/mujoco/meshes/cad_left_103.obj',ROOT/'assets/mujoco/meshes/cad_right_47.obj',
                        ROOT/'assets/mujoco/meshes/link2.STL',ROOT/'assets/mujoco/meshes/link3.STL',
                        ROOT/'configs/appearance.json']}
manifest=dict(schema_version=1,units='metres',generator='Blender '+bpy.app.version_string,
    mesh_archive='meshes.npz',parts=parts,materials=materials,hide_environment=['blue_panel'],source_files=source_files,
    mujoco_material_bindings=material_bindings,usd_material_bindings=usd_bindings,
    status='Photo-inspired estimated room and surface details; original workcell dimensions and camera calibration unchanged',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for img in bpy.data.images:
    if img.source=='FILE':img.pack()
for attachment in {o['attachment'] for o in authored}:
    col=bpy.data.collections.new(attachment.replace('/','_'));scene.collection.children.link(col)
    for o in authored:
        if o['attachment']==attachment:
            for old in list(o.users_collection):old.objects.unlink(o)
            col.objects.link(o)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'stage.blend'),compress=True)
# Place link-local pieces into the reference pose for a full editable scene.
for o in authored:
    if o['attachment']!='world':
        b=source['bodies'][o['attachment']];tf=Matrix(b['rotation']).to_4x4();tf.translation=b['position'];o.matrix_world=tf@o.matrix_world
with bpy.data.libraries.load(str(ROOT/'outputs/blender_clean_table/scene.blend'),link=False) as (data,loaded):
    loaded.objects=[name for name in data.objects if name.startswith(('CAD_','Environment_','Scene_','Floor'))]
for o in loaded.objects:
    if o:
        scene.collection.objects.link(o)
        if 'blue_panel' in o.name:o.hide_render=True
        if o.name=='Floor':o.hide_render=True
        if o.name.startswith('CAD_') and o.name[4:] in material_bindings:
            o.data.materials.clear();o.data.materials.append(blmats[material_bindings[o.name[4:]]])
scene.world=bpy.data.worlds.new('Lab ambient');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.16
scene.render.engine='CYCLES';scene.cycles.samples=64;scene.cycles.use_denoising=True
prefs=bpy.context.preferences.addons['cycles'].preferences
try:
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    for d in prefs.devices:d.use=d.type!='CPU'
    if any(d.use for d in prefs.devices):scene.cycles.device='GPU'
except (RuntimeError,TypeError):pass
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.3
for name,pos,power,size,color in [('Ceiling',[.2,-.6,1.62],95,1.2,[1.,.97,.92]),('Window',[.2,2.1,1.0],160,1.5,[.89,.95,1.])]:
    light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size;light.color=color
    o=bpy.data.objects.new(name,light);scene.collection.objects.link(o);o.location=pos
    o.rotation_euler=(Vector([.4,0,.05])-o.location).to_track_quat('-Z','Y').to_euler()
def camera(name,eye,target,lens):
    cam=bpy.data.cameras.new(name);o=bpy.data.objects.new(name,cam);scene.collection.objects.link(o)
    o.location=eye;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler();cam.lens=lens;cam.clip_start=.005
    return o
scene.camera=camera('Lab_overview',[1.35,-1.65,1.05],[.35,.08,.20],38)
scene.render.resolution_x=1440;scene.render.resolution_y=1080;scene.render.resolution_percentage=100
for area in bpy.context.screen.areas:
    if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
bpy.ops.wm.save_as_mainfile(filepath=str(PREVIEW/'scene.blend'),compress=True)
for cam,name in [(scene.camera,'blender_overview'),(camera('Lab_context',[-.6,-.85,.9],[1.2,.1,.3],28),'blender_context')]:
    scene.camera=cam;scene.render.filepath=str(PREVIEW/(name+'.png'));bpy.ops.render.render(write_still=True)
print('LAB_STAGE_COMPLETE',len(parts),sum(len(v) for k,v in arrays.items() if k.endswith('_faces')),'triangles',flush=True)
