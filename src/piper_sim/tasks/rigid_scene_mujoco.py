"""Task-neutral composition of textured rigid objects and static surroundings."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def nums(values):
    return ' '.join(f'{float(v):.10g}' for v in np.asarray(values).reshape(-1))


def compose_model(path,task):
    path=Path(path);root=ET.parse(path).getroot()
    if task.config.get('variant')=='rigid_stack_v1':
        size=root.find('size')
        if size is None: size=ET.SubElement(root,'size')
        size.attrib.pop('njmax',None);size.attrib.pop('nconmax',None)
        size.set('memory','128M')
    compiler=root.find('compiler')
    compiler.set('meshdir',str((path.parent/compiler.get('meshdir','.')).resolve()))
    asset=root.find('asset');world=root.find('worldbody')
    old=world.find("body[@name='object']")
    if old is not None: world.remove(old)
    keys=root.find('keyframe')
    if keys is not None: root.remove(keys)
    # A task-local softer lighting preset preserves white porcelain detail.
    visual=root.find('visual')
    headlight=visual.find('headlight')
    if headlight is None: headlight=ET.SubElement(visual,'headlight')
    headlight.attrib.update(ambient='.15 .15 .15',diffuse='.20 .20 .20',specular='.10 .10 .10')
    quality=visual.find('quality')
    if quality is None: quality=ET.SubElement(visual,'quality')
    quality.set('shadowsize','4096')
    for i,light in enumerate(world.findall('light')):
        light.set('diffuse','.60 .60 .60' if i==0 else '.20 .20 .20')
        light.set('ambient','.08 .08 .08' if i==0 else '0 0 0')
    prefix='clean' if task.config['name']=='clean_table' else task.config['name']
    for name,m in task.materials.items():
        attrs={'name':prefix+'/'+name,'rgba':nums([*m['color'],1]),
               'specular':str(m.get('mujoco_specular',.06+.50*(1-m['roughness']))),
               'shininess':str(m.get('mujoco_shininess',1-m['roughness'])), 'reflectance':str(m['metallic']*.15)}
        if 'texture' in m:
            ET.SubElement(asset,'texture',name=prefix+'/'+name,type='2d',file=str(task.material_path.parent/m['texture']),
                          vflip=str(m.get('mujoco_vflip',False)).lower())
            attrs.update(texture=prefix+'/'+name,texuniform='false',texrepeat='1 1')
            if name in ('white_table','table_gray','table_wood'):
                attrs.update(texuniform='true',texrepeat='1 1')
        ET.SubElement(asset,'material',**attrs)
    for n,s in task.config.get('environment',{}).items():
        ET.SubElement(world,'geom',name=prefix+'_environment/'+n,type='box',
                      size=nums(np.array(s['size_m'])/2),pos=nums(s['position_m']),
                      material=prefix+'/'+s['material'])
    if task.config.get('appearance',{}).get('preset')=='indoor_lab':
        from .indoor_lab import configure_mujoco
        configure_mujoco(root,task,prefix)
        # Original CAD OBJ has no UVs. Add planar coordinates without changing
        # any vertex, face, dimension or pose; otherwise 2D albedo is invisible.
        table_mesh=asset.find("mesh[@name='cad_static_0']")
        if table_mesh is not None and table_mesh.get('file'):
            lines=(Path(compiler.get('meshdir'))/table_mesh.get('file')).read_text().splitlines()
            vertices=np.array([[float(v) for v in line.split()[1:4]] for line in lines if line.startswith('v ')])
            faces=[[int(v.split('/')[0])-1 for v in line.split()[1:4]] for line in lines if line.startswith('f ')]
            axes=np.argsort(np.ptp(vertices,axis=0))[-2:]
            uv=(vertices[:,axes]-vertices[:,axes].min(0))/np.ptp(vertices[:,axes],axis=0)
            table_mesh.attrib.pop('file')
            table_mesh.attrib.update(vertex=nums(vertices),face=nums(faces),texcoord=nums(uv))
    mesh_names={}
    for n,parts in task.parts.items():
        pose=task.layout['poses'][n]
        body=ET.SubElement(world,'body',name='task/'+n,pos=nums(pose['position']),quat=nums(pose['quaternion_wxyz']))
        ET.SubElement(body,'freejoint',name='task/'+n+'/free')
        lo,hi=task.bounds[n];size=np.maximum(hi-lo,.001);mass=task.specs[n]['mass_kg']
        inertia=mass/12*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
        ET.SubElement(body,'inertial',pos=nums((lo+hi)/2),mass=str(mass),diaginertia=nums(inertia))
        for p in parts:
            full=f'task/{n}/{p["name"]}'
            attrs=dict(name=full,vertex=nums(p['points']),face=nums(p['faces']),inertia='shell')
            if 'uv' in p: attrs['texcoord']=nums(p['uv'])
            # Identical cups share one immutable geometry asset, not a body.
            if id(p) not in mesh_names:
                ET.SubElement(asset,'mesh',**attrs)
                mesh_names[id(p)]=full
            attrs=dict(name=full,type='mesh',mesh=mesh_names[id(p)],density='0',material=prefix+'/'+p['material'],
                       contype='1' if p['collision'] else '0',conaffinity='1' if p['collision'] else '0',
                       group='3' if p['hidden'] else '1',
                       friction=f'{task.config["contact"]["friction"]} .003 .0001',
                       priority=str(task.config['contact'].get('mujoco_priority',0)),
                       solref='.006 1',solimp='.95 .99 .001',margin='0')
            if p['hidden']: attrs['rgba']='0 0 0 0'
            # Analytic boxes give stable multi-point contact for thin broad mats
            # and basket floors; a triangulated flat hull yields one soft contact.
            if p['collision'] and 'box' in p:
                attrs.pop('mesh')
                attrs.update(type='box',size=nums(np.array(p['box']['size'])/2),pos=nums(p['box']['center']))
            if p['collision'] and 'cylinder' in p:
                attrs.pop('mesh')
                c=p['cylinder']
                attrs.update(type='cylinder',size=nums([c['radius'],c['height']/2]),pos=nums(c['center']))
            ET.SubElement(body,'geom',**attrs)
    return ET.tostring(root,encoding='unicode')


def reset_task(model,data,task):
    for n,p in task.layout['poses'].items():
        j=model.joint(f'task/{n}/free');q=j.qposadr[0];v=j.dofadr[0]
        data.qpos[q:q+7]=[*p['position'],*p['quaternion_wxyz']];data.qvel[v:v+6]=0
        # Slots are allocated once; only active inventory participates in an
        # episode. Restore exactly from source parts on every reset.
        active=n in getattr(task,'active_names',task.names)
        bid=model.body('task/'+n).id
        model.body_gravcomp[bid]=0 if active else 1
        for part in task.parts[n]:
            gid=model.geom(f'task/{n}/{part["name"]}').id
            model.geom_contype[gid]=model.geom_conaffinity[gid]=int(active and part['collision'])
            model.geom_rgba[gid,3]=float(active and not part['hidden'])


def task_poses(data,task):
    return {n:{'position':data.body('task/'+n).xpos.tolist(),
               'quaternion_wxyz':data.body('task/'+n).xquat.tolist()}
            for n in getattr(task,'active_names',task.names)}
