"""Compose task XML with an unchanged workcell MJCF; no asset conversion at runtime."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def nums(values):
    return ' '.join(f'{float(v):.12g}' for v in np.asarray(values).reshape(-1))


def compose_model(path,task):
    path=Path(path)
    root=ET.parse(path).getroot()
    # All base references remain relative to their original bundle, not a cache.
    compiler=root.find('compiler')
    compiler.set('meshdir',str((path.parent/compiler.get('meshdir','.')).resolve()))
    asset=root.find('asset'); world=root.find('worldbody')
    old=world.find("body[@name='object']")
    if old is not None: world.remove(old)
    keys=root.find('keyframe')
    if keys is not None: root.remove(keys)  # task adds free bodies; reset handles qpos explicitly
    color=[.43,.44,.46,1]
    pose=task.layout['poses']['board']
    board=ET.SubElement(world,'body',name='task/board',pos=nums(pose['position']),quat=nums(pose['quaternion_wxyz']))
    ET.SubElement(asset,'mesh',name='task/board_visual',file=str(task.asset_path.parent/task.assets['visual_mesh']),inertia='shell')
    ET.SubElement(board,'geom',name='task/board_visual',type='mesh',mesh='task/board_visual',
                  rgba=nums(color),contype='0',conaffinity='0',group='1')
    friction=task.config['contact']['friction']
    for index,piece in enumerate(task.assets['collision_parts']):
        name=f'task/board_collision_{index}'
        ET.SubElement(asset,'mesh',name=name,vertex=nums(piece['points']),face=nums(piece['faces']),inertia='shell')
        ET.SubElement(board,'geom',name=name,type='mesh',mesh=name,rgba='0 0 0 0',group='3',
                      friction=f'{friction} .003 .0001',solref='.0042 1',solimp='.95 .99 .001',margin='0')
    for name,parts in task.parts.items():
        pose=task.layout['poses'][name]
        body=ET.SubElement(world,'body',name='task/'+name,pos=nums(pose['position']),quat=nums(pose['quaternion_wxyz']))
        ET.SubElement(body,'freejoint',name='task/'+name+'/free')
        points=np.concatenate([p['points'] for p in parts]); lo,hi=points.min(0),points.max(0)
        # Estimated box inertia is declared independently of decorative mesh volume.
        size=hi-lo; mass=task.config['plugs'][name]['mass_kg']
        inertia=mass/12*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
        ET.SubElement(body,'inertial',pos=nums((hi+lo)/2),mass=str(mass),diaginertia=nums(inertia))
        for p in parts:
            full=f'task/{name}/{p["name"]}'
            ET.SubElement(asset,'mesh',name=full,vertex=nums(p['points']),face=nums(p['faces']),inertia='shell')
            ET.SubElement(body,'geom',name=full,type='mesh',mesh=full,rgba=nums(p['color']),density='0',
                          friction=f'{friction} .003 .0001',solref='.0042 1',solimp='.95 .99 .001',margin='0')
    return ET.tostring(root,encoding='unicode')


def reset_task(model,data,task):
    board=model.body('task/board')
    pose=task.layout['poses']['board']
    model.body_pos[board.id]=pose['position']; model.body_quat[board.id]=pose['quaternion_wxyz']
    for name in task.names:
        joint=model.joint(f'task/{name}/free')
        q=int(joint.qposadr[0]); v=int(joint.dofadr[0]); pose=task.layout['poses'][name]
        data.qpos[q:q+7]=[*pose['position'],*pose['quaternion_wxyz']]
        data.qvel[v:v+6]=0


def task_poses(data,task):
    return {name:{'position':data.body('task/'+name).xpos.tolist(),
                  'quaternion_wxyz':data.body('task/'+name).xquat.tolist()}
            for name in ('board',*task.names)}
