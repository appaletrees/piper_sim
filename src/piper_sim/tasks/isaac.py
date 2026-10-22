"""USD/PhysX adapter for the same task parts and hole-preserving convex meshes."""
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaacsim.core.utils.stage import get_current_stage


def spawn_task(task):
    stage=get_current_stage()
    UsdGeom.Xform.Define(stage,'/World/Task')
    material=UsdShade.Material.Define(stage,'/World/Task/ContactMaterial')
    contact_material=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact_material.CreateStaticFrictionAttr(task.config['contact']['friction'])
    contact_material.CreateDynamicFrictionAttr(task.config['contact']['friction'])
    contact_material.CreateRestitutionAttr(0.)
    objects={}
    for name in ('board',*task.names):
        path='/World/Task/'+name
        xform=UsdGeom.Xform.Define(stage,path)
        prim=xform.GetPrim()
        pose=task.layout['poses'][name]
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose['position']))
        q=pose['quaternion_wxyz']
        xform.AddOrientOp().Set(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
        rigid=UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(name=='board')
        physx=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx.CreateEnableCCDAttr(name!='board')
        physx.CreateSolverPositionIterationCountAttr(16)
        physx.CreateSolverVelocityIterationCountAttr(4)
        mass=UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(.25 if name=='board' else task.config['plugs'][name]['mass_kg'])
        if name=='board':
            points=[]; faces=[]
            for line in (task.asset_path.parent/task.assets['visual_mesh']).read_text().splitlines():
                if line.startswith('v '): points.append(list(map(float,line.split()[1:])))
                elif line.startswith('f '): faces.append([int(i)-1 for i in line.split()[1:]])
            parts=[{'name':'visual','points':points,'faces':faces,'color':[.43,.44,.46,1],'collision':False}]
            parts += [dict(piece,name=f'collision_{i}',color=[.43,.44,.46,1],hidden=True)
                      for i,piece in enumerate(task.assets['collision_parts'])]
        else: parts=task.parts[name]
        for part in parts:
            mesh=UsdGeom.Mesh.Define(stage,path+'/'+part['name'])
            mesh.CreatePointsAttr(part['points'])
            mesh.CreateFaceVertexCountsAttr([3]*len(part['faces']))
            mesh.CreateFaceVertexIndicesAttr(np.array(part['faces']).reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*part['color'][:3])])
            if part.get('hidden'): mesh.CreateVisibilityAttr('invisible')
            if part.get('collision',True):
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
                contact=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                contact.CreateContactOffsetAttr(task.config['contact']['contact_offset_m'])
                contact.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
        objects[name]=RigidObject(RigidObjectCfg(prim_path=path,spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose['position']),rot=tuple(q))))
    return objects


def reset_task(objects,task):
    for name,obj in objects.items():
        pose=task.layout['poses'][name]
        state=obj.data.default_root_state.clone()
        state[0,:3]=state.new_tensor(pose['position'])
        state[0,3:7]=state.new_tensor(pose['quaternion_wxyz'])
        state[0,7:]=0
        obj.write_root_state_to_sim(state)
        obj.reset()


def task_poses(objects):
    return {name:{'position':obj.data.root_pos_w[0].cpu().tolist(),
                  'quaternion_wxyz':obj.data.root_quat_w[0].cpu().tolist()} for name,obj in objects.items()}
