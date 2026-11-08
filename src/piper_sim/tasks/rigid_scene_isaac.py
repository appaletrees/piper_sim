"""Task-neutral USD/PhysX adapter sharing SI geometry and UVs with MuJoCo."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaacsim.core.utils.stage import get_current_stage


def _materials(stage,root,task):
    result={}
    for name,s in task.materials.items():
        path=root+'/Materials/'+name
        mat=UsdShade.Material.Define(stage,path)
        shader=UsdShade.Shader.Define(stage,path+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
        color=np.asarray(s['color'])
        if not s.get('linear_color'):
            color=np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4)
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(s['roughness'])
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(s['metallic'])
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(s.get('opacity',1.))
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(s.get('ior',1.5))
        base_texture=s.get('texture',s.get('detail_texture'))
        if base_texture or s.get('roughness_texture') or s.get('normal_texture'):
            reader=UsdShade.Shader.Define(stage,path+'/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        if base_texture:
            tex=UsdShade.Shader.Define(stage,path+'/Texture');tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(task.material_path.parent/base_texture))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
            tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color,1.))
            shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
        for field,node_name,input_name in [('roughness_texture','RoughnessTexture','roughness'),
                                            ('normal_texture','NormalTexture','normal')]:
            if field not in s:continue
            tex=UsdShade.Shader.Define(stage,path+'/'+node_name);tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(task.material_path.parent/s[field]))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
            tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            if input_name=='roughness':
                tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*([2*s['roughness']]*3),1.))
                shader.GetInput('roughness').ConnectToSource(tex.ConnectableAPI(),'r')
            else:
                strength=s.get('normal_strength',1.)
                tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2*strength,2*strength,2.,1.))
                tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-strength,-strength,-1.,0.))
                shader.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        result[name]=mat
    contact=UsdShade.Material.Define(stage,root+'/Materials/Contact')
    api=UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    api.CreateStaticFrictionAttr(task.config['contact']['friction'])
    api.CreateDynamicFrictionAttr(task.config['contact']['friction'])
    api.CreateRestitutionAttr(0.)
    return result,contact


def spawn_task(task,root_path='/World/Task',object_root_path=None):
    stage=get_current_stage();UsdGeom.Xform.Define(stage,root_path)
    materials,contact_material=_materials(stage,root_path,task)
    for name,s in task.config.get('environment',{}).items():
        cube=UsdGeom.Cube.Define(stage,root_path+'/Environment/'+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*s['position_m']))
        cube.AddScaleOp().Set(Gf.Vec3f(*s['size_m']))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(materials[s['material']])
    objects={}
    for name,parts in task.parts.items():
        path=root_path+'/'+name;p=task.layout['poses'][name];q=p['quaternion_wxyz']
        x=UsdGeom.Xform.Define(stage,path);prim=x.GetPrim()
        x.AddTranslateOp().Set(Gf.Vec3d(*p['position']))
        x.AddOrientOp().Set(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        physx=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx.CreateEnableCCDAttr(True)
        physx.CreateSolverPositionIterationCountAttr(16)
        physx.CreateSolverVelocityIterationCountAttr(4)
        physx.CreateMaxDepenetrationVelocityAttr(1.)
        mass=UsdPhysics.MassAPI.Apply(prim);m=task.specs[name]['mass_kg']
        lo,hi=task.bounds[name];size=np.maximum(hi-lo,.001)
        inertia=m/12*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
        mass.CreateMassAttr(m);mass.CreateCenterOfMassAttr(Gf.Vec3f(*((lo+hi)/2)))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        for part in parts:
            mesh=UsdGeom.Mesh.Define(stage,path+'/'+part['name'])
            points=np.asarray(part['points']);faces=np.asarray(part['faces'])
            mesh.CreatePointsAttr(part['points']);mesh.CreateFaceVertexCountsAttr([3]*len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*task.materials[part['material']]['color'])])
            if not part['hidden']:
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[part['material']])
                if 'uv' in part:
                    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
                        UsdGeom.Tokens.vertex).Set(part['uv'])
                if 'normals' in part:
                    mesh.CreateNormalsAttr(part['normals']);mesh.SetNormalsInterpolation('vertex')
                elif part.get('smooth',name!='paper'):
                    normals=np.zeros_like(points)
                    fn=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
                    for i in range(3):np.add.at(normals,faces[:,i],fn)
                    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
                    mesh.CreateNormalsAttr(normals.tolist());mesh.SetNormalsInterpolation('vertex')
            else: mesh.CreateVisibilityAttr('invisible')
            if part['collision']:
                collision_prim=mesh.GetPrim()
                if 'box' in part:
                    # Analytic thin boxes avoid GPU convex-cooking degeneration.
                    cube=UsdGeom.Cube.Define(stage,path+'/'+part['name']+'_box_contact')
                    cube.CreateSizeAttr(1.)
                    cube.AddTranslateOp().Set(Gf.Vec3d(*part['box']['center']))
                    cube.AddScaleOp().Set(Gf.Vec3f(*part['box']['size']))
                    cube.CreateVisibilityAttr('invisible')
                    collision_prim=cube.GetPrim()
                elif 'cylinder' in part:
                    c=part['cylinder']
                    cylinder=UsdGeom.Cylinder.Define(stage,path+'/'+part['name']+'_cylinder_contact')
                    cylinder.CreateRadiusAttr(c['radius']);cylinder.CreateHeightAttr(c['height'])
                    cylinder.CreateAxisAttr('Z');cylinder.AddTranslateOp().Set(Gf.Vec3d(*c['center']))
                    cylinder.CreateVisibilityAttr('invisible');collision_prim=cylinder.GetPrim()
                else:
                    UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr('convexHull')
                UsdPhysics.CollisionAPI.Apply(collision_prim).CreateCollisionEnabledAttr(True)
                contact=PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
                contact.CreateContactOffsetAttr(task.config['contact']['contact_offset_m'])
                contact.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(contact_material,materialPurpose='physics')
        objects[name]=RigidObject(RigidObjectCfg(prim_path=(object_root_path or root_path)+'/'+name,
            spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(p['position']),rot=tuple(q))))
    configure_appearance(task,root_path)
    return objects


def configure_appearance(task,root_path='/World/Task'):
    """Optional scene appearance, reapplied after generic CAD material binding."""
    if task.config.get('appearance',{}).get('preset')=='indoor_lab':
        from .indoor_lab import configure_isaac
        configure_isaac(get_current_stage(),root_path,task)
    from .stage_visuals import spawn_isaac
    spawn_isaac(get_current_stage(),root_path,task)


def reset_task(objects,task):
    for name,obj in objects.items():
        p=task.layout['poses'][name];state=obj.data.default_root_state.clone()
        state[0,:7]=state.new_tensor([*p['position'],*p['quaternion_wxyz']]);state[0,7:]=0
        obj.write_root_state_to_sim(state);obj.reset()
        obj._task_active=name in getattr(task,'active_names',task.names)
        prim=get_current_stage().GetPrimAtPath('/World/Task/'+name)
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited' if obj._task_active else 'invisible')


def task_poses(objects):
    return {n:{'position':obj.data.root_pos_w[0].cpu().tolist(),
               'quaternion_wxyz':obj.data.root_quat_w[0].cpu().tolist()}
            for n,obj in objects.items() if getattr(obj,'_task_active',True)}
