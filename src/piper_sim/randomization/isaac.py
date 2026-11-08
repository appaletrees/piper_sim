"""Isaac Lab reset adapter, using live PhysX property setters, not stale USD masses."""
import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from isaacsim.core.utils.stage import get_current_stage


class IsaacRandomizer:
    def __init__(self, objects, task):
        self.objects, self.task = objects, task
        self.base = {}
        for name, obj in objects.items():
            view = obj.root_physx_view
            self.base[name] = (view.get_masses().clone(), view.get_inertias().clone(),
                               view.get_material_properties().clone())
        self.stage = get_current_stage()
        self.lights = []
        for path in ('/World/Light', '/World/Task/Lighting/Ceiling_0', '/World/Task/Lighting/Ceiling_1'):
            prim = self.stage.GetPrimAtPath(path)
            api = UsdLux.LightAPI(prim)
            if prim and api:
                op = next((op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                           if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                self.lights.append((api, api.GetIntensityAttr().Get(), op, np.array(op.Get()) if op else None,
                                    np.array(api.GetColorAttr().Get())))

    def apply(self, parameters):
        name=(parameters or {}).get('tabletop_material') or self.task.config['appearance']['tabletop_material']
        material=UsdShade.Material(self.stage.GetPrimAtPath('/World/Task/Materials/'+name))
        workcell=self.stage.GetPrimAtPath('/World/Workcell')
        if workcell:
            for prim in Usd.PrimRange(workcell):
                if prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith('part_000_'):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        rear=self.stage.GetPrimAtPath('/World/Task/Environment/rear_table')
        if rear:UsdShade.MaterialBindingAPI.Apply(rear).Bind(material)
        for name, obj in self.objects.items():
            mass, inertia, material = (x.clone() for x in self.base[name])
            if parameters:
                p = parameters['objects'][name]
                mass *= p['mass_scale']; inertia *= p['mass_scale']
                material[..., :2] *= p['friction_scale']
            ids = torch.arange(len(mass), dtype=torch.int32, device='cpu')
            view = obj.root_physx_view
            view.set_masses(mass, ids); view.set_inertias(inertia, ids)
            view.set_material_properties(material, ids)
        for name, nominal in self.task.materials.items():
            p = parameters['materials'][name] if parameters else {'color_scale': [1, 1, 1], 'roughness_delta': 0}
            color = np.clip(p.get('color_rgb',np.asarray(nominal['color'])*p['color_scale']), 0, 1)
            if not nominal.get('linear_color'):
                color = np.where(color <= .04045, color/12.92, ((color+.055)/1.055)**2.4)
            root = '/World/Task/Materials/'+name
            shader = UsdShade.Shader(self.stage.GetPrimAtPath(root+'/Surface'))
            roughness=float(np.clip(nominal['roughness']+p['roughness_delta'], 0, 1))
            shader.GetInput('roughness').Set(roughness)
            if 'roughness_texture' in nominal:
                texture=UsdShade.Shader(self.stage.GetPrimAtPath(root+'/RoughnessTexture'))
                texture.GetInput('scale').Set(Gf.Vec4f(*([2*roughness]*3),1.))
            if 'texture' in nominal or 'detail_texture' in nominal:
                texture = UsdShade.Shader(self.stage.GetPrimAtPath(root+'/Texture'))
                texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color, 1.))
            else:
                shader.GetInput('diffuseColor').Set(Gf.Vec3f(*color))
        for api, intensity, op, position, color in self.lights:
            api.GetIntensityAttr().Set(float(intensity*(parameters['light_scale'] if parameters else 1)))
            light_color=color*np.array(parameters['light_color_scale']) if parameters else color
            api.GetColorAttr().Set(Gf.Vec3f(*map(float,light_color)))
            if op:
                p = position+(np.array(parameters['light_position_delta_m']) if parameters else 0)
                op.Set(Gf.Vec3d(*p))
