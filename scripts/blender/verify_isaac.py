#!/usr/bin/env python3
"""Inspect portable Blender materials in RTX; keeps all CAD cameras unchanged."""
import argparse
import json
from pathlib import Path
import yaml
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--legacy',action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app
backend=None
try:
    import torch
    from PIL import Image
    from pxr import UsdShade
    from isaacsim.core.utils.stage import get_current_stage
    from isaaclab.sensors import Camera,CameraCfg
    import isaaclab.sim as sim_utils
    from piper_sim.config import ROOT,load
    from piper_sim.backends.isaac import IsaacBackend
    c=load('configs/clean_table_workcell.yaml')
    c['task_scene']=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    if args.legacy:c['task_scene']['appearance'].pop('asset_library')
    backend=IsaacBackend(c,args.device,cameras=True)
    inspection=Camera(CameraCfg(prim_path='/World/BlenderInspection',height=960,width=1280,
        data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=42.,horizontal_aperture=36.,
            clipping_range=(.005,20.))))
    backend.sim.reset();backend.reset(0,False)
    for _ in range(60):backend.step()
    inspection.reset()
    for label,eye,target in [('scene',[1.,-1.,.85],[.36,-.10,.09]),
                              ('details',[.02,.10,.26],[.215,0,.075]),
                              ('pen_charger',[.20,-.43,.32],[.29,-.28,.035])]:
        inspection.set_world_poses_from_view(torch.tensor([eye],device=backend.sim.device),
            torch.tensor([target],device=backend.sim.device))
        for _ in range(24):backend.sim.render()
        inspection.update(.033,force_recompute=True)
        Image.fromarray(inspection.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output/(label+'.png'))
    for name,image in backend.read().images.items():Image.fromarray(image).save(args.output/(name+'.png'))
    connections={};stage=get_current_stage()
    for name,material in backend.task.materials.items():
        root='/World/Task/Materials/'+name
        shader=UsdShade.Shader(stage.GetPrimAtPath(root+'/Surface'))
        for field,input in [('normal_texture','normal'),('roughness_texture','roughness')]:
            if field in material:
                assert shader.GetInput(input).HasConnectedSource(),(name,field)
                connections[name+'/'+input]=str(shader.GetInput(input).GetConnectedSource()[0].GetPath())
    report={'passed':True,'legacy':args.legacy,'material_connections':connections,
            'visual_assets':backend.task.visual_asset_metadata,
            'scope':'actual RTX renderer, CAD cameras unchanged; inspection camera is presentation-only; no policy rollout'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
finally:
    if backend:backend.close()
    from isaaclab.sim import SimulationContext
    context=SimulationContext.instance()
    if context:context._disable_app_control_on_stop_handle=True
    app.close(wait_for_replicator=False)
