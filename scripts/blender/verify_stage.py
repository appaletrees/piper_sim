#!/usr/bin/env python3
"""Capture actual sensor images plus presentation views for visual review.

Reference-pose images are static calibration fixtures, not successful policy
rollouts. Same workcell, optics and task assets are used with/without the stage.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import yaml

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend',choices=['mujoco','isaac'],default='mujoco')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--baseline',action='store_true')
selected,_=parser.parse_known_args()
if selected.backend=='isaac':
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
if (args.output/'report.json').exists():parser.error('Use a new output directory to preserve capture evidence')
app=AppLauncher(args).app if selected.backend=='isaac' else None
b=None
try:
    from PIL import Image
    from piper_sim.config import ROOT,load
    c=load('configs/clean_table_workcell.yaml')
    t=yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text());c['task_scene']=t
    if args.baseline:
        t['appearance'].pop('stage_library',None);t['appearance'].pop('lighting_profile',None)
        t['appearance'].update(dome_intensity=700.,rect_light_intensity=1800.,mujoco_light_scale=.75)
    if args.backend=='mujoco':
        os.environ.setdefault('MUJOCO_GL','egl')
        import mujoco
        from piper_sim.backends.mujoco import MujocoBackend
        b=MujocoBackend(c,cameras=True)
    else:
        import torch
        from isaaclab.sensors import Camera,CameraCfg
        import isaaclab.sim as sim_utils
        from piper_sim.backends.isaac import IsaacBackend
        b=IsaacBackend(c,args.device,cameras=True)
        inspection=Camera(CameraCfg(prim_path='/World/StageInspection',height=960,width=1280,
            data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=30.,horizontal_aperture=36.,
                clipping_range=(.005,20.))))
        b.sim.reset();b.reset(0,False)
    for _ in range(30):b.step()
    stats={}
    def capture(label):
        obs=b.read()
        for name,pixels in obs.images.items():
            Image.fromarray(pixels).save(args.output/(label+'_'+name+'.png'))
            stats[label+'/'+name]={'mean_rgb':pixels.mean((0,1)).tolist(),
                'std_rgb':pixels.std((0,1)).tolist(),'fraction_all_channels_ge_250':float(np.mean(np.all(pixels>=250,axis=2))),
                'fraction_all_channels_le_5':float(np.mean(np.all(pixels<=5,axis=2)))}
        return obs.extras['cameras']
    home=capture('home')
    views=[('overview',[1.35,-1.65,1.05],[.35,.08,.20],38.),
           ('context',[-.6,-.85,.9],[1.2,.1,.3],28.),
           ('hardware',[.34,.06,.48],[.065,.29,.31],52.)]
    for name,eye,target,lens in views:
        if args.backend=='mujoco':
            cam=mujoco.MjvCamera();cam.lookat[:]=target;delta=np.array(eye)-target
            cam.distance=float(np.linalg.norm(delta));cam.azimuth=180+float(np.degrees(np.arctan2(delta[1],delta[0])))
            cam.elevation=-float(np.degrees(np.arctan2(delta[2],np.linalg.norm(delta[:2]))))
            b.model.vis.global_.fovy=float(np.degrees(2*np.arctan(27/(2*lens))))
            pixels=b._render(cam,1280,960)
        else:
            from pxr import UsdGeom
            from isaacsim.core.utils.stage import get_current_stage
            UsdGeom.Camera(get_current_stage().GetPrimAtPath('/World/StageInspection')).GetFocalLengthAttr().Set(lens)
            inspection.set_world_poses_from_view(torch.tensor([eye],device=b.sim.device),torch.tensor([target],device=b.sim.device))
            for _ in range(24):b.sim.render()
            inspection.update(.033,force_recompute=True)
            pixels=inspection.data.output['rgb'][0,:,:,:3].cpu().numpy()
        Image.fromarray(pixels).save(args.output/(name+'.png'))
    # Frame-zero angles define the target. MuJoCo supports an exact static
    # fixture; Isaac's Fabric path needs a control step, so use and report its
    # measured state for FK validation rather than calling it exact playback.
    q=np.array(t['reference']['joint14_rad_fraction']);q[[6,13]]*=.095
    if args.backend=='mujoco':b.set_observation_state(q)
    else:
        for i,(side,arm) in enumerate(b.arms.items()):
            a=q[7*i:7*i+7];position=arm.data.default_joint_pos.clone()
            position[:,b.indices[side]]=position.new_tensor([*a[:6],a[6]/2,a[6]/2])
            arm.write_joint_state_to_sim(position,torch.zeros_like(position));arm.set_joint_position_target(position)
            arm.write_data_to_sim();arm.update(0.)
        b.step()
        for _ in range(24):b.sim.render()
        for camera in b.cameras.values():camera.update(.033,force_recompute=True)
    reference=capture('reference')
    observed=b.read().position
    state_error=float(np.max(np.abs(observed-q)))
    (args.output/'reference_state.json').write_text(json.dumps({'requested':q.tolist(),
        'observed':observed.tolist(),'max_error':state_error})+'\n')
    if args.backend=='mujoco':assert state_error<1e-12
    for name in ('left_wrist','right_wrist'):
        assert np.linalg.norm(np.array(reference[name]['position_world'])-home[name]['position_world'])>.05,('stale camera pose',name)
    fk_errors={}
    if args.backend=='isaac':
        # Independent FK of the measured joints, not the requested target. This
        # rejects stale rendered sensor transforms even if joint buffers updated.
        from piper_sim.kinematics import forward
        from piper_sim.transforms import load_camera_rig,pose_matrix
        rig=load_camera_rig(c)
        for name,pose in reference.items():
            mount=rig['cameras'][name]
            expected=pose_matrix(mount['position'],mount['rotation_wxyz_ros'])
            if name!='overhead':
                side=name.split('_')[0];i=('left','right').index(side)
                base=c['robot'][side]
                fk=forward(ROOT/c['robot']['urdf'],{f'joint{j+1}':observed[i*7+j] for j in range(6)},
                    pose_matrix(base['position'],base['rotation']))
                expected=fk['link6']@expected
            error=float(np.linalg.norm(expected[:3,3]-pose['position_world']))
            assert error<.002,(name,'sensor/observed-state FK mismatch',error)
            fk_errors[name]=error
    if args.backend=='isaac' and not args.baseline:
        from pxr import Usd,UsdPhysics
        from isaacsim.core.utils.stage import get_current_stage
        stage=get_current_stage();n=0
        for prim in stage.Traverse():
            if '/StageVisuals/' in str(prim.GetPath()):
                assert not prim.HasAPI(UsdPhysics.CollisionAPI)
                assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                assert not prim.HasAPI(UsdPhysics.MassAPI)
                n+=1
        assert n==b.task.stage_visual_metadata['parts'],n
    report=dict(backend=args.backend,baseline=args.baseline,scope='renderer and sensor fixtures, not policy validation',
        cameras_home=home,cameras_reference=reference,reference_joint14_si=q.tolist(),
        observed_reference_joint14_si=observed.tolist(),max_reference_state_error_si=state_error,image_statistics=stats,
        reference_fixture_physics_seconds=b.dt*b.decimation if args.backend=='isaac' else 0.,
        camera_position_fk_error_m=fk_errors,
        stage_visuals=b.task.stage_visual_metadata,
        visual_acceptance='requires visual review; pixel statistics do not certify benchmark equivalence')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('STAGE_CAPTURE_COMPLETE',args.output,flush=True)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    (args.output/'failure.json').write_text(json.dumps({'error':repr(exc)})+'\n')
    raise
finally:
    if b:b.close()
    if app:
        from isaaclab.sim import SimulationContext
        context=SimulationContext.instance()
        if context:context._disable_app_control_on_stop_handle=True
        app.close(wait_for_replicator=False)
