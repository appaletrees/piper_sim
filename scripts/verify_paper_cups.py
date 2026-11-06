#!/usr/bin/env python3
"""Render and validate randomized paper cups in either engine, without hardware."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    probe=argparse.ArgumentParser(add_help=False);probe.add_argument('--backend',default='mujoco')
    selected,_=probe.parse_known_args()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['mujoco','isaac'],default='mujoco')
    p.add_argument('--config',default='configs/paper_cups_workcell.yaml')
    p.add_argument('--task-scene',default='configs/tasks/paper_cups.yaml')
    p.add_argument('--output',type=Path,required=True)
    if selected.backend=='isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    else:
        p.add_argument('--headless',action='store_true');p.add_argument('--enable_cameras',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    args.output.mkdir(parents=True)
    app=AppLauncher(args).app if args.backend=='isaac' else None
    b=None
    try:
        from PIL import Image
        from piper_sim.config import load
        from piper_sim.simulation import SimulationSession
        from piper_sim.tasks import create_task
        cfg=load(args.config);cfg['task_scene']=args.task_scene
        t=create_task(args.task_scene,cfg)
        for seed in range(1000):
            a=t.reset(seed,True);assert a==t.reset(seed,True)
            xy=np.array([p['position'][:2] for p in a['poses'].values()])
            for i in range(len(xy)):
                assert np.all(np.linalg.norm(xy[i+1:]-xy[i],axis=1)>=2*t.radius+t.config['randomization']['min_gap_m'])
        if args.backend=='isaac':
            from piper_sim.backends.isaac import IsaacBackend
            b=IsaacBackend(cfg,args.device,args.enable_cameras)
        else:
            from piper_sim.backends.mujoco import MujocoBackend
            b=MujocoBackend(cfg,cameras=args.enable_cameras)
        results=[]
        for seed in (42,7,23):
            b.reset(seed,True)
            for _ in range(60):b.step()
            obs=b.read();metrics=obs.extras['task']['metrics']['summary']
            if args.enable_cameras:
                assert set(obs.images)=={'overhead','left_wrist','right_wrist'}
                assert all(pixels.ndim==3 and pixels.shape[2]==3 and pixels.dtype==np.uint8
                           and min(pixels.shape[:2])>0 for pixels in obs.images.values())
            assert metrics['upright_count']==len(b.task.names) and metrics['on_table_count']==len(b.task.names),metrics
            results.append({'seed':seed,'objects':obs.extras['objects'],'metrics':metrics})
            for name,pixels in obs.images.items():Image.fromarray(pixels).save(args.output/f'seed{seed}_{name}.png')
        b.reset(42,False)
        for _ in range(45):b.step()
        if args.backend=='mujoco':
            import mujoco
            cam=mujoco.MjvCamera();cam.lookat[:]=[.365,0,.078]
            cam.distance=.32;cam.azimuth=8;cam.elevation=-22
            Image.fromarray(b._render(cam,1280,900)).save(args.output/'detail.png')
            b.save_overview(args.output/'overview.png')
            # Small sphere dropped through the open mouth verifies the convex
            # contacts do not fill the cavity. Test fixture is a separate model.
            import xml.etree.ElementTree as ET
            from piper_sim.backends.mujoco import MujocoBackend
            def add_probe(xml):
                root=ET.fromstring(xml);w=root.find('worldbody')
                body=ET.SubElement(w,'body',name='validation_ball',pos='0 0 1')
                ET.SubElement(body,'freejoint',name='validation_ball/free')
                ET.SubElement(body,'inertial',pos='0 0 0',mass='.0005',diaginertia='3.2e-9 3.2e-9 3.2e-9')
                ET.SubElement(body,'geom',type='sphere',size='.004',mass='0',rgba='1 0 0 1')
                return ET.tostring(root,encoding='unicode')
            check=MujocoBackend(cfg,model_xml_transform=add_probe)
            try:
                check.reset(42,False);cup=check.read().extras['objects']['cup_00']['position']
                j=check.model.joint('validation_ball/free');q=j.qposadr[0]
                check.data.qpos[q:q+3]=np.array(cup)+[0,0,.13]
                for _ in range(120):check.step()
                ball=check.data.body('validation_ball').xpos.copy()
                expected=b.task.table_top+b.task.config['cup']['floor_height_m']+.004
                assert abs(ball[2]-expected)<.001,(ball,expected)
                cavity={'passed':True,'ball_position_m':ball.tolist(),'expected_center_height_m':expected}
            finally:check.close()
        else:
            # Additional fixed camera only for product evidence; original three
            # optical poses and capture interfaces are untouched.
            from isaaclab.sensors import Camera,CameraCfg
            import isaaclab.sim as sim_utils
            import torch
            camera=Camera(CameraCfg(prim_path='/World/PaperCupDetail',height=900,width=1280,
                data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=36.,
                clipping_range=(.005,20.))))
            # Trigger the sensor's registered initialization callback.
            b.sim.reset()
            b.reset(42,False)
            for _ in range(45):b.step()
            for _ in range(3):b.sim.render()
            camera.reset()
            camera.set_world_poses_from_view(torch.tensor([[.065,-.05,.23]],device=b.sim.device),
                                             torch.tensor([[.365,0.,.075]],device=b.sim.device))
            for _ in range(24):b.sim.render()
            camera.update(.033,force_recompute=True)
            Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output/'detail.png')
            cavity={'passed':None,'note':'Dynamic sphere-in-cup fixture is verified in the MuJoCo run'}
        session=SimulationSession(b,record_dir=args.output/'episodes')
        session.reset(12,True);session.start_recording('paper_cups_smoke',b.task.config['instruction'])
        for _ in range(5):session.step()
        rec=session.stop_recording(False)
        import h5py
        with h5py.File(rec['path']) as f:
            extras=json.loads(f['observation/extras_json'][0])
            assert len(extras['objects'])==len(b.task.names)
            assert extras['task']['name']=='paper_cups'
            assert json.loads(f.attrs['metadata'])['task_scene']['layout']['seed']==12
        report={'passed':True,'backend':args.backend,'cameras_enabled':args.enable_cameras,
                'random_seeds_checked':1000,'settling':results,
                'cavity_probe':cavity,'recording':rec,'metadata':b.task.metadata(),
                'scope':'randomization, appearance, rigid-body stability, state recording and optional RGB; no grasp policy'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'report':str(args.output/'report.json')}),flush=True)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        (args.output/'failure.json').write_text(json.dumps({'passed':False,'error':repr(exc)})+'\n')
        raise
    finally:
        if b:b.close()
        if app:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
