#!/usr/bin/env python3
"""Headless scene/contact/recording audit. Placement fixtures are not a policy."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['mujoco','isaac'],default='mujoco')
    parser.add_argument('--config',default='configs/clean_table_workcell.yaml')
    parser.add_argument('--task-scene',default='configs/tasks/clean_table.yaml')
    parser.add_argument('--output',type=Path,required=True)
    select=argparse.ArgumentParser(add_help=False);select.add_argument('--backend',default='mujoco')
    selected,_=select.parse_known_args()
    if selected.backend=='isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument('--headless',action='store_true')
        parser.add_argument('--enable_cameras',action='store_true')
    args=parser.parse_args()
    if (args.output/'report.json').exists(): raise FileExistsError(args.output/'report.json')
    args.output.mkdir(parents=True,exist_ok=True)
    app=AppLauncher(args).app if args.backend=='isaac' else None
    b=None
    try:
        from piper_sim.config import load
        from piper_sim.simulation import SimulationSession
        from PIL import Image
        cfg=load(args.config);cfg['task_scene']=args.task_scene
        if args.backend=='isaac':
            from piper_sim.backends.isaac import IsaacBackend
            b=IsaacBackend(cfg,args.device,args.enable_cameras)
        else:
            from piper_sim.backends.mujoco import MujocoBackend
            b=MujocoBackend(cfg,cameras=args.enable_cameras)
        t=b.task
        for seed in range(100):
            a=t.reset(seed,True)
            assert t._valid_layout(a['poses']) and a==t.reset(seed,True)
        b.reset(42,False)
        nominal=copy.deepcopy(t.layout)
        for _ in range(60): b.step()
        obs=b.read();settled=copy.deepcopy(obs.extras)
        if args.enable_cameras:
            assert set(obs.images)=={'overhead','left_wrist','right_wrist'}
        assert not settled['task']['metrics']['summary']['success']
        for n in ('placemat','basket','trash_bin'):
            assert abs(settled['objects'][n]['position'][2]-t.table_top)<.003,(n,settled['objects'][n])
        for name,pixels in obs.images.items():Image.fromarray(pixels).save(args.output/(name+'.png'))
        if args.backend=='mujoco' or not args.headless:
            b.save_overview(args.output/'overview.png')
        if args.backend=='mujoco' and args.enable_cameras:
            import mujoco
            # Independent close-up camera avoids the overhead housing/mast.
            cam=mujoco.MjvCamera();cam.lookat[:]=[.44,0,.045]
            cam.distance=1.2;cam.azimuth=8;cam.elevation=-67
            Image.fromarray(b._render(cam,1280,900)).save(args.output/'tabletop.png')
            cam.lookat[:]=[.335,0,.055];cam.distance=.65;cam.azimuth=40;cam.elevation=-56
            Image.fromarray(b._render(cam,1280,900)).save(args.output/'detail.png')
            # Same real frame-0 joint vector, only fractions converted to metres.
            q=np.array(t.config['reference']['joint14_rad_fraction']);q[[6,13]]*=.095
            b.set_observation_state(q)
            Image.fromarray(b.read().images['overhead']).save(args.output/'reference_pose.png')
            b.reset(42,False)
            for _ in range(30):b.step()
        if args.backend=='isaac' and args.enable_cameras:
            # Extra evidence camera; it never replaces or moves the CAD sensors.
            from isaaclab.sensors import Camera,CameraCfg
            import isaaclab.sim as sim_utils
            import torch
            camera=Camera(CameraCfg(prim_path='/World/CleanTableInspection',height=900,width=1280,
                data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=30.,horizontal_aperture=36.,
                clipping_range=(.005,20.))))
            b.sim.reset();b.reset(42,False)
            for _ in range(60):b.step()
            camera.reset()
            for filename,eye,target in [
                ('tabletop.png',[.02,-.06,1.12],[.44,0.,.045]),
                ('detail.png',[.02,-.30,.58],[.335,0.,.055])]:
                camera.set_world_poses_from_view(torch.tensor([eye],device=b.sim.device),
                                                 torch.tensor([target],device=b.sim.device))
                for _ in range(24):b.sim.render()
                camera.update(.033,force_recompute=True)
                Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(args.output/filename)
        # Drop items into open containers; never attach, weld or teleport as a policy.
        for name,target in [('plate','basket'),('paper','trash_bin')]:
            p=b.read().extras['objects'][target]
            pose=[*(np.array(p['position'])+[0,0,.04]),*p['quaternion_wxyz']]
            if args.backend=='mujoco':
                j=b.model.joint('task/'+name+'/free');q=j.qposadr[0]
                b.data.qpos[q:q+7]=pose;b.data.qvel[j.dofadr[0]:j.dofadr[0]+6]=0
            else:
                obj=b.task_objects[name];state=obj.data.root_state_w.clone()
                state[0,:7]=state.new_tensor(pose);state[0,7:]=0
                obj.write_root_state_to_sim(state);obj.reset()
        for _ in range(90):b.step()
        dropped=copy.deepcopy(b.read().extras)
        assert dropped['task']['metrics']['plate']['success'],dropped['task']['metrics']['plate']
        assert dropped['task']['metrics']['paper']['success'],dropped['task']['metrics']['paper']
        for name,pixels in b.read().images.items():Image.fromarray(pixels).save(args.output/('sorted_'+name+'.png'))
        session=SimulationSession(b,record_dir=args.output/'episodes')
        session.reset(17,True)
        session.start_recording('smoke',t.config['instruction'])
        for _ in range(3):session.step()
        recording=session.stop_recording(False)
        import h5py
        with h5py.File(recording['path']) as f:
            extras=json.loads(f['observation/extras_json'][0])
            assert extras['task']['name']=='clean_table' and extras['task']['seed']==17
            assert len(extras['objects'])==11
        report={'passed':True,'backend':args.backend,'random_seeds_tested':100,
                'nominal_layout':nominal,'settled':settled,'container_drop':dropped,
                'recorded_frames':recording['frames'],'metadata':t.metadata(),
                'scope':'scene rendering, stability, free-body container drops and recording; no autonomous clearing policy'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'report':str(args.output/'report.json')}),flush=True)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        (args.output/'failure.json').write_text(json.dumps({'passed':False,'error':repr(exc)})+'\n')
        raise
    finally:
        if b:b.close()
        elif app:
            from isaaclab.sim import SimulationContext
            context=SimulationContext.instance()
            if context is not None:context._disable_app_control_on_stop_handle=True
        if app:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
