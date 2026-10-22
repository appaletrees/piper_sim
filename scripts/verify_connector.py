#!/usr/bin/env python3
"""Pure-simulation contact and reset audit. Direct object placement is a test fixture,
not a robot insertion controller and never contacts a physical CAN device.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from piper_sim.config import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('mujoco','isaac'),default='mujoco')
    parser.add_argument('--task-scene',default='configs/tasks/connector.yaml')
    parser.add_argument('--output',type=Path,required=True)
    select=argparse.ArgumentParser(add_help=False);select.add_argument('--backend',default='mujoco')
    selected,_=select.parse_known_args()
    if selected.backend=='isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    if (args.output/'report.json').exists(): raise FileExistsError(args.output/'report.json')
    args.output.mkdir(parents=True,exist_ok=True)
    app=AppLauncher(args).app if args.backend=='isaac' else None
    backend=None
    try:
        cfg=load();cfg['task_scene']=args.task_scene
        if args.backend=='isaac':
            from piper_sim.backends.isaac import IsaacBackend
            backend=IsaacBackend(cfg,args.device,cameras=args.enable_cameras)
        else:
            from piper_sim.backends.mujoco import MujocoBackend
            backend=MujocoBackend(cfg,cameras=args.render)
        task=backend.task
        def place(name,pose):
            if args.backend=='isaac':
                obj=backend.task_objects[name]
                state=obj.data.root_state_w.clone()
                state[0,:7]=state.new_tensor([*pose['position'],*pose['quaternion_wxyz']]);state[0,7:]=0
                obj.write_root_state_to_sim(state);obj.reset()
            else:
                joint=backend.model.joint('task/'+name+'/free');q=joint.qposadr[0];v=joint.dofadr[0]
                backend.data.qpos[q:q+7]=[*pose['position'],*pose['quaternion_wxyz']]
                backend.data.qvel[v:v+6]=0
        def capture(stem):
            if args.backend=='mujoco' and args.render:
                import mujoco
                from PIL import Image
                cam=mujoco.MjvCamera();cam.lookat[:]=[.36,0,.05]
                cam.distance=.65;cam.azimuth=90;cam.elevation=-65
                Image.fromarray(backend._render(cam,1280,900)).save(args.output/(stem+'.png'))
                for name,pixels in backend.read().images.items():
                    Image.fromarray(pixels).save(args.output/(stem+'_'+name+'.png'))
        layouts=[]
        for seed in range(100):
            layout=task.reset(seed)
            assert task._valid_layout(layout['poses'])
            assert task.reset(seed)==layout
            layouts.append(layout)
        backend.reset(seed=42,randomize=False)
        initial=copy.deepcopy(task.layout)
        for _ in range(60):backend.step()
        state=backend.read()
        assert not any(m['success'] for m in state.extras['task']['metrics'].values())
        for name in task.names:
            assert state.extras['objects'][name]['position'][2]>task.table_top
        capture('layout')
        if args.backend=='mujoco' and args.render:
            backend.save_overview(args.output/'overview.png')
        # At the estimated nominal pin spacing both objects must enter real CAD holes.
        for name in task.names:
            pose=task.target_pose(name);pose['position'][2]+=.005
            place(name,pose)
        for _ in range(90):backend.step()
        seated=copy.deepcopy(backend.read().extras)
        assert all(m['success'] for m in seated['task']['metrics'].values()), seated['task']
        capture('inserted')
        # A prong shifted onto solid board material must remain blocked above the face.
        backend.reset(seed=42,randomize=False)
        for name in task.names:
            pose=task.target_pose(name);pose['position'][0]+=.010;pose['position'][2]+=.021
            place(name,pose)
        for _ in range(60):backend.step()
        blocked=copy.deepcopy(backend.read().extras)
        assert not any(m['success'] for m in blocked['task']['metrics'].values()), blocked['task']
        # Assert no solid-board tunnelling: free objects remain above the board's bottom.
        assert all(p['position'][2]>task.table_top for n,p in blocked['objects'].items() if n!='board')
        capture('misaligned')
        report={'passed':True,'backend':args.backend,'random_seeds_tested':len(layouts),
                'source_sha256':task.assets['source_sha256'],'board_size_m':[.17,.06,.033],
                'convex_collision_parts':len(task.assets['collision_parts']),
                'source_volume_m3':task.assets['source_volume_m3'],
                'collision_volume_m3':task.assets['collision_volume_m3'],
                'dimensions':'board from CAD; both plugs estimated from photos',
                'initial_layout':initial,'seated':seated,'misaligned':blocked,
                'scope':'geometry/contact fixtures; does not demonstrate a robot grasp/insertion policy'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'backend':args.backend,'report':str(args.output/'report.json')}),flush=True)
    finally:
        if backend:backend.close()
        elif app:
            from isaaclab.sim import SimulationContext
            context=SimulationContext.instance()
            if context is not None:context._disable_app_control_on_stop_handle=True
        if app:app.close(wait_for_replicator=False)


if __name__=='__main__':main()
