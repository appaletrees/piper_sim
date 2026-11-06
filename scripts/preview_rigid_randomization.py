#!/usr/bin/env python3
"""Render reproducible domain samples without moving or recording the robot."""
import argparse
import copy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/rigid_table_collection'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 4, 42, 43])
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import yaml
    from PIL import Image, ImageDraw
    from piper_sim.config import ROOT, load
    from piper_sim.backends.mujoco import MujocoBackend

    cfg = load('configs/clean_table_workcell.yaml')
    cfg['task_scene'] = yaml.safe_load((ROOT/'configs/tasks/clean_table_rigid.yaml').read_text())
    cfg['task_scene']['domain_randomization'] = 'configs/domain_randomization/clean_table.yaml'
    args.output.mkdir(parents=True, exist_ok=True)
    backend = MujocoBackend(cfg)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.40, -.06, .04]
    camera.distance, camera.azimuth, camera.elevation = 1.22, 12, -73
    panels, reports = [], []
    try:
        for seed in args.seeds:
            backend.reset(seed, True)
            for _ in range(30):
                backend.step()
            image = Image.fromarray(backend._render(camera, 880, 660))
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, 880, 35], fill=(25, 25, 25))
            count = backend.task.layout['content']['cup_count']
            material = backend.task.domain_parameters['tabletop_material']
            draw.text((12, 12), f'SEED {seed} | CUPS {count} | {material}', fill='white')
            image.save(args.output/f'domain_{seed}.png')
            panels.append(image.resize((660, 495)))
            reports.append(copy.deepcopy(backend.task.metadata()))
        canvas = Image.new('RGB', (1980, 495*((len(panels)+2)//3)))
        for index, panel in enumerate(panels):
            canvas.paste(panel, ((index % 3)*660, (index//3)*495))
        canvas.save(args.output/'domain_comparison.jpg', quality=90)
        (args.output/'domain_samples.json').write_text(json.dumps(reports, indent=2)+'\n')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
