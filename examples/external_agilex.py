#!/usr/bin/env python3
"""External read/control/record/reset loop in example_agilex units; simulation only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from piper_sim.client import SimulationClient
from piper_sim.agilex import AgilexAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--name', default='external_'+time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('outputs/external_client'))
    parser.add_argument('--lerobot-output', type=Path, help='Optional NEW LeRobot v2.1 dataset directory after collection')
    parser.add_argument('--lerobot-image-size', type=int, nargs=2, metavar=('WIDTH','HEIGHT'),
                        help='Optional letterbox size for exported videos; default preserves native resolution')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if args.lerobot_image_size and not args.lerobot_output:
        parser.error('--lerobot-image-size requires --lerobot-output')
    if args.lerobot_image_size and any(x <= 0 or x % 2 for x in args.lerobot_image_size):
        parser.error('--lerobot-image-size needs positive even width and height')
    if args.lerobot_output:
        from piper_sim.lerobot_export import check_dependencies
        check_dependencies()
        if args.lerobot_output.exists(): raise FileExistsError(args.lerobot_output)
    client = SimulationClient(args.url, timeout=15.)
    meta = client.metadata()
    if meta['mode'] != 'sim' or not meta['allow_commands']:
        raise RuntimeError('This example requires pure simulation with external command control')
    robot = AgilexAdapter(client)
    recorded = []
    for episode in range(args.episodes):
        name = f'{args.name}_{episode:03d}'
        robot.reset(seed=args.seed+episode, randomize=True)
        robot.send()  # Align initial target with measured state before sending new targets.
        start = [robot.get_current_gripper_open_value(arm) for arm in ('left', 'right')]
        robot.start_recording(name, task='open both grippers to 95 mm in a randomized connector scene')
        try:
            for frame in range(180):
                tick = time.monotonic()
                # Four-second ramp plus two seconds holding. Keep arm joint targets unchanged.
                for arm, initial in zip(('left', 'right'), start):
                    robot.set_gripper_open_value(float(np.clip(initial+(1-initial)*min(frame/120, 1), 0, 1)), arm)
                robot.send()
                time.sleep(max(0., 1/30-(time.monotonic()-tick)))
        finally:
            result = robot.stop_recording(success=False)  # Gripper demo is not insertion success.
            print(json.dumps(result), flush=True)
        recorded.append(result['path'])
        robot.refresh()
        out = args.output/name
        out.mkdir(parents=True, exist_ok=False)
        for camera, pixels in robot.observation.images.items():
            Image.fromarray(pixels).save(out/(camera+'.png'))
        (out/'state.json').write_text(json.dumps(client.state(), indent=2)+'\n')
        print('Measured grippers:', {a: robot.get_current_gripper_open_value(a) for a in ('left', 'right')})
    if args.lerobot_output:
        from piper_sim.lerobot_export import export_lerobot_v2
        # Convert only after all control/recording is finished. Paths refer to this host.
        report = export_lerobot_v2(recorded, args.lerobot_output, image_size=args.lerobot_image_size,
                                  progress=lambda s:print(s,flush=True))
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
