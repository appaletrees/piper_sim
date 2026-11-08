#!/usr/bin/env python3
"""Build a contact sheet and optional video from the RECORDED camera frames."""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video', action='store_true')
    p.add_argument('--speed', type=int, choices=range(1, 17), default=4)
    args = p.parse_args()
    if args.output.exists(): p.error('Output already exists')
    args.output.mkdir(parents=True)
    with h5py.File(args.episode) as f:
        if not f.attrs.get('complete'): p.error('Episode is incomplete')
        names = ['overhead', 'left_wrist', 'right_wrist']
        if set(f.get('observation/images', {})) != set(names): p.error('Requires three recorded RGB streams')
        n = int(f.attrs['num_transitions']); fps = json.loads(f.attrs['metadata'])['fps']
        indices = np.linspace(0, n-1, 6).astype(int).tolist()
        def panel(i, width=384):
            teacher = json.loads(f['observation/extras_json'][i]).get('teacher', {})
            images = [Image.fromarray(f['observation/images/'+name][i]) for name in names]
            height = round(images[0].height*width/images[0].width)
            out = Image.new('RGB', (width*3, height+40), '#121a24')
            draw = ImageDraw.Draw(out)
            for col, (name, im) in enumerate(zip(names, images)):
                out.paste(im.resize((width, height)), (col*width, 40))
                draw.text((col*width+8, 5), name, fill='white')
            t = f['observation/sim_time'][i]
            draw.text((8, 23), f'RECORDED RGB | frame {i} | t={t:.2f}s | {teacher.get("object")} : {teacher.get("phase")}', fill='white')
            return out
        panels = [panel(i) for i in indices]
        sheet = Image.new('RGB', (panels[0].width, sum(im.height for im in panels)))
        for row, im in enumerate(panels): sheet.paste(im, (0, row*im.height))
        sheet.save(args.output/'camera_review.jpg', quality=92)
        if args.video:
            sample = panel(0)
            proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                '-pixel_format', 'rgb24', '-video_size', f'{sample.width}x{sample.height}',
                '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart',
                str(args.output/'recorded_rgb_preview.mp4')], stdin=subprocess.PIPE)
            try:
                for i in range(0, n, args.speed):
                    im = panel(i)
                    ImageDraw.Draw(im).text((im.width-140, 23), f'{args.speed}x PLAYBACK', fill='white')
                    proc.stdin.write(im.tobytes())
                proc.stdin.close()
                if proc.wait(timeout=120): raise RuntimeError('Preview encoding failed')
            except BaseException:
                proc.kill(); proc.wait(); raise
        (args.output/'review.json').write_text(json.dumps({'episode': str(args.episode.resolve()),
            'frames': n, 'sample_indices': indices, 'source': 'observation/images; no rerender or state placement',
            'playback_speed': args.speed if args.video else None}, indent=2)+'\n')


if __name__ == '__main__': main()
