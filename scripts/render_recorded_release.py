#!/usr/bin/env python3
"""Render a 1x diagnostic release clip from recorded native states.

Every selected 30 Hz state is displayed once. This is kinematic reconstruction,
not dynamics replay or original synchronized sensor RGB, and is never training
data. The source HDF5 and its acceptance fields remain read-only.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.config import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--object', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Preserve existing previews; choose a new output')
    args.output.mkdir(parents=True)
    with h5py.File(args.source, 'r') as f:
        if not f.attrs.get('complete'): raise ValueError('Closed source recording required')
        meta = json.loads(f.attrs['metadata'])
        events = [e for e in meta['collection_outcome']['teacher_events'] if e['object']==args.object]
        start_time = next(e['sim_time'] for e in events if e['phase']=='lower')-1.
        end_time = next(e['sim_time'] for e in events if e['phase']=='placed')
        fps = float(meta['fps'])
        times = np.r_[f['observation/sim_time'][:], f['next_observation/sim_time'][-1]]
        if fps != 30. or not np.allclose(np.diff(times), 1/fps, rtol=0, atol=1e-6):
            raise ValueError('Expected intact 30 Hz source clocks; no resampling or speed changes')
        first, stop = int(np.searchsorted(times, start_time)), min(len(times), int(np.searchsorted(times, end_time))+1)
        if stop <= first: raise ValueError('Empty release interval')
        changed = []
        for name, digest in meta.get('provenance', {}).items():
            path = ROOT/name
            if not path.is_file() or sha256(path) != digest: changed.append(name)
        cfg = copy.deepcopy(meta['config'])
        cfg['task_scene']['domain_randomization'] = meta['task_scene'].get('domain_profile')
        backend = MujocoBackend(cfg, cameras=False)
        partial = args.output/'release.partial.mp4'
        encoder = None
        try:
            layout = meta['task_scene']['layout'];backend.reset(layout['seed'], layout['randomized'])
            camera = mujoco.MjvCamera();camera.lookat[:] = [.6, -.43, .12]
            camera.distance = .7;camera.azimuth = 60;camera.elevation = -60
            encoder = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                '-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r',str(fps),'-i','pipe:0','-an',
                '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],
                stdin=subprocess.PIPE)
            snapshots = []
            for frame in range(first, stop):
                prefix, index = ('observation',frame) if frame<len(times)-1 else ('next_observation',frame-1)
                extras = json.loads(f[prefix+'/extras_json'][index])
                native = extras['teacher']['physics']['arms']
                if set(native) != {'left', 'right'}: raise ValueError('Missing native arm poses')
                for side, values in native.items():
                    q = np.asarray(values['joint_position'])
                    if q.shape != (8,) or not np.isfinite(q).all(): raise ValueError('Invalid native arm pose')
                    backend.data.qpos[backend.indices[side]['qpos']] = q
                for name, pose in extras['objects'].items():
                    adr = backend.model.joint('task/'+name+'/free').qposadr[0]
                    backend.data.qpos[adr:adr+7] = pose['position']+pose['quaternion_wxyz']
                mujoco.mj_forward(backend.model, backend.data)
                im = Image.fromarray(backend._render(camera,640,480))
                draw = ImageDraw.Draw(im);draw.rectangle((0,0,640,35),fill=(15,20,28))
                draw.text((8,3),'RECORDED STATE RECONSTRUCTION | 1x / 30 Hz | NOT SENSOR RGB',fill='white')
                draw.text((8,18),f'{args.object}: {extras["teacher"]["phase"]} | source t={times[frame]:.3f}s',fill='white')
                encoder.stdin.write(np.asarray(im).tobytes())
                if frame in (first,(first+stop)//2,stop-1): snapshots.append(im.copy())
            encoder.stdin.close()
            if encoder.wait(timeout=30): raise RuntimeError('Preview encoding failed')
            encoder = None
            movie = args.output/'release.mp4';partial.rename(movie)
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=nb_read_frames,avg_frame_rate,duration','-of','json',str(movie)],text=True))['streams'][0]
            if int(probe['nb_read_frames']) != stop-first or probe['avg_frame_rate'] != '30/1':
                raise ValueError('Encoded preview frame/rate mismatch')
            sheet = Image.new('RGB',(640*len(snapshots),480))
            for i,im in enumerate(snapshots): sheet.paste(im,(640*i,0))
            sheet.save(args.output/'review.jpg')
            atomic_json(args.output/'preview.json', {'scope': __doc__, 'training_ready': False,
                'source': str(args.source.resolve()), 'source_sha256': sha256(args.source),
                'script_sha256': sha256(__file__), 'source_provenance_changes': changed,
                'frame_range_exclusive': [first,stop], 'frames': stop-first, 'fps': fps,
                'first_last_source_sim_time': [float(times[first]),float(times[stop-1])],
                'duration_s': (stop-first)/fps, 'duration_note': 'Each displayed sample, including the last, occupies one frame interval.',
                'video_sha256': sha256(movie), 'ffprobe': probe,
                'render_note': 'Diagnostic overview camera; sensor noise/postprocessing are not replayed.'})
            print(json.dumps({'output':str(movie),'frames':stop-first,'duration_s':(stop-first)/fps}))
        finally:
            if encoder is not None:
                if encoder.stdin and not encoder.stdin.closed: encoder.stdin.close()
                try: encoder.wait(timeout=10)
                except subprocess.TimeoutExpired: encoder.kill();encoder.wait(timeout=10)
            backend.close()


if __name__ == '__main__': main()
