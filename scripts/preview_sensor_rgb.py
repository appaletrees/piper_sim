#!/usr/bin/env python3
"""Preview a phase of an accepted episode using its original three RGB streams.

All selected 30 Hz samples appear once at 1x. This review montage is not an
exported training episode; the complete independently accepted HDF5 is retained.
Source pixels are copied without resizing, with labels outside the image area.
"""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image, ImageDraw

from piper_sim.collection_jobs import atomic_json, sha256
from piper_sim.training_acceptance import verify_training_acceptance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--object',default='plate')
    parser.add_argument('--policy',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory')
    cameras=('overhead','left_wrist','right_wrist')
    with h5py.File(args.source,'r') as f:
        audit,digest=verify_training_acceptance(f,args.source,policy=args.policy)
        if not audit['training_ready']:raise ValueError('Accepted three-camera recording required')
        metadata=json.loads(f.attrs['metadata']);fps=float(metadata['fps'])
        events=metadata['collection_outcome']['teacher_events']
        selected=[e for e in events if e['object']==args.object]
        begin=next(e['sim_time'] for e in selected if e['phase']=='grasp')
        end=next(e['sim_time'] for e in selected if e['phase']=='placed')
        times=f['observation/sim_time'][:]
        if fps!=30 or not np.allclose(np.diff(times),1/fps,rtol=0,atol=1e-6):
            raise ValueError('Expected contiguous 30 Hz samples; no resampling permitted')
        first=int(np.searchsorted(times,begin));stop=int(np.searchsorted(times,end,side='right'))
        streams=[f['observation/images/'+name] for name in cameras]
        height,width,channels=streams[0].shape[1:]
        if (not first<stop<=len(times) or channels!=3
                or any(s.shape!=(len(times),height,width,3) or s.dtype!=np.uint8 for s in streams)):
            raise ValueError('Invalid frame interval or matching RGB streams')
        args.output.mkdir(parents=True)
        movie=args.output/'sensor_rgb.partial.mp4'
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width*3}x{height+32}',
            '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for i in range(first,stop):
                panel=Image.new('RGB',(width*3,height+32),(15,20,28));draw=ImageDraw.Draw(panel)
                for j,(name,stream) in enumerate(zip(cameras,streams)):
                    panel.paste(Image.fromarray(stream[i]),(j*width,32))
                    draw.text((j*width+8,5),f'SENSOR RGB | {name} | 1x / 30 Hz | t={times[i]:.3f}s',fill='white')
                process.stdin.write(np.asarray(panel).tobytes())
                if i==(first+stop)//2:panel.save(args.output/'review.jpg')
            process.stdin.close()
            if process.wait(timeout=30):raise RuntimeError('Video encoding failed')
        finally:
            if not process.stdin.closed:process.stdin.close()
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        target=args.output/'sensor_rgb.mp4';movie.rename(target)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=nb_read_frames,avg_frame_rate,duration',
            '-of','json',str(target)],text=True))['streams'][0]
        if int(probe['nb_read_frames'])!=stop-first or probe['avg_frame_rate']!='30/1':
            raise ValueError('Encoded frame count or rate differs from selected source')
    if sha256(args.source)!=digest:raise ValueError('Source recording changed')
    atomic_json(args.output/'preview.json',{'scope':__doc__,'training_ready':False,
        'source':str(args.source.resolve()),'source_sha256':digest,'source_unchanged':True,
        'source_training_ready':True,'acceptance_policy':audit['acceptance_policy'],
        'frame_range_exclusive':[first,stop],'frames':stop-first,'fps':fps,
        'source_sim_time_first_last_s':[float(times[first]),float(times[stop-1])],
        'cameras':list(cameras),'video_sha256':sha256(target),'ffprobe':probe,
        'script_sha256':sha256(__file__),'source_pixels_resized':False})
    print(json.dumps({'video':str(target),'frames':stop-first,'duration_s':(stop-first)/fps}))


if __name__=='__main__':main()
