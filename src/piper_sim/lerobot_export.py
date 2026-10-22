"""Offline piper.transitions.v1 -> LeRobot Dataset v2.1; no simulator imports.

Each row is obs_t + the applied action_t. Video frame i is obs_i, never obs_i+1.
Source HDF5 remains authoritative; auxiliary files retain task/calibration data.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import tempfile

import h5py
import numpy as np
from PIL import Image

from .contracts import FEATURE_NAMES
from .lerobot_alignment import CAMERA_MAPPING, CONNECTOR_PLUGIN_NAMES, CONNECTOR_PLUGIN_GRIPPER_SCALE_M

DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO_PATH = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'


def check_dependencies():
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install export dependencies: pip install -e '.[lerobot-export]'") from error
    for name in ('ffmpeg', 'ffprobe'):
        if shutil.which(name) is None: raise RuntimeError(f'{name} is required for LeRobot video export')


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for value in values: f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def collect_episodes(inputs):
    if isinstance(inputs, (str, Path)): inputs = [inputs]
    paths = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir(): paths.extend(sorted(path.glob('*.h5')))
        elif path.is_file() and path.suffix == '.h5': paths.append(path)
        else: raise ValueError(f'Expected a completed .h5 episode or directory: {path}')
    if not paths: raise ValueError('No completed .h5 episodes found')
    if len(set(paths)) != len(paths): raise ValueError('Duplicate input episodes')
    return paths


def _inspect(path):
    with h5py.File(path, 'r') as f:
        if f.attrs.get('schema_version') != 'piper.transitions.v1' or not f.attrs.get('complete', False):
            raise ValueError(f'Not a complete piper.transitions.v1 episode: {path}')
        if tuple(json.loads(f.attrs['features'])) != FEATURE_NAMES or json.loads(f.attrs['units']) != ['rad']*6+['m']+['rad']*6+['m']:
            raise ValueError(f'Unexpected input feature order/units: {path}')
        meta = json.loads(f.attrs['metadata'])
        fps = float(meta['fps'])
        if not math.isfinite(fps) or fps <= 0: raise ValueError('Invalid episode fps')
        n = int(f.attrs['num_transitions'])
        if n < 1: raise ValueError('Empty episode')
        cameras = sorted(f.get('observation/images', {}))
        if not set(cameras).issubset(CAMERA_MAPPING.values()): raise ValueError('Unknown camera names')
        for prefix in ('observation', 'next_observation'):
            if sorted(f.get(prefix+'/images', {})) != cameras: raise ValueError('Camera set changed')
            q = f[prefix+'/state'][:]
            if q.shape != (n,14) or not np.isfinite(q).all(): raise ValueError('Invalid state array')
            ts = f[prefix+'/timestamp'][:]
            if ts.shape != (n,) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                raise ValueError('Invalid source observation timestamps')
        actions = f['action/position'][:]
        if actions.shape != (n,14) or not np.isfinite(actions).all(): raise ValueError('Invalid actions')
        before, after = f['observation/sim_time'][:], f['next_observation/sim_time'][:]
        if (before.shape != (n,) or after.shape != (n,) or not np.isfinite([before,after]).all()
                or not np.allclose(after-before, 1/fps, atol=1e-6, rtol=1e-5)
                or not np.allclose(np.diff(before), 1/fps, atol=1e-6, rtol=1e-5)):
            raise ValueError('Simulation timestamps must have one continuous step per frame; split resets/gaps first')
        shapes = {}
        for camera in cameras:
            pixels = f['observation/images/'+camera]
            if pixels.ndim != 4 or pixels.shape[0] != n or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
                raise ValueError('Expected NxHxWx3 RGB uint8')
            shapes[camera] = tuple(pixels.shape[1:])
            if f['next_observation/images/'+camera].shape != pixels.shape: raise ValueError('Camera shape changed')
            for prefix, stamps in (('observation',before), ('next_observation',after)):
                captured = f[prefix+'/image_timestamps/'+camera][:]
                if captured.shape != (n,) or not np.allclose(captured, stamps, atol=1e-6, rtol=0):
                    raise ValueError('Camera timestamp does not match the corresponding state')
        return dict(path=path, length=n, fps=fps, metadata=meta, shapes=shapes,
                    success=bool(f.attrs.get('success',False)))


def _agilex(values):
    q = np.asarray(values, dtype=np.float64).copy()
    q[:, np.r_[0:6,7:13]] = np.rad2deg(q[:, np.r_[0:6,7:13]])
    q[:, [6,13]] /= CONNECTOR_PLUGIN_GRIPPER_SCALE_M
    return q.astype(np.float32)  # Preserve physical solver overshoot; no observation clipping.


def _stats(array):
    x = np.asarray(array, dtype=np.float64)
    if x.ndim == 1: x = x[:,None]
    return {**{k: fn(x, axis=0).tolist() for k,fn in
               [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]}, 'count':[len(x)]}


def _image_transform(shape, image_size):
    h,w,_ = shape
    width,height = image_size or (w,h)
    if width % 2 or height % 2: raise ValueError('H.264 yuv420p needs even width and height')
    scale = min(width/w,height/h)
    rw,rh = max(1,round(w*scale)),max(1,round(h*scale))
    left,top = (width-rw)//2,(height-rh)//2
    # Integer pixel centres: resized u=(u+.5)*scale-.5+padding.
    matrix = [[rw/w,0,left+(rw/w-1)/2], [0,rh/h,top+(rh/h-1)/2], [0,0,1]]
    return {'source_size':[w,h], 'output_size':[width,height], 'resized_size':[rw,rh],
            'pad_left_top':[left,top], 'pixel_transform':matrix, 'mode':'letterbox' if image_size else 'native'}


def _resize(pixels, transform):
    if transform['source_size'] == transform['output_size']: return pixels
    out = Image.new('RGB', tuple(transform['output_size']), (0,0,0))
    out.paste(Image.fromarray(pixels).resize(tuple(transform['resized_size']), Image.Resampling.BILINEAR),
              tuple(transform['pad_left_top']))
    return np.asarray(out)


def _video(dataset, destination, fps, transform):
    destination.parent.mkdir(parents=True,exist_ok=True)
    width,height = transform['output_size']
    command = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
               '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
               '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
               '-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',
               '-movflags','+faststart',str(destination)]
    total = np.zeros(3); squares = np.zeros(3)
    low = np.full(3,255.); high = np.zeros(3); count = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for frame in dataset:
                pixels = _resize(frame, transform)
                process.stdin.write(pixels.tobytes())
                flat = pixels.reshape(-1,3).astype(np.float64)/255
                total += flat.sum(0); squares += np.einsum('ij,ij->j',flat,flat)
                low = np.minimum(low,flat.min(0)); high = np.maximum(high,flat.max(0)); count += len(flat)
            process.stdin.close()
            if process.wait(timeout=120):
                errors.seek(0); raise RuntimeError('ffmpeg failed: '+errors.read().decode(errors='replace'))
        except BaseException:
            process.kill(); process.wait()
            raise
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=nb_read_frames,width,height,codec_name,pix_fmt',
        '-of','json',str(destination)]))['streams'][0]
    if int(probe['nb_read_frames']) != len(dataset) or (probe['width'],probe['height']) != (width,height):
        raise RuntimeError('Encoded video frame count/shape mismatch')
    mean = total/count
    statistics = {k: v.reshape(3,1,1).tolist() for k,v in
                  [('min',low),('max',high),('mean',mean),('std',np.sqrt(np.maximum(squares/count-mean**2,0)))]}
    statistics['count'] = [len(dataset)]
    return statistics, probe


def _aggregate(items):
    result = {}
    for key in items[0]:
        counts = np.array([s[key]['count'][0] for s in items],dtype=float)
        weights = counts/counts.sum()
        means = np.array([s[key]['mean'] for s in items])
        shape = (len(items),)+ (1,)*(means.ndim-1)
        mean = (means*weights.reshape(shape)).sum(0)
        variance = ((np.array([s[key]['std'] for s in items])**2+(means-mean)**2)*weights.reshape(shape)).sum(0)
        result[key] = {'min':np.min([s[key]['min'] for s in items],axis=0).tolist(),
                       'max':np.max([s[key]['max'] for s in items],axis=0).tolist(),
                       'mean':mean.tolist(),'std':np.sqrt(variance).tolist(),'count':[int(counts.sum())]}
    return result


def export_lerobot_v2(inputs, output, *, image_size=None, progress=None):
    """Create a NEW local v2.1 dataset from completed H5 episodes; never upload.

    image_size=(width,height) optionally letterboxes RGB and records adjusted K.
    This is an offline operation: call after collection, outside a real-time loop.
    """
    check_dependencies()
    import pyarrow as pa
    import pyarrow.parquet as pq
    if image_size is not None:
        if len(image_size)!=2 or any(type(x) is not int or x<=0 for x in image_size):
            raise ValueError('image_size must be two positive integers')
    output = Path(output).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    episodes = [_inspect(p) for p in collect_episodes(inputs)]
    first = episodes[0]
    transforms = {name:_image_transform(shape,image_size) for name,shape in first['shapes'].items()}
    for ep in episodes:
        if ep['shapes'] != first['shapes'] or ep['fps'] != first['fps']:
            raise ValueError('All episodes must use identical source camera shapes and fps')
    fps = first['fps']
    features = {k:{'dtype':'float32','shape':[14],'names':list(CONNECTOR_PLUGIN_NAMES)}
                for k in ('observation.state','action')}
    for key,dtype in [('timestamp','float32'),('source_timestamp','float64'),('sim_timestamp','float64'),
                      *[(k,'int64') for k in ('frame_index','episode_index','index','task_index')]]:
        features[key] = {'dtype':dtype,'shape':[1],'names':None}
    camera_keys = {name:'observation.images.'+legacy for legacy,name in CAMERA_MAPPING.items() if name in transforms}
    for name,key in camera_keys.items():
        w,h = transforms[name]['output_size']
        features[key] = {'dtype':'video','shape':[3,h,w],'names':['channels','height','width'],
            'info':{'video.height':h,'video.width':w,'video.codec':'h264','video.pix_fmt':'yuv420p',
                    'video.is_depth_map':False,'video.fps':fps,'video.channels':3,'has_audio':False}}
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.partial-',dir=output.parent))
    task_names=[]; records=[]; all_stats=[]; provenance=[]; total=0
    try:
        for index,ep in enumerate(episodes):
            if progress: progress(f'Exporting episode {index+1}/{len(episodes)}: {ep["path"].name}')
            chunk=index//1000; n=ep['length']; task=str(ep['metadata'].get('task','dual Piper demonstration'))
            if task not in task_names: task_names.append(task)
            with h5py.File(ep['path'],'r') as source:
                columns={'observation.state':_agilex(source['observation/state'][:]),
                         'action':_agilex(source['action/position'][:]),
                         'timestamp':(np.arange(n)/fps).astype(np.float32),
                         'source_timestamp':source['observation/timestamp'][:].astype(np.float64),
                         'sim_timestamp':source['observation/sim_time'][:].astype(np.float64),
                         'frame_index':np.arange(n,dtype=np.int64), 'index':np.arange(total,total+n,dtype=np.int64),
                         'episode_index':np.full(n,index,dtype=np.int64),
                         'task_index':np.full(n,task_names.index(task),dtype=np.int64)}
                arrays={k:pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)),14) if v.ndim==2 else pa.array(v)
                        for k,v in columns.items()}
                path=staging/DATA_PATH.format(episode_chunk=chunk,episode_index=index)
                path.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.table(arrays),path,compression='snappy')
                statistics={k:_stats(v) for k,v in columns.items()}
                for name,key in camera_keys.items():
                    path=staging/VIDEO_PATH.format(episode_chunk=chunk,episode_index=index,video_key=key)
                    statistics[key],_= _video(source['observation/images/'+name],path,fps,transforms[name])
                def extra_frames():
                    for frame in range(n):
                        extras=json.loads(source['observation/extras_json'][frame])
                        calibration={name:{**pose,'intrinsics':(np.array(transforms[name]['pixel_transform'])@
                                     np.array(pose['intrinsics'])).tolist()}
                                     for name,pose in extras.get('cameras',{}).items() if name in transforms}
                        yield {'frame_index':frame,'source_timestamp':float(columns['source_timestamp'][frame]),
                               'sim_timestamp':float(columns['sim_timestamp'][frame]),
                               'state_si':source['observation/state'][frame].tolist(),
                               'action_si':source['action/position'][frame].tolist(),
                               'next_state_si':source['next_observation/state'][frame].tolist(),
                               'observation_extras':extras,'output_camera_calibration':calibration,
                               'next_observation_extras':json.loads(source['next_observation/extras_json'][frame]),
                               'image_timestamps':{k:float(source['observation/image_timestamps/'+k][frame]) for k in transforms}}
                extras_path=f'extras/chunk-{chunk:03d}/episode_{index:06d}.jsonl'
                _jsonl(staging/extras_path,extra_frames())
            records.append({'episode_index':index,'tasks':[task],'length':n})
            all_stats.append(statistics)
            provenance.append({'episode_index':index,'source_h5':str(ep['path']), 'source_sha256':_sha256(ep['path']),
                               'success':ep['success'],'metadata':ep['metadata'],'image_transforms':transforms,
                               'frame_extras_path':extras_path})
            total+=n
        info={'codebase_version':'v2.1','robot_type':'piper','total_episodes':len(episodes),'total_frames':total,
              'total_tasks':len(task_names),'total_videos':len(episodes)*len(camera_keys),
              'total_chunks':math.ceil(len(episodes)/1000),'chunks_size':1000,'fps':fps,
              'splits':{'train':f'0:{len(episodes)}'},'data_path':DATA_PATH,'video_path':VIDEO_PATH,'features':features}
        _json(staging/'meta/info.json',info)
        _jsonl(staging/'meta/episodes.jsonl',records)
        _jsonl(staging/'meta/tasks.jsonl',({'task_index':i,'task':t} for i,t in enumerate(task_names)))
        _jsonl(staging/'meta/episodes_stats.jsonl',({'episode_index':i,'stats':s} for i,s in enumerate(all_stats)))
        _json(staging/'meta/stats.json',_aggregate(all_stats))
        _jsonl(staging/'meta/sim_episodes.jsonl',provenance)
        report={'format':'LeRobot Dataset v2.1','output':str(output),'episodes':len(episodes),'frames':total,
                'videos':info['total_videos'],'fps':fps,'units':'joint degrees; gripper width / 0.095 m',
                'action_semantics':'applied absolute joint position targets, not next-state or desired user targets',
                'timestamp_clock':'episode simulation time, frame_index/fps',
                'source_timestamp_clock':'source host monotonic; not Unix time',
                'sim_timestamp_clock':'original episode simulation clock',
                'image_statistics':'all pre-encoding output RGB pixels, normalized 0..1',
                'video_validation':'all encoded streams decoded with ffprobe; frame counts and shapes match',
                'image_transforms':transforms,'source_h5_preserved':True,'uploaded':False}
        _json(staging/'meta/export.json',report)
        if output.exists(): raise FileExistsError(output)
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise
