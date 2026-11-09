"""Lossless compaction of closed transition recordings using HDF5 virtual datasets.

obs[i+1].RGB and next_obs[i].RGB share storage after exact continuity checks.
All public datasets retain their names, shapes, dtypes and pixel values.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from .recording_json import CompressedJsonStreams


def compress_episode_json(source, destination, progress=None):
    """Copy a closed episode with lossless JSON compression; never replace source.

    Every source dataset is compared after copying. Old acceptance is cleared;
    the copy requires its own closed-file audit. Interrupted copies stay partial.
    """
    from .collection_jobs import sha256
    source, destination = Path(source), Path(destination)
    partial = destination.with_name(destination.name+'.partial')
    if destination.exists() or partial.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    with h5py.File(source, 'r') as src, h5py.File(partial, 'x', libver='latest') as dst:
        dst.attrs['complete'] = False
        if not src.attrs.get('complete') or src.attrs.get('schema_version') != 'piper.transitions.v1':
            raise ValueError('A closed complete transition recording is required')
        if src.attrs.get('extras_storage') == 'compressed_utf8_buckets_v1' or 'recording_extras' in src:
            raise ValueError('Already compressed or reserved JSON storage exists')
        n = int(src.attrs['num_transitions'])
        if n < 1: raise ValueError('Empty episode')
        datasets = []
        def inspect(name, item):
            if isinstance(item, h5py.Dataset):
                if item.shape[0] != n: raise ValueError('Inconsistent length: '+name)
                if item.is_virtual and any(s.file_name != '.' for s in item.virtual_sources()):
                    raise ValueError('External virtual sources are not self-contained: '+name)
                datasets.append(name)
        src.visititems(inspect)
        for prefix in ('observation', 'next_observation'):
            ds = src[prefix+'/extras_json']
            info = h5py.check_string_dtype(ds.dtype)
            if ds.shape != (n,) or info is None or info.encoding != 'utf-8' or info.length is not None:
                raise ValueError('Expected variable-length UTF-8 JSON stream')
        for key, value in src.attrs.items():
            if key not in ('complete', 'accepted', 'training_ready', 'dataset_audit', 'dataset_audit_sidecar'):
                dst.attrs[key] = value
        dst.attrs['accepted'] = dst.attrs['training_ready'] = False
        for name in src:
            if name not in ('observation', 'next_observation'):
                src.copy(name, dst)
            else:
                group = dst.create_group(name)
                for key, value in src[name].attrs.items(): group.attrs[key] = value
                for field in src[name]:
                    if field != 'extras_json': src.copy(src[name+'/'+field], group, name=field)
        streams = CompressedJsonStreams(dst)
        json_bytes = 0
        for prefix in ('observation', 'next_observation'):
            name = prefix+'/extras_json'
            for start in range(0, n, 128):
                for i, raw in enumerate(src[name][start:start+128], start):
                    streams.append(name, raw, i);json_bytes += len(raw)
            if progress: progress(name)
        streams.finalize(n)
        for prefix in ('observation', 'next_observation'):
            name = prefix+'/extras_json'
            for key, value in src[name].attrs.items(): dst[name].attrs[key] = value
        meta = json.loads(dst.attrs['metadata'])
        meta.setdefault('collection_outcome', {}).update(accepted=False, training_ready=False)
        meta['json_storage_derivation'] = {'source_sha256': source_hash, 'source': str(source.resolve()),
            'method': 'compressed_utf8_buckets_v1', 'implementation_sha256': sha256(__file__),
            'json_writer_sha256': sha256(Path(__file__).with_name('recording_json.py'))}
        dst.attrs['metadata'] = json.dumps(meta)
        dst.flush()
        for name in datasets:
            before, after = src[name], dst[name]
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError('Dataset shape/dtype changed: '+name)
            step = 8 if before.ndim > 2 else 128
            for start in range(0, n, step):
                a, b = before[start:start+step], after[start:start+step]
                equal = np.array_equal(a, b) if a.dtype.kind == 'O' else a.tobytes() == b.tobytes()
                if not equal: raise ValueError('Dataset changed: '+name)
        if sha256(source) != source_hash: raise ValueError('Source changed during copy')
        dst.attrs['complete'] = True
    partial.rename(destination)
    return {'source': str(source.resolve()), 'destination': str(destination.resolve()),
        'source_sha256': source_hash, 'destination_sha256': sha256(destination),
        'frames': n, 'source_bytes': source.stat().st_size, 'output_bytes': destination.stat().st_size,
        'uncompressed_json_bytes': json_bytes, 'verified_datasets': datasets,
        'verification': 'All source dataset values, shapes and dtypes compared; source unchanged.',
        'training_ready': False}


def compact_episode(source, destination, progress=None):
    source, destination = Path(source), Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.partial')
    if partial.exists(): raise FileExistsError(partial)
    try:
        with h5py.File(source, 'r') as src, h5py.File(partial, 'x', libver='latest') as dst:
            if not src.attrs.get('complete') or src.attrs.get('schema_version') != 'piper.transitions.v1':
                raise ValueError('A closed complete transition recording is required')
            if src.attrs.get('image_storage') == 'shared_transition_frames_v1':
                raise ValueError('Already compacted')
            n = int(src.attrs['num_transitions'])
            if n < 1: raise ValueError('Empty episode')
            names = sorted(src.get('observation/images', {}))
            if not names or names != sorted(src.get('next_observation/images', {})):
                raise ValueError('Matching RGB streams are required')
            for key, value in src.attrs.items(): dst.attrs[key] = value
            dst.attrs['complete'] = False
            dst.attrs['accepted'] = dst.attrs['training_ready'] = False
            for name in src:
                if name not in ('observation', 'next_observation'):
                    src.copy(name, dst)
                    continue
                group = dst.create_group(name)
                for key, value in src[name].attrs.items(): group.attrs[key] = value
                for field in src[name]:
                    if field != 'images': src.copy(src[name+'/'+field], group, name=field)
                group.create_group('images')
                for key, value in src[name+'/images'].attrs.items():
                    group['images'].attrs[key] = value
            for name in names:
                before, after = (src[p+'/images/'+name] for p in ('observation', 'next_observation'))
                if (before.shape != after.shape or before.shape[0] != n or before.ndim != 4
                        or before.dtype != np.uint8 or after.dtype != before.dtype
                        or before.chunks != (1, *before.shape[1:]) or after.chunks != before.chunks
                        or before.compression != after.compression or before.is_virtual or after.is_virtual
                        or before.compression_opts != after.compression_opts
                        or before.shuffle or after.shuffle or before.fletcher32 or after.fletcher32
                        or before.scaleoffset is not None or after.scaleoffset is not None):
                    raise ValueError('Expected matching, frame-chunked uint8 RGB datasets')
                kwargs = dict(shape=before.shape, dtype=before.dtype, chunks=before.chunks,
                              compression=before.compression, compression_opts=before.compression_opts)
                shared = dst.create_dataset('observation/images/'+name, **kwargs)
                for key, value in before.attrs.items(): shared.attrs[key] = value
                terminal = dst.create_dataset('recording_terminal_images/'+name, **kwargs)
                last = None
                for i in range(n):
                    current = before.id.read_direct_chunk((i, 0, 0, 0))
                    if last is not None and last != current:
                        if not np.array_equal(after[i-1], before[i]):
                            raise ValueError('Noncontiguous RGB frames: '+name)
                    shared.id.write_direct_chunk((i, 0, 0, 0), current[1], filter_mask=current[0])
                    last = after.id.read_direct_chunk((i, 0, 0, 0))
                terminal.id.write_direct_chunk((n-1, 0, 0, 0), last[1], filter_mask=last[0])
                layout = h5py.VirtualLayout(shape=before.shape, dtype=before.dtype)
                # '.' refers to this HDF5 file, including after it is moved.
                shared_source = h5py.VirtualSource('.', shared.name, shape=shared.shape)
                final_source = h5py.VirtualSource('.', terminal.name, shape=terminal.shape)
                if n > 1: layout[:-1] = shared_source[1:]
                layout[-1] = final_source[-1]
                virtual = dst.create_virtual_dataset('next_observation/images/'+name, layout)
                for key, value in after.attrs.items(): virtual.attrs[key] = value
                if progress: progress(name)
            dst.attrs['image_storage'] = 'shared_transition_frames_v1'
            # A compacted file needs a fresh checksum before training export.
            dst.attrs['accepted'] = dst.attrs['training_ready'] = False
            for key in ('dataset_audit', 'dataset_audit_sidecar'):
                if key in dst.attrs: del dst.attrs[key]
            meta = json.loads(dst.attrs['metadata'])
            meta.setdefault('collection_outcome', {}).update(accepted=False, training_ready=False)
            dst.attrs['metadata'] = json.dumps(meta)
            dst.attrs['complete'] = True
        partial.rename(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {'frames': n, 'cameras': names, 'lossless': True,
            'source_bytes': source.stat().st_size, 'output_bytes': destination.stat().st_size,
            'verified': 'Every RGB chunk copied exactly; every shared frame checked for byte/pixel equality'}
