"""Lossless UTF-8 JSON buckets with the original variable-string read interface.

HDF5 filters on variable-length strings do not compress their heap payloads.
Fixed-string sources make the payload compressible; a self-contained virtual
dataset converts them back to variable UTF-8 strings at the public paths.
"""
import h5py
import numpy as np


class CompressedJsonStreams:
    def __init__(self, file):
        if 'recording_extras' in file:
            raise ValueError('Reserved recording_extras group already exists')
        self.file = file
        self.streams = {}

    def append(self, name, text, index):
        if name not in ('observation/extras_json', 'next_observation/extras_json'):
            raise ValueError('Unsupported JSON stream: '+name)
        raw = text.encode('utf-8') if isinstance(text, str) else bytes(text)
        raw.decode('utf-8')  # reject invalid UTF-8 before creating a dataset
        if b'\0' in raw:
            raise ValueError('JSON strings cannot contain unescaped null bytes')
        stream = self.streams.setdefault(name, {'count': 0, 'buckets': {}, 'segments': []})
        if index != stream['count']:
            raise ValueError('Noncontiguous JSON stream: '+name)
        # Grow by actual byte length. Never cast into a smaller fixed string:
        # NumPy/HDF5 would silently truncate it. Capacity changes are lossless.
        capacity = max(256, 1 << max(0, len(raw)-1).bit_length())
        path = 'recording_extras/'+name.split('/')[0]+'/utf8_'+str(capacity)
        if capacity not in stream['buckets']:
            self.file.create_dataset(path, shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype('utf-8', length=capacity), chunks=(1,),
                compression='gzip', compression_opts=1)
            stream['buckets'][capacity] = 0
        source_index = stream['buckets'][capacity]
        ds = self.file[path]
        ds.resize(source_index+1, axis=0)
        ds[source_index] = np.asarray(raw, dtype=ds.dtype)
        segments = stream['segments']
        if (segments and segments[-1][3] == path and segments[-1][0]+segments[-1][2] == index
                and segments[-1][1]+segments[-1][2] == source_index):
            segments[-1][2] += 1
        else:
            segments.append([index, source_index, 1, path])
        stream['buckets'][capacity] += 1
        stream['count'] += 1

    def finalize(self, count):
        expected = {'observation/extras_json', 'next_observation/extras_json'}
        if set(self.streams) != expected or any(s['count'] != count for s in self.streams.values()):
            raise ValueError('Incomplete compressed JSON streams')
        for name, stream in self.streams.items():
            # Like terminal RGB storage, internal datasets retain logical N
            # length for existing schema validators. Unused rows allocate no
            # chunks and are never exposed by the public virtual mapping.
            for capacity in stream['buckets']:
                path = 'recording_extras/'+name.split('/')[0]+'/utf8_'+str(capacity)
                self.file[path].resize(count, axis=0)
            layout = h5py.VirtualLayout((count,), dtype=h5py.string_dtype('utf-8'))
            for index, source_index, length, path in stream['segments']:
                source = h5py.VirtualSource('.', path, shape=self.file[path].shape)
                layout[index:index+length] = source[source_index:source_index+length]
            self.file.create_virtual_dataset(name, layout)
        self.file.attrs['extras_storage'] = 'compressed_utf8_buckets_v1'
