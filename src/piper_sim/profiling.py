"""Opt-in, bounded-memory wall-time profiling; never changes simulation clocks.

Inclusive spans nest. Exclusive times can be summed; inclusive times cannot.
GPU samples describe the entire device (other applications may be running).
"""
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time


class PipelineProfiler:
    def __init__(self, *, sample_interval_s=2., sample_gpu=True):
        self.stages = {}
        self._stack = []
        self._patches = []
        self._start = time.perf_counter()
        self._cpu = time.process_time()
        self._stop = threading.Event()
        self._interval = sample_interval_s
        if sample_interval_s <= 0:
            raise ValueError('sample_interval_s must be positive')
        self._gpu = sample_gpu
        self.resources = {'samples': 0, 'peak_rss_bytes': 0, 'cpu_percent_one_core': [],
                          'gpus': {}, 'gpu_sampling_errors': 0}
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    @contextmanager
    def stage(self, name):
        frame = [time.perf_counter(), 0.]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter()-frame[0]
            self._stack.pop()
            if self._stack:
                self._stack[-1][1] += elapsed
            item = self.stages.setdefault(name, {'calls': 0, 'inclusive_s': 0., 'exclusive_s': 0.,
                                                  'max_call_s': 0.})
            item['calls'] += 1
            item['inclusive_s'] += elapsed
            item['exclusive_s'] += elapsed-frame[1]
            item['max_call_s'] = max(item['max_call_s'], elapsed)

    def wrap(self, owner, attribute, label):
        original = getattr(owner, attribute)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            name = label(*args, **kwargs) if callable(label) else label
            with self.stage(name):
                return original(*args, **kwargs)
        setattr(owner, attribute, measured)
        self._patches.append((owner, attribute, original))

    def _monitor(self):
        last_cpu, last_wall = time.process_time(), time.perf_counter()
        cpu_sum, cpu_peak, cpu_n = 0., 0., 0
        while not self._stop.wait(self._interval):
            now_cpu, now_wall = time.process_time(), time.perf_counter()
            value = 100*(now_cpu-last_cpu)/(now_wall-last_wall)
            cpu_sum += value; cpu_peak = max(cpu_peak, value); cpu_n += 1
            last_cpu, last_wall = now_cpu, now_wall
            self.resources['cpu_percent_one_core'] = {'mean': cpu_sum/cpu_n, 'peak': cpu_peak}
            self.resources['samples'] += 1
            self.resources['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            if not self._gpu:
                continue
            try:
                output = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=uuid,name,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=2)
                for line in output.splitlines():
                    uuid, name, utilization, memory = [x.strip() for x in line.split(',')]
                    gpu = self.resources['gpus'].setdefault(uuid, {'name': name, 'samples': 0,
                        'utilization_sum_percent': 0., 'peak_utilization_percent': 0., 'peak_used_bytes': 0})
                    gpu['samples'] += 1
                    gpu['utilization_sum_percent'] += float(utilization)
                    gpu['peak_utilization_percent'] = max(gpu['peak_utilization_percent'], float(utilization))
                    gpu['peak_used_bytes'] = max(gpu['peak_used_bytes'], int(float(memory)*1024**2))
            except (OSError, ValueError, subprocess.SubprocessError):
                self.resources['gpu_sampling_errors'] += 1

    def close(self):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()
        wall, cpu = time.perf_counter()-self._start, time.process_time()-self._cpu
        self._stop.set()
        self._thread.join(timeout=3)
        result = {'schema_version': 'piper.pipeline_profile.v1', 'pid': os.getpid(),
                  'wall_seconds': wall, 'process_cpu_seconds': cpu,
                  'mean_cpu_percent_one_core': 100*cpu/wall,
                  'stages': self.stages, 'resources': self.resources,
                  'scope': 'Single synchronous process; GPU totals include unrelated applications. '
                           'Exclusive stage time excludes nested measured calls; unwrapped work remains unattributed. '
                           'RSS is process lifetime high-water mark. Rendering includes GPU synchronization/readback.'}
        result['resources']['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result['unattributed_wall_seconds'] = wall-sum(v['exclusive_s'] for v in self.stages.values())
        return result


def instrument_collection(profiler):
    """Attach timings only in an explicitly profiled process, restoring on close."""
    import h5py
    import mujoco
    from .backends.mujoco import MujocoBackend
    from .experts.mujoco_ik import PiperIK
    from .experts import clean_table
    from . import collection_audit, lerobot_export, randomization
    from .recording import EpisodeWriter
    for owner, attribute, label in (
        (MujocoBackend, '__init__', 'scene_load'),
        (MujocoBackend, 'step', 'backend_step'),
        (MujocoBackend, 'apply', 'control_apply'),
        (MujocoBackend, 'read', 'observation_read'),
        (MujocoBackend, '_render', 'camera_render'),
        (mujoco, 'mj_step', 'physics_integration'),
        (mujoco, 'mj_forward', 'physics_forward'),
        (mujoco.Renderer, 'update_scene', 'render_scene_update'),
        (mujoco.Renderer, 'render', 'render_and_readback'),
        (randomization, 'sensor_images', 'sensor_postprocess'),
        (PiperIK, 'solve', 'expert_ik'),
        (clean_table.CleanTableTeacher, 'check_joint_path', 'expert_collision_plan'),
        (clean_table, 'path_trajectory', 'expert_time_parameterization'),
        (clean_table, 'smooth_trajectory', 'expert_time_parameterization'),
        (EpisodeWriter, 'append', 'recording_append'),
        (EpisodeWriter, '_append', lambda self, name, value, **kwargs: 'rgb_compress_write'
            if '/images/' in name or name.startswith('recording_terminal_images/') else 'metadata_write'),
        (EpisodeWriter, 'close', 'recording_close'),
        (h5py.File, 'flush', 'hdf5_flush'),
        (collection_audit, 'audit_episode', 'audit_total'),
        (collection_audit, 'audit_streams', 'audit_streams'),
        (collection_audit, 'audit_task', 'audit_task'),
        (collection_audit, 'save_audit', 'audit_save_and_checksum'),
        (lerobot_export, 'export_lerobot_v2', 'lerobot_export'),
        (lerobot_export, '_inspect', 'lerobot_input_validation'),
        (lerobot_export, '_video', 'lerobot_video_and_statistics'),
    ):
        profiler.wrap(owner, attribute, label)


def save_profile(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
