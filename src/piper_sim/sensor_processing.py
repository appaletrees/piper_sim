"""Owned, bounded CPU workers for synchronous camera postprocessing."""
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

from . import randomization


class SensorImageProcessor:
    """At most one frame in flight per owner, with up to three camera workers.

    Rendering and simulation stay on the calling thread. Workers hold no random
    state or previous images. The owning backend must call close().
    """
    def __init__(self, workers=1):
        if type(workers) is not int or not 1 <= workers <= 3:
            raise ValueError('sensor_postprocess_workers must be an integer from 1 to 3')
        self.workers = workers
        self._pool = None
        self._closed = False
        self._lock = RLock()

    def __call__(self, images, parameters, tick):
        with self._lock:
            if self._closed:
                raise RuntimeError('Sensor image processor is closed')
            if self._pool is None and self.workers > 1 and parameters and len(images) > 1:
                self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='piper-sensor')
            return randomization.sensor_images(images, parameters, tick, executor=self._pool)

    def close(self):
        with self._lock:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None
