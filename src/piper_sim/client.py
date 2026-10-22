"""External Python client; only HTTP/numpy/Pillow, no simulator or CAN imports."""
from io import BytesIO
import json
import time
import urllib.request

import numpy as np


class SimulationClient:
    def __init__(self, url="http://127.0.0.1:8765", timeout=5.):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self._sequence = None

    def _request(self, path, body=None):
        request = urllib.request.Request(self.url+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"} if body is not None else {})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read(), dict(response.headers)

    def metadata(self):
        return json.loads(self._request('/metadata')[0])

    def read(self):
        """Same Observation contract as the in-process and real-arm connections."""
        from .contracts import Observation
        bundle = self.observation()
        s = bundle['state']
        return Observation(s['position'], s['timestamp'], s['sim_time'], bundle['images'],
                           s['image_timestamps'], s['extras'])

    def state(self):
        return json.loads(self._request('/state')[0])

    def observation(self):
        """An atomic frame: SI state, RGB, camera calibration/poses, task objects/metrics."""
        data, _ = self._request('/observation.npz')
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            state = json.loads(str(archive['state_json']))
            images = {key[4:]: archive[key].copy() for key in archive.files if key.startswith('rgb_')}
        return {"state": state, "images": images}

    def image(self, camera):
        """Returns (RGB uint8 array, frame_id); use observation() for synchronized cameras."""
        if camera not in ('left_wrist', 'right_wrist', 'overhead'):
            raise ValueError('Unknown camera')
        from PIL import Image
        data, headers = self._request('/cameras/'+camera+'.png')
        return np.asarray(Image.open(BytesIO(data)).convert('RGB')), int(headers['X-Observation-Id'])

    def send_target(self, position, sequence=None):
        """204 means queued. Check state()['control'] for application/rejection."""
        if sequence is None:
            if self._sequence is None:
                self._sequence = self.state()['last_received_sequence']+1
            sequence = self._sequence
        self._request('/target', {"position": np.asarray(position).tolist(), "sequence": sequence})
        self._sequence = int(sequence)+1
        return sequence

    def reset(self, seed=None, randomize=None):
        """Queue a simulation-thread reset and wait for its published acknowledgement."""
        result = json.loads(self._request('/reset', {"seed": seed, "randomize": randomize})[0])
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            state = self.state()
            if state['reset_id'] >= result['reset_id']: return state
            time.sleep(.01)
        raise TimeoutError('Reset has not been applied; inspect the simulator process')

    def recording_status(self):
        return json.loads(self._request('/recording')[0])

    def _recording_request(self, operation, **body):
        result = json.loads(self._request('/recording/'+operation, body)[0])
        deadline = time.monotonic()+self.timeout
        while time.monotonic() < deadline:
            status = self.recording_status()
            if status['request_id'] >= result['request_id']:
                if status['error']: raise RuntimeError(status['error'])
                return status
            time.sleep(.01)
        raise TimeoutError('Recording lifecycle request has not completed')

    def start_recording(self, name, task='dual Piper demonstration'):
        return self._recording_request('start', name=name, task=task)

    def stop_recording(self, success=False):
        return self._recording_request('stop', success=success)
