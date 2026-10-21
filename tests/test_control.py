import json
import time
import urllib.request
import urllib.error
import numpy as np
import pytest
from piper_sim.control import CommandServer
from piper_sim.contracts import Observation


def test_loopback_ingress_and_staleness():
    server = CommandServer(0)
    base = f"http://127.0.0.1:{server.server.server_port}"
    q = [0, .8, -.6, 0, 0, 0, .04]*2
    try:
        server.publish(Observation(q, time.monotonic()))
        assert json.load(urllib.request.urlopen(base+"/state"))["position"] == q
        def post(sequence):
            request = urllib.request.Request(base+"/target", data=json.dumps(
                {"position": q, "sequence": sequence}).encode())
            return urllib.request.urlopen(request)
        with post(0) as response:
            assert response.status == 204
        assert np.allclose(server.sample().position, q)
        for sequence in (0, -1, 1.5, True):
            with pytest.raises(urllib.error.HTTPError) as error:
                post(sequence)
            assert error.value.code == 400
        time.sleep(.02)
        with pytest.raises(TimeoutError):
            server.sample(max_age=.01)
    finally:
        server.close()


def test_read_only_real_connection_does_not_accept_targets():
    server = CommandServer(0, allow_commands=False)
    base = f"http://127.0.0.1:{server.server.server_port}"
    q = [0, .8, -.6, 0, 0, 0, .04]*2
    try:
        server.publish(Observation(q, time.monotonic()))
        assert json.load(urllib.request.urlopen(base+"/state"))["position"] == q
        request = urllib.request.Request(base+"/target", data=json.dumps(
            {"position": q, "sequence": 0}).encode())
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        assert server.sample() is None
    finally:
        server.close()
