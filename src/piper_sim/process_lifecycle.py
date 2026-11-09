"""Supervise only process groups explicitly created by this invocation."""
from contextlib import contextmanager
import os
import signal
import subprocess


class CollectionTerminated(SystemExit):
    def __init__(self, signum):
        super().__init__(128+signum)
        self.signum = signum

    def __str__(self):
        return signal.Signals(self.signum).name


@contextmanager
def termination_guard():
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise CollectionTerminated(signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def stop_process_groups(processes):
    """Also reap descendants after an abnormal leader exit; leave other jobs alone."""
    owned = [p for p in processes if p.poll() not in (0, 2)]
    for process in owned:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    for process in owned:
        if process.poll() is None:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: pass
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
