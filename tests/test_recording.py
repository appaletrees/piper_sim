import h5py
import numpy as np
import pytest
from piper_sim.contracts import Command, Observation
from piper_sim.recording import EpisodeWriter
from piper_sim.dataset import validate_episode, sequence_windows


def obs(t):
    return Observation(np.zeros(14), t, t, {"overhead": np.zeros((8, 8, 3), np.uint8)},
                       {"overhead": t})


def test_transitions_images_and_windows(tmp_path):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path, {"task": "pick"}) as writer:
        for i in range(4):
            writer.append(obs(i), Command(np.zeros(14), i+.1, i), obs(i+1))
    info = validate_episode(path)
    assert info["transitions"] == 4
    windows = list(sequence_windows(path, horizon=3))
    assert len(windows) == 2
    assert windows[0]["images"]["overhead"].shape == (3, 8, 8, 3)
    with h5py.File(path) as f:
        assert f["next_observation/timestamp"][0] == 1
        assert not f.attrs["success"]
    with pytest.raises(FileExistsError):
        EpisodeWriter(path, {})


def test_failed_episode_is_not_published(tmp_path):
    path = tmp_path / "broken.h5"
    with pytest.raises(RuntimeError):
        with EpisodeWriter(path, {}) as writer:
            writer.append(obs(0), Command(np.zeros(14), 0, 1), obs(1))
            raise RuntimeError("interrupted")
    assert not path.exists()
    with pytest.raises(ValueError, match="incomplete"):
        validate_episode(str(path)+".partial")


def test_camera_change_rejected(tmp_path):
    with EpisodeWriter(tmp_path / "episode.h5", {}) as writer:
        writer.append(obs(0), Command(np.zeros(14), 0, 1), obs(1))
        with pytest.raises(ValueError, match="Camera"):
            writer.append(Observation(np.zeros(14), 2), Command(np.zeros(14), 2, 2), obs(3))
