import importlib.util
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from piper_sim.config import ROOT

spec = importlib.util.spec_from_file_location('video_audit', ROOT/'scripts/audit_exported_video.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


@pytest.fixture
def movie(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('Video tools unavailable')
    frames = np.zeros((8, 32, 48, 3), dtype=np.uint8)
    for i in range(len(frames)):
        frames[i] = [20+i*20, 180-i*15, 40+i*10]
        frames[i, :, i*4:i*4+8] = [230, 20, 120]
    path = tmp_path/'movie.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '48x32', '-r', '30', '-i', 'pipe:0', '-c:v', 'libx264', '-crf', '18',
        '-pix_fmt', 'yuv420p', str(path)], input=frames.tobytes(), check=True)
    return path, frames


def test_every_decoded_frame_is_compared_at_its_expected_index(movie):
    path, frames = movie
    result, errors = cli.inspect_video(path, frames, 30.)
    assert result['frames'] == 8 and result['max_pts_error_s'] < 1e-6
    assert errors.shape == (8, 3) and np.isfinite(errors).all()
    assert result['mean_absolute_pixel_error_0_to_255'] < 5
    assert result['psnr_db'] > 25
    # Frame count and timestamps alone cannot discover a wrong image order.
    # The independent indexed pixel comparison must expose this corruption.
    reordered, _ = cli.inspect_video(path, frames[::-1], 30.)
    assert reordered['mean_absolute_pixel_error_0_to_255'] > 10*result['mean_absolute_pixel_error_0_to_255']


@pytest.mark.parametrize('fault', ['short_source', 'wrong_fps', 'wrong_shape'])
def test_frame_count_timestamp_and_dimensions_cannot_silently_change(movie, fault):
    path, frames = movie
    fps = 30.
    if fault == 'short_source': frames = frames[:-1]
    elif fault == 'wrong_fps': fps = 15.
    else: frames = frames[:, :-2]
    with pytest.raises(ValueError): cli.inspect_video(path, frames, fps)
