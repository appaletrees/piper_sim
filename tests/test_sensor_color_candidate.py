import importlib.util
from pathlib import Path

import numpy as np
import pytest

from piper_sim.randomization import sensor_images

spec = importlib.util.spec_from_file_location('colour_experiment',
    Path(__file__).resolve().parents[1]/'scripts/benchmark_sensor_color_lut.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('method', ['broadcast', 'channels'])
@pytest.mark.parametrize('gamma', [.5, .83, 1., 1.17, 2.2])
@pytest.mark.parametrize('noise,blur', [(0.,0.), (.7,0.), (1.3,.4)])
def test_fused_lookup_preserves_all_rgb_levels_and_noise(method, gamma, noise, blur):
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    images = {'overhead': np.stack([levels,np.flipud(levels),np.fliplr(levels)], axis=-1)}
    images['left_wrist'] = images['overhead'][::-1, ::-1]
    images['right_wrist'] = images['overhead'].copy()
    parameters = {'seed': 7, 'cameras': {name: {'gamma': gamma, 'gain':1.09,
        'white_balance': [.92,1.03,1.14], 'noise_std_uint8':noise, 'blur_sigma_px':blur} for name in images}}
    unchanged = {k:v.copy() for k,v in images.items()}
    expected = sensor_images(images, parameters, 19)
    actual = module.candidate(images, parameters, 19, method)
    for name in images:
        np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(images[name], unchanged[name])
