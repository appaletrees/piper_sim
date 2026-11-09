"""Optimizations must preserve the original sensor pixels, not just statistics."""
import numpy as np
import pytest
from PIL import Image, ImageFilter

from piper_sim.randomization import sensor_images, stream_seed


@pytest.mark.parametrize('gamma', [.5, .83, 1., 1.17, 2.2])
@pytest.mark.parametrize('noise,blur', [(0., 0.), (1.4, 0.), (2.1, .6)])
def test_sensor_matches_original_expression_all_levels(gamma, noise, blur):
    # All uint8 levels, channel permutations, a non-contiguous input and
    # repeated same-tick reads exercise exact rounding and RNG stream identity.
    levels = np.arange(256, dtype=np.uint8)
    pixels = np.tile(np.column_stack((levels, levels[::-1], np.roll(levels, 37))), (17, 1, 1))[:, ::-1]
    original = pixels.copy()
    p = {'gamma': gamma, 'gain': 1.09, 'white_balance': [.92, 1.03, 1.14],
         'noise_std_uint8': noise, 'blur_sigma_px': blur}
    params = {'seed': 53, 'cameras': {'overhead': p}}
    raw = np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(blur))) if blur else pixels
    expected = 255.*(raw.astype(np.float32)/255.)**gamma
    expected *= p['gain']*np.asarray(p['white_balance'], np.float32)
    if noise:
        rng = np.random.default_rng(stream_seed(53, 'sensor/overhead/19'))
        expected += rng.normal(0, noise, expected.shape).astype(np.float32)
    expected = np.clip(np.rint(expected), 0, 255).astype(np.uint8)
    for _ in range(2):
        result = sensor_images({'overhead': pixels}, params, 19)['overhead']
        np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(pixels, original)
