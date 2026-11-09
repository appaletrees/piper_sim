import numpy as np
import pytest

from piper_sim.lerobot_export import RGBStatistics


def test_uniform_and_binary_channels_have_exact_moments():
    stats = RGBStatistics()
    a = np.empty((8, 8, 3), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 51, 0, 255
    b = a.copy()
    b[..., 1] = 255
    stats.add(a[:, ::-1])
    stats.add(b)
    result = stats.result()
    assert result['count'] == [2]
    np.testing.assert_allclose(np.array(result['mean']).ravel(), [.2, .5, 1.], atol=1e-15)
    np.testing.assert_allclose(np.array(result['std']).ravel(), [0., .5, 0.], atol=1e-15)
    np.testing.assert_array_equal(np.array(result['min']).ravel(), [.2, 0., 1.])
    np.testing.assert_array_equal(np.array(result['max']).ravel(), [.2, 1., 1.])


def test_all_levels_match_discrete_uniform_distribution():
    stats = RGBStatistics()
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    stats.add(np.stack([levels, levels[::-1], levels.T], axis=-1))
    result = stats.result()
    np.testing.assert_allclose(np.array(result['mean']).ravel(), .5, atol=1e-15)
    np.testing.assert_allclose(np.array(result['std']).ravel(), np.sqrt((256**2-1)/12)/255., atol=1e-15)


def test_empty_and_invalid_input_are_rejected():
    stats = RGBStatistics()
    with pytest.raises(ValueError, match='empty'):
        stats.result()
    for pixels in (np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 4), np.uint8)):
        with pytest.raises(ValueError, match='uint8 RGB'):
            stats.add(pixels)
