import numpy as np
import pytest
from cielostitch_core.stitching.illumination import IlluminationNormalizer


def test_compute_linear_correction_grayscale():
    canvas = np.ones((10, 10), dtype=np.uint8) * 100
    warped = np.ones((10, 10), dtype=np.uint8) * 120
    overlap_mask = np.ones((10, 10), dtype=bool)
    gain, offset = IlluminationNormalizer.compute_linear_correction(canvas, warped, overlap_mask)
    assert isinstance(gain, float)
    assert isinstance(offset, float)


def test_compute_linear_correction_rgb():
    canvas = np.ones((10, 10, 3), dtype=np.uint8) * 100
    warped = np.ones((10, 10, 3), dtype=np.uint8) * 120
    overlap_mask = np.ones((10, 10), dtype=bool)
    gain, offset = IlluminationNormalizer.compute_linear_correction(canvas, warped, overlap_mask)
    assert isinstance(gain, float)
    assert isinstance(offset, float)


def test_apply_linear_correction_grayscale():
    image = np.ones((10, 10), dtype=np.uint8) * 100
    out = IlluminationNormalizer.apply_linear_correction(image, 1.1, 5)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_apply_linear_correction_rgb():
    image = np.ones((10, 10, 3), dtype=np.uint8) * 100
    out = IlluminationNormalizer.apply_linear_correction(image, 0.9, -10)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_apply_linear_correction_rgba():
    image = np.ones((10, 10, 4), dtype=np.uint8) * 100
    image[..., 3] = 255  # alpha channel
    out = IlluminationNormalizer.apply_linear_correction(image, 1.2, 10)
    # Alpha should remain unchanged
    assert np.all(out[..., 3] == 255)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_apply_linear_correction_float32():
    image = np.ones((10, 10), dtype=np.float32) * 100.0
    out = IlluminationNormalizer.apply_linear_correction(image, 0.5, -20.0)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_compute_linear_correction_with_nan_inf():
    canvas = np.ones((10, 10), dtype=np.float32) * 100
    warped = np.ones((10, 10), dtype=np.float32) * 120
    warped[0, 0] = np.nan
    warped[1, 1] = np.inf
    overlap_mask = np.ones((10, 10), dtype=bool)
    gain, offset = IlluminationNormalizer.compute_linear_correction(canvas, warped, overlap_mask)
    assert isinstance(gain, float)
    assert isinstance(offset, float)


def test_flatten_low_frequency_grayscale():
    image = np.ones((20, 20), dtype=np.uint8) * 100
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    out = IlluminationNormalizer.flatten_low_frequency(image, mask)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_flatten_low_frequency_rgb():
    image = np.ones((20, 20, 3), dtype=np.uint8) * 100
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    out = IlluminationNormalizer.flatten_low_frequency(image, mask)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_flatten_low_frequency_empty_mask():
    image = np.ones((10, 10), dtype=np.uint8) * 100
    mask = np.zeros((10, 10), dtype=bool)
    out = IlluminationNormalizer.flatten_low_frequency(image, mask)
    assert np.all(out == image)


def test_flatten_low_frequency_full_mask():
    image = np.ones((10, 10), dtype=np.uint8) * 100
    mask = np.ones((10, 10), dtype=bool)
    out = IlluminationNormalizer.flatten_low_frequency(image, mask)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_flatten_low_frequency_single_pixel():
    image = np.ones((10, 10), dtype=np.uint8) * 100
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    out = IlluminationNormalizer.flatten_low_frequency(image, mask)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_compute_linear_correction_low_overlap():
    canvas = np.ones((10, 10), dtype=np.uint8) * 100
    warped = np.ones((10, 10), dtype=np.uint8) * 120
    overlap_mask = np.zeros((10, 10), dtype=bool)
    gain, offset = IlluminationNormalizer.compute_linear_correction(canvas, warped, overlap_mask)
    assert gain == 1.0
    assert offset == 0.0


def test_apply_linear_correction_extreme_gain_offset():
    image = np.ones((10, 10), dtype=np.uint8) * 100
    out = IlluminationNormalizer.apply_linear_correction(image, 10.0, 1000.0)
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    # Should be clipped to dtype max
    assert np.all(out <= 255)


def test_compute_linear_correction_mismatched_shapes():
    canvas = np.ones((10, 10), dtype=np.uint8)
    warped = np.ones((8, 10), dtype=np.uint8)
    overlap_mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(Exception):
        IlluminationNormalizer.compute_linear_correction(canvas, warped, overlap_mask)
