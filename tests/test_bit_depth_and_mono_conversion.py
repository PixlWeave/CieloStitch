# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.utils.image_manipulation import (
    ensure_mono_2d,
    from_internal_float32,
    integer_content_bit_depth,
)


def test_ensure_mono_2d_preserves_float_precision_for_color_input():
    # Float input in [0,1] should not be quantized to integer-like values.
    img = np.zeros((2, 2, 3), dtype=np.float32)
    img[..., 0] = 0.2  # B
    img[..., 1] = 0.4  # G
    img[..., 2] = 0.6  # R

    mono = ensure_mono_2d(img, "float_color")

    assert mono.dtype == np.float32
    expected = 0.114 * 0.2 + 0.587 * 0.4 + 0.299 * 0.6
    assert np.allclose(mono, expected, atol=1e-6)


def test_integer_content_bit_depth_signed_int_uses_absolute_range():
    # Large negative magnitudes should not be misclassified as low-bit depth.
    img = np.array([[-32768, -10], [-1, 0]], dtype=np.int16)
    depth = integer_content_bit_depth(img, "signed")
    assert depth == 16


def test_from_internal_float32_supports_32bit_float_output():
    img = np.array([[-0.5, 0.25], [1.2, 0.75]], dtype=np.float32)

    out = from_internal_float32(img, 32)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([[0.0, 0.25], [1.0, 0.75]], dtype=np.float32))
