# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.stitching.local_gain_normalizer import LocalGainNormalizer


def test_resize_image_supports_float16_without_opencv_assertion():
    img = np.linspace(0.0, 1.0, 32 * 24 * 3, dtype=np.float16).reshape((24, 32, 3))

    out = LocalGainNormalizer._resize_image(img, 16, 12)

    assert out.shape == (12, 16, 3)
    assert out.dtype == np.float16


def test_resize_image_supports_more_than_four_channels():
    rng = np.random.default_rng(123)
    img = rng.random((18, 22, 5), dtype=np.float32)

    out = LocalGainNormalizer._resize_image(img, 11, 9)

    assert out.shape == (9, 11, 5)
    assert out.dtype == np.float32


def test_resize_image_bool_roundtrips_as_bool_mask():
    img = np.zeros((20, 30), dtype=np.bool_)
    img[4:16, 8:24] = True

    out = LocalGainNormalizer._resize_image(img, 15, 10)

    assert out.shape == (10, 15)
    assert out.dtype == np.bool_
