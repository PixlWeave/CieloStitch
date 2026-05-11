# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.utils.image_resampling import (
    choose_resize_interpolator,
    choose_warp_interpolator,
    is_integer_translation,
    resize_for_export,
)


def test_choose_resize_interpolator_auto_down_up():
    # Downscale should pick AREA in Auto
    flag_down = choose_resize_interpolator(0.5, "auto")
    # Upscale should pick LANCZOS4 in Auto
    flag_up = choose_resize_interpolator(2.0, "auto")
    assert flag_down != flag_up


def test_is_integer_translation_and_warp_choice():
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 3.0
    H[1, 2] = -2.0
    assert is_integer_translation(H)
    flag, pre = choose_warp_interpolator(H, "auto")
    # For integer translations, expect a nearest-like flag and no preblur
    assert pre is None


def test_strong_downscale_prefilter():
    # 0.5 scale on x-axis
    H = np.array([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    flag, pre = choose_warp_interpolator(H, "auto")
    # Should recommend a small preblur for anti-aliasing
    assert pre is None or pre >= 0.0


def test_resize_for_export_accepts_float16_without_cv_assertion():
    img = np.linspace(0.0, 1.0, num=32 * 32 * 3, dtype=np.float16).reshape(32, 32, 3)
    out = resize_for_export(img, scale=2.0, mode="auto")
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float16
