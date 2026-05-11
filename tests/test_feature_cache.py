# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.core.detector import FeatureDetector
from cielostitch_core.core.feature_cache import (
    clear_global_feature_cache,
    detect_with_cache,
    get_feature_cache_stats,
)


def _make_test_image(seed: int = 123, shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 255).astype(np.uint8)


def test_feature_cache_reuses_detection_for_same_image():
    detector = FeatureDetector(downscale_factor=0.5, max_features=500, feature_sensitivity=0.01)
    image = _make_test_image()

    clear_global_feature_cache()
    first = detect_with_cache(detector, image, "frame-001.fit", True)
    second = detect_with_cache(detector, image, "frame-001.fit", True)
    stats = get_feature_cache_stats()

    kp1, des1, scale1 = first
    kp2, des2, scale2 = second
    assert len(kp1) == len(kp2)
    assert scale1 == scale2
    assert des1 is not None
    assert des2 is not None
    assert des1.shape == des2.shape
    assert int(stats["requests"]) == 2
    assert int(stats["hits"]) == 1
    assert int(stats["entries"]) == 1


def test_feature_cache_invalidates_when_detector_settings_change():
    image = _make_test_image(seed=456)
    detector_a = FeatureDetector(downscale_factor=0.5, max_features=500, feature_sensitivity=0.01)
    detector_b = FeatureDetector(downscale_factor=1.0, max_features=500, feature_sensitivity=0.01)

    clear_global_feature_cache()
    detect_with_cache(detector_a, image, "frame-002.fit", True)
    detect_with_cache(detector_b, image, "frame-002.fit", True)
    stats = get_feature_cache_stats()

    assert int(stats["requests"]) == 2
    assert int(stats["hits"]) == 0
    assert int(stats["misses"]) == 2
    assert int(stats["entries"]) == 2
