# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

from types import SimpleNamespace

import numpy as np

from cielostitch_core.core.matcher import FeatureMatcher


def test_describe_pair_evidence_reports_descriptor_missing():
    matcher = FeatureMatcher(min_inliers=8, min_inlier_ratio=0.15)

    metrics = matcher.describe_pair_evidence({
        "has_descriptors": False,
        "min_kp": 0,
        "match_count": 0,
        "inlier_count": 0,
        "transform_found": False,
    })

    assert metrics["diagnostic_label"] == "descriptor_missing"
    assert metrics["failure_stage"] == "descriptors"


def test_describe_pair_evidence_reports_sparse_features():
    matcher = FeatureMatcher(min_inliers=8, min_inlier_ratio=0.15)

    metrics = matcher.describe_pair_evidence({
        "has_descriptors": True,
        "min_kp": 12,
        "match_count": 14,
        "inlier_count": 0,
        "transform_found": False,
    })

    assert metrics["diagnostic_label"] == "sparse_features"
    assert metrics["failure_stage"] == "detection"


def test_describe_pair_evidence_reports_ambiguous_geometry_failure():
    matcher = FeatureMatcher(min_inliers=8, min_inlier_ratio=0.15)

    metrics = matcher.describe_pair_evidence({
        "has_descriptors": True,
        "min_kp": 140,
        "match_count": 17,
        "inlier_count": 0,
        "transform_found": False,
    })

    assert metrics["diagnostic_label"] == "ambiguous_or_inconsistent"
    assert metrics["failure_stage"] == "geometry"


def test_describe_pair_evidence_reports_insufficient_geometric_support():
    matcher = FeatureMatcher(min_inliers=8, min_inlier_ratio=0.15)

    metrics = matcher.describe_pair_evidence({
        "has_descriptors": True,
        "min_kp": 140,
        "match_count": 10,
        "inlier_count": 0,
        "transform_found": False,
    })

    assert metrics["diagnostic_label"] == "insufficient_geometric_support"
    assert metrics["failure_stage"] == "geometry"


def test_describe_pair_evidence_reports_strong_pair():
    matcher = FeatureMatcher(min_inliers=8, min_inlier_ratio=0.15)

    metrics = matcher.describe_pair_evidence({
        "has_descriptors": True,
        "min_kp": 140,
        "match_count": 24,
        "inlier_count": 12,
        "transform_found": True,
    })

    assert metrics["diagnostic_label"] == "strong_pair"
    assert metrics["failure_stage"] == "none"
    assert metrics["inlier_ratio"] == 0.5


def test_compute_transform_returns_flat_bool_inlier_mask(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    def _fake_estimate_affine_partial_2d(*_args, **_kwargs):
        affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]], dtype=np.float64)
        inliers = np.array([[1], [1], [0], [1], [1], [1]], dtype=np.uint8)
        return affine, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.estimateAffinePartial2D",
        _fake_estimate_affine_partial_2d,
    )

    kp1 = [SimpleNamespace(pt=(float(i + 5), float(i + 7))) for i in range(6)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(6)]

    h_locked, inliers_locked = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        allow_rotation=False,
    )
    h_rot, inliers_rot = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        allow_rotation=True,
    )

    assert h_locked is not None
    assert h_rot is not None
    assert inliers_locked is not None
    assert inliers_rot is not None
    assert inliers_locked.dtype == np.bool_
    assert inliers_rot.dtype == np.bool_
    assert inliers_locked.ndim == 1
    assert inliers_rot.ndim == 1
    assert inliers_locked.shape == (6,)
    assert inliers_rot.shape == (6,)


def test_compute_transform_lock_rotation_skips_affine_and_accepts_translation(monkeypatch):
    matcher = FeatureMatcher(min_inliers=4, min_inlier_ratio=0.5, ransac_thresh=1.5)

    def _fail_if_affine_called(*_args, **_kwargs):
        raise AssertionError("Affine estimation should not run when rotation is locked")

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.estimateAffinePartial2D",
        _fail_if_affine_called,
    )

    pts2 = [
        (0.0, 0.0),
        (10.0, 5.0),
        (20.0, 12.0),
        (30.0, 18.0),
        (40.0, 23.0),
        (50.0, 29.0),
    ]
    translation = np.array([5.0, 7.0])
    pts1 = [tuple(np.array(point) + translation) for point in pts2]

    # One outlier should be rejected by the translation consensus step.
    pts1[-1] = (67.0, 48.0)

    kp1 = [SimpleNamespace(pt=point) for point in pts1]
    kp2 = [SimpleNamespace(pt=point) for point in pts2]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(len(pts1))]

    h_locked, inliers_locked = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        allow_rotation=False,
    )

    assert h_locked is not None
    assert inliers_locked is not None
    assert inliers_locked.dtype == np.bool_
    assert inliers_locked.shape == (6,)
    assert int(np.count_nonzero(inliers_locked)) == 5
    np.testing.assert_allclose(h_locked[0, 2], 5.0)
    np.testing.assert_allclose(h_locked[1, 2], 7.0)


def test_match_returns_empty_when_train_descriptors_less_than_knn_k():
    matcher = FeatureMatcher()

    des_query = np.ones((4, 128), dtype=np.float32)
    des_train = np.ones((1, 128), dtype=np.float32)

    matches = matcher.match([], des_query, [], des_train)

    assert matches == []


def test_match_returns_empty_for_non_2d_descriptors():
    matcher = FeatureMatcher()

    des_query = np.ones((128,), dtype=np.float32)
    des_train = np.ones((3, 128), dtype=np.float32)

    matches = matcher.match([], des_query, [], des_train)

    assert matches == []


def test_compute_transform_lock_rotation_handles_large_match_sets_with_bounded_consensus():
    matcher = FeatureMatcher(min_inliers=20, min_inlier_ratio=0.5, ransac_thresh=0.6)

    base_count = 1200
    translation = np.array([3.25, -4.75], dtype=np.float32)

    pts2 = [(float(i % 50), float(i // 50)) for i in range(base_count)]
    pts1 = [tuple(np.array(point, dtype=np.float32) + translation) for point in pts2]

    # Inject outliers while keeping a strong inlier majority.
    for i in range(0, base_count, 10):
        pts1[i] = (pts1[i][0] + 30.0, pts1[i][1] - 20.0)

    kp1 = [SimpleNamespace(pt=point) for point in pts1]
    kp2 = [SimpleNamespace(pt=point) for point in pts2]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(base_count)]

    h_locked, inliers_locked = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        allow_rotation=False,
    )

    assert h_locked is not None
    assert inliers_locked is not None
    assert inliers_locked.dtype == np.bool_
    assert inliers_locked.shape == (base_count,)
    assert int(np.count_nonzero(inliers_locked)) >= 1000
    np.testing.assert_allclose(h_locked[0, 2], translation[0], atol=0.2)
    np.testing.assert_allclose(h_locked[1, 2], translation[1], atol=0.2)
