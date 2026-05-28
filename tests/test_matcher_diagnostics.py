# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import pytest
from types import SimpleNamespace

import cv2
import numpy as np

from cielostitch_core.core.apap import APAPBootstrapResult, APAPConfig, APAPEstimator, APAPRegistrationResult
from cielostitch_core.core.matcher import FeatureMatcher
from cielostitch_core.stitching.base_mode import BaseStitchMode


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
        transform_mode="translation",
    )
    h_rot, inliers_rot = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
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


def test_compute_transform_homography_caps_match_count_before_ransac(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    captured = {}

    def _fake_find_homography(src_pts, dst_pts, **_kwargs):
        captured["src_count"] = int(src_pts.shape[0])
        captured["src_x_values"] = src_pts[:, 0].copy()
        homography = np.eye(3, dtype=np.float64)
        inliers = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.findHomography",
        _fake_find_homography,
    )

    base_count = 1000
    kp1 = [SimpleNamespace(pt=(float(i), float(i + 1))) for i in range(base_count)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i + 1))) for i in range(base_count)]
    matches = [
        SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i))
        for i in range(base_count - 1, -1, -1)
    ]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="homography",
    )

    assert h is not None
    assert inliers is not None
    assert captured["src_count"] == matcher._MAX_HOMOGRAPHY_MATCHES
    np.testing.assert_array_equal(captured["src_x_values"], np.arange(matcher._MAX_HOMOGRAPHY_MATCHES, dtype=np.float32))


def test_compute_transform_homography_rejects_implausible_growth(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    def _fake_find_homography(*_args, **_kwargs):
        # 50x growth on both axes should be rejected by plausibility guard.
        h = np.array(
            [[50.0, 0.0, 0.0],
             [0.0, 50.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        inliers = np.ones((8, 1), dtype=np.uint8)
        return h, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.findHomography",
        _fake_find_homography,
    )

    pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 80.0),
        (0.0, 80.0),
        (20.0, 10.0),
        (80.0, 10.0),
        (80.0, 70.0),
        (20.0, 70.0),
    ]
    kp1 = [SimpleNamespace(pt=point) for point in pts]
    kp2 = [SimpleNamespace(pt=point) for point in pts]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(8)]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="homography",
    )

    assert h is None
    assert inliers is None
    assert matcher.last_transform_diagnostics["rejected_mode"] == "homography"
    assert matcher.last_transform_diagnostics["rejection_reason"] == "homography_implausible"


def test_compute_transform_homography_records_affine_fallback_reason(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0, homography_allow_affine_fallback=True)

    def _fake_find_homography(*_args, **_kwargs):
        h = np.array(
            [[50.0, 0.0, 0.0],
             [0.0, 50.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        inliers = np.ones((8, 1), dtype=np.uint8)
        return h, inliers

    affine = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0]], dtype=np.float64)
    affine_inliers = np.ones((8, 1), dtype=np.uint8)

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.findHomography",
        _fake_find_homography,
    )
    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.estimateAffinePartial2D",
        lambda *_args, **_kwargs: (affine, affine_inliers),
    )

    pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 80.0),
        (0.0, 80.0),
        (20.0, 10.0),
        (80.0, 10.0),
        (80.0, 70.0),
        (20.0, 70.0),
    ]
    kp1 = [SimpleNamespace(pt=point) for point in pts]
    kp2 = [SimpleNamespace(pt=point) for point in pts]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(8)]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="homography",
    )

    assert h is not None
    assert inliers is not None
    assert matcher.last_transform_diagnostics["requested_mode"] == "homography"
    assert matcher.last_transform_diagnostics["resolved_mode"] == "affine"
    assert matcher.last_transform_diagnostics["rejected_mode"] == "homography"
    assert matcher.last_transform_diagnostics["rejection_reason"] == "homography_implausible"
    assert matcher.last_transform_diagnostics["used_fallback"] is True
    assert matcher.last_transform_diagnostics["fallback_mode"] == "affine"


def test_compute_transform_apap_uses_projective_bootstrap_path(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    captured = {}

    def _fake_find_homography(src_pts, dst_pts, **_kwargs):
        captured["src_count"] = int(src_pts.shape[0])
        homography = np.eye(3, dtype=np.float64)
        inliers = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.findHomography",
        _fake_find_homography,
    )

    pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 80.0),
        (0.0, 80.0),
        (20.0, 10.0),
        (80.0, 10.0),
        (80.0, 70.0),
        (20.0, 70.0),
    ]
    kp1 = [SimpleNamespace(pt=point) for point in pts]
    kp2 = [SimpleNamespace(pt=point) for point in pts]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(8)]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="apap",
    )

    assert h is not None
    assert inliers is not None
    assert captured["src_count"] == len(matches)


def test_apap_estimator_bootstrap_reports_selected_match_count(monkeypatch):
    estimator = APAPEstimator()

    def _fake_find_homography(src_pts, dst_pts, **_kwargs):
        homography = np.eye(3, dtype=np.float64)
        inliers = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.apap.cv2.findHomography",
        _fake_find_homography,
    )

    pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 80.0),
        (0.0, 80.0),
        (20.0, 10.0),
        (80.0, 10.0),
        (80.0, 70.0),
        (20.0, 70.0),
    ]
    kp1 = [SimpleNamespace(pt=point) for point in pts]
    kp2 = [SimpleNamespace(pt=point) for point in pts]
    valid_pairs = [(i, i, float(i)) for i in range(len(pts))]

    result = estimator.compute_bootstrap(
        kp1,
        kp2,
        valid_pairs,
        scale1=1.0,
        scale2=1.0,
        ransac_thresh=5.0,
        max_matches=64,
    )

    assert result.homography is not None
    assert result.inlier_mask is not None
    assert result.diagnostics["bootstrap_mode"] == "global_homography"
    assert result.diagnostics["selected_match_count"] == len(valid_pairs)


def test_apap_estimator_builds_registration_result_from_bootstrap() -> None:
    estimator = APAPEstimator()
    bootstrap = APAPBootstrapResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.array([True, False, True], dtype=bool),
        selected_pairs=((0, 0, 0.0), (1, 1, 1.0), (2, 2, 2.0)),
        diagnostics={"stage": "bootstrap", "selected_match_count": 3},
    )

    result = estimator.build_registration_result(bootstrap)

    assert result.homography is not None
    assert result.inlier_mask is not None
    assert result.has_local_warp is False
    assert result.diagnostics["local_warp_ready"] is False
    assert result.diagnostics["kernel_sigma_px"] == estimator.config.kernel_sigma_px
    assert result.diagnostics["min_local_support"] == estimator.config.min_local_support
    assert result.diagnostics["mesh_cols"] == estimator.config.mesh_cols


def test_feature_matcher_compute_apap_registration_returns_registration_result(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    def _fake_estimate_registration(*_args, **_kwargs):
        return APAPRegistrationResult(
            homography=np.eye(3, dtype=np.float64),
            inlier_mask=np.ones(8, dtype=bool),
            diagnostics={"mesh_cols": 16, "mesh_rows": 16, "local_warp_ready": False},
        )

    monkeypatch.setattr(matcher.apap_estimator, "estimate_registration", _fake_estimate_registration)

    pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 80.0),
        (0.0, 80.0),
        (20.0, 10.0),
        (80.0, 10.0),
        (80.0, 70.0),
        (20.0, 70.0),
    ]
    kp1 = [SimpleNamespace(pt=point) for point in pts]
    kp2 = [SimpleNamespace(pt=point) for point in pts]
    valid_pairs = [(i, i, float(i)) for i in range(len(pts))]

    registration = matcher.compute_apap_registration(
        kp1,
        kp2,
        valid_pairs,
        scale1=1.0,
        scale2=1.0,
    )

    assert isinstance(registration, APAPRegistrationResult)
    assert registration.homography is not None
    assert registration.inlier_mask is not None


def test_apap_estimator_estimate_registration_records_support_points(monkeypatch):
    estimator_config = APAPConfig(min_local_support=4)
    estimator = APAPEstimator(config=estimator_config)

    def _fake_find_homography(_src_pts, _dst_pts, **_kwargs):
        homography = np.eye(3, dtype=np.float64)
        homography[0, 2] = -1.0
        inliers = np.ones((8, 1), dtype=np.uint8)
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.apap.cv2.findHomography",
        _fake_find_homography,
    )

    pts2 = [
        (0.0, 0.0),
        (10.0, 0.0),
        (20.0, 0.0),
        (30.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
        (20.0, 10.0),
        (30.0, 10.0),
    ]
    pts1 = [(x + 1.0, y) for x, y in pts2]
    kp1 = [SimpleNamespace(pt=point) for point in pts1]
    kp2 = [SimpleNamespace(pt=point) for point in pts2]
    valid_pairs = [(i, i, float(i)) for i in range(8)]

    registration = estimator.estimate_registration(
        kp1,
        kp2,
        valid_pairs,
        scale1=1.0,
        scale2=1.0,
        ransac_thresh=5.0,
        max_matches=64,
    )

    assert registration.support_source_points is not None
    assert registration.support_destination_points is not None
    assert registration.support_residuals is not None
    assert registration.diagnostics["support_point_count"] == 8
    assert registration.diagnostics["local_warp_ready"] is True
    assert registration.diagnostics["min_local_support"] == estimator_config.min_local_support
    assert registration.mesh_homographies is not None


def test_apap_real_world_pair_builds_mesh():
    import os

    img1_path = os.path.join(os.path.dirname(__file__), "data", "027.jpg")
    img2_path = os.path.join(os.path.dirname(__file__), "data", "028.jpg")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    assert img1 is not None and img2 is not None, f"Could not load test images: {img1_path}, {img2_path}"

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    assert des1 is not None and des2 is not None, "SIFT failed to compute descriptors"

    matcher = FeatureMatcher(
        min_inliers=8,
        min_inlier_ratio=0.15,
        apap_config=APAPConfig(mesh_cols=8, mesh_rows=8, min_local_support=8, kernel_sigma_px=64.0),
    )
    matches = matcher.match(kp1, des1, kp2, des2)
    assert len(matches) >= 16, f"Not enough matches for APAP: {len(matches)}"

    registration = matcher.compute_apap_registration(
        kp1,
        kp2,
        [(m.queryIdx, m.trainIdx, float(m.distance)) for m in matches],
        scale1=1.0,
        scale2=1.0,
    )

    assert registration.homography is not None, "APAP did not estimate a homography"
    assert registration.diagnostics.get("local_warp_ready", False), f"APAP did not enable local warping: {registration.diagnostics}"
    assert registration.mesh_homographies is not None, "APAP mesh was not built"


def test_apap_estimator_builds_local_homography_mesh_when_support_is_sufficient(monkeypatch):
    estimator = APAPEstimator(config=APAPConfig(mesh_cols=3, mesh_rows=3, min_local_support=4, kernel_sigma_px=32.0))

    def _fake_find_homography(src_pts, dst_pts, **_kwargs):
        homography = np.eye(3, dtype=np.float64)
        inliers = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.apap.cv2.findHomography",
        _fake_find_homography,
    )

    pts1 = [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (0.0, 20.0), (20.0, 20.0), (40.0, 20.0), (0.0, 40.0), (20.0, 40.0), (40.0, 40.0)]
    pts2 = pts1
    kp1 = [SimpleNamespace(pt=point) for point in pts1]
    kp2 = [SimpleNamespace(pt=point) for point in pts2]
    valid_pairs = [(i, i, float(i)) for i in range(len(pts1))]

    registration = estimator.estimate_registration(
        kp1,
        kp2,
        valid_pairs,
        scale1=1.0,
        scale2=1.0,
        ransac_thresh=5.0,
        max_matches=64,
    )

    assert registration.mesh_x is not None
    assert registration.mesh_y is not None
    assert registration.mesh_homographies is not None
    assert registration.diagnostics.get("mesh_valid_node_count", 0) >= 4


def test_apap_estimator_with_canvas_offset_shifts_mesh_coordinates() -> None:
    estimator = APAPEstimator()
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        mesh_x=np.array([0.0, 10.0], dtype=np.float64),
        mesh_y=np.array([5.0, 15.0], dtype=np.float64),
        mesh_homographies=np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], 4, axis=0).reshape(2, 2, 3, 3),
        diagnostics={"local_warp_ready": True},
    )

    shifted = estimator.with_canvas_offset(registration, 7.0, 11.0)

    np.testing.assert_allclose(shifted.homography, np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 11.0], [0.0, 0.0, 1.0]], dtype=np.float64))
    np.testing.assert_allclose(shifted.mesh_x, np.array([7.0, 17.0], dtype=np.float64))
    np.testing.assert_allclose(shifted.mesh_y, np.array([16.0, 26.0], dtype=np.float64))


def test_apap_estimator_with_canvas_offset_left_multiplies_projective_homography() -> None:
    estimator = APAPEstimator()
    registration = APAPRegistrationResult(
        homography=np.array(
            [[1.0, 0.0, 3.0], [0.0, 1.0, 5.0], [0.02, 0.01, 1.0]],
            dtype=np.float64,
        ),
        inlier_mask=np.ones(4, dtype=bool),
        diagnostics={"local_warp_ready": False},
    )

    shifted = estimator.with_canvas_offset(registration, 7.0, 11.0)

    expected = np.array(
        [[1.14, 0.07, 10.0], [0.22, 1.11, 16.0], [0.02, 0.01, 1.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(shifted.homography, expected)


def test_apap_estimator_compose_registration_updates_mesh_homographies() -> None:
    estimator = APAPEstimator()
    base_h = np.eye(3, dtype=np.float64)
    refine_h = np.eye(3, dtype=np.float64)
    refine_h[0, 2] = 2.5
    registration = APAPRegistrationResult(
        homography=base_h,
        inlier_mask=np.ones(4, dtype=bool),
        mesh_x=np.array([0.0, 1.0], dtype=np.float64),
        mesh_y=np.array([0.0, 1.0], dtype=np.float64),
        mesh_homographies=np.repeat(base_h[None, None, :, :], 4, axis=0).reshape(2, 2, 3, 3),
        diagnostics={"local_warp_ready": True},
    )

    composed = estimator.compose_registration(registration, refine_h)

    np.testing.assert_allclose(composed.homography, refine_h)
    np.testing.assert_allclose(composed.mesh_homographies[0, 0], refine_h)


def test_apap_estimator_compose_registration_shifts_mesh_axes_and_invalidates_dense_remap() -> None:
    estimator = APAPEstimator()
    refine_h = np.eye(3, dtype=np.float64)
    refine_h[0, 2] = 1.25
    refine_h[1, 2] = -0.5
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        remap_x=np.zeros((2, 2), dtype=np.float32),
        remap_y=np.zeros((2, 2), dtype=np.float32),
        mesh_x=np.array([0.0, 10.0], dtype=np.float64),
        mesh_y=np.array([5.0, 15.0], dtype=np.float64),
        mesh_homographies=np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], 4, axis=0).reshape(2, 2, 3, 3),
        diagnostics={"local_warp_ready": True},
    )

    composed = estimator.compose_registration(registration, refine_h)

    np.testing.assert_allclose(composed.mesh_x, np.array([1.25, 11.25], dtype=np.float64))
    np.testing.assert_allclose(composed.mesh_y, np.array([4.5, 14.5], dtype=np.float64))
    assert composed.remap_x is None
    assert composed.remap_y is None


def test_apap_estimator_compose_registration_uses_refine_transform_for_nonseparable_mesh_positions() -> None:
    estimator = APAPEstimator()
    refine_h = np.array(
        [[1.0, 0.2, 3.0], [0.1, 1.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        mesh_x=np.array([0.0, 10.0], dtype=np.float64),
        mesh_y=np.array([0.0, 20.0], dtype=np.float64),
        mesh_homographies=np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], 4, axis=0).reshape(2, 2, 3, 3),
        diagnostics={"local_warp_ready": True},
    )

    composed = estimator.compose_registration(registration, refine_h)

    assert composed.mesh_node_positions is not None
    expected_positions = np.array(
        [
            [[3.0, 4.0], [13.0, 5.0]],
            [[7.0, 24.0], [17.0, 25.0]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(composed.mesh_node_positions, expected_positions)
    np.testing.assert_allclose(composed.mesh_x, np.array([3.0, 17.0], dtype=np.float64))
    np.testing.assert_allclose(composed.mesh_y, np.array([4.0, 25.0], dtype=np.float64))


def test_base_mode_expand_bounds_with_apap_registration_uses_mesh_homographies() -> None:
    # Mesh nodes at canvas x=[0,20] with pad=20 would try to expand max_x to 40,
    # but the natural content bounds are (0,0,20,10) so the result must be clamped
    # to the natural bounds — no expansion beyond what compute_warp_bounds gives.
    bounds = BaseStitchMode._expand_bounds_with_apap_registration(
        np.zeros((10, 20), dtype=np.uint8),
        (0.0, 0.0, 20.0, 10.0),
        APAPRegistrationResult(
            homography=np.eye(3, dtype=np.float64),
            inlier_mask=np.ones(4, dtype=bool),
            mesh_x=np.array([0.0, 20.0], dtype=np.float64),
            mesh_y=np.array([0.0, 10.0], dtype=np.float64),
            mesh_homographies=np.array(
                [
                    [np.eye(3, dtype=np.float64), np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)],
                    [np.eye(3, dtype=np.float64), np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)],
                ],
                dtype=np.float64,
            ),
            diagnostics={"local_warp_ready": True},
        ),
    )

    # Expansion is clamped to the natural content bounds — never beyond.
    assert bounds[0] == pytest.approx(0.0)
    assert bounds[1] == pytest.approx(0.0)
    assert bounds[2] == pytest.approx(20.0)
    assert bounds[3] == pytest.approx(10.0)


def test_base_mode_expand_bounds_with_apap_registration_stays_within_natural_bounds() -> None:
    # Mesh nodes that fall strictly inside the natural bounds should NOT push the
    # result outside the natural bounds even with the ±pad cell margin.
    # Natural bounds: x=[0,100], y=[0,50].  Mesh x=[30,70] → pad=40 → tries
    # x=[−10,110] but must be clamped to [0,100].
    bounds = BaseStitchMode._expand_bounds_with_apap_registration(
        np.zeros((50, 100), dtype=np.uint8),
        (0.0, 0.0, 100.0, 50.0),
        APAPRegistrationResult(
            homography=np.eye(3, dtype=np.float64),
            inlier_mask=np.ones(4, dtype=bool),
            mesh_x=np.array([30.0, 70.0], dtype=np.float64),
            mesh_y=np.array([10.0, 40.0], dtype=np.float64),
            mesh_homographies=np.stack([
                [np.eye(3), np.eye(3)],
                [np.eye(3), np.eye(3)],
            ]).astype(np.float64),
            diagnostics={"local_warp_ready": True},
        ),
    )

    assert bounds[0] == pytest.approx(0.0)   # clamped, not -10
    assert bounds[1] == pytest.approx(0.0)   # clamped
    assert bounds[2] == pytest.approx(100.0) # clamped, not 110
    assert bounds[3] == pytest.approx(50.0)  # clamped


def test_apap_estimator_uses_only_capped_bootstrap_pairs_for_support(monkeypatch):
    estimator = APAPEstimator(config=APAPConfig(min_local_support=2))

    def _fake_find_homography(src_pts, _dst_pts, **_kwargs):
        homography = np.eye(3, dtype=np.float64)
        inliers = np.zeros((src_pts.shape[0], 1), dtype=np.uint8)
        inliers[:2] = 1
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.apap.cv2.findHomography",
        _fake_find_homography,
    )

    kp1 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(12)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(12)]
    valid_pairs = [(i, i, float(i)) for i in range(12)]

    registration = estimator.estimate_registration(
        kp1,
        kp2,
        valid_pairs,
        scale1=1.0,
        scale2=1.0,
        ransac_thresh=5.0,
        max_matches=4,
    )

    assert registration.diagnostics["selected_match_count"] == 4
    assert registration.diagnostics["support_point_count"] == 2
    assert registration.support_destination_points.shape[0] == 2


def test_apap_estimator_disables_local_warp_when_support_is_too_sparse(monkeypatch):
    estimator = APAPEstimator(config=APAPConfig(min_local_support=3))

    def _fake_find_homography(src_pts, _dst_pts, **_kwargs):
        homography = np.eye(3, dtype=np.float64)
        inliers = np.zeros((src_pts.shape[0], 1), dtype=np.uint8)
        inliers[:2] = 1
        return homography, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.apap.cv2.findHomography",
        _fake_find_homography,
    )

    kp1 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    valid_pairs = [(i, i, float(i)) for i in range(6)]

    registration = estimator.estimate_registration(
        kp1,
        kp2,
        valid_pairs,
        scale1=1.0,
        scale2=1.0,
        ransac_thresh=5.0,
        max_matches=6,
    )

    assert registration.diagnostics["support_point_count"] == 2
    assert registration.diagnostics["local_warp_ready"] is False
    assert registration.diagnostics["fallback_reason"] == "insufficient_local_support"


def test_base_mode_match_feature_pair_with_metrics_threads_apap_registration(monkeypatch):
    mode = object.__new__(BaseStitchMode)
    mode.profile = SimpleNamespace(transform_mode="apap")
    mode._record_phase_time = lambda *_args, **_kwargs: None
    mode._refine_transform_with_phase_correlation = lambda *_args, **_kwargs: (_args[2], None)

    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(8, dtype=bool),
        diagnostics={"mesh_cols": 16, "mesh_rows": 12, "local_warp_ready": False},
    )

    mode.matcher = SimpleNamespace(
        min_matches_for_transform=6,
        match=lambda *_args, **_kwargs: [
            SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(8)
        ],
        compute_apap_registration=lambda *_args, **_kwargs: registration,
        describe_pair_evidence=lambda metrics: metrics,
    )

    keypoints = [SimpleNamespace(pt=(float(i), float(i))) for i in range(8)]
    descriptors = np.ones((8, 4), dtype=np.float32)

    local_transform, inlier_mask, metrics = mode._match_feature_pair_with_metrics(
        keypoints,
        descriptors,
        1.0,
        keypoints,
        descriptors,
        1.0,
    )

    assert np.array_equal(local_transform, np.eye(3, dtype=np.float64))
    assert np.array_equal(inlier_mask, np.ones(8, dtype=bool))
    assert metrics["transform_found"] is True
    assert metrics["apap_enabled"] is True
    assert metrics["apap_has_local_warp"] is False
    assert metrics["apap_local_warp_ready"] is False
    assert metrics["apap_mesh_cols"] == 16
    assert metrics["apap_mesh_rows"] == 12
    assert metrics["apap_support_point_count"] == 0
    assert isinstance(metrics["apap_registration"], APAPRegistrationResult)
    assert np.array_equal(metrics["apap_registration"].homography, registration.homography)
    assert np.array_equal(metrics["apap_registration"].inlier_mask, registration.inlier_mask)


def test_compute_transform_homography_rejects_implausible_growth_for_narrow_spread(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    def _fake_find_homography(*_args, **_kwargs):
        h = np.array(
            [[50.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        inliers = np.ones((8, 1), dtype=np.uint8)
        return h, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.findHomography",
        _fake_find_homography,
    )

    pts = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 100.0),
        (0.0, 100.0),
        (2.0, 20.0),
        (8.0, 20.0),
        (8.0, 80.0),
        (2.0, 80.0),
    ]
    kp1 = [SimpleNamespace(pt=point) for point in pts]
    kp2 = [SimpleNamespace(pt=point) for point in pts]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(8)]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="homography",
    )

    assert h is None
    assert inliers is None


def test_compute_transform_homography_rejects_projective_overfit_outside_matched_cluster(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    def _fake_find_homography(*_args, **_kwargs):
        h = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [-0.004, 0.0, 1.0]],
            dtype=np.float64,
        )
        inliers = np.ones((8, 1), dtype=np.uint8)
        return h, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.findHomography",
        _fake_find_homography,
    )

    matched_cluster = [
        (0.0, 0.0),
        (8.0, 0.0),
        (8.0, 8.0),
        (0.0, 8.0),
        (2.0, 2.0),
        (6.0, 2.0),
        (6.0, 6.0),
        (2.0, 6.0),
    ]
    wide_support = [
        (160.0, 20.0),
        (180.0, 50.0),
        (200.0, 80.0),
        (220.0, 110.0),
    ]
    points = matched_cluster + wide_support
    kp1 = [SimpleNamespace(pt=point) for point in points]
    kp2 = [SimpleNamespace(pt=point) for point in points]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(len(matched_cluster))]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="homography",
    )

    assert h is None
    assert inliers is None


def test_compute_transform_translation_mode_skips_affine_and_accepts_translation(monkeypatch):
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
        transform_mode="translation",
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


def test_match_returns_empty_for_non_finite_descriptors():
    matcher = FeatureMatcher()

    des_query = np.ones((4, 128), dtype=np.float32)
    des_train = np.ones((4, 128), dtype=np.float32)
    des_query[0, 0] = np.nan

    matches = matcher.match([], des_query, [], des_train)

    assert matches == []


def test_match_returns_empty_when_knn_match_raises_cv2_error():
    matcher = FeatureMatcher()

    class _BrokenMatcher:
        def knnMatch(self, *_args, **_kwargs):
            raise cv2.error("mock", "knnMatch", "forced failure")

    matcher._matcher = _BrokenMatcher()

    des_query = np.ones((4, 128), dtype=np.float32)
    des_train = np.ones((4, 128), dtype=np.float32)

    matches = matcher.match([], des_query, [], des_train)

    assert matches == []


def test_compute_transform_translation_mode_handles_large_match_sets_with_bounded_consensus():
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
        transform_mode="translation",
    )

    assert h_locked is not None
    assert inliers_locked is not None
    assert inliers_locked.dtype == np.bool_
    assert inliers_locked.shape == (base_count,)
    assert int(np.count_nonzero(inliers_locked)) >= 1000
    np.testing.assert_allclose(h_locked[0, 2], translation[0], atol=0.2)
    np.testing.assert_allclose(h_locked[1, 2], translation[1], atol=0.2)


def test_compute_transform_ignores_invalid_match_indices(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)
    captured = {}

    def _fake_estimate_affine_partial_2d(src_pts, dst_pts, *_args, **_kwargs):
        captured["src_count"] = int(src_pts.shape[0])
        captured["dst_count"] = int(dst_pts.shape[0])
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        inliers = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return affine, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.estimateAffinePartial2D",
        _fake_estimate_affine_partial_2d,
    )

    kp1 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(6)]
    matches.extend([
        SimpleNamespace(queryIdx=999, trainIdx=0, distance=10.0),
        SimpleNamespace(queryIdx=0, trainIdx=999, distance=10.0),
    ])

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="affine",
    )

    assert h is not None
    assert inliers is not None
    assert captured["src_count"] == 6
    assert captured["dst_count"] == 6


def test_compute_transform_rejects_non_positive_scale():
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    kp1 = [SimpleNamespace(pt=(float(i + 5), float(i + 7))) for i in range(6)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(6)]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=0.0,
        scale2=1.0,
        transform_mode="affine",
    )

    assert h is None
    assert inliers is None


def test_compute_transform_rejects_non_finite_affine(monkeypatch):
    matcher = FeatureMatcher(min_inliers=1, min_inlier_ratio=0.0)

    def _fake_estimate_affine_partial_2d(*_args, **_kwargs):
        affine = np.array([[np.nan, 0.0, 5.0], [0.0, 1.0, 7.0]], dtype=np.float64)
        inliers = np.ones((6, 1), dtype=np.uint8)
        return affine, inliers

    monkeypatch.setattr(
        "cielostitch_core.core.matcher.cv2.estimateAffinePartial2D",
        _fake_estimate_affine_partial_2d,
    )

    kp1 = [SimpleNamespace(pt=(float(i + 5), float(i + 7))) for i in range(6)]
    kp2 = [SimpleNamespace(pt=(float(i), float(i))) for i in range(6)]
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(6)]

    h, inliers = matcher.compute_transform(
        kp1,
        kp2,
        matches,
        scale1=1.0,
        scale2=1.0,
        transform_mode="affine",
    )

    assert h is None
    assert inliers is None
