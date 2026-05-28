# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np
from types import SimpleNamespace

from cielostitch_core.stitching.base_mode import BaseStitchMode
from cielostitch_core.state.preferences import AppPreferences


def test_hidden_subpixel_preference_loads_from_ini_value(monkeypatch):
    values = {
        "stitch/enable_subpixel_refinement": "false",
    }

    def fake_read_pref(key, default=None, value_type=None, type=None):
        return values.get(key, default)

    monkeypatch.setattr("cielostitch_core.state.preferences.read_pref", fake_read_pref)

    prefs = AppPreferences.load()

    assert prefs.enable_subpixel_refinement is False


def test_subpixel_refinement_composes_small_residual_translation(monkeypatch):
    mode = object.__new__(BaseStitchMode)
    monkeypatch.setattr(mode, "_subpixel_refinement_enabled", lambda: True)

    ref = np.zeros((96, 96), dtype=np.float32)
    ref[30:50, 40:60] = 1.0
    current = ref.copy()

    base_transform = np.eye(3, dtype=np.float64)

    def fake_estimate_translation(_ref_roi, _warped_roi, min_response=0.02, max_shift_ratio=0.08):
        residual = np.eye(3, dtype=np.float64)
        residual[0, 2] = 0.75
        residual[1, 2] = -0.5
        return residual, 0.42

    monkeypatch.setattr(mode, "_estimate_translation_fallback", fake_estimate_translation)

    refined, metrics = mode._refine_transform_with_phase_correlation(ref, current, base_transform)

    assert metrics is not None
    assert metrics["applied"] is True
    np.testing.assert_allclose(refined[0, 2], 0.75)
    np.testing.assert_allclose(refined[1, 2], -0.5)


def test_subpixel_refinement_rejects_large_residual_translation(monkeypatch):
    mode = object.__new__(BaseStitchMode)
    monkeypatch.setattr(mode, "_subpixel_refinement_enabled", lambda: True)

    ref = np.zeros((96, 96), dtype=np.float32)
    ref[30:50, 40:60] = 1.0
    current = ref.copy()

    base_transform = np.eye(3, dtype=np.float64)
    base_transform[0, 2] = 5.0

    def fake_estimate_translation(_ref_roi, _warped_roi, min_response=0.02, max_shift_ratio=0.08):
        residual = np.eye(3, dtype=np.float64)
        residual[0, 2] = 4.0
        residual[1, 2] = 0.0
        return residual, 0.55

    monkeypatch.setattr(mode, "_estimate_translation_fallback", fake_estimate_translation)

    refined, metrics = mode._refine_transform_with_phase_correlation(ref, current, base_transform)

    assert metrics is not None
    assert metrics["applied"] is False
    np.testing.assert_allclose(refined, base_transform)


def test_subpixel_refinement_uses_geometric_support_not_signal_threshold(monkeypatch):
    mode = object.__new__(BaseStitchMode)
    monkeypatch.setattr(mode, "_subpixel_refinement_enabled", lambda: True)

    ref = np.zeros((96, 96), dtype=np.float32)
    current = np.zeros((96, 96), dtype=np.float32)
    ref[30:50, 40:60] = 1e-8
    current[30:50, 40:60] = 1e-8

    called = {"count": 0}

    def fake_estimate_translation(_ref_roi, _warped_roi, min_response=0.02, max_shift_ratio=0.08):
        called["count"] += 1
        residual = np.eye(3, dtype=np.float64)
        return residual, 0.25

    monkeypatch.setattr(mode, "_estimate_translation_fallback", fake_estimate_translation)

    refined, metrics = mode._refine_transform_with_phase_correlation(ref, current, np.eye(3, dtype=np.float64))

    assert called["count"] == 1
    assert metrics is not None
    assert metrics["applied"] is True
    np.testing.assert_allclose(refined, np.eye(3, dtype=np.float64))


def test_match_feature_pair_skips_subpixel_refinement_for_homography(monkeypatch):
    mode = object.__new__(BaseStitchMode)
    mode.profile = SimpleNamespace(transform_mode="homography")
    mode.matcher = SimpleNamespace(
        match=lambda *_args, **_kwargs: [SimpleNamespace(queryIdx=0, trainIdx=0, distance=0.0)],
        compute_transform=lambda *_args, **_kwargs: (np.eye(3, dtype=np.float64), np.ones(1, dtype=bool)),
        describe_pair_evidence=lambda metrics: metrics,
    )
    mode._record_phase_time = lambda *_args, **_kwargs: None

    called = {"count": 0}

    def fake_refine(*_args, **_kwargs):
        called["count"] += 1
        return np.eye(3, dtype=np.float64), {"applied": True, "response": 1.0, "dx": 0.0, "dy": 0.0}

    monkeypatch.setattr(mode, "_refine_transform_with_phase_correlation", fake_refine)

    ref_keypoints = [SimpleNamespace(pt=(0.0, 0.0))]
    current_keypoints = [SimpleNamespace(pt=(0.0, 0.0))]
    local_transform, inlier_mask, feature_metrics = mode._match_feature_pair_with_metrics(
        ref_keypoints,
        np.ones((1, 8), dtype=np.float32),
        1.0,
        current_keypoints,
        np.ones((1, 8), dtype=np.float32),
        1.0,
        ref_image=np.zeros((64, 64), dtype=np.float32),
        current_image=np.zeros((64, 64), dtype=np.float32),
    )

    assert called["count"] == 0
    assert local_transform is not None
    assert inlier_mask is not None
    assert feature_metrics.get("subpixel_refinement_applied", False) is False