# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np
from types import SimpleNamespace

from cielostitch_core.config.config import cfg
from cielostitch_core.core.apap import APAPEstimator, APAPRegistrationResult
from cielostitch_core.core.bundle import BundleAdjustmentEdge
from cielostitch_core.stitching import free_mode as free_mode_module
from cielostitch_core.stitching.free_mode import FreeMode


class _MatcherStub:
    min_matches_for_transform = 6


class _ProfileStub:
    lock_rotation = False


class PanoramaProfile:
    lock_rotation = False


def test_auto_mode_records_subpixel_count_once_per_placed_panel(monkeypatch):
    auto = object.__new__(FreeMode)
    auto._subpixel_refinement_keys = set()

    monkeypatch.setattr(auto, "_warp_and_blend_single_image", lambda *_args, **_kwargs: True)

    best_match = {
        "h_local": np.eye(3),
        "ref_global_h": np.eye(3),
        "ref_name": "ref0",
        "score": 12,
        "subpixel_refinement_applied": True,
        "subpixel_refinement_dx": 0.4,
        "subpixel_refinement_dy": -0.2,
        "subpixel_refinement_response": 0.31,
    }
    global_transform = np.eye(3)

    success = auto._process_matched_image(
        canvas_system=object(),
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        best_match=best_match,
        global_transform=global_transform,
        image_index=1,
        cancel_cb=None,
        progress_cb=None,
    )

    assert success is True
    assert auto._subpixel_refinement_keys == {1}


def test_auto_mode_marks_warp_failures_as_skipped_for_badges(monkeypatch):
    auto = object.__new__(FreeMode)
    auto._panel_source_by_index = {}
    auto._panel_evidence_by_index = {1: {"diagnostic_label": "strong_pair"}}

    monkeypatch.setattr(
        auto,
        "_find_best_match_or_fallback",
        lambda *_args, **_kwargs: {
            "h_local": np.eye(3),
            "ref_global_h": np.eye(3),
            "ref_name": "ref0",
            "score": 12,
        },
    )
    monkeypatch.setattr(auto, "_process_matched_image", lambda *_args, **_kwargs: False)

    success = auto._try_place_image(
        canvas_system=object(),
        image_items=[
            ("anchor", np.zeros((8, 8), dtype=np.float32)),
            ("panel1", np.zeros((8, 8), dtype=np.float32)),
        ],
        features=[
            ([], None, 1.0),
            ([], None, 1.0),
        ],
        placed_refs=[{"kp": [], "name": "anchor", "global_h": np.eye(3)}],
        global_transform=np.eye(3),
        image_index=1,
        image_count=2,
        cancel_cb=None,
        progress_cb=None,
    )

    assert success is False
    assert auto._panel_source_by_index == {1: "skipped"}
    assert auto._panel_evidence_by_index == {}


def test_auto_mode_phase_fallback_uses_blend_overlap_guard(monkeypatch):
    auto = object.__new__(FreeMode)
    auto._panel_source_by_index = {}
    auto._panel_evidence_by_index = {}
    auto._subpixel_refinement_keys = set()

    captured = {}

    def fake_warp_and_blend(*_args, **kwargs):
        captured["placement_score"] = kwargs.get("placement_score")
        captured["min_blend_score"] = kwargs.get("min_blend_score")
        return True

    monkeypatch.setattr(auto, "_warp_and_blend_single_image", fake_warp_and_blend)

    best_match = {
        "h_local": np.eye(3),
        "ref_global_h": np.eye(3),
        "ref_name": "ref0",
        "score": 90,
        "fallback": True,
    }

    success = auto._process_matched_image(
        canvas_system=object(),
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        best_match=best_match,
        global_transform=np.eye(3),
        image_index=1,
        cancel_cb=None,
        progress_cb=None,
    )

    assert success is True
    assert captured.get("placement_score") == 90.0
    assert captured.get("min_blend_score") == 120.0


def test_auto_mode_passes_apap_registration_into_warp_stage(monkeypatch):
    auto = object.__new__(FreeMode)
    auto._panel_source_by_index = {}
    auto._panel_evidence_by_index = {}
    auto._subpixel_refinement_keys = set()
    auto.matcher = SimpleNamespace(apap_estimator=APAPEstimator())

    captured = {}

    def fake_warp_and_blend(*_args, **kwargs):
        captured["diagnostic_context"] = kwargs.get("diagnostic_context")
        return True

    monkeypatch.setattr(auto, "_warp_and_blend_single_image", fake_warp_and_blend)

    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        diagnostics={"local_warp_ready": True, "support_point_count": 4},
    )
    best_match = {
        "h_local": np.eye(3),
        "ref_global_h": np.eye(3),
        "ref_name": "ref0",
        "score": 18,
        "apap_registration": registration,
    }

    success = auto._process_matched_image(
        canvas_system=object(),
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        best_match=best_match,
        global_transform=np.eye(3),
        image_index=1,
        cancel_cb=None,
        progress_cb=None,
    )

    assert success is True
    captured_registration = captured["diagnostic_context"]["apap_registration"]
    assert isinstance(captured_registration, APAPRegistrationResult)
    np.testing.assert_allclose(captured_registration.homography, registration.homography)
    assert np.array_equal(captured_registration.inlier_mask, registration.inlier_mask)


def test_auto_mode_composes_apap_registration_into_global_frame(monkeypatch):
    auto = object.__new__(FreeMode)
    auto._panel_source_by_index = {}
    auto._panel_evidence_by_index = {}
    auto._subpixel_refinement_keys = set()
    auto.matcher = SimpleNamespace(apap_estimator=APAPEstimator())

    captured = {}

    def fake_warp_and_blend(*_args, **kwargs):
        captured["diagnostic_context"] = kwargs.get("diagnostic_context")
        return True

    monkeypatch.setattr(auto, "_warp_and_blend_single_image", fake_warp_and_blend)

    local_registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        support_destination_points=np.array([[1.0, 2.0]], dtype=np.float64),
        diagnostics={"local_warp_ready": True, "support_point_count": 1},
    )
    ref_global_h = np.eye(3, dtype=np.float64)
    ref_global_h[0, 2] = 25.0
    ref_global_h[1, 2] = -7.0
    best_match = {
        "h_local": np.eye(3),
        "ref_global_h": ref_global_h,
        "ref_name": "ref0",
        "score": 18,
        "apap_registration": local_registration,
    }

    success = auto._process_matched_image(
        canvas_system=object(),
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        best_match=best_match,
        global_transform=np.eye(3),
        image_index=1,
        cancel_cb=None,
        progress_cb=None,
    )

    assert success is True
    composed_registration = captured["diagnostic_context"]["apap_registration"]
    assert isinstance(composed_registration, APAPRegistrationResult)
    np.testing.assert_allclose(composed_registration.homography, ref_global_h)
    np.testing.assert_allclose(
        composed_registration.support_destination_points,
        np.array([[26.0, -5.0]], dtype=np.float64),
    )


def test_phase_fallback_rejects_low_overlap_for_panorama(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.profile = PanoramaProfile()

    monkeypatch.setattr(cfg.state, "profile", "panorama")

    # Large shift leaves almost no overlap: should be rejected.
    def fake_phase_estimate(*_args, **_kwargs):
        h = np.eye(3, dtype=np.float64)
        h[0, 2] = 60.0
        h[1, 2] = 60.0
        return h, 0.9

    monkeypatch.setattr(auto, "_estimate_translation_fallback", fake_phase_estimate)

    refs = [{
        "name": "ref0",
        "image": np.zeros((64, 64), dtype=np.float32),
        "global_h": np.eye(3),
    }]

    result = auto._try_phase_fallback(refs, np.zeros((64, 64), dtype=np.float32), cancel_cb=None)

    assert result is None


def test_phase_fallback_returns_overlap_ratio_when_accepted(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.profile = PanoramaProfile()

    monkeypatch.setattr(cfg.state, "profile", "panorama")

    def fake_phase_estimate(*_args, **_kwargs):
        h = np.eye(3, dtype=np.float64)
        h[0, 2] = 8.0
        h[1, 2] = 6.0
        return h, 0.8

    monkeypatch.setattr(auto, "_estimate_translation_fallback", fake_phase_estimate)

    refs = [{
        "name": "ref0",
        "image": np.zeros((64, 64), dtype=np.float32),
        "global_h": np.eye(3),
    }]

    result = auto._try_phase_fallback(refs, np.zeros((64, 64), dtype=np.float32), cancel_cb=None)

    assert result is not None
    assert result.get("fallback") is True
    assert float(result.get("phase_overlap_ratio", 0.0)) > 0.0


def test_auto_mode_prefers_phase_only_after_weak_feature_failure(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    def fake_match_with_metrics(*_args, **_kwargs):
        return None, None, {
            "kp_ref": 18,
            "kp_curr": 20,
            "min_kp": 18,
            "match_count": 4,
            "inlier_count": 0,
            "has_descriptors": True,
        }

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", lambda *_args, **_kwargs: {"score": 350, "ref_name": "ref0", "h_local": np.eye(3), "ref_global_h": np.eye(3), "fallback": True})

    placed_refs = [
        {"name": "ref0", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=2,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is not None
    assert result["fallback"] is True
    assert result["score"] == 350


def test_auto_mode_keeps_feature_match_even_when_counts_look_weak(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    phase_calls = {"count": 0}

    def fake_match_with_metrics(*_args, **_kwargs):
        return np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[1], [1], [1], [1], [1], [1]], dtype=np.uint8), {
            "kp_ref": 20,
            "kp_curr": 18,
            "min_kp": 18,
            "match_count": 6,
            "inlier_count": 6,
            "has_descriptors": True,
        }

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return {"score": 999, "ref_name": "ref0", "h_local": np.eye(3), "ref_global_h": np.eye(3), "fallback": True}

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    placed_refs = [
        {"name": "ref0", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=2,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is not None
    assert "fallback" not in result
    assert result["score"] == 6
    assert phase_calls["count"] == 0


def test_auto_mode_skips_phase_when_feature_evidence_is_not_weak(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    phase_calls = {"count": 0}

    def fake_match_with_metrics(*_args, **_kwargs):
        return None, None, {
            "kp_ref": 140,
            "kp_curr": 150,
            "min_kp": 140,
            "match_count": 18,
            "inlier_count": 0,
            "has_descriptors": True,
        }

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return {"score": 999, "ref_name": "ref0", "h_local": np.eye(3), "ref_global_h": np.eye(3), "fallback": True}

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    placed_refs = [
        {"name": "ref0", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=2,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is None
    assert phase_calls["count"] == 0


def test_auto_mode_tries_phase_when_feature_geometry_is_rejected(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    def fake_match_with_metrics(*_args, **_kwargs):
        return (
            np.eye(3, dtype=np.float64),
            np.ones((8,), dtype=np.bool_),
            {
                "kp_ref": 140,
                "kp_curr": 150,
                "min_kp": 140,
                "match_count": 24,
                "inlier_count": 20,
                "has_descriptors": True,
            },
        )

    phase_calls = {"count": 0}

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return {"score": 777, "ref_name": "ref0", "h_local": np.eye(3), "ref_global_h": np.eye(3), "fallback": True}

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    # Force geometry rejection by returning huge projected bounds.
    monkeypatch.setattr(
        "cielostitch_core.stitching.free_mode.MosaicCanvas.compute_warp_bounds",
        lambda *_args, **_kwargs: (0.0, 0.0, 100_000.0, 80_000.0),
    )

    placed_refs = [
        {"name": "ref0", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=2,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is not None
    assert result.get("fallback") is True
    assert phase_calls["count"] == 1


def test_auto_mode_tries_phase_when_feature_fails_geometry_stage(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    def fake_match_with_metrics(*_args, **_kwargs):
        return None, None, {
            "kp_ref": 140,
            "kp_curr": 150,
            "min_kp": 140,
            "match_count": 1200,
            "inlier_count": 0,
            "has_descriptors": True,
            "failure_stage": "geometry",
            "diagnostic_label": "ambiguous_or_inconsistent",
        }

    phase_calls = {"count": 0}

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return {"score": 888, "ref_name": "ref0", "h_local": np.eye(3), "ref_global_h": np.eye(3), "fallback": True}

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    placed_refs = [
        {"name": "ref0", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=2,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is not None
    assert result.get("fallback") is True
    assert phase_calls["count"] == 1


def test_auto_mode_panorama_respects_phase_fallback_flag_when_false(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = PanoramaProfile()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", False)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    def fake_match_with_metrics(*_args, **_kwargs):
        return None, None, {
            "kp_ref": 140,
            "kp_curr": 150,
            "min_kp": 140,
            "match_count": 1200,
            "inlier_count": 0,
            "has_descriptors": True,
            "failure_stage": "geometry",
            "diagnostic_label": "ambiguous_or_inconsistent",
        }

    phase_calls = {"count": 0}

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return {"score": 999, "ref_name": "ref0", "h_local": np.eye(3), "ref_global_h": np.eye(3), "fallback": True}

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    placed_refs = [
        {"name": "ref0", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=2,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is None
    assert phase_calls["count"] == 0


def test_auto_mode_phase_fallback_stays_within_staged_reference_window(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "manual")
    monkeypatch.setattr(cfg.state.ssm, "refs_per_stage", 4)

    def fake_find_best_match_staged(*_args, **_kwargs):
        return None, 0, {
            "kp_ref": 18,
            "kp_curr": 20,
            "min_kp": 18,
            "match_count": 4,
            "inlier_count": 0,
            "has_descriptors": True,
        }

    phase_refs = {}

    def fake_phase(references, *_args, **_kwargs):
        phase_refs["names"] = [reference["name"] for reference in references]
        return None

    monkeypatch.setattr(auto, "_find_best_match_staged", fake_find_best_match_staged)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    placed_refs = [
        {"name": f"ref{i}", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
        for i in range(6)
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=12,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is None
    assert phase_refs["names"] == ["ref2", "ref3", "ref4", "ref5"]


def test_auto_mode_uses_all_refs_for_phase_fallback_when_staging_disabled(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.matcher = _MatcherStub()
    auto.profile = _ProfileStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)
    monkeypatch.setattr(cfg.state.ssm, "staged_matching_mode", "disabled")

    def fake_match_with_metrics(*_args, **_kwargs):
        return None, None, {
            "kp_ref": 18,
            "kp_curr": 20,
            "min_kp": 18,
            "match_count": 4,
            "inlier_count": 0,
            "has_descriptors": True,
        }

    phase_refs = {}

    def fake_phase(references, *_args, **_kwargs):
        phase_refs["names"] = [reference["name"] for reference in references]
        return None

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(auto, "_try_phase_fallback", fake_phase)

    placed_refs = [
        {"name": f"ref{i}", "image": np.zeros((64, 64), dtype=np.float32), "kp": [], "des": np.ones((8, 8), dtype=np.float32), "scale": 1.0, "global_h": np.eye(3)}
        for i in range(6)
    ]

    result = auto._find_best_match_or_fallback(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="img1",
        current_image=np.zeros((64, 64), dtype=np.float32),
        placed_refs=placed_refs,
        total_panels=12,
        current_shape=(64, 64),
        cancel_cb=None,
        progress_cb=None,
    )

    assert result is None
    assert phase_refs["names"] == ["ref0", "ref1", "ref2", "ref3", "ref4", "ref5"]


def test_auto_mode_composes_global_transform_in_reference_space_order(monkeypatch):
    auto = object.__new__(FreeMode)

    captured = {}

    def fake_warp(*_args, **_kwargs):
        captured["global_transform"] = _args[2].copy()
        return True

    monkeypatch.setattr(auto, "_warp_and_blend_single_image", fake_warp)

    ref_global = np.array([
        [0.0, -1.0, 100.0],
        [1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0],
    ])
    local_transform = np.array([
        [1.0, 0.0, 15.0],
        [0.0, 1.0, 7.0],
        [0.0, 0.0, 1.0],
    ])
    expected = ref_global @ local_transform

    best_match = {
        "h_local": local_transform,
        "ref_global_h": ref_global,
        "ref_name": "ref0",
        "score": 42,
    }
    global_transform = np.eye(3, dtype=np.float64)

    success = auto._process_matched_image(
        canvas_system=None,
        current_name="img1",
        current_image=np.zeros((16, 16), dtype=np.float32),
        best_match=best_match,
        global_transform=global_transform,
        image_index=1,
        cancel_cb=None,
        progress_cb=None,
    )

    assert success is True
    assert np.allclose(global_transform, expected)
    assert np.allclose(captured["global_transform"], expected)


def test_auto_mode_homography_prefers_compact_near_tie_candidate(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.profile = SimpleNamespace(transform_mode="homography")

    references = [
        {"name": "ref0", "kp": [], "des": None, "scale": 1.0, "global_h": np.eye(3, dtype=np.float64)},
        {"name": "ref1", "kp": [], "des": None, "scale": 1.0, "global_h": np.eye(3, dtype=np.float64)},
    ]

    calls = {"index": -1}

    def fake_match_pair(*_args, **_kwargs):
        calls["index"] += 1
        if calls["index"] == 0:
            return (
                np.eye(3, dtype=np.float64),
                np.ones(11, dtype=bool),
                {"match_count": 14, "inlier_count": 11, "inlier_ratio": 11.0 / 14.0, "apap_registration": None},
            )
        return (
            np.eye(3, dtype=np.float64),
            np.ones(10, dtype=bool),
            {"match_count": 12, "inlier_count": 10, "inlier_ratio": 10.0 / 12.0, "apap_registration": None},
        )

    monkeypatch.setattr(auto, "_match_feature_pair_with_metrics", fake_match_pair)

    def fake_bounds(_image, _global_h):
        if calls["index"] == 0:
            return (0.0, 0.0, 300.0, 300.0)
        return (0.0, 0.0, 120.0, 120.0)

    monkeypatch.setattr(free_mode_module.MosaicCanvas, "compute_warp_bounds", staticmethod(fake_bounds))

    best_match, _best_match_count, _feature_metrics = auto._find_best_match_among_references_with_evidence(
        current_keypoints=[],
        current_descriptors=np.ones((8, 8), dtype=np.float32),
        current_scale=1.0,
        current_name="current",
        current_image=np.zeros((64, 64), dtype=np.float32),
        references=references,
        cancel_cb=None,
        progress_cb=None,
    )

    assert best_match is not None
    assert best_match["ref_name"] == "ref1"


def test_free_mode_bundle_rebuild_reduces_translation_chain_drift(monkeypatch):
    auto = object.__new__(FreeMode)
    monkeypatch.setattr(cfg.state.ssm, "bundle_adjustment_mode", "translation")

    anchor = np.zeros((16, 16), dtype=np.float32)
    auto._bundle_candidates = [
        [],
        [BundleAdjustmentEdge(0, 1, np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), 10.0, "feature")],
        [
            BundleAdjustmentEdge(1, 2, np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), 10.0, "feature"),
            BundleAdjustmentEdge(0, 2, np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), 12.0, "feature"),
        ],
        [
            BundleAdjustmentEdge(2, 3, np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), 10.0, "feature"),
            BundleAdjustmentEdge(1, 3, np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), 12.0, "feature"),
        ],
    ]
    auto._freeform_global_transforms = [
        np.eye(3, dtype=np.float64),
        np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 23.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 36.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    ]
    auto._freeform_placed_panels = {
        0: {"name": "anchor", "image": anchor, "global_h": np.eye(3, dtype=np.float64)},
        1: {"name": "p1", "image": anchor.copy(), "global_h": auto._freeform_global_transforms[1].copy()},
        2: {"name": "p2", "image": anchor.copy(), "global_h": auto._freeform_global_transforms[2].copy()},
        3: {"name": "p3", "image": anchor.copy(), "global_h": auto._freeform_global_transforms[3].copy()},
    }

    captured_transforms = []

    class _DummyCanvas:
        def __init__(self, image, enable_overlap_diagnostics=True, enable_seam_diagnostics=True):
            self.image = image
            self.enable_overlap_diagnostics = enable_overlap_diagnostics
            self.enable_seam_diagnostics = enable_seam_diagnostics

    monkeypatch.setattr(free_mode_module, "MosaicCanvas", _DummyCanvas)
    monkeypatch.setattr(
        auto,
        "_warp_and_blend_single_image",
        lambda *_args, **_kwargs: captured_transforms.append(np.asarray(_args[2], dtype=np.float64).copy()) or True,
    )

    rebuilt = auto._rebuild_canvas_with_bundle_adjustment(object(), progress_cb=None, cancel_cb=None)

    assert isinstance(rebuilt, _DummyCanvas)
    assert len(captured_transforms) == 3
    assert abs(float(captured_transforms[1][0, 2]) - 20.0) < abs(23.0 - 20.0)
    assert abs(float(captured_transforms[2][0, 2]) - 30.0) < abs(36.0 - 30.0)
    assert auto.last_bundle_adjustment_diagnostics is not None
    assert auto.last_bundle_adjustment_diagnostics.status == "ok"
    assert auto.last_bundle_adjustment_diagnostics.adjusted_panels >= 2


def test_free_mode_bundle_edge_skips_apap_registration() -> None:
    auto = object.__new__(FreeMode)
    auto._bundle_candidates = [[], []]

    auto._record_bundle_edge(
        {
            "ref_image_index": 0,
            "h_local": np.eye(3, dtype=np.float64),
            "score": 12,
            "apap_registration": APAPRegistrationResult(
                homography=np.eye(3, dtype=np.float64),
                inlier_mask=np.ones(4, dtype=bool),
                diagnostics={"local_warp_ready": True},
            ),
        },
        1,
    )

    assert auto._bundle_candidates == [[], []]


def test_partial_init_free_mode_falls_back_when_local_normalizer_missing(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.profile = SimpleNamespace(gain_compensation="local", local_gain_tile_size=32)

    class _Normalizer:
        def __init__(self):
            self.calls = []

        def compute_linear_correction(self, base, new, overlap, **kwargs):
            self.calls.append((base.copy(), new.copy(), overlap.copy(), kwargs))
            return 1.0, 0.0

        def apply_linear_correction(self, image, gain, offset):
            return image

    class _Blender:
        def blend(self, canvas, mask, warped_img, warped_mask, **kwargs):
            return canvas, mask

    auto.normalizer = _Normalizer()
    auto.blender = _Blender()

    psm = cfg.state.psm
    monkeypatch.setattr(psm, "enable_panel_flatten", False, raising=False)
    monkeypatch.setattr(psm, "gain_clamp", (0.7, 1.3), raising=False)
    monkeypatch.setattr(psm, "illumination_min_overlap_px", 10, raising=False)
    monkeypatch.setattr(psm, "illumination_full_confidence_px", 100, raising=False)
    monkeypatch.setattr(psm, "blend_offset_match", False, raising=False)
    monkeypatch.setattr(psm, "blend_offset_clamp", 0.0, raising=False)

    canvas_system = SimpleNamespace(
        canvas=np.ones((4, 4), dtype=np.float32),
        mask=np.ones((4, 4), dtype=np.uint8),
        coverage_count=None,
        blended_mask=None,
        seam_heatmap=None,
    )
    warped_img = np.ones((4, 4), dtype=np.float32)
    warped_mask = np.ones((4, 4), dtype=bool)

    canvas, mask = auto._apply_illumination_and_blend(canvas_system, warped_img, warped_mask)

    assert canvas is canvas_system.canvas
    assert mask is canvas_system.mask
    assert len(auto.normalizer.calls) == 1


def test_partial_init_free_mode_normalizes_gain_mode_before_blend(monkeypatch):
    auto = object.__new__(FreeMode)
    auto.profile = SimpleNamespace(gain_compensation=" Local ", local_gain_tile_size=32)

    class _Normalizer:
        def __init__(self):
            self.linear_calls = 0

        def compute_linear_correction(self, *args, **kwargs):
            self.linear_calls += 1
            return 1.0, 0.0

        def apply_linear_correction(self, image, gain, offset):
            return image

    class _LocalNormalizer:
        def __init__(self):
            self.compute_calls = 0
            self.apply_calls = 0

        def compute_gain_map(self, base, new, overlap, tile_size=None):
            self.compute_calls += 1
            return np.ones_like(new, dtype=np.float32), np.zeros_like(new, dtype=np.float32)

        def apply_gain(self, image, gain_map, offset_map=None):
            self.apply_calls += 1
            return image

    class _Blender:
        def __init__(self):
            self.kwargs = None

        def blend(self, canvas, mask, warped_img, warped_mask, **kwargs):
            self.kwargs = kwargs
            return canvas, mask

    auto.normalizer = _Normalizer()
    auto.local_normalizer = _LocalNormalizer()
    auto.blender = _Blender()

    psm = cfg.state.psm
    monkeypatch.setattr(psm, "enable_panel_flatten", False, raising=False)
    monkeypatch.setattr(psm, "gain_clamp", (0.7, 1.3), raising=False)
    monkeypatch.setattr(psm, "illumination_min_overlap_px", 10, raising=False)
    monkeypatch.setattr(psm, "illumination_full_confidence_px", 100, raising=False)
    monkeypatch.setattr(psm, "blend_offset_match", True, raising=False)
    monkeypatch.setattr(psm, "blend_offset_clamp", 0.0, raising=False)

    canvas_system = SimpleNamespace(
        canvas=np.ones((4, 4), dtype=np.float32),
        mask=np.ones((4, 4), dtype=np.uint8),
        coverage_count=None,
        blended_mask=None,
        seam_heatmap=None,
    )
    warped_img = np.ones((4, 4), dtype=np.float32)
    warped_mask = np.ones((4, 4), dtype=bool)

    auto._apply_illumination_and_blend(canvas_system, warped_img, warped_mask)

    assert auto.normalizer.linear_calls == 0
    assert auto.local_normalizer.compute_calls == 1
    assert auto.local_normalizer.apply_calls == 1
    assert auto.blender.kwargs["use_gain"] is False
    assert auto.blender.kwargs["use_offset"] is False
