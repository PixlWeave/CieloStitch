# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.config.config import cfg
from cielostitch_core.core.stitching.free_mode import FreeMode


class _MatcherStub:
    min_matches_for_transform = 6


class _ProfileStub:
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
