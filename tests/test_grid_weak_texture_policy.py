# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.config.config import cfg
from cielostitch_core.core.stitching.base_mode import BaseStitchMode
from cielostitch_core.core.stitching.grid_mode import GridMode


class _MatcherStub:
    min_matches_for_transform = 6


def test_init_stitch_session_skips_detection_when_disabled(monkeypatch):
    stitch = object.__new__(BaseStitchMode)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("feature detection should be skipped")

    monkeypatch.setattr(stitch, "_detect_features_parallel", _should_not_run)

    image_items = [("img0", np.zeros((32, 32), dtype=np.float32))]

    names, images, count, features, _start = stitch._init_stitch_session(
        image_items,
        cancel_cb=None,
        progress_cb=None,
        detect_features=False,
    )

    assert names == ["img0"]
    assert len(images) == 1
    assert count == 1
    assert features == []


def test_grid_stitch_disables_detection_in_deterministic_mode(monkeypatch):
    grid = object.__new__(GridMode)
    grid.profile = type("ProfileStub", (), {"lock_rotation": False})()

    called = {"detect_features": None}

    def fake_init_stitch_session(_image_items, cancel_cb=None, progress_cb=None, detect_features=True):
        called["detect_features"] = detect_features
        image = np.zeros((64, 64), dtype=np.float32)
        return ["img0"], [image], 1, [], 0.0

    monkeypatch.setattr(cfg.state, "stitch_mode", "fixed-overlap")
    monkeypatch.setattr(grid, "_init_stitch_session", fake_init_stitch_session)
    monkeypatch.setattr(grid, "_init_grid_layout", lambda _n: None)
    monkeypatch.setattr(grid, "_compute_global_transforms", lambda *args, **kwargs: ([np.eye(3)], [1e9], 0))
    monkeypatch.setattr(grid, "_preallocate_canvas_once", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid, "_warp_and_blend_panels", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid, "_sync_random_counters", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid, "_finalize_stitch", lambda *args, **kwargs: (None, None, None))
    monkeypatch.setattr("cielostitch_core.core.stitching.grid_mode.MosaicCanvas", lambda *args, **kwargs: object())

    image_items = [("img0", np.zeros((64, 64), dtype=np.float32))]
    grid.stitch(image_items, progress_cb=None, cancel_cb=None)

    assert called["detect_features"] is False


def test_is_weak_feature_evidence_when_keypoints_are_sparse():
    stitch = object.__new__(BaseStitchMode)
    stitch.matcher = _MatcherStub()

    metrics = {
        "has_descriptors": True,
        "min_kp": 12,
        "match_count": 14,
    }

    assert stitch._is_weak_feature_evidence(metrics) is True


def test_is_weak_feature_evidence_when_matches_are_sparse():
    stitch = object.__new__(BaseStitchMode)
    stitch.matcher = _MatcherStub()

    metrics = {
        "has_descriptors": True,
        "min_kp": 120,
        "match_count": 5,
    }

    assert stitch._is_weak_feature_evidence(metrics) is True


def test_is_not_weak_feature_evidence_with_healthy_counts():
    stitch = object.__new__(BaseStitchMode)
    stitch.matcher = _MatcherStub()

    metrics = {
        "has_descriptors": True,
        "min_kp": 120,
        "match_count": 24,
    }

    assert stitch._is_weak_feature_evidence(metrics) is False


def test_grid_pair_prefers_phase_for_weak_texture(monkeypatch):
    grid = object.__new__(GridMode)
    grid.matcher = _MatcherStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)

    def fake_match_with_metrics(*_args, **_kwargs):
        return None, None, {
            "kp_ref": 18,
            "kp_curr": 20,
            "min_kp": 18,
            "match_count": 3,
            "inlier_count": 0,
            "has_descriptors": True,
        }

    monkeypatch.setattr(grid, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(grid, "_is_plausible_neighbor_shift", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        grid,
        "_estimate_translation_fallback",
        lambda *_args, **_kwargs: (np.array([[1.0, 0.0, 11.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 0.42),
    )
    monkeypatch.setattr(grid, "_try_nominal_fallback", lambda *_args, **_kwargs: ("nominal", -1))

    features = [([], np.ones((8, 8), dtype=np.float32), 1.0), ([], np.ones((8, 8), dtype=np.float32), 1.0)]
    images = [np.zeros((128, 128), dtype=np.float32), np.zeros((128, 128), dtype=np.float32)]
    global_h = [np.eye(3), None]

    result = grid._pair_global(0, 1, "left", global_h, features, images, False, None)

    assert result is not None
    h, score = result
    assert score == 420
    assert float(h[0, 2]) == 11.0


def test_grid_pair_keeps_feature_path_when_evidence_is_healthy(monkeypatch):
    grid = object.__new__(GridMode)
    grid.matcher = _MatcherStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)

    h_feature = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    inliers = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype=np.uint8)

    def fake_match_with_metrics(*_args, **_kwargs):
        return h_feature, inliers, {
            "kp_ref": 150,
            "kp_curr": 144,
            "min_kp": 144,
            "match_count": 28,
            "inlier_count": 12,
            "has_descriptors": True,
        }

    phase_calls = {"count": 0}

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return np.eye(3), 0.5

    monkeypatch.setattr(grid, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(grid, "_is_plausible_neighbor_shift", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(grid, "_estimate_translation_fallback", fake_phase)
    monkeypatch.setattr(grid, "_try_nominal_fallback", lambda *_args, **_kwargs: ("nominal", -1))

    features = [([], np.ones((8, 8), dtype=np.float32), 1.0), ([], np.ones((8, 8), dtype=np.float32), 1.0)]
    images = [np.zeros((128, 128), dtype=np.float32), np.zeros((128, 128), dtype=np.float32)]
    global_h = [np.eye(3), None]

    result = grid._pair_global(0, 1, "left", global_h, features, images, False, None)

    assert result is not None
    h, score = result
    assert score == 12
    assert float(h[0, 2]) == 7.0
    assert phase_calls["count"] == 0


def test_grid_pair_keeps_feature_match_even_when_counts_look_weak(monkeypatch):
    grid = object.__new__(GridMode)
    grid.matcher = _MatcherStub()

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_phase_fallback", True)

    h_feature = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    inliers = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype=np.uint8)

    def fake_match_with_metrics(*_args, **_kwargs):
        return h_feature, inliers, {
            "kp_ref": 22,
            "kp_curr": 20,
            "min_kp": 20,
            "match_count": 7,
            "inlier_count": 10,
            "has_descriptors": True,
        }

    phase_calls = {"count": 0}

    def fake_phase(*_args, **_kwargs):
        phase_calls["count"] += 1
        return np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 0.7

    monkeypatch.setattr(grid, "_match_feature_pair_with_metrics", fake_match_with_metrics)
    monkeypatch.setattr(grid, "_is_plausible_neighbor_shift", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(grid, "_estimate_translation_fallback", fake_phase)
    monkeypatch.setattr(grid, "_try_nominal_fallback", lambda *_args, **_kwargs: ("nominal", -1))

    features = [([], np.ones((8, 8), dtype=np.float32), 1.0), ([], np.ones((8, 8), dtype=np.float32), 1.0)]
    images = [np.zeros((128, 128), dtype=np.float32), np.zeros((128, 128), dtype=np.float32)]
    global_h = [np.eye(3), None]

    result = grid._pair_global(0, 1, "left", global_h, features, images, False, None)

    assert result is not None
    h, score = result
    assert score == 10
    assert float(h[0, 2]) == 9.0
    assert phase_calls["count"] == 0


def test_grid_mode_does_not_let_nominal_fallback_outrank_real_neighbor_match(monkeypatch):
    grid = object.__new__(GridMode)
    grid.profile = type("ProfileStub", (), {"lock_rotation": False})()
    grid.grid_cols = 2
    grid._slot_of_index = [0, 1, 2]
    grid._slot_to_index = {0: 0, 1: 1, 2: 2}
    grid._rows = 2

    monkeypatch.setattr(cfg.state.ssm, "enable_grid_nominal_fallback", True)
    monkeypatch.setattr(cfg.state.ssm.grid_guide, "grid_nominal_score", 2.0)

    feature_transform = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    nominal_transform = np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def fake_pair_global(ref_idx, *_args, **_kwargs):
        if ref_idx == 0:
            return feature_transform, 9
        if ref_idx == 1:
            return nominal_transform, 2
        return None

    monkeypatch.setattr(grid, "_pair_global", fake_pair_global)

    images = [
        np.zeros((128, 128), dtype=np.float32),
        np.zeros((128, 128), dtype=np.float32),
        np.zeros((128, 128), dtype=np.float32),
    ]
    names = ["img0", "img1", "img2"]
    features = [([], np.ones((8, 8), dtype=np.float32), 1.0) for _ in range(3)]
    global_h = [np.eye(3), np.eye(3), None]
    placement_scores = [1e9, 1e9, 0.0]

    skipped = grid._compute_feature_transforms(
        image_count=3,
        images=images,
        names=names,
        features=features,
        global_transforms=global_h,
        placement_scores=placement_scores,
        cancel_cb=None,
        progress_cb=None,
    )

    assert skipped == 0
    assert placement_scores[2] == 9
    assert float(global_h[2][0, 2]) == 7.0


def test_grid_mode_uses_image_indices_for_non_linear_scan_order(monkeypatch):
    grid = object.__new__(GridMode)
    grid.profile = type("ProfileStub", (), {"lock_rotation": False})()
    grid.grid_cols = 3
    grid._slot_of_index = [0, 3, 4, 1, 2, 5]
    grid._slot_to_index = {slot: idx for idx, slot in enumerate(grid._slot_of_index)}
    grid._rows = 2

    calls = []

    def fake_pair_global(ref_idx, cur_idx, relation, *_args, **_kwargs):
        calls.append((ref_idx, cur_idx, relation))
        transform = np.eye(3)
        transform[0, 2] = float(cur_idx)
        return transform, 15

    monkeypatch.setattr(grid, "_pair_global", fake_pair_global)

    images = [np.zeros((128, 128), dtype=np.float32) for _ in range(6)]
    names = [f"img{i}" for i in range(6)]
    features = [([], np.ones((8, 8), dtype=np.float32), 1.0) for _ in range(6)]
    global_h = [None] * 6
    global_h[0] = np.eye(3)
    placement_scores = [0.0] * 6
    placement_scores[0] = 1e9

    skipped = grid._compute_feature_transforms(
        image_count=6,
        images=images,
        names=names,
        features=features,
        global_transforms=global_h,
        placement_scores=placement_scores,
        cancel_cb=None,
        progress_cb=None,
    )

    assert skipped == 0
    assert calls[:3] == [(0, 1, "top"), (1, 2, "left"), (2, 3, "bottom")]
    assert global_h[3] is not None
    assert global_h[4] is not None
    assert global_h[1] is not None
    assert placement_scores[3] == 15
    assert placement_scores[4] == 15
    assert placement_scores[1] == 15


def test_grid_mode_keeps_step1_nominal_when_no_other_neighbors(monkeypatch):
    grid = object.__new__(GridMode)
    grid.profile = type("ProfileStub", (), {"lock_rotation": False})()
    grid.grid_cols = 2
    grid._slot_of_index = [1, 0]
    grid._slot_to_index = {slot: idx for idx, slot in enumerate(grid._slot_of_index)}
    grid._rows = 1

    nominal_transform = np.eye(3)
    nominal_transform[0, 2] = -10.0

    def fake_pair_global(ref_idx, cur_idx, relation, *_args, **_kwargs):
        assert (ref_idx, cur_idx, relation) == (0, 1, "right")
        return nominal_transform, 2, "nominal"

    monkeypatch.setattr(grid, "_pair_global", fake_pair_global)

    images = [np.zeros((128, 128), dtype=np.float32) for _ in range(2)]
    names = ["img0", "img1"]
    features = [([], np.ones((8, 8), dtype=np.float32), 1.0) for _ in range(2)]
    global_h = [None] * 2
    global_h[0] = np.eye(3)
    placement_scores = [0.0] * 2
    placement_scores[0] = 1e9

    skipped = grid._compute_feature_transforms(
        image_count=2,
        images=images,
        names=names,
        features=features,
        global_transforms=global_h,
        placement_scores=placement_scores,
        cancel_cb=None,
        progress_cb=None,
    )

    assert skipped == 0
    assert global_h[1] is not None
    assert float(global_h[1][0, 2]) == -10.0
    assert placement_scores[1] == 2


def test_grid_mode_records_subpixel_count_once_for_chosen_panel(monkeypatch):
    grid = object.__new__(GridMode)
    grid.profile = type("ProfileStub", (), {"lock_rotation": False})()
    grid.grid_cols = 2
    grid._slot_of_index = [0, 1, 2]
    grid._slot_to_index = {0: 0, 1: 1, 2: 2}
    grid._rows = 2
    grid._subpixel_refinement_keys = set()

    def fake_pair_global(ref_idx, cur_idx, *_args, **_kwargs):
        if cur_idx == 1:
            return np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 7, "feature", {
                "ref_idx": ref_idx,
                "subpixel_refinement_applied": False,
                "subpixel_refinement_dx": 0.0,
                "subpixel_refinement_dy": 0.0,
                "subpixel_refinement_response": 0.0,
            }
        if ref_idx == 0:
            return np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 9, "feature", {
                "ref_idx": 0,
                "subpixel_refinement_applied": True,
                "subpixel_refinement_dx": 0.2,
                "subpixel_refinement_dy": 0.1,
                "subpixel_refinement_response": 0.4,
            }
        if ref_idx == 1:
            return np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 8, "feature", {
                "ref_idx": 1,
                "subpixel_refinement_applied": True,
                "subpixel_refinement_dx": 0.5,
                "subpixel_refinement_dy": -0.3,
                "subpixel_refinement_response": 0.5,
            }
        return None

    monkeypatch.setattr(grid, "_pair_global", fake_pair_global)

    images = [
        np.zeros((128, 128), dtype=np.float32),
        np.zeros((128, 128), dtype=np.float32),
        np.zeros((128, 128), dtype=np.float32),
    ]
    names = ["img0", "img1", "img2"]
    features = [([], np.ones((8, 8), dtype=np.float32), 1.0) for _ in range(3)]
    global_h = [np.eye(3), np.eye(3), None]
    placement_scores = [1e9, 1e9, 0.0]

    skipped = grid._compute_feature_transforms(
        image_count=3,
        images=images,
        names=names,
        features=features,
        global_transforms=global_h,
        placement_scores=placement_scores,
        cancel_cb=None,
        progress_cb=None,
    )

    assert skipped == 0
    assert grid._subpixel_refinement_keys == {2}
    assert placement_scores[2] == 9
