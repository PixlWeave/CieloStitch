# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import time
from dataclasses import replace

import numpy as np

from cielostitch_core.config.config import cfg
from cielostitch_core.core.stitching.free_mode import FreeMode
from cielostitch_core.core.stitching.grid_mode import GridMode
from cielostitch_core.state.profiles import GeneralProfile


def _make_two_panels(shape=(64, 64)):
    h, w = shape
    panel_a = np.zeros((h, w), dtype=np.float32)
    panel_b = np.ones((h, w), dtype=np.float32)
    return [("panel_0", panel_a), ("panel_1", panel_b)]


def _assert_canvas_diagnostics(canvas_system) -> None:
    assert canvas_system is not None

    coverage = getattr(canvas_system, "coverage_count", None)
    seam = getattr(canvas_system, "seam_heatmap", None)

    assert coverage is not None
    assert seam is not None

    overlap = coverage >= 2
    assert np.any(overlap)
    assert np.max(seam[overlap]) > 0.0
    assert np.all(seam[~overlap] == 0.0)


def _set_valid_blender_knobs(monkeypatch) -> None:
    monkeypatch.setattr(cfg.state.psm, "adaptive_mb_risk_boost_threshold", 0.45)
    monkeypatch.setattr(cfg.state.psm, "adaptive_mb_low_risk_threshold", 0.10)
    monkeypatch.setattr(cfg.state.psm, "adaptive_mb_max_boost", 1)
    monkeypatch.setattr(cfg.state.psm, "photometric_min_overlap_px", 2_000)
    monkeypatch.setattr(cfg.state.psm, "photometric_full_confidence_px", 250_000)



def test_grid_mode_stitch_exposes_overlap_and_seam_diagnostics(monkeypatch):
    monkeypatch.setattr(
        cfg,
        "prefs",
        replace(
            cfg.prefs,
            retain_overlap_diagnostics=True,
            retain_seam_diagnostics=True,
        ),
    )
    monkeypatch.setattr(cfg.state.psm, "blend_type", "seamless")
    _set_valid_blender_knobs(monkeypatch)

    monkeypatch.setattr(cfg.state, "stitch_mode", "fixed-overlap")
    monkeypatch.setattr(cfg.state.ssm, "current_stitch_mode", "fixed-overlap")
    monkeypatch.setattr(cfg.state.ssm, "overlap_x_pct", 50.0)
    monkeypatch.setattr(cfg.state.ssm, "overlap_y_pct", 0.0)

    image_items = _make_two_panels((64, 64))
    engine = GridMode(GeneralProfile(), grid_cols=2)
    result = engine.stitch(image_items, progress_cb=None, cancel_cb=None)

    assert result is not None
    _assert_canvas_diagnostics(engine.last_canvas_system)



def test_free_mode_stitch_exposes_overlap_and_seam_diagnostics(monkeypatch):
    monkeypatch.setattr(
        cfg,
        "prefs",
        replace(
            cfg.prefs,
            retain_overlap_diagnostics=True,
            retain_seam_diagnostics=True,
        ),
    )
    monkeypatch.setattr(cfg.state.psm, "blend_type", "seamless")
    _set_valid_blender_knobs(monkeypatch)

    image_items = _make_two_panels((64, 64))

    engine = FreeMode(GeneralProfile())

    def fake_init_stitch_session(_image_items, cancel_cb=None, progress_cb=None):
        del cancel_cb, progress_cb
        names = [name for name, _ in _image_items]
        images = [image for _, image in _image_items]
        features = [([], None, 1.0), ([], None, 1.0)]
        return names, images, len(images), features, time.perf_counter()

    def fake_find_best_match_or_fallback(
        current_keypoints,
        current_descriptors,
        current_scale,
        current_name,
        current_image,
        placed_refs,
        total_panels,
        current_shape,
        cancel_cb,
        progress_cb,
        skip_color="yellow",
    ):
        del (
            current_keypoints,
            current_descriptors,
            current_scale,
            current_name,
            current_image,
            total_panels,
            current_shape,
            cancel_cb,
            progress_cb,
            skip_color,
        )
        h_local = np.eye(3, dtype=np.float64)
        h_local[0, 2] = 32.0
        return {
            "score": 10,
            "ref_name": placed_refs[0]["name"],
            "h_local": h_local,
            "ref_global_h": placed_refs[0]["global_h"],
            "ref_index": 0,
        }

    monkeypatch.setattr(engine, "_init_stitch_session", fake_init_stitch_session)
    monkeypatch.setattr(engine, "_find_best_match_or_fallback", fake_find_best_match_or_fallback)

    result = engine.stitch(image_items, progress_cb=None, cancel_cb=None)

    assert result is not None
    _assert_canvas_diagnostics(engine.last_canvas_system)
