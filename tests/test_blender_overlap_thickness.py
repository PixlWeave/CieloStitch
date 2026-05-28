# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np
import pytest

from cielostitch_core.core.blender import Blender


def _make_blender(blend_type: str = "adaptive-feather", multiband_levels: int = 5) -> Blender:
    return Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type=blend_type,
        multiband_levels=multiband_levels,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
    )


def test_estimate_overlap_thickness_stays_bounded_for_full_mask() -> None:
    blender = _make_blender()
    overlap = np.ones((40, 60), dtype=bool)

    thickness = blender.estimate_overlap_thickness(overlap)

    assert 1 <= thickness <= 40


def test_estimate_overlap_thickness_normalizes_non_255_binary_masks() -> None:
    blender = _make_blender()
    overlap = np.ones((25, 25), dtype=np.uint8)

    thickness = blender.estimate_overlap_thickness(overlap)

    assert 1 <= thickness <= 25


def test_compute_shared_blend_bbox_unions_overlap_and_respects_padding() -> None:
    blender = _make_blender(blend_type="multiband", multiband_levels=3)
    canvas_mask = np.zeros((80, 100), dtype=bool)
    img_mask = np.zeros((80, 100), dtype=bool)

    canvas_mask[10:35, 20:45] = True
    img_mask[30:55, 40:70] = True

    bbox = blender._compute_shared_blend_bbox(canvas_mask, img_mask)

    assert bbox is not None
    y0, y1, x0, x1 = bbox

    overlap = canvas_mask & img_mask
    img_bbox = blender._mask_bbox(img_mask)
    overlap_bbox = blender._mask_bbox(overlap)
    union_bbox = blender._union_bbox(img_bbox, overlap_bbox)
    expected_pad = blender._compute_blend_roi_padding()

    assert union_bbox is not None
    uy0, uy1, ux0, ux1 = union_bbox
    assert y0 <= uy0
    assert x0 <= ux0
    assert y1 >= uy1
    assert x1 >= ux1
    assert y0 == max(0, uy0 - expected_pad)
    assert x0 == max(0, ux0 - expected_pad)
    assert y1 == min(img_mask.shape[0], uy1 + expected_pad)
    assert x1 == min(img_mask.shape[1], ux1 + expected_pad)


def test_adaptive_multiband_blend_does_not_mutate_configured_levels() -> None:
    blender = _make_blender(blend_type="adaptive-multiband", multiband_levels=6)
    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    blender._adaptive_multiband_blend(canvas, img, canvas_mask, img_mask)

    assert blender.multiband_levels == 6


def test_seamless_blend_does_not_mutate_configured_levels() -> None:
    blender = _make_blender(blend_type="seamless", multiband_levels=6)
    blender.seamless_quality = "fast"
    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    blender._seamless_blend(canvas, img, canvas_mask, img_mask)

    assert blender.multiband_levels == 6


def test_seamless_blend_populates_seam_heatmap() -> None:
    blender = _make_blender(blend_type="seamless", multiband_levels=6)
    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    seam_heatmap = np.zeros((64, 64), dtype=np.float32)

    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    blender._seamless_blend(canvas, img, canvas_mask, img_mask, seam_heatmap=seam_heatmap)

    overlap = canvas_mask & img_mask
    assert np.any(overlap)
    assert np.max(seam_heatmap[overlap]) > 0.0


def test_adaptive_feather_blend_does_not_mutate_feather_px() -> None:
    blender = _make_blender(blend_type="adaptive-feather")
    blender.feather_px = 40
    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    blender._adaptive_feather_blend(canvas, img, canvas_mask, img_mask)

    assert blender.feather_px == 40


def test_seamless_quality_controls_levels_and_weight_sigma() -> None:
    blender = _make_blender(blend_type="seamless", multiband_levels=6)

    fast = blender._setup_seamless_params("fast")
    balanced = blender._setup_seamless_params("balanced")
    best = blender._setup_seamless_params("best")

    assert fast["multiband_levels"] == 3
    assert balanced["multiband_levels"] == 4
    assert best["multiband_levels"] == 5
    assert fast["weight_sigma"] < balanced["weight_sigma"] < best["weight_sigma"]


def test_blend_gain_is_attenuated_for_low_overlap_confidence() -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.5, 3.0),
        blend_type="none",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
    )

    # Tiny overlap should suppress photometric correction strongly.
    canvas_small = np.full((64, 64), 100.0, dtype=np.float32)
    img_small = np.full((64, 64), 50.0, dtype=np.float32)
    canvas_mask_small = np.zeros((64, 64), dtype=bool)
    img_mask_small = np.zeros((64, 64), dtype=bool)
    canvas_mask_small[:, :33] = True
    img_mask_small[:, 32:] = True

    out_small, _ = blender.blend(
        canvas_small,
        canvas_mask_small,
        img_small,
        img_mask_small,
        use_gain=True,
        use_offset=False,
    )
    new_small = img_mask_small & (~canvas_mask_small)
    mean_small = float(np.mean(out_small[new_small]))

    # Large overlap should apply almost full correction.
    canvas_large = np.full((600, 600), 100.0, dtype=np.float32)
    img_large = np.full((600, 600), 50.0, dtype=np.float32)
    canvas_mask_large = np.zeros((600, 600), dtype=bool)
    img_mask_large = np.zeros((600, 600), dtype=bool)
    canvas_mask_large[:, :500] = True
    img_mask_large[:, 100:] = True

    out_large, _ = blender.blend(
        canvas_large,
        canvas_mask_large,
        img_large,
        img_mask_large,
        use_gain=True,
        use_offset=False,
    )
    new_large = img_mask_large & (~canvas_mask_large)
    mean_large = float(np.mean(out_large[new_large]))

    assert mean_small < 60.0
    assert mean_large > 95.0


def test_blend_offset_is_attenuated_for_low_overlap_confidence() -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.5, 3.0),
        blend_type="none",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
    )

    # Tiny overlap should suppress additive offset strongly.
    canvas_small = np.full((64, 64), 100.0, dtype=np.float32)
    img_small = np.full((64, 64), 50.0, dtype=np.float32)
    canvas_mask_small = np.zeros((64, 64), dtype=bool)
    img_mask_small = np.zeros((64, 64), dtype=bool)
    canvas_mask_small[:, :33] = True
    img_mask_small[:, 32:] = True

    out_small, _ = blender.blend(
        canvas_small,
        canvas_mask_small,
        img_small,
        img_mask_small,
        use_gain=False,
        use_offset=True,
        offset_clamp=500.0,
    )
    new_small = img_mask_small & (~canvas_mask_small)
    mean_small = float(np.mean(out_small[new_small]))

    # Large overlap should apply almost full offset.
    canvas_large = np.full((600, 600), 100.0, dtype=np.float32)
    img_large = np.full((600, 600), 50.0, dtype=np.float32)
    canvas_mask_large = np.zeros((600, 600), dtype=bool)
    img_mask_large = np.zeros((600, 600), dtype=bool)
    canvas_mask_large[:, :500] = True
    img_mask_large[:, 100:] = True

    out_large, _ = blender.blend(
        canvas_large,
        canvas_mask_large,
        img_large,
        img_mask_large,
        use_gain=False,
        use_offset=True,
        offset_clamp=500.0,
    )
    new_large = img_mask_large & (~canvas_mask_large)
    mean_large = float(np.mean(out_large[new_large]))

    assert mean_small < 60.0
    assert mean_large > 95.0


def test_blend_gain_respects_configured_photometric_overlap_thresholds() -> None:
    # 64x64 setup with overlap stripe width=1 => overlap area=64 px.
    canvas = np.full((64, 64), 100.0, dtype=np.float32)
    img = np.full((64, 64), 50.0, dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :33] = True
    img_mask[:, 32:] = True

    # Strict thresholds keep confidence very low for this overlap size.
    strict = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.5, 3.0),
        blend_type="none",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        photometric_min_overlap_px=2_000,
        photometric_full_confidence_px=250_000,
    )
    out_strict, _ = strict.blend(
        canvas.copy(),
        canvas_mask,
        img.copy(),
        img_mask,
        use_gain=True,
        use_offset=False,
    )
    only_img = img_mask & (~canvas_mask)
    mean_strict = float(np.mean(out_strict[only_img]))

    # Relaxed thresholds should allow near-full confidence and stronger correction.
    relaxed = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.5, 3.0),
        blend_type="none",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        photometric_min_overlap_px=0,
        photometric_full_confidence_px=64,
    )
    out_relaxed, _ = relaxed.blend(
        canvas.copy(),
        canvas_mask,
        img.copy(),
        img_mask,
        use_gain=True,
        use_offset=False,
    )
    mean_relaxed = float(np.mean(out_relaxed[only_img]))

    assert mean_relaxed > mean_strict
    assert mean_relaxed > 95.0


def test_blend_gain_handles_integer_dtype_without_casting_error() -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.5, 3.0),
        blend_type="none",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        photometric_min_overlap_px=0,
        photometric_full_confidence_px=64,
    )

    canvas = np.full((64, 64), 100, dtype=np.uint16)
    img = np.full((64, 64), 50, dtype=np.uint16)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :33] = True
    img_mask[:, 32:] = True

    out, _ = blender.blend(
        canvas,
        canvas_mask,
        img,
        img_mask,
        use_gain=True,
        use_offset=False,
    )

    new_pixels = img_mask & (~canvas_mask)

    assert out.dtype == np.uint16
    assert np.any(out[new_pixels] != 50)


def test_blend_gain_offset_integer_matches_legacy_staged_behavior() -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.5, 3.0),
        blend_type="none",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        photometric_min_overlap_px=0,
        photometric_full_confidence_px=64,
    )

    canvas = np.full((64, 64), 100, dtype=np.uint16)
    img = np.full((64, 64), 50, dtype=np.uint16)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :33] = True
    img_mask[:, 32:] = True

    # Legacy staged behavior: gain pass writes integer image first, then offset pass reads that.
    canvas_legacy = canvas.copy()
    img_legacy = img.copy()
    overlap = canvas_mask & img_mask
    img_bbox, _ = blender._compute_blend_regions(canvas_mask, img_mask)
    assert img_bbox is not None
    y0, y1, x0, x1 = img_bbox
    overlap_roi = overlap[y0:y1, x0:x1]
    assert np.any(overlap_roi)
    overlap_confidence = blender._photometric_overlap_confidence(overlap_roi)

    canvas_roi = canvas_legacy[y0:y1, x0:x1]
    img_mask_roi = img_mask[y0:y1, x0:x1]
    info = np.iinfo(np.uint16)

    img_roi_f32 = img_legacy[y0:y1, x0:x1].astype(np.float32, copy=True)
    gain = blender._compute_gain(canvas_roi, img_legacy[y0:y1, x0:x1], overlap_roi)
    gain = blender._attenuate_gain(gain, overlap_confidence)
    if np.ndim(gain) == 0:
        img_roi_f32[img_mask_roi] *= float(gain)
    else:
        img_roi_f32[img_mask_roi] *= gain[:1]
    img_roi_f32 = np.clip(img_roi_f32, info.min, info.max)
    img_legacy[y0:y1, x0:x1] = img_roi_f32.astype(np.uint16, copy=False)

    img_roi_f32 = img_legacy[y0:y1, x0:x1].astype(np.float32, copy=True)
    offset = blender._compute_offset(canvas_roi, img_roi_f32, overlap_roi, offset_clamp=800.0)
    offset = blender._attenuate_offset(offset, overlap_confidence)
    if np.ndim(offset) == 0:
        img_roi_f32[img_mask_roi] += float(offset)
    else:
        img_roi_f32[img_mask_roi] += offset[:1]
    img_roi_f32 = np.clip(img_roi_f32, info.min, info.max)
    img_legacy[y0:y1, x0:x1] = img_roi_f32.astype(np.uint16, copy=False)

    expected = canvas_legacy.copy()
    new_pixels = img_mask & (~canvas_mask)
    expected[new_pixels] = img_legacy[new_pixels]

    out, _ = blender.blend(
        canvas.copy(),
        canvas_mask,
        img.copy(),
        img_mask,
        use_gain=True,
        use_offset=True,
        offset_clamp=800.0,
    )

    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)


def test_edge_aware_params_adapt_to_scale_and_noise() -> None:
    blender = _make_blender(blend_type="multiband", multiband_levels=5)

    small_clean = np.full((32, 32), 128, dtype=np.uint8)
    large_clean = np.full((512, 512), 128, dtype=np.uint8)

    d_small, sc_small, ss_small = blender._compute_edge_aware_params(small_clean)
    d_large, sc_large, ss_large = blender._compute_edge_aware_params(large_clean)

    assert d_large >= d_small
    assert ss_large >= ss_small
    assert sc_large >= 12.0
    assert sc_small >= 12.0

    rng = np.random.default_rng(0)
    noisy = np.clip(128 + rng.normal(0, 30, size=(128, 128)), 0, 255).astype(np.uint8)
    clean = np.full((128, 128), 128, dtype=np.uint8)

    _, sc_clean, _ = blender._compute_edge_aware_params(clean)
    _, sc_noisy, _ = blender._compute_edge_aware_params(noisy)
    assert sc_noisy > sc_clean


def test_adaptive_multiband_levels_consider_seam_risk() -> None:
    blender = _make_blender(blend_type="adaptive-multiband", multiband_levels=5)

    overlap = np.zeros((240, 240), dtype=bool)
    overlap[:, 40:200] = True

    # Same overlap geometry, different photometric consistency.
    canvas = np.full((240, 240), 100.0, dtype=np.float32)
    img_low_risk = np.full((240, 240), 100.0, dtype=np.float32)
    img_high_risk = np.full((240, 240), 140.0, dtype=np.float32)

    levels_low = blender._compute_adaptive_levels(overlap, canvas=canvas, img=img_low_risk)
    levels_high = blender._compute_adaptive_levels(overlap, canvas=canvas, img=img_high_risk)

    assert 2 <= levels_low <= 5
    assert 2 <= levels_high <= 5
    assert levels_high >= levels_low


def test_adaptive_multiband_levels_fallback_to_two_when_overlap_thickness_missing() -> None:
    blender = _make_blender(blend_type="adaptive-multiband", multiband_levels=5)

    levels = blender._compute_adaptive_levels(np.zeros((32, 32), dtype=bool))

    assert levels == 2


def test_adaptive_feather_uses_configured_base_radius_for_wide_overlap() -> None:
    blender = _make_blender(blend_type="adaptive-feather")
    blender.feather_px = 40
    overlap = np.zeros((160, 160), dtype=bool)
    overlap[:, 20:140] = True

    feather = blender.get_effective_feather_radius(overlap)

    assert feather == 40


def test_adaptive_multiband_levels_respect_configured_risk_knobs(monkeypatch) -> None:
    overlap = np.zeros((240, 240), dtype=bool)
    overlap[:, 40:200] = True
    canvas = np.full((240, 240), 100.0, dtype=np.float32)
    img = np.full((240, 240), 120.0, dtype=np.float32)

    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="adaptive-multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        adaptive_mb_risk_boost_threshold=0.45,
        adaptive_mb_low_risk_threshold=0.10,
        adaptive_mb_max_boost=1,
    )

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.60)

    boosted = blender._compute_adaptive_levels(overlap, canvas=canvas, img=img)
    assert boosted == 5

    blender.adaptive_mb_risk_boost_threshold = 0.90
    suppressed_by_threshold = blender._compute_adaptive_levels(overlap, canvas=canvas, img=img)
    assert suppressed_by_threshold == 4

    blender.adaptive_mb_risk_boost_threshold = 0.45
    blender.adaptive_mb_max_boost = 0
    suppressed_by_max_boost = blender._compute_adaptive_levels(overlap, canvas=canvas, img=img)
    assert suppressed_by_max_boost == 4


def test_ghost_guard_routes_multiband_to_fallback_on_high_risk(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    calls = {"content_aware": 0, "multiband": 0}

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.9)

    def _fake_content_aware(canvas_in, img_in, canvas_mask_in, img_mask_in, overlap_in, blended_mask=None, seam_heatmap=None):
        del blended_mask, seam_heatmap
        calls["content_aware"] += 1
        assert np.any(overlap_in)
        return canvas_in

    def _fake_multiband(*_args, **_kwargs):
        calls["multiband"] += 1
        return canvas

    monkeypatch.setattr(blender, "_content_aware_feather_blend", _fake_content_aware)
    monkeypatch.setattr(blender, "_multiband_blend_roi", _fake_multiband)

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert calls["content_aware"] == 1
    assert calls["multiband"] == 0


def test_ghost_guard_allows_multiband_when_risk_is_low(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    calls = {"fallback": 0, "multiband": 0}

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.1)

    def _fake_fallback(canvas_in, *_args, **_kwargs):
        calls["fallback"] += 1
        return canvas_in

    def _fake_multiband(canvas_in, *_args, **_kwargs):
        calls["multiband"] += 1
        return canvas_in

    monkeypatch.setattr(blender, "_fallback_feather_blend", _fake_fallback)
    monkeypatch.setattr(blender, "_multiband_blend_roi", _fake_multiband)

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert calls["fallback"] == 0
    assert calls["multiband"] == 1


def test_ghost_guard_feather_mode_uses_legacy_fallback_on_high_risk(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_mode="feather",
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    calls = {"fallback": 0, "content_aware": 0}

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.9)

    def _fake_fallback(canvas_in, img_in, canvas_mask_in, img_mask_in, overlap_in, blended_mask_in=None):
        calls["fallback"] += 1
        assert np.any(overlap_in)
        return canvas_in

    def _fake_content_aware(*_args, **_kwargs):
        calls["content_aware"] += 1
        return canvas

    monkeypatch.setattr(blender, "_fallback_feather_blend", _fake_fallback)
    monkeypatch.setattr(blender, "_content_aware_feather_blend", _fake_content_aware)

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert calls["fallback"] == 1
    assert calls["content_aware"] == 0


def test_ghost_guard_routes_adaptive_feather_to_fallback_on_high_risk(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="adaptive-feather",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    calls = {"content_aware": 0, "adaptive_feather": 0}

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.9)

    def _fake_content_aware(canvas_in, img_in, canvas_mask_in, img_mask_in, overlap_in, blended_mask=None, seam_heatmap=None):
        del blended_mask, seam_heatmap
        calls["content_aware"] += 1
        assert np.any(overlap_in)
        return canvas_in

    def _fake_adaptive_feather(*_args, **_kwargs):
        calls["adaptive_feather"] += 1
        return canvas

    monkeypatch.setattr(blender, "_content_aware_feather_blend", _fake_content_aware)
    monkeypatch.setattr(blender, "_adaptive_feather_blend", _fake_adaptive_feather)

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert calls["content_aware"] == 1
    assert calls["adaptive_feather"] == 0


def test_ghost_guard_routes_to_content_aware_seam_fallback_on_high_risk(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    calls = {"content_aware": 0, "multiband": 0}

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.9)

    def _fake_content_aware(canvas_in, img_in, canvas_mask_in, img_mask_in, overlap_in, blended_mask=None, seam_heatmap=None):
        del img_in, canvas_mask_in, img_mask_in, blended_mask, seam_heatmap
        calls["content_aware"] += 1
        assert np.any(overlap_in)
        return canvas_in

    def _fake_multiband(*_args, **_kwargs):
        calls["multiband"] += 1
        return canvas

    monkeypatch.setattr(blender, "_content_aware_feather_blend", _fake_content_aware)
    monkeypatch.setattr(blender, "_multiband_blend_roi", _fake_multiband)

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert calls["content_aware"] == 1
    assert calls["multiband"] == 0


def test_ghost_guard_logs_selected_mode_and_risk(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_mode="content-aware",
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    debug_messages = []

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.9)
    monkeypatch.setattr(
        "cielostitch_core.core.blender.logger.debug",
        lambda msg, *args: debug_messages.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        blender,
        "_content_aware_feather_blend",
        lambda canvas_in, *_args, **_kwargs: canvas_in,
    )

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert any("ghost_guard triggered" in message for message in debug_messages)
    assert any("blend_type=multiband" in message for message in debug_messages)
    assert any("mode=content-aware" in message for message in debug_messages)


def test_ghost_guard_feather_mode_logs_legacy_fallback(monkeypatch) -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="multiband",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        ghost_guard_enabled=True,
        ghost_guard_mode="feather",
        ghost_guard_risk_threshold=0.40,
        ghost_guard_feather_px=12,
    )

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)
    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    debug_messages = []

    monkeypatch.setattr(blender, "_compute_overlap_seam_risk", lambda *_args, **_kwargs: 0.9)
    monkeypatch.setattr(
        "cielostitch_core.core.blender.logger.debug",
        lambda msg, *args: debug_messages.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        blender,
        "_fallback_feather_blend",
        lambda canvas_in, *_args, **_kwargs: canvas_in,
    )

    blender.blend(canvas, canvas_mask, img, img_mask, use_gain=False)

    assert any("ghost_guard triggered" in message for message in debug_messages)
    assert any("ghost_guard fallback applied" in message for message in debug_messages)
    assert any("blend_type=multiband ghost_guard_mode=feather" in message for message in debug_messages)


@pytest.mark.parametrize(
    "field_name",
    [
        "adaptive_mb_risk_boost_threshold",
        "adaptive_mb_low_risk_threshold",
        "adaptive_mb_max_boost",
        "photometric_min_overlap_px",
        "photometric_full_confidence_px",
        "ghost_guard_risk_threshold",
        "ghost_guard_feather_px",
    ],
)
def test_blender_init_rejects_invalid_numeric_knobs(field_name: str) -> None:
    kwargs = dict(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="seamless",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        adaptive_mb_risk_boost_threshold=0.45,
        adaptive_mb_low_risk_threshold=0.10,
        adaptive_mb_max_boost=1,
        photometric_min_overlap_px=2_000,
        photometric_full_confidence_px=250_000,
        ghost_guard_enabled=False,
        ghost_guard_risk_threshold=0.55,
        ghost_guard_feather_px=18,
    )
    kwargs[field_name] = "invalid"

    with pytest.raises(ValueError, match=f"Invalid value for '{field_name}'"):
        Blender(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "adaptive_mb_max_boost",
        "photometric_min_overlap_px",
        "photometric_full_confidence_px",
        "ghost_guard_feather_px",
    ],
)
def test_blender_init_rejects_non_integral_numeric_for_integer_knobs(field_name: str) -> None:
    kwargs = dict(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="seamless",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        adaptive_mb_risk_boost_threshold=0.45,
        adaptive_mb_low_risk_threshold=0.10,
        adaptive_mb_max_boost=1,
        photometric_min_overlap_px=2_000,
        photometric_full_confidence_px=250_000,
        ghost_guard_enabled=False,
        ghost_guard_risk_threshold=0.55,
        ghost_guard_feather_px=18,
    )
    kwargs[field_name] = 1.9

    with pytest.raises(ValueError, match=f"Invalid value for '{field_name}'"):
        Blender(**kwargs)


def test_blender_init_coalesces_none_for_ghost_guard_numeric_knobs() -> None:
    blender = Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type="seamless",
        multiband_levels=5,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
        adaptive_mb_risk_boost_threshold=0.45,
        adaptive_mb_low_risk_threshold=0.10,
        adaptive_mb_max_boost=1,
        photometric_min_overlap_px=2_000,
        photometric_full_confidence_px=250_000,
        ghost_guard_enabled=True,
        ghost_guard_risk_threshold=None,
        ghost_guard_feather_px=None,
    )

    assert blender.ghost_guard_enabled is True
    assert blender.ghost_guard_risk_threshold == pytest.approx(0.55)
    assert blender.ghost_guard_feather_px == 20
