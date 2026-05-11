# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

from cielostitch_core.core.session_resolver import ResolverInputs, compute_resolved_params


def test_resolver_basic_lunar_grid():
    inp = ResolverInputs(
        subject="lunar",
        panel_w=4000,
        panel_h=3000,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        stitch_mode="grid-guided",
        mount_precision="tight",
        bit_depth=16,
    )
    out = compute_resolved_params(inp)
    # Key expectations
    assert 0.4 <= out["detector_downscale"] <= 1.0
    assert 500 <= out["detector_max_points"] <= 12000
    assert out["ratio_test"] <= 0.75  # lunar base 0.70 tightened a bit
    assert out["ransac_thresh"] <= 4.0
    assert out["blend_type"] in {"feather", "multiband"}
    assert out["multiband_levels"] >= 2
    # Grid values present
    # assert out.get("overlap_x_pct") == 25.0
    # assert out.get("overlap_y_pct") == 25.0
    assert out.get("enable_grid_shift_guard") is True


def test_resolver_handheld_random():
    inp = ResolverInputs(
        subject="milky-way",
        panel_w=6000,
        panel_h=4000,
        overlap_x_pct=20.0,
        overlap_y_pct=20.0,
        stitch_mode="auto",
        mount_precision="manual/handheld",
        bit_depth=14,
    )
    out = compute_resolved_params(inp)
    assert out["ransac_thresh"] >= 6.0
    assert out["min_inlier_ratio"] >= 0.12
    assert out["gain_compensation"] == "uniform"
    assert out["blend_offset_match"] is True
    assert out["blend_offset_clamp"] >= 300.0


def test_resolver_disables_grid_shift_guard_for_solar_grid_guided():
    inp = ResolverInputs(
        subject="solar-h-alpha",
        panel_w=3000,
        panel_h=3000,
        overlap_x_pct=20.0,
        overlap_y_pct=20.0,
        stitch_mode="grid-guided",
        mount_precision="tight",
        bit_depth=16,
    )
    out = compute_resolved_params(inp)

    assert out.get("enable_grid_shift_guard") is False
    assert out.get("enable_grid_phase_fallback") is True


def test_resolver_speed_preset_fast_reduces_quality_knobs():
    base = ResolverInputs(
        subject="lunar",
        panel_w=4000,
        panel_h=3000,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        stitch_mode="auto",
        mount_precision="normal",
        bit_depth=16,
        speed_preset="balanced",
    )
    fast = ResolverInputs(**{**base.__dict__, "speed_preset": "fast"})

    out_balanced = compute_resolved_params(base)
    out_fast = compute_resolved_params(fast)

    assert out_fast["detector_max_points"] <= out_balanced["detector_max_points"]
    assert out_fast["multiband_levels"] <= out_balanced["multiband_levels"]
    assert out_fast["gain_compensation"] == "none"


def test_resolver_speed_preset_best_increases_quality_knobs():
    base = ResolverInputs(
        subject="lunar",
        panel_w=4000,
        panel_h=3000,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        stitch_mode="auto",
        mount_precision="normal",
        bit_depth=16,
        speed_preset="balanced",
    )
    best = ResolverInputs(**{**base.__dict__, "speed_preset": "best"})

    out_balanced = compute_resolved_params(base)
    out_best = compute_resolved_params(best)

    assert out_best["detector_downscale"] >= out_balanced["detector_downscale"]
    assert out_best["detector_max_points"] >= out_balanced["detector_max_points"]
    assert out_best["multiband_levels"] >= out_balanced["multiband_levels"]
