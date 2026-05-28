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
    assert out.get("enable_grid_shift_guard") is False


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
    assert out["local_gain_tile_size"] >= 64
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


def test_resolver_disables_phase_fallback_for_panorama():
    inp = ResolverInputs(
        subject="panorama",
        panel_w=4032,
        panel_h=3024,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        stitch_mode="freeform",
        mount_precision="manual/handheld",
        bit_depth=16,
    )
    out = compute_resolved_params(inp)

    assert out.get("enable_grid_phase_fallback") is False


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
        panel_count=6,
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


def test_resolver_computes_additional_geometry_driven_user_inputs():
    inp = ResolverInputs(
        subject="general",
        panel_w=6000,
        panel_h=4000,
        panel_count=14,
        overlap_x_pct=12.0,
        overlap_y_pct=10.0,
        stitch_mode="freeform",
        mount_precision="normal",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["transform_mode"] == "affine"  # general is treated as a stable/scientific profile
    assert out["enable_grid_nominal_fallback"] is True
    assert out["staged_matching_mode"] == "manual"
    assert 4 <= out["refs_per_stage"] <= 8
    assert out["photometric_min_overlap_px"] >= 2000
    assert out["photometric_full_confidence_px"] >= out["photometric_min_overlap_px"]


def test_resolver_grid_guided_tight_prefers_translation_and_auto_staging():
    inp = ResolverInputs(
        subject="solar",
        panel_w=3000,
        panel_h=3000,
        panel_count=9,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        stitch_mode="grid-guided",
        mount_precision="tight",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["transform_mode"] == "translation"
    assert out["enable_grid_nominal_fallback"] is True
    assert out["staged_matching_mode"] == "auto"
    assert out["refs_per_stage"] == 6


def test_resolver_cylindrical_projection_prefers_affine_over_homography():
    inp = ResolverInputs(
        subject="panorama",
        panel_w=6000,
        panel_h=4000,
        panel_count=8,
        stitch_mode="freeform",
        # projection_mode="cylindrical",
        fpx_mode="fov",
        hfov_deg=50.0,
        mount_precision="normal",
        overlap_x_pct=20.0,
        overlap_y_pct=20.0,
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["transform_mode"] == "apap"  # 8 panels cylindrical: count≥5 → APAP even post-projection


def test_resolver_cylindrical_projection_tightens_ransac_threshold():
    native = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=6000,
            panel_h=4000,
            panel_count=8,
            stitch_mode="freeform",
            # projection_mode="native",
            fpx_mode="fov",
            hfov_deg=10.0,
            mount_precision="normal",
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            bit_depth=16,
        )
    )
    cylindrical = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=6000,
            panel_h=4000,
            panel_count=8,
            stitch_mode="freeform",
            fpx_mode="fov",
            hfov_deg=60.0,
            # projection_mode="cylindrical",
            mount_precision="normal",
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            bit_depth=16,
        )
    )

    assert native["projection_mode"] == "native"
    assert cylindrical["projection_mode"] == "cylindrical"
    assert cylindrical["ransac_thresh"] < native["ransac_thresh"]


def test_resolver_recommends_cylindrical_for_wide_natural_scene_when_projection_is_unset():
    out = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=6000,
            panel_h=4000,
            panel_count=8,
            stitch_mode="freeform",
            # projection_mode=None,
            mount_precision="normal",
            fpx_factor=0.5,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            bit_depth=16,
        )
    )

    assert out["projection_mode"] == "cylindrical"


def test_resolver_does_not_emit_camera_intrinsics_in_output():
    out = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=6000,
            panel_h=4000,
            panel_count=6,
            focal_length_mm=50.0,
            sensor_width_mm=25.0,
            sensor_height_mm=16.0,
            camera_angle_deg=90.0,
            stitch_mode="freeform",
            # projection_mode=None,
            mount_precision="normal",
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            bit_depth=16,
        )
    )

    assert "fpx_mode" not in out
    assert "focal_length_mm" not in out
    assert "sensor_width_mm" not in out
    assert "sensor_height_mm" not in out
    assert "camera_angle_deg" not in out


def test_resolver_does_not_emit_fov_intrinsics_in_output():
    out = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=6000,
            panel_h=4000,
            panel_count=6,
            fpx_mode="fov",
            focal_length_mm=50.0,
            sensor_width_mm=25.0,
            sensor_height_mm=16.0,
            hfov_deg=42.0,
            camera_angle_deg=90.0,
            stitch_mode="freeform",
            mount_precision="normal",
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            bit_depth=16,
        )
    )

    assert "fpx_mode" not in out
    assert "hfov_deg" not in out
    assert "focal_length_mm" not in out


def test_resolver_does_not_emit_factor_intrinsics_in_output():
    out = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=6000,
            panel_h=4000,
            panel_count=6,
            fpx_mode="factor",
            focal_length_mm=50.0,
            sensor_width_mm=25.0,
            sensor_height_mm=16.0,
            hfov_deg=42.0,
            stitch_mode="freeform",
            mount_precision="normal",
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            bit_depth=16,
        )
    )

    assert "fpx_mode" not in out
    assert "fpx_factor" not in out
    assert "hfov_deg" not in out


def test_resolver_disables_ghost_guard_for_general_freeform_manual_capture():
    inp = ResolverInputs(
        subject="general",
        panel_w=6000,
        panel_h=4000,
        panel_count=10,
        overlap_x_pct=15.0,
        overlap_y_pct=12.0,
        stitch_mode="freeform",
        mount_precision="manual",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["ghost_guard_enabled"] is False
    assert out["ghost_guard_mode"] == "feather"
    assert out["ghost_guard_risk_threshold"] == 0.55
    assert 12 <= out["ghost_guard_feather_px"] <= 36


def test_resolver_keeps_ghost_guard_off_for_stable_lunar_grid_capture():
    inp = ResolverInputs(
        subject="lunar",
        panel_w=4000,
        panel_h=3000,
        panel_count=6,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        stitch_mode="grid-guided",
        mount_precision="tight",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["ghost_guard_enabled"] is False
    assert out["ghost_guard_mode"] == "feather"
    assert out["ghost_guard_risk_threshold"] == 0.55
    assert 12 <= out["ghost_guard_feather_px"] <= 36


def test_resolver_prefers_feather_ghost_guard_for_stable_solar_capture():
    out = compute_resolved_params(
        ResolverInputs(
            subject="solar",
            panel_w=3000,
            panel_h=3000,
            panel_count=6,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="tight",
            bit_depth=16,
        )
    )

    assert out["ghost_guard_mode"] == "feather"


def test_resolver_enables_content_aware_ghost_guard_for_milky_way_manual_low_overlap():
    out = compute_resolved_params(
        ResolverInputs(
            subject="milky-way",
            panel_w=6000,
            panel_h=4000,
            panel_count=8,
            overlap_x_pct=14.0,
            overlap_y_pct=14.0,
            stitch_mode="freeform",
            mount_precision="manual",
            bit_depth=16,
        )
    )

    assert out["ghost_guard_enabled"] is True
    assert out["ghost_guard_mode"] == "content-aware"


def test_resolver_calculates_subject_aware_local_gain_tile_size():
    solar = compute_resolved_params(
        ResolverInputs(
            subject="solar",
            panel_w=3000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="tight",
            bit_depth=16,
        )
    )
    lunar = compute_resolved_params(
        ResolverInputs(
            subject="lunar",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=25.0,
            overlap_y_pct=25.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
            bit_depth=16,
        )
    )
    general = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=6000,
            panel_h=4000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="normal",
            bit_depth=16,
        )
    )

    assert 32 <= solar["local_gain_tile_size"] <= 96
    assert 48 <= lunar["local_gain_tile_size"] <= 128
    assert 64 <= general["local_gain_tile_size"] <= 192
    assert solar["local_gain_tile_size"] <= lunar["local_gain_tile_size"] <= general["local_gain_tile_size"]


def test_resolver_keeps_non_solar_low_overlap_grid_capture_on_no_gain_compensation():
    inp = ResolverInputs(
        subject="general",
        panel_w=6000,
        panel_h=4000,
        overlap_x_pct=1.0,
        overlap_y_pct=1.0,
        stitch_mode="grid-guided",
        mount_precision="tight",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["gain_compensation"] == "none"


def test_resolver_uses_simple_gain_for_tight_mid_overlap_milky_way_capture():
    inp = ResolverInputs(
        subject="milky-way",
        panel_w=6000,
        panel_h=4000,
        panel_count=8,
        overlap_x_pct=4.0,
        overlap_y_pct=4.0,
        stitch_mode="freeform",
        mount_precision="normal",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["gain_compensation"] == "simple"


def test_resolver_keeps_nightscape_handheld_capture_on_uniform_gain():
    inp = ResolverInputs(
        subject="nightscape",
        panel_w=6000,
        panel_h=4000,
        panel_count=8,
        overlap_x_pct=4.0,
        overlap_y_pct=4.0,
        stitch_mode="freeform",
        mount_precision="manual/handheld",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["gain_compensation"] == "uniform"


def test_resolver_uses_simple_gain_for_tight_compact_nightscape_capture():
    inp = ResolverInputs(
        subject="nightscape",
        panel_w=6000,
        panel_h=4000,
        panel_count=6,
        overlap_x_pct=4.0,
        overlap_y_pct=4.0,
        stitch_mode="freeform",
        mount_precision="normal",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["gain_compensation"] == "simple"


def test_resolver_keeps_larger_nightscape_session_on_uniform_gain():
    inp = ResolverInputs(
        subject="nightscape",
        panel_w=6000,
        panel_h=4000,
        panel_count=12,
        overlap_x_pct=4.0,
        overlap_y_pct=4.0,
        stitch_mode="freeform",
        mount_precision="normal",
        bit_depth=16,
    )

    out = compute_resolved_params(inp)

    assert out["gain_compensation"] == "uniform"


def test_resolver_grid_ransac_threshold_increases_with_mount_instability():
    tight = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="tight",
        )
    )
    normal = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )
    sloppy = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="sloppy",
        )
    )
    manual = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="manual",
        )
    )
    handheld = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="handheld",
        )
    )

    assert tight["ransac_thresh"] <= normal["ransac_thresh"] <= sloppy["ransac_thresh"]
    assert sloppy["ransac_thresh"] <= manual["ransac_thresh"] <= handheld["ransac_thresh"]


def test_resolver_freeform_min_inliers_grows_with_image_size():
    small = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=1200,
            panel_h=800,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="normal",
        )
    )
    large = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=8000,
            panel_h=6000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="normal",
        )
    )

    assert small["min_inliers"] >= 8
    assert large["min_inliers"] > small["min_inliers"]


def test_resolver_grid_guided_keeps_min_inliers_no_higher_than_freeform_for_same_panel_size():
    grid = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=6000,
            panel_h=4000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )
    freeform = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=6000,
            panel_h=4000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="normal",
        )
    )

    assert grid["min_inliers"] <= freeform["min_inliers"]


def test_resolver_phase_fallback_response_is_stricter_for_smooth_gradient_low_overlap_than_natural_scene():
    smooth = compute_resolved_params(
        ResolverInputs(
            subject="solar",
            panel_w=3000,
            panel_h=3000,
            overlap_x_pct=8.0,
            overlap_y_pct=8.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )
    natural = compute_resolved_params(
        ResolverInputs(
            subject="panorama",
            panel_w=3000,
            panel_h=3000,
            overlap_x_pct=8.0,
            overlap_y_pct=8.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )

    assert smooth["grid_phase_fallback_min_response"] > natural["grid_phase_fallback_min_response"]


def test_resolver_phase_fallback_response_relaxes_with_healthier_overlap_for_smooth_gradient_subjects():
    low_overlap = compute_resolved_params(
        ResolverInputs(
            subject="solar-h-alpha",
            panel_w=3000,
            panel_h=3000,
            overlap_x_pct=6.0,
            overlap_y_pct=6.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )
    higher_overlap = compute_resolved_params(
        ResolverInputs(
            subject="solar-h-alpha",
            panel_w=3000,
            panel_h=3000,
            overlap_x_pct=24.0,
            overlap_y_pct=24.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )

    assert low_overlap["grid_phase_fallback_min_response"] >= higher_overlap["grid_phase_fallback_min_response"]


def test_resolver_grid_min_inlier_ratio_tracks_mount_instability():
    tight = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="tight",
        )
    )
    normal = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="normal",
        )
    )
    sloppy = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="grid-guided",
            mount_precision="sloppy",
        )
    )

    assert tight["min_inlier_ratio"] <= normal["min_inlier_ratio"] <= sloppy["min_inlier_ratio"]


def test_resolver_freeform_min_inlier_ratio_tracks_mount_instability():
    tight = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="tight",
        )
    )
    normal = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="normal",
        )
    )
    handheld = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=4000,
            panel_h=3000,
            overlap_x_pct=20.0,
            overlap_y_pct=20.0,
            stitch_mode="freeform",
            mount_precision="handheld",
        )
    )

    assert tight["min_inlier_ratio"] <= normal["min_inlier_ratio"] <= handheld["min_inlier_ratio"]


def test_resolver_freeform_low_overlap_requires_higher_inlier_ratio_than_healthy_overlap():
    low_overlap = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=6000,
            panel_h=4000,
            overlap_x_pct=8.0,
            overlap_y_pct=8.0,
            stitch_mode="freeform",
            mount_precision="normal",
        )
    )
    healthy_overlap = compute_resolved_params(
        ResolverInputs(
            subject="general",
            panel_w=6000,
            panel_h=4000,
            overlap_x_pct=24.0,
            overlap_y_pct=24.0,
            stitch_mode="freeform",
            mount_precision="normal",
        )
    )

    assert low_overlap["min_inlier_ratio"] >= healthy_overlap["min_inlier_ratio"]
