# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from dataclasses import dataclass

@dataclass
class PanoramaProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    # Uniform works well for most panoramas; simple is lighter-weight, local helps uneven lighting.
    gain_compensation = "uniform"
    # Tile size for local gain computation (only used when gain_compensation="local").
    local_gain_tile_size = 128
    # Affine-friendly default to reduce ghosting/diplopia on handheld sweeps.
    blend_type = "adaptive-feather"
    multiband_levels = 4
    blend_feather_px = 24
    # Gain computation method: "mean", "median", or "trimmed" (excludes outliers).
    gain_method = "median"
    # Compute gain from luminance only (prevents color shifts in RGB).
    lum_only_gain = True
    # Pre-blend histogram matching to reduce vignetting artifacts.
    histogram_matching = True
    edge_aware_smoothing = True
    adaptive_mb_risk_boost_threshold = 0.45
    adaptive_mb_low_risk_threshold = 0.10
    adaptive_mb_max_boost = 1
    ghost_guard_enabled = True
    ghost_guard_risk_threshold = 0.55
    ghost_guard_feather_px = 20 #18
    # Extras
    blend_offset_match = True
    blend_offset_clamp = 350.0
    gain_clamp = (0.92, 1.10)
    photometric_min_overlap_px = 2_000
    photometric_full_confidence_px = 250_000
    # Illumination normalization parameters
    illumination_min_overlap_px = 2_000
    illumination_full_confidence_px = 250_000

    # --- Feature Detection ---
    # Detection settings
    detector_max_points = 9000
    detector_feature_sensitivity = 0.018
    detector_downscale = 0.6

    # --- Feature Matching ---
    # Feature matching and geometry (general landscape/panorama)
    ratio_test = 0.82
    ransac_thresh = 10.0
    min_inlier_ratio = 0.10
    min_inliers = 8
    # Transform model for feature matching: 'auto', 'affine', 'homography', 'translation'
    transform_mode = "homography"
    # Panorama-specific homography controls for wide-FOV handheld captures.
    homography_max_matches = 512
    # Keep projective solves flexible but reject extreme warp expansion that usually
    # indicates an unstable fit; affine fallback remains enabled.
    homography_axis_growth = 10.0
    homography_linear_cond_max = 5e4
    homography_allow_affine_fallback = True

    # --- Additional processing ---
    # Panel flattening (rare for daytime panos)
    enable_panel_flatten = False
    flatten_sigma_px = 160
    flatten_strength = 0.20
