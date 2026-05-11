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
    blend_type = "multiband"
    multiband_levels = 4
    blend_feather_px = 60
    # Gain computation method: "mean", "median", or "trimmed" (excludes outliers).
    gain_method = "median"
    # Compute gain from luminance only (prevents color shifts in RGB).
    lum_only_gain = True
    # Pre-blend histogram matching to reduce vignetting artifacts.
    histogram_matching = False
    edge_aware_smoothing = True
    adaptive_mb_risk_boost_threshold = 0.45
    adaptive_mb_low_risk_threshold = 0.10
    adaptive_mb_max_boost = 1
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
    detector_max_points = 6000
    detector_feature_sensitivity = 0.020
    detector_downscale = 0.5

    # --- Feature Matching ---
    # Feature matching and geometry (general landscape/panorama)
    ratio_test = 0.80
    ransac_thresh = 8.0
    min_inlier_ratio = 0.15
    min_inliers = 12
    lock_rotation = False

    # --- Additional processing ---
    # Panel flattening (rare for daytime panos)
    enable_panel_flatten = False
    flatten_sigma_px = 160
    flatten_strength = 0.20
