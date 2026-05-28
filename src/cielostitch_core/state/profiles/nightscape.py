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
class NightscapeProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    # Using uniform for now; simple is lighter-weight, local helps light-pollution gradients.
    gain_compensation = "uniform"
    # Tile size for local gain computation (only used when gain_compensation="local").
    # Larger tiles for smooth sky gradient handling.
    local_gain_tile_size = 128
    blend_type = "multiband"
    multiband_levels = 4
    blend_feather_px = 120
    # Gain computation method: "mean", "median", or "trimmed" (excludes outliers).
    gain_method = "trimmed"
    # Compute gain from luminance only (prevents color shifts in RGB).
    lum_only_gain = True
    # Pre-blend histogram matching to reduce vignetting artifacts.
    histogram_matching = True
    edge_aware_smoothing = True
    adaptive_mb_risk_boost_threshold = 0.45
    adaptive_mb_low_risk_threshold = 0.10
    adaptive_mb_max_boost = 1
    # Extra blending tuning
    blend_offset_match = True
    blend_offset_clamp = 450.0
    gain_clamp = (0.85, 1.20)
    photometric_min_overlap_px = 2_000
    photometric_full_confidence_px = 250_000
    # Illumination normalization parameters
    illumination_min_overlap_px = 2_000
    illumination_full_confidence_px = 250_000

    # --- Feature Detection ---
    # Detection settings (retain small stars)
    detector_max_points = 10000
    detector_feature_sensitivity = 0.008
    detector_downscale = 1.0

    # --- Feature Matching ---
    # GridMode feature matching tuned for night scenes (stars + foreground)
    ratio_test = 0.75
    ransac_thresh = 3.0
    min_inlier_ratio = 0.10
    min_inliers = 10
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    # Transform model for feature matching: 'auto', 'affine', 'homography', 'translation'
    transform_mode = "affine"
    
    # --- Additional processing ---
    # Panel flattening helps night gradients
    enable_panel_flatten = True
    flatten_sigma_px = 200
    flatten_strength = 0.35
