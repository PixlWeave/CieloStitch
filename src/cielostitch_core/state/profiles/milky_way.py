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
class MilkyWayProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    # Using uniform for now; simple is lighter-weight, local helps challenging gradient cases.
    gain_compensation = "uniform"
    # Tile size for local gain computation (only used when gain_compensation="local").
    # Larger tiles for sky gradients and airglow variations.
    local_gain_tile_size = 128
    blend_type = "multiband"
    multiband_levels = 4
    blend_feather_px = 160
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
    blend_offset_clamp = 650.0
    gain_clamp = (0.80, 1.25)
    photometric_min_overlap_px = 2_000
    photometric_full_confidence_px = 250_000
    # Illumination normalization parameters
    illumination_min_overlap_px = 2_000
    illumination_full_confidence_px = 250_000

    # --- Feature Detection ---
    # Detection settings (lots of small points)
    detector_max_points = 16000
    detector_feature_sensitivity = 0.006
    detector_downscale = 1.0

    # --- Feature Matching ---
    # Very dense star fields require stricter matching and higher evidence
    ratio_test = 0.70
    ransac_thresh = 2.5
    min_inlier_ratio = 0.12
    min_inliers = 16
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    lock_rotation = False

    # --- Additional processing ---
    # Flatten to suppress airglow/LP gradients
    enable_panel_flatten = True
    flatten_sigma_px = 280
    flatten_strength = 0.45

