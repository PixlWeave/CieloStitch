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
class GeneralProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    gain_compensation = "none"
    # Tile size for local gain computation (only used when gain_compensation="local").
    local_gain_tile_size = 128
    # Blend method: multiband recommended to hide seams in general composites.
    blend_type = "none"  # options: "multiband", "feather", "none"
    # Multiband pyramid depth.
    multiband_levels = 4
    # Feather width used in feather mode or as helper window in multiband.
    blend_feather_px = 40
    # Gain computation method: "mean", "median", or "trimmed" (excludes outliers).
    gain_method = "median"
    # Compute gain from luminance only (prevents color shifts in RGB).
    lum_only_gain = False
    # Pre-blend histogram matching to reduce vignetting artifacts.
    histogram_matching = False
    edge_aware_smoothing = True
    adaptive_mb_risk_boost_threshold = 0.45
    adaptive_mb_low_risk_threshold = 0.10
    adaptive_mb_max_boost = 1
    # Enable additive offset matching across overlap to reduce residual steps.
    blend_offset_match = True
    # Clamp for offset magnitude to bound correction.
    blend_offset_clamp = 350.0
    gain_clamp = (0.92, 1.10)
    photometric_min_overlap_px = 2_000
    photometric_full_confidence_px = 250_000
    # Illumination normalization parameters
    illumination_min_overlap_px = 2_000
    illumination_full_confidence_px = 250_000

    # --- Feature Detection ---
    # Detection settings (user-friendly labels):
    # Max keypoints per image (caps detected points to keep speed reasonable)
    detector_max_points = 6000
    # Feature sensitivity (higher = fewer, stronger points; lower = more points)
    detector_feature_sensitivity = 0.02  # maps to SIFT contrastThreshold
    # Detection scale (analyze a smaller copy for speed; 0.5 = half-size)
    detector_downscale = 0.5

    # --- Feature Matching ---
    # Lowe’s ratio threshold for feature matching (constructor-required for GridMode); not used when forced layout.
    ratio_test = 0.82
    # RANSAC reprojection threshold (px) for inlier acceptance; unused when `force_grid_layout=True`.
    ransac_thresh = 5.0
    # Minimum ratio of inliers to matches to accept solve; improves robustness; unused in forced layout.
    min_inlier_ratio = 0.08
    # Absolute minimum inlier count; guards against tiny, unreliable solves; unused in forced layout.
    min_inliers = 8
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    lock_rotation = True

