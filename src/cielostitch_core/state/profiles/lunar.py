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
class LunarProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    gain_compensation = "simple"
    # Tile size for local gain computation (only used when gain_compensation="local").
    # Medium tiles for illumination gradient and crater detail.
    local_gain_tile_size = 64
    # Blending method: feather (sharper seams) or multiband (hides broad gradients better).
    blend_type = "feather"  # options: "multiband", "feather", "none"
    # Pyramid depth for multiband blending; more levels blend larger-scale differences.
    multiband_levels = 4
    # Default feather radius for Feather blending (pixels). Exposed in Advanced → Blending.
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
    # Extra blending tuning (not in Advanced groups ordering but kept here for convenience):
    blend_offset_match = True
    blend_offset_clamp = 350.0
    gain_clamp = (0.9, 1.1)
    photometric_min_overlap_px = 2_000
    photometric_full_confidence_px = 250_000
    # Illumination normalization parameters
    illumination_min_overlap_px = 2_000
    illumination_full_confidence_px = 250_000

    # --- Feature Detection ---
    # Max keypoints per image (caps detected points to keep speed reasonable)
    detector_max_points = 6000
    # Feature sensitivity (higher = fewer, stronger points; lower = more points)
    detector_feature_sensitivity = 0.015  # maps to SIFT contrastThreshold
    # Detection scale (analyze a smaller copy for speed; 0.5 = half-size)
    detector_downscale = 0.6

    # --- Feature Matching ---
    # Lowe’s ratio threshold for feature matching; adjust to trade recall vs. mismatch risk.
    ratio_test = 0.7
    # RANSAC reprojection threshold (px) for accepting inliers in homography.
    ransac_thresh = 3.0
    # Minimum ratio of inliers to total matches to accept a match.
    min_inlier_ratio = 0.10
    # Absolute minimum inlier count to accept a match.
    min_inliers = 10
    # Note: UI also exposes detector_downscale under Feature Matching; value shared with detection.
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    # Transform model for feature matching: 'auto', 'affine', 'homography', 'translation'
    transform_mode = "affine"