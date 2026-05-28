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
class LandscapeProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    gain_compensation = "none"
    # Tile size for local gain computation (only used when gain_compensation="local").
    local_gain_tile_size = 128
    # Blend method: feather (fast) or multiband (better for gradients/uneven lighting).
    blend_type = "multiband"  # options: "multiband", "feather", "none"
    # Multiband pyramid depth.
    multiband_levels = 4
    # Feather width used in feather mode or as helper window in multiband.
    blend_feather_px = 40
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
    # Max keypoints per image (caps detected points to keep speed reasonable)
    detector_max_points = 6000
    # Feature sensitivity (higher = fewer, stronger points; lower = more points)
    detector_feature_sensitivity = 0.02  # maps to SIFT contrastThreshold
    # Detection scale (analyze a smaller copy for speed; 0.5 = half-size)
    detector_downscale = 0.5

    # --- Feature Matching ---
    # Lowe’s ratio threshold for descriptor match filtering; lower is stricter.
    ratio_test = 0.80
    # RANSAC reprojection threshold (px) for inlier acceptance in homography.
    ransac_thresh = 8.0
    # Minimum inlier ratio to total matches to accept a match.
    min_inlier_ratio = 0.08
    # Absolute minimum inlier count to accept a match.
    min_inliers = 8
    # Note: detector_downscale shared with Detection section.
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    # Transform model for feature matching: 'auto', 'affine', 'homography', 'translation'
    transform_mode = "affine"
