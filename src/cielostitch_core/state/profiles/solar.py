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
class SolarProfile:
    # --- Blending (order matches gui/profile_states.py: PROFILE_ADVANCED_FIELD_GROUPS → blending) ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    gain_compensation = "simple"
    # Tile size for local gain computation (only used when gain_compensation="local").
    # Fine tiles for capturing limb darkening gradient details.
    local_gain_tile_size = 48
    # Blending algorithm selection: feather (fast, sharp) or multiband (multi‑scale, hides gradients).
    blend_type = "multiband"  # options: "multiband", "feather", "none"
    # Number of pyramid levels for multiband blending; higher hides broader seams.
    multiband_levels = 3
    # Feather width (px) for seam transition in feather mode.
    blend_feather_px = 110
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
    # Confidence schedule for illumination normalization
    illumination_min_overlap_px = 3_000
    illumination_full_confidence_px = 600_000

    # --- Feature Detection ---
    # Max keypoints per image (caps detected points to keep speed reasonable)
    detector_max_points = 7000
    # Feature sensitivity (higher = fewer, stronger points; lower = more points)
    detector_feature_sensitivity = 0.01  # maps to SIFT contrastThreshold
    # Detection scale (analyze a smaller copy for speed; 0.5 = half-size)
    detector_downscale = 1.0

    # --- Feature Matching ---
    # Lowe’s ratio threshold for feature matching; lower is stricter, helps reject bad matches.
    ratio_test = 0.75
    # RANSAC inlier reprojection error (px) for homography; smaller favors precision over recall.
    ransac_thresh = 3.0
    # Minimum ratio of inliers to total matches to accept a solve.
    min_inlier_ratio = 0.10
    # Absolute minimum inlier count to accept a solve.
    min_inliers = 10
    # Note: UI also exposes detector_downscale under Feature Matching; value shared with detection.
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    # Transform model for feature matching: 'auto', 'affine', 'homography', 'translation'
    transform_mode = "affine"
    
    # --- Additional processing (not part of Advanced groups) ---
    # Remove low‑frequency illumination variations inside each warped panel before blending.
    enable_panel_flatten = False
    # Gaussian sigma (px) for estimating low‑frequency field in panel flatten.
    flatten_sigma_px = 160
    # Strength (0..1) of low‑frequency subtraction applied to the panel.
    flatten_strength = 0.25
