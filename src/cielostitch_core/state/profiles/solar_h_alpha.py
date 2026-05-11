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
class SolarHAlphaProfile:
    # --- Blending ---
    seamless_quality = "balanced"
    # Gain compensation mode: "none", "simple", "uniform" (global), or "local" (sub-tile map).
    gain_compensation = "simple"
    # Tile size for local gain computation (only used when gain_compensation="local").
    # Fine tiles for capturing limb darkening and prominence details.
    local_gain_tile_size = 48
    blend_type = "multiband"
    multiband_levels = 3
    blend_feather_px = 130  # broader seam for smooth gradients
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
    # Extra blending tuning
    blend_offset_match = True
    blend_offset_clamp = 350.0
    gain_clamp = (0.90, 1.10)
    photometric_min_overlap_px = 2_000
    photometric_full_confidence_px = 250_000
    # Illumination normalization parameters
    illumination_min_overlap_px = 2_000
    illumination_full_confidence_px = 250_000

    # --- Feature Detection ---
    # Detection settings (maps to SIFT)
    detector_max_points = 7000
    detector_feature_sensitivity = 0.010  # more permissive for low-contrast detail
    detector_downscale = 0.8

    # --- Feature Matching ---
    # Feature matching and geometry (H-alpha: lower micro-contrast away from active regions)
    ratio_test = 0.80  # a bit looser to retain more tentative matches
    # RANSAC inlier reprojection error (px) for homography; smaller favors precision over recall.
    ransac_thresh = 3.0  # px at full-res
    # Minimum ratio of inliers to total matches to accept a solve.
    min_inlier_ratio = 0.10
    # Absolute minimum inlier count to accept a solve.
    min_inliers = 10
    # Note: UI also exposes detector_downscale under Feature Matching; value shared with detection.
    # Prohibit rotation in transform solve when True; stabilizes grid mosaics on tracking mounts.
    lock_rotation = False

    # --- Additional processing (not part of Advanced groups) ---
    # Remove low‑frequency illumination variations inside each warped panel before blending.
    enable_panel_flatten = False
    # Gaussian sigma (px) for estimating low‑frequency field in panel flatten.
    flatten_sigma_px = 160
    # Strength (0..1) of low‑frequency subtraction applied to the panel.
    flatten_strength = 0.25
