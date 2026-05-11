# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Dict, Optional
from ..config.constants import MAX_MULTIBAND_LEVELS


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _norm_profile(name: Optional[str]) -> str:
    n = (name or "").strip().lower().replace(" ", "_")
    # Common aliases
    if n in {"milky way", "milkyway", "milky-way"}:
        return "milky-way"
    if n in {"nightscape", "night_scape", "night scape"}:
        return "nightscape"
    if n in {"ha", "solar_ha", "solar_h_alpha", "solar h alpha", "solar h-alpha"}:
        return "solar-h-alpha"
    return n


def _is_smooth_subject(profile: str) -> bool:
    # Subjects with broad gradients where multiband + larger feather helps
    return profile in {"solar", "solar-h-alpha", "lunar", "milky-way", "nightscape"}


def _is_high_freq_subject(profile: str) -> bool:
    return profile in {"landscape", "panorama", "general"}


def _subject_feature_sensitivity(profile: str) -> float:
    # Lower values → more/key weaker features (like SIFT contrastThreshold)
    if profile == "lunar":
        return 0.014  # within 0.012–0.02
    if profile in {"solar", "solar-h-alpha"}:
        return 0.010  # within 0.008–0.012
    if profile in {"milky-way", "nightscape", "panorama", "landscape", "general"}:
        return 0.018  # within 0.015–0.02
    return 0.015


def _subject_ratio_test(profile: str) -> float:
    if profile == "lunar":
        return 0.70
    if profile in {"solar", "solar-h-alpha"}:
        return 0.78
    if profile in {"milky-way", "nightscape", "panorama", "landscape", "general"}:
        return 0.80
    return 0.76


def _mount_factor(mount_precision: str) -> float:
    mp = (mount_precision or "normal").strip().lower()
    if "tight" in mp:
        return -0.03  # tighten
    if "sloppy" in mp:
        return +0.02
    if "manual" in mp:
        return +0.03
    if "handheld" in mp:
        return +0.04
    return 0.0


def _ransac_thresh(stitch_mode: str, mount_precision: str) -> float:
    mode = (stitch_mode or "freeform").lower()
    mp = (mount_precision or "normal").lower()
    if mode == "grid-guided":
        base = 3.0
        if "sloppy" in mp:
            base = 4.0
        elif "manual" in mp:
            base = 6.0
        elif "handheld" in mp:
            base = 7.0
        return base
    # auto (no layout hints)
    return 8.0 if ("handheld" in mp or "manual" in mp) else 6.0


def _min_inliers(megapx: float) -> int:
    # max(8, round(1.5 * MP)) capped 20
    return int(min(20, max(8, round(1.5 * float(megapx)))))


def _min_inlier_ratio(stitch_mode: str, mount_precision: str) -> float:
    if (stitch_mode or "freeform").lower() == "grid-guided":
        # Grid provides extra structure; accept slightly lower ratio when tight
        mp = (mount_precision or "normal").lower()
        if "tight" in mp:
            return 0.08
        if "sloppy" in mp or "handheld" in mp or "manual" in mp:
            return 0.12
        return 0.10
    # Auto mode needs more evidence
    return 0.12 if ("tight" in (mount_precision or "").lower()) else 0.15


def _grid_tolerances(mount_precision: str) -> tuple[float, float]:
    mp = (mount_precision or "normal").lower()
    if "tight" in mp:
        return 0.12, 0.12
    if "sloppy" in mp:
        return 0.28, 0.20
    if "handheld" in mp or "manual" in mp:
        return 0.35, 0.25
    return 0.20, 0.18


def _grid_bias(profile: str) -> float:
    # Weak texture → bias towards nominal placement more strongly
    if profile in {"solar", "solar-h-alpha"}:
        return 0.50
    if profile == "lunar":
        return 0.20
    return 0.30


def _blend_mode(profile: str) -> str:
    if _is_smooth_subject(profile):
        return "multiband"
    if _is_high_freq_subject(profile):
        return "feather"
    return "multiband"


def _blend_feather_px(overlap_width_px: float, profile: str) -> int:
    # Use higher end for smooth subjects
    factor = 0.30 if _is_smooth_subject(profile) else 0.25
    val = int(round(_clamp(factor * float(overlap_width_px), 20.0, 200.0)))
    return max(20, min(200, val))


def _multiband_levels(overlap_width_px: float, profile: str) -> int:
    base = max(64.0, float(overlap_width_px))
    lvl = int(round(log2(base))) - 1
    if profile in ["lunar", "solar", "solar-h-alpha"]:
        return int(_clamp(lvl, 2, min(4, MAX_MULTIBAND_LEVELS)))
    else:
        return int(_clamp(lvl, 2, min(6, MAX_MULTIBAND_LEVELS)))


@dataclass(frozen=True)
class ResolverInputs:
    subject: str
    panel_w: int
    panel_h: int
    # focal_length_mm: Optional[float] = None
    # f_ratio: Optional[float] = None
    stitch_mode: str = "freeform"  # "grid-guided" | "freeform"
    mount_precision: str = "normal"  # tight|normal|sloppy|manual|handheld
    bit_depth: int = 16
    # Intended overlaps are percentage values in the 0.0-95.0 range.
    overlap_x_pct: Optional[float] = None
    overlap_y_pct: Optional[float] = None
    speed_preset: str = "balanced"  # "fast" | "balanced" | "best"


def compute_resolved_params(inp: ResolverInputs) -> Dict[str, float | int | bool]:
    """Compute advanced stitching/blending parameters from minimal session inputs.

    This function is UI-agnostic and safe to call from the GUI layer. All inputs
    are explicit; panel size is in pixels.
    """
    subj = _norm_profile(inp.subject)
    w = max(0, int(inp.panel_w or 0))
    h = max(0, int(inp.panel_h or 0))
    maxdim = max(w, h) or 1
    panel_area = float(w * h)
    # Intended overlap is expressed in percent and normalized here.
    ox_pct = _clamp(float(inp.overlap_x_pct or 0.0), 0.0, 100.0)
    oy_pct = _clamp(float(inp.overlap_y_pct or 0.0), 0.0, 100.0)
    ox = ox_pct / 100.0
    oy = oy_pct / 100.0
    overlap_width_px = float(w) * ox
    # Conservative area proxy uses the smaller of the two
    overlap_area_px = panel_area * min(ox, oy)
    mpix = panel_area / 1_000_000.0  # megapixels

    # 2) Detection defaults
    detector_downscale = _clamp(1200.0 / float(maxdim), 0.4, 1.0)
    # Interpret "4·MP" as 4k features per megapixel to reach usable counts
    detector_max_points = int(min(12_000, max(500, round(4_000.0 * mpix))))
    detector_feature_sensitivity = _subject_feature_sensitivity(subj)

    # 3) Matching
    base_ratio = _subject_ratio_test(subj)
    ratio_adj = _mount_factor(inp.mount_precision)
    ratio_test = float(_clamp(base_ratio + ratio_adj, 0.60, 0.90))
    ransac_thresh = float(_ransac_thresh(inp.stitch_mode, inp.mount_precision))
    min_inliers = int(_min_inliers(mpix))
    min_inlier_ratio = float(_min_inlier_ratio(inp.stitch_mode, inp.mount_precision))

    # 4) Grid-related fallback heuristics (used by all modes, not just grid-guided)
    # These flags are applied regardless of stitch mode for consistent behavior across modes.
    tol_exp, tol_ortho = _grid_tolerances(inp.mount_precision)
    enable_shift_guard = subj not in {"solar", "solar-h-alpha"}
    grid_guide: Dict[str, float | bool] = {
        "enable_grid_shift_guard": enable_shift_guard,
        "grid_expected_shift_tolerance": float(_clamp(tol_exp, 0.05, 0.40)),
        "grid_orthogonal_shift_tolerance": float(_clamp(tol_ortho, 0.05, 0.35)),
        "grid_nominal_shift_bias": float(_clamp(_grid_bias(subj), 0.0, 1.5)),
        # Phase fallback for weak texture (used by both grid and free modes)
        "enable_grid_phase_fallback": subj in {"solar", "solar-h-alpha", "milky-way"},
        "grid_phase_fallback_min_response": 0.02 if _is_smooth_subject(subj) else 0.012,
    }

    # 5) Blending
    blend_type = _blend_mode(subj)
    blend_feather_px = int(_blend_feather_px(overlap_width_px, subj))
    multiband_levels = int(_multiband_levels(overlap_width_px, subj))

    # 6) Gain compensation
    should_enable_illum = bool(overlap_area_px >= 0.05 * panel_area or (inp.stitch_mode or "").lower() == "freeform")
    # For Solar/Lunar subjects, use local compensation (sub-tile) for limb darkening.
    # For other subjects with sufficient overlap, use uniform compensation.
    if subj in {"solar", "solar-h-alpha", "lunar"}:
        gain_compensation = "simple"  # Handle limb darkening with spatial variation
    elif should_enable_illum:
        gain_compensation = "uniform"  # Standard global gain+offset normalization
    else:
        gain_compensation = "none"  # No gain compensation
    # Clamp for bit depth (offset units are in pixel values)
    bd = max(8, min(32, int(inp.bit_depth or 16)))
    blend_offset_clamp = float(30.0 * (2 ** (bd - 8)))
    blend_offset_match = subj in {"solar", "solar-h-alpha", "milky-way", "nightscape"}

    out: Dict[str, float | int | bool] = {
        # detection
        "detector_downscale": float(detector_downscale),
        "detector_max_points": int(detector_max_points),
        "detector_feature_sensitivity": float(detector_feature_sensitivity),
        # matching
        "ratio_test": float(ratio_test),
        "ransac_thresh": float(ransac_thresh),
        "min_inliers": int(min_inliers),
        "min_inlier_ratio": float(min_inlier_ratio),
        # blending
        "blend_type": str(blend_type),
        "blend_feather_px": int(blend_feather_px),
        "multiband_levels": int(multiband_levels),
        # illumination
        "gain_compensation": str(gain_compensation),
        "blend_offset_match": bool(blend_offset_match),
        "blend_offset_clamp": float(blend_offset_clamp),
    }
    out.update(grid_guide)
    _apply_speed_preset(out, inp.speed_preset)
    return out


def _apply_speed_preset(out: Dict[str, float | int | bool], preset: str) -> None:
    """Modify resolver output in-place based on speed preset.

    "balanced" (default) leaves all resolver-computed values unchanged.
    "fast" trades quality for speed: lower-res detection, feather blend, no gain correction.
    "best" pushes quality higher: full-res detection, more features, deeper multiband.
    """
    p = (preset or "balanced").strip().lower()
    if p == "fast":
        out["detector_downscale"] = float(_clamp(out["detector_downscale"] * 0.65, 0.25, 1.0))
        out["detector_max_points"] = int(max(300, out["detector_max_points"] // 2))
        # out["blend_type"] = "feather"
        out["multiband_levels"] = 2
        out["gain_compensation"] = "none"
    elif p == "best":
        out["detector_downscale"] = 1.0
        out["detector_max_points"] = int(min(12_000, out["detector_max_points"] * 2))
        out["multiband_levels"] = int(min(MAX_MULTIBAND_LEVELS, out["multiband_levels"] + 1))


__all__ = ["ResolverInputs", "compute_resolved_params"]
