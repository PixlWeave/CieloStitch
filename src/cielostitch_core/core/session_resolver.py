# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# from config.constants import MAX_MULTIBAND_LEVELS

# --- Constants for magic numbers ---
MIN_OVERLAP_PCT = 0.02
MAX_OVERLAP_PCT = 0.95
MIN_OVERLAP_AREA = 2000.0
MAX_OVERLAP_AREA = 100000.0
MAX_CONFIDENCE_AREA = 1000000.0
MIN_BLEND_FEATHER = 20.0
MAX_BLEND_FEATHER = 200.0
SMOOTH_GRADIENT_FEATHER_FACTOR = 0.30
DEFAULT_FEATHER_FACTOR = 0.25
MIN_MULTIBAND_LEVELS_SMOOTH = 2
MAX_MULTIBAND_LEVELS_SMOOTH = 4
MIN_MULTIBAND_LEVELS = 2
MAX_MULTIBAND_LEVELS = 6
MIN_LOCAL_GAIN_TILE = 32.0
MAX_LOCAL_GAIN_TILE = 192.0
MIN_LOCAL_GAIN_TILE_SOLAR = 32.0
MAX_LOCAL_GAIN_TILE_SOLAR = 96.0
MIN_LOCAL_GAIN_TILE_LUNAR = 48.0
MAX_LOCAL_GAIN_TILE_LUNAR = 128.0
MIN_PANEL_HINT = 1.0
MIN_OVERLAP_HINT = 32.0
MIN_PANEL_HINT_FACTOR = 0.02
MIN_DETECTOR_DOWNSCALE = 0.4
MAX_DETECTOR_DOWNSCALE = 1.0
DEFAULT_DETECTOR_DOWNSCALE = 1200.0
MIN_DETECTOR_POINTS = 500.0
MAX_DETECTOR_POINTS = 12000.0
DETECTOR_POINTS_BASE = 2500.0
DETECTOR_POINTS_GROWTH = 1800.0
MIN_INLIERS = 8
MAX_INLIERS = 20
BLEND_OFFSET_CLAMP_FACTOR = 30.0
GHOST_GUARD_FALLBACK_FEATHER_MIN = 12.0
GHOST_GUARD_FALLBACK_FEATHER_MAX = 36.0
RISK_THRESHOLD_MIN = 0.35
RISK_THRESHOLD_MAX = 0.70
TELEPHOTO_THRESHOLD = 1.4


def compute_resolved_params(inp: ResolverInputs) -> dict:
    return SessionResolver(inp).resolve()


@dataclass(frozen=True)
class ResolverInputs:
    subject: str
    panel_w: int
    panel_h: int
    overlap_x_pct: float
    overlap_y_pct: float
    panel_count: int = 1

    fpx_mode: Optional[str] = None
    focal_length_mm: Optional[float] = None  # combined with sensor_width_mm → fpx_factor
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    hfov_deg: Optional[float] = None  # alternative to focal/sensor pair
    fpx_factor: Optional[float] = None
    camera_angle_deg: Optional[float] = None

    stitch_mode: str = "freeform"  # "grid-guided" | "freeform"
    # projection_mode: Optional[str] = None  # None/"auto" => recommend, else "native" | "cylindrical"
    mount_precision: str = "normal"  # tight|normal|sloppy|manual|handheld
    bit_depth: int = 16
    speed_preset: str = "balanced"  # "fast" | "balanced" | "best"


# --- Class-based refactor ---
class SessionResolver:
    def __init__(self, inp: ResolverInputs):
        self.inp = inp
        self._init_state()

    def _init_state(self):
        # normalize subject / profile
        self.profile = self._norm_profile(self.inp.subject)

        if self.inp.overlap_x_pct is None or self.inp.overlap_y_pct is None:
            raise ValueError("overlap_x_pct and overlap_y_pct are required")

        self.w = max(0, int(self.inp.panel_w or 0))
        self.h = max(0, int(self.inp.panel_h or 0))
        self.maxdim = max(self.w, self.h) or 1
        self.panel_area = float(self.w * self.h)
        self.ox_pct = float(self.inp.overlap_x_pct)
        self.oy_pct = float(self.inp.overlap_y_pct)
        if not 0.0 <= self.ox_pct <= 100.0:
            raise ValueError("overlap_x_pct must be between 0.00 and 100.00")
        if not 0.0 <= self.oy_pct <= 100.0:
            raise ValueError("overlap_y_pct must be between 0.00 and 100.00")
        self.ox = self.ox_pct / 100.0
        self.oy = self.oy_pct / 100.0
        self.overlap_width_px = max(float(self.w) * self.ox, float(self.h) * self.oy)
        self.fpx = self._fpx_factor_estimate()
        self.primary_overlap = max(self.ox, self.oy)
        self.secondary_overlap = min(self.ox, self.oy)
        self.overlap_area_px = self.panel_area * (self.primary_overlap * (0.35 + 0.65 * self.secondary_overlap))
        self.min_overlap_pct = min(self.ox_pct, self.oy_pct)
        self.mpix = self.panel_area / 1_000_000.0
        self._set_flags()
        self.detector_downscale = self._clamp(1200.0 / float(self.maxdim), 0.4, 1.0)
        self.detector_max_points = int(self._clamp(2500.0 + 1800.0 * math.log2(self.mpix + 1.0), 500.0, 12000.0))

    def _set_flags(self):
        mp = (self.inp.mount_precision or "normal").strip().lower()
        self.is_natural_scene = self.profile in {"general", "panorama", "landscape", "nightscape"}
        self.is_smooth_gradient = self.profile in {"solar", "solar-h-alpha", "lunar", "milky-way"}
        self.has_weak_texture = self.profile in {"solar", "solar-h-alpha", "milky-way"}
        self.has_ambiguous_features = self.profile in {"lunar", "milky-way"}
        self.is_solar_lunar = self.profile in {"solar", "solar-h-alpha", "lunar"}
        self.supports_local_warp = self.is_natural_scene and not self.has_ambiguous_features
        self.has_parallax_risk = (
                self.is_natural_scene and (
                "handheld" in mp or "manual" in mp or "sloppy" in mp or (self.inp.panel_count or 0) >= 5)
        )

    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def _fpx_factor_estimate(self):
        if self.inp.fpx_mode == "camera":
            fl = float(self.inp.focal_length_mm) if self.inp.focal_length_mm else 0.0
            sw = float(self.inp.sensor_width_mm) if self.inp.sensor_width_mm else 0.0
            if fl > 0.0 and sw > 0.0:
                return fl / sw
        elif self.inp.fpx_mode == "fov":
            hf = float(self.inp.hfov_deg) if self.inp.hfov_deg else 0.0
            if 0.0 < hf < 180.0:
                return 0.5 / math.tan(math.radians(hf) / 2.0)
        elif self.inp.fpx_mode == "factor":
            return self.inp.fpx_factor

        return None

    def resolve(self):
        # Local variables for parameters only used in this resolution
        detector_feature_sensitivity = self._subject_feature_sensitivity()
        ratio_test = self._ratio_test()
        grid_tolerances = self._get_grid_tolerances()
        blending = self._get_blending()
        # projection_intrinsics = self._get_projection_intrinsics()
        ransac_thresh = self._ransac_thresh(blending["projection_mode"])
        min_inliers = self._min_inliers()
        min_inlier_ratio = self._min_inlier_ratio()

        gain_comp = self._get_gain_compensation()
        misc = self._get_misc()
        apap = self._get_apap()
        speed = self._apply_speed_preset(blending, gain_comp, misc, apap)

        out = {
            "detector_downscale": speed["detector_downscale"],
            "detector_max_points": speed["detector_max_points"],
            "detector_feature_sensitivity": detector_feature_sensitivity,
            "ratio_test": ratio_test,
            "ransac_thresh": ransac_thresh,
            "min_inliers": min_inliers,
            "min_inlier_ratio": min_inlier_ratio,
            **blending,
            # **projection_intrinsics,
            **gain_comp,
            **misc,
            **apap,
            **grid_tolerances,
        }
        return out

    # The following methods are refactored to return dictionaries of their results instead of setting self attributes
    def _get_grid_tolerances(self):
        mp = (self.inp.mount_precision or "normal").lower()
        if "tight" in mp:
            tol_exp, tol_ortho = 0.12, 0.12
        elif "sloppy" in mp:
            tol_exp, tol_ortho = 0.28, 0.20
        elif "handheld" in mp or "manual" in mp:
            tol_exp, tol_ortho = 0.35, 0.25
        else:
            tol_exp, tol_ortho = 0.20, 0.18
        enable_grid_shift_guard = not self.is_solar_lunar
        grid_expected_shift_tolerance = self._clamp(tol_exp, 0.05, 0.40)
        grid_orthogonal_shift_tolerance = self._clamp(tol_ortho, 0.05, 0.35)
        if self.has_weak_texture:
            bias = 0.50
        elif self.profile == "lunar":
            bias = 0.20
        else:
            bias = 0.30
        grid_nominal_shift_bias = self._clamp(bias, 0.0, 1.5)
        enable_grid_phase_fallback = self.has_weak_texture or self.is_solar_lunar
        if self.is_smooth_gradient:
            base_response = 0.018
            if self.min_overlap_pct < 10.0:
                base_response += 0.004
            elif self.min_overlap_pct >= 20.0:
                base_response -= 0.002
        else:
            base_response = 0.012
            if self.min_overlap_pct < 10.0:
                base_response += 0.001
            elif self.min_overlap_pct >= 20.0:
                base_response -= 0.001
        if "handheld" in mp:
            base_response += 0.001
        grid_phase_fallback_min_response = float(self._clamp(base_response, 0.010, 0.030))
        return {
            "enable_grid_shift_guard": enable_grid_shift_guard,
            "grid_expected_shift_tolerance": grid_expected_shift_tolerance,
            "grid_orthogonal_shift_tolerance": grid_orthogonal_shift_tolerance,
            "grid_nominal_shift_bias": grid_nominal_shift_bias,
            "enable_grid_phase_fallback": enable_grid_phase_fallback,
            "grid_phase_fallback_min_response": grid_phase_fallback_min_response,
        }

    def _get_blending(self):
        if self.is_smooth_gradient:
            blend_type = "multiband"
        elif self.is_natural_scene:
            blend_type = "feather"
        else:
            blend_type = "multiband"
        factor = SMOOTH_GRADIENT_FEATHER_FACTOR if self.is_smooth_gradient else DEFAULT_FEATHER_FACTOR
        val = int(round(self._clamp(factor * float(self.overlap_width_px), MIN_BLEND_FEATHER, MAX_BLEND_FEATHER)))
        blend_feather_px = max(int(MIN_BLEND_FEATHER), min(int(MAX_BLEND_FEATHER), val))
        base = max(64.0, float(self.overlap_width_px))
        lvl = int(round(math.log2(base))) - 1
        if self.is_smooth_gradient:
            multiband_levels = int(
                self._clamp(lvl, MIN_MULTIBAND_LEVELS_SMOOTH, min(MAX_MULTIBAND_LEVELS_SMOOTH, MAX_MULTIBAND_LEVELS)))
        else:
            multiband_levels = int(
                self._clamp(lvl, MIN_MULTIBAND_LEVELS, min(MAX_MULTIBAND_LEVELS, MAX_MULTIBAND_LEVELS)))
        # requested_projection_mode = (self.inp.projection_mode or "").strip().lower()
        # if requested_projection_mode in {"native", "cylindrical"}:
        #     projection_mode = requested_projection_mode
        # else:
        projection_mode = self._projection_mode_recommendation()
        transform_mode = self._transform_mode()
        photometric_min_overlap_px, photometric_full_confidence_px = self._photometric_overlap_thresholds()
        local_gain_tile_size = self._local_gain_tile_size()
        ghost_guard = self._get_ghost_guard(blend_type, blend_feather_px)
        return {
            "blend_type": blend_type,
            "blend_feather_px": blend_feather_px,
            "multiband_levels": multiband_levels,
            "projection_mode": projection_mode,
            "transform_mode": transform_mode,
            "photometric_min_overlap_px": photometric_min_overlap_px,
            "photometric_full_confidence_px": photometric_full_confidence_px,
            "local_gain_tile_size": local_gain_tile_size,
            **ghost_guard,
        }

    # def _get_projection_intrinsics(self):
    #     fl = float(self.inp.focal_length_mm) if self.inp.focal_length_mm else 0.0
    #     sw = float(self.inp.sensor_width_mm) if self.inp.sensor_width_mm else 0.0
    #     sh = float(self.inp.sensor_height_mm) if self.inp.sensor_height_mm else 0.0
    #     hf = float(self.inp.hfov_deg) if self.inp.hfov_deg else 0.0
    #     angle = float(self.inp.camera_angle_deg) if self.inp.camera_angle_deg else 0.0
    #     requested_mode = (self.inp.fpx_mode or "").strip().lower()
    #
    #     if requested_mode == "camera":
    #         if fl > 0.0 and sw > 0.0 and sh > 0.0:
    #             return {
    #                 "fpx_mode": "camera",
    #                 "focal_length_mm": fl,
    #                 "sensor_width_mm": sw,
    #                 "sensor_height_mm": sh,
    #                 "camera_angle_deg": angle,
    #             }
    #         return {}
    #
    #     if requested_mode == "fov":
    #         if 0.0 < hf < 180.0:
    #             return {
    #                 "fpx_mode": "fov",
    #                 "hfov_deg": hf,
    #             }
    #         return {}
    #
    #     if requested_mode == "factor":
    #         if self.fpx is None:
    #             return {}
    #         return {
    #             "fpx_mode": "factor",
    #             "fpx_factor": float(self.fpx),
    #         }
    #
    #     if fl > 0.0 and sw > 0.0 and sh > 0.0:
    #         return {
    #             "fpx_mode": "camera",
    #             "focal_length_mm": fl,
    #             "sensor_width_mm": sw,
    #             "sensor_height_mm": sh,
    #             "camera_angle_deg": angle,
    #         }
    #     if 0.0 < hf < 180.0:
    #         return {
    #             "fpx_mode": "fov",
    #             "hfov_deg": hf,
    #         }
    #     if self.fpx is None:
    #         return {}
    #     return {
    #         "fpx_mode": "factor",
    #         "fpx_factor": float(self.fpx),
    #     }

    def _get_ghost_guard(self, blend_type, blend_feather_px):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        if mode in {"fixed-overlap", "zero-overlap"}:
            return {
                "ghost_guard_enabled": False,
                "ghost_guard_risk_threshold": 0.55,
                "ghost_guard_feather_px": 18,
                "ghost_guard_mode": "feather",
            }
        eligible_blend = blend_type in {"feather", "multiband", "seamless"}
        if not eligible_blend:
            return {
                "ghost_guard_enabled": False,
                "ghost_guard_risk_threshold": 0.55,
                "ghost_guard_feather_px": 18,
                "ghost_guard_mode": "feather",
            }
        is_high_risk_scene = self.profile in {"panorama", "landscape", "nightscape"}
        is_weak_texture_scene = self.profile == "milky-way"
        is_stable_scene = self.is_solar_lunar
        enabled = False
        if is_high_risk_scene and ("handheld" in mp or "sloppy" in mp):
            enabled = True
        elif is_weak_texture_scene and ("manual" in mp or "handheld" in mp or self.min_overlap_pct < 18.0):
            enabled = True
        elif not is_stable_scene and not self.is_natural_scene and (
                "manual" in mp or "handheld" in mp or "sloppy" in mp):
            enabled = True
        risk_threshold = 0.55
        if enabled:
            if is_high_risk_scene:
                risk_threshold -= 0.10
            if "manual" in mp or "handheld" in mp:
                risk_threshold -= 0.10
            elif "sloppy" in mp:
                risk_threshold -= 0.05
            if self.min_overlap_pct < 18.0:
                risk_threshold -= 0.05
        fallback_feather = int(round(self._clamp(min(float(blend_feather_px), float(self.overlap_width_px) * 0.12),
                                                 GHOST_GUARD_FALLBACK_FEATHER_MIN, GHOST_GUARD_FALLBACK_FEATHER_MAX)))
        ghost_guard_mode = "feather"
        if enabled:
            if self.is_natural_scene:
                ghost_guard_mode = "content-aware"
            elif self.profile == "milky-way" and ("manual" in mp or "handheld" in mp or self.min_overlap_pct < 18.0):
                ghost_guard_mode = "content-aware"
        return {
            "ghost_guard_enabled": enabled,
            "ghost_guard_risk_threshold": float(self._clamp(risk_threshold, RISK_THRESHOLD_MIN, RISK_THRESHOLD_MAX)),
            "ghost_guard_feather_px": fallback_feather,
            "ghost_guard_mode": ghost_guard_mode,
        }

    def _get_gain_compensation(self):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        overlap_ratio = 0.0 if self.panel_area <= 0.0 else float(self.overlap_area_px) / float(self.panel_area)
        min_overlap_pct = float(self.min_overlap_pct)
        count = max(1, int(self.inp.panel_count or 1))
        profile = self.profile
        is_solar_lunar = self.is_solar_lunar
        is_natural_scene = self.is_natural_scene
        gain_compensation = "none"
        if is_solar_lunar:
            gain_compensation = "simple"
        elif profile == "milky-way":
            if 3.0 <= min_overlap_pct < 8.0 and count <= 12 and "manual" not in mp and "handheld" not in mp and "sloppy" not in mp:
                gain_compensation = "simple"
            elif min_overlap_pct >= 4.0 or overlap_ratio >= 0.05 or (mode == "freeform" and overlap_ratio >= 0.02):
                gain_compensation = "uniform"
        elif is_natural_scene:
            if 3.0 <= min_overlap_pct < 6.0 and count <= 8 and mp in {"tight", "normal"} and mode != "grid-guided":
                gain_compensation = "simple"
            elif min_overlap_pct >= 4.0 or overlap_ratio >= 0.05 or (mode == "freeform" and overlap_ratio >= 0.02):
                gain_compensation = "uniform"
        elif overlap_ratio >= 0.05:
            gain_compensation = "uniform"
        elif mode == "freeform" and overlap_ratio >= 0.02:
            gain_compensation = "uniform"
        return {"gain_compensation": gain_compensation}

    def _get_misc(self):
        bd = max(8, min(32, int(self.inp.bit_depth or 16)))
        blend_offset_clamp = float(30.0 * (2 ** (bd - 8)))
        blend_offset_match = self.profile in {"solar", "solar-h-alpha", "milky-way", "nightscape"}
        enable_grid_nominal_fallback = self._enable_nominal_fallback()
        staged_matching = self._get_staged_matching()
        histogram_matching = self._histogram_matching()
        return {
            "blend_offset_clamp": blend_offset_clamp,
            "blend_offset_match": blend_offset_match,
            "enable_grid_nominal_fallback": enable_grid_nominal_fallback,
            **staged_matching,
            "histogram_matching": histogram_matching,
        }

    def _get_staged_matching(self):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        count = max(1, int(self.inp.panel_count or 1))
        refs = 6
        if mode == "grid-guided":
            staged_matching_mode = "auto"
            refs_per_stage = refs
        elif count >= 12 or self.min_overlap_pct < 18.0 or "handheld" in mp or "manual" in mp:
            refs = 4 + int(count >= 9) + int(count >= 20) + int(self.min_overlap_pct < 20.0)
            staged_matching_mode = "manual"
            refs_per_stage = int(self._clamp(float(refs), 4.0, 8.0))
        else:
            staged_matching_mode = "auto"
            refs_per_stage = refs
        return {"staged_matching_mode": staged_matching_mode, "refs_per_stage": refs_per_stage}

    def _get_apap(self):
        ow = max(64.0, float(self.overlap_width_px))
        feature_density = self._clamp(self.detector_max_points / max(256.0, self.overlap_width_px), 2.0, 18.0)
        mesh_scale = (ow / 140.0) * (0.55 + 0.08 * feature_density)
        mesh_cols = int(self._clamp(round(mesh_scale), 6.0, 20.0))
        if self.profile in {"solar", "solar-h-alpha", "lunar", "milky-way"}:
            mesh_cols = min(mesh_cols, 10)
        aspect = float(self.h) / float(max(1, self.w))
        mesh_rows = int(self._clamp(round(mesh_cols * max(0.6, aspect)), 6.0, 20.0))
        kernel_sigma_px = float(self._clamp(ow * 0.06, 32.0, 128.0))
        min_local_support = int(self._clamp(round(6.0 + feature_density * 0.45), 6.0, 16.0))
        return {
            "apap_mesh_cols": mesh_cols,
            "apap_mesh_rows": mesh_rows,
            "apap_kernel_sigma_px": kernel_sigma_px,
            "apap_min_local_support": min_local_support,
        }

    def _apply_speed_preset(self, blending, gain_comp, misc, apap):
        p = (self.inp.speed_preset or "balanced").strip().lower()
        result = {
            "detector_downscale": self.detector_downscale,
            "detector_max_points": self.detector_max_points,
        }
        if p == "fast":
            result["detector_downscale"] = float(
                self._clamp(self.detector_downscale * 0.65, MIN_DETECTOR_DOWNSCALE, MAX_DETECTOR_DOWNSCALE))
            result["detector_max_points"] = int(max(300, self.detector_max_points // 2))
            if blending["transform_mode"] == "apap":
                blending["transform_mode"] = "homography"
            if blending["blend_type"] == "multiband":
                blending["multiband_levels"] = MIN_MULTIBAND_LEVELS_SMOOTH
            gain_comp["gain_compensation"] = "none"
            misc["histogram_matching"] = False
            apap["apap_mesh_cols"] = min(apap["apap_mesh_cols"], 8)
            apap["apap_mesh_rows"] = min(apap["apap_mesh_rows"], 8)
            blending["ghost_guard_enabled"] = False
            blending["ghost_guard_mode"] = "feather"
            blending["ghost_guard_risk_threshold"] = 0.55
            blending["ghost_guard_feather_px"] = 18
        elif p == "best":
            result["detector_downscale"] = 1.0
            result["detector_max_points"] = int(min(MAX_DETECTOR_POINTS, self.detector_max_points * 2))
            blending["multiband_levels"] = int(min(MAX_MULTIBAND_LEVELS, blending["multiband_levels"] + 1))
        return result

    def _subject_feature_sensitivity(self):
        if self.profile == "lunar":
            return 0.014
        if self.has_weak_texture:
            return 0.010
        if self.is_natural_scene:
            return 0.018
        return 0.015

    def _ratio_test(self):
        if self.profile == "lunar":
            base = 0.70
        elif self.has_ambiguous_features:
            base = 0.72
        elif self.has_weak_texture:
            base = 0.78
        elif self.is_natural_scene:
            base = 0.80
        else:
            base = 0.76
        mp = (self.inp.mount_precision or "normal").strip().lower()
        if "tight" in mp:
            base += -0.03
        if "sloppy" in mp:
            base += 0.02
        if "manual" in mp:
            base += 0.03
        if "handheld" in mp:
            base += 0.04
        return self._clamp(base, 0.60, 0.90)

    def _ransac_thresh(self, proj=None):
        mode = (self.inp.stitch_mode or "freeform").lower()
        mp = (self.inp.mount_precision or "normal").lower()
        low_overlap = self.min_overlap_pct < 12.0
        if mode == "grid-guided":
            base = 3.5
            if "tight" in mp:
                base = 3.0
            if "sloppy" in mp:
                base = 4.5
            elif "manual" in mp:
                base = 6.0
            elif "handheld" in mp:
                base = 7.0
            if low_overlap:
                base += 0.5
            return base
        if "handheld" in mp:
            base = 8.0
        elif "manual" in mp:
            base = 7.0
        elif "sloppy" in mp:
            base = 6.5
        else:
            base = 6.0
        if low_overlap:
            base += 0.5
        if proj == "cylindrical":
            base = max(2.5, base - 0.75)
        return base

    def _min_inliers(self):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        growth = 1.35 if mode == "grid-guided" else 1.5
        min_inliers = round(growth * float(self.mpix))
        if self.min_overlap_pct < 10.0:
            min_inliers += 1
        if mode != "grid-guided" and ("manual" in mp or "handheld" in mp):
            min_inliers += 1
        return int(min(MAX_INLIERS, max(MIN_INLIERS, min_inliers)))

    def _min_inlier_ratio(self):
        mode = (self.inp.stitch_mode or "freeform").lower()
        mp = (self.inp.mount_precision or "normal").lower()
        if mode == "grid-guided":
            base = 0.10
            if "tight" in mp:
                base = 0.08
            elif "sloppy" in mp:
                base = 0.115
            elif "manual" in mp:
                base = 0.125
            elif "handheld" in mp:
                base = 0.13
            if self.min_overlap_pct < 10.0:
                base += 0.01
            elif self.min_overlap_pct >= 20.0:
                base -= 0.005
            return float(self._clamp(base, 0.08, 0.14))

        base = 0.14
        if "tight" in mp:
            base = 0.12
        elif "sloppy" in mp:
            base = 0.155
        elif "manual" in mp:
            base = 0.16
        elif "handheld" in mp:
            base = 0.17
        if self.min_overlap_pct < 10.0:
            base += 0.01
        elif self.min_overlap_pct >= 20.0:
            base -= 0.005
        return float(self._clamp(base, 0.12, 0.18))

    def _projection_mode_recommendation(self):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        count = max(1, int(self.inp.panel_count or 1))
        if mode in {"grid-guided", "fixed-overlap", "zero-overlap"}:
            return "native"
        if self.has_weak_texture:
            return "native"
        if self.is_natural_scene:
            is_telephoto = (
                    self.fpx is not None and
                    self.fpx > TELEPHOTO_THRESHOLD
            )
            if is_telephoto:
                return "native"
            if "handheld" in mp or "manual" in mp:
                return "cylindrical"
            if count >= 4:
                return "cylindrical"
            return "native"
        return "native"

    def _transform_mode(self):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        # proj = (projection_mode or self.inp.projection_mode or "native").strip().lower()
        if mode == "grid-guided":
            return "translation" if "tight" in mp else "affine"
        if mode in {"fixed-overlap", "zero-overlap"}:
            return "translation"
        if self.is_smooth_gradient:
            return "affine"
        if self.profile == "general":
            return "affine"
        if self.is_natural_scene:
            if self.supports_local_warp and self.has_parallax_risk:
                return "apap"
            return "homography"
        return "affine"

    def _photometric_overlap_thresholds(self):
        effective_overlap = max(0.0, float(self.overlap_area_px))
        min_overlap = int(round(self._clamp(effective_overlap * MIN_OVERLAP_PCT, MIN_OVERLAP_AREA, MAX_OVERLAP_AREA)))
        full_confidence = int(round(self._clamp(effective_overlap * 0.35, float(min_overlap), MAX_CONFIDENCE_AREA)))
        return min_overlap, max(min_overlap, full_confidence)

    def _local_gain_tile_size(self):
        min_dim = max(MIN_PANEL_HINT, float(min(self.w, self.h)))
        overlap_hint = max(MIN_OVERLAP_HINT, float(self.overlap_width_px) * 0.5)
        panel_hint = min_dim * MIN_PANEL_HINT_FACTOR
        if self.profile in {"solar", "solar-h-alpha"}:
            raw = min(panel_hint, overlap_hint, MAX_LOCAL_GAIN_TILE_SOLAR)
            return int(
                self._clamp(float(self._round_to_step(raw, 16)), MIN_LOCAL_GAIN_TILE_SOLAR, MAX_LOCAL_GAIN_TILE_SOLAR))
        if self.profile == "lunar":
            raw = min(max(panel_hint, MIN_LOCAL_GAIN_TILE_LUNAR), max(overlap_hint, MIN_LOCAL_GAIN_TILE_LUNAR))
            return int(
                self._clamp(float(self._round_to_step(raw, 16)), MIN_LOCAL_GAIN_TILE_LUNAR, MAX_LOCAL_GAIN_TILE_LUNAR))
        raw = max(96.0, min(panel_hint * 2.5, max(overlap_hint, 128.0)))
        return int(self._clamp(float(self._round_to_step(raw, 16)), MIN_LOCAL_GAIN_TILE, MAX_LOCAL_GAIN_TILE))

    @staticmethod
    def _round_to_step(value, step):
        step_i = max(1, int(step))
        return int(round(float(value) / float(step_i))) * step_i

    def _enable_nominal_fallback(self):
        mode = (self.inp.stitch_mode or "freeform").strip().lower()
        mp = (self.inp.mount_precision or "normal").strip().lower()
        if mode == "grid-guided":
            return True
        if self.is_solar_lunar:
            return False
        if self.min_overlap_pct < 15.0 and "handheld" not in mp and "manual" not in mp:
            return True
        return False

    def _histogram_matching(self):
        mp = (self.inp.mount_precision or "normal").strip().lower()
        if self.is_natural_scene:
            return True
        if self.profile == "milky-way" and ("handheld" in mp or "manual" in mp):
            return True
        return False

    @staticmethod
    def _norm_profile(name: Optional[str]) -> str:
        n = (name or "").strip().lower().replace(" ", "_")
        if n in {"milky_way", "milkyway", "milky-way"}:
            return "milky-way"
        if n in {"nightscape", "night_scape"}:
            return "nightscape"
        if n in {"ha", "solar_ha", "solar_h_alpha", "solar_h-alpha"}:
            return "solar-h-alpha"
        return n
