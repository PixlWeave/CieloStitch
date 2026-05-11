# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import logging
import os

from dataclasses import dataclass
from PySide6.QtCore import QSettings, QStandardPaths
from .. import app_name

logger = logging.getLogger(__name__)


def _normalize_speed_preset(value: str) -> str:
    preset = str(value or "balanced").strip().lower()
    if preset in {"fast", "balanced", "best"}:
        return preset
    return "balanced"


def _normalize_hidden_bool_pref(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off", ""}:
        return False
    return bool(default)


def _normalize_hidden_csv_pref(value, default: str) -> str:
    raw = default if value is None else value
    parts = [str(part).strip().lower() for part in str(raw).split(",")]
    parts = [part for part in parts if part]
    return ",".join(parts)


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

def _normalize_log_level(value, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().upper()
    if normalized in _VALID_LOG_LEVELS:
        return normalized
    return default


def _normalize_input_zoom_mode(value) -> str:
    """Normalize persisted input zoom mode to supported string values."""
    if isinstance(value, bool):
        return "fit" if value else "auto"
    mode = str(value or "auto").strip().lower()
    if mode in {"fit", "auto"}:
        return mode
    if mode in {"true", "1", "yes", "on"}:
        return "fit"
    if mode in {"false", "0", "no", "off"}:
        return "auto"
    return "auto"

def _prefs_file_path() -> str:
    appdata = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    prefs_folder = os.path.join(appdata, app_name())
    os.makedirs(prefs_folder, exist_ok=True)
    return os.path.join(prefs_folder, "preferences.ini")


# Export a shared QSettings instance configured to use an INI file under AppData
_preferences = QSettings(_prefs_file_path(), QSettings.IniFormat)


def read_pref(key: str, default=None, value_type=None, type=None):
    """Read a setting from QSettings with optional type conversion.

    Args:
        key: Setting key
        default: Default value if not found
        value_type: Type to convert the value to (preferred parameter name)
        type: Legacy parameter name, use value_type instead
    """
    resolved_type = value_type if value_type is not None else type
    if resolved_type is None:
        return _preferences.value(key, default)
    return _preferences.value(key, default, type=resolved_type)


def write_pref(key: str, value) -> None:
    _preferences.setValue(key, value)


def sync_prefs() -> None:
    _preferences.sync()


@dataclass(frozen=True)
class AppPreferences:
    #ui
    start_maximized: bool = True
    show_advanced_controls: bool = False
    show_tooltips: bool = True
    theme: str = "moonlight"

    #input
    input_display_mode: str = "Show Margin"
    zoom_mode: str = "auto" # "auto", "fit"
    input_show_badges: bool = False

    input_preview_mode: str = "tone-enhance"
    input_speedy_preview: bool = False
    default_alpha_policy: str = "auto"
    allow_mouse_wheel: bool = False

    #stitch
    feature_cache_enabled: bool = True
    default_profile: str = "lunar"
    default_stitch_mode: str = "auto"
    default_grid_cols: int = 3
    stitch_interpolator: str = "auto"
    stitch_precision_mode: str = "auto"
    default_resolve_for: str = "balanced"
    retain_overlap_diagnostics: bool = True
    retain_seam_diagnostics: bool = True
    enable_candidate_debug: bool = False

    #output
    confirm_unsaved_mosaic: bool = True
    coverage_dim_mode: str  = "inside"
    coverage_dim_percent: int = 100

    output_preview_mode: str = "tone-enhance"
    output_speedy_preview: bool = False
    output_interpolator: str = "auto"

    #export
    export_jpeg_quality: int = 95
    export_scale_percent: int = 100
    default_export_format: str = ".tif"
    default_export_dir: str = ""
    export_interpolator: str = "auto"
    save_overlap_mask: bool = False
    save_seam_coverage: bool = False

    #extra
    enable_subpixel_refinement: bool = True
    log_message_colors: str = "default,red,yellow,green,debug"
    stderr_log_level: str = "WARNING"
    check_for_updates_on_startup: bool = True

    @classmethod
    def load(cls) -> "AppPreferences":
        return cls(
            #ui
            start_maximized=bool(read_pref(
                "ui/start_maximized", cls.start_maximized, type=bool)),
            show_advanced_controls=bool(read_pref(
                "ui/show_advanced_controls", cls.show_advanced_controls, type=bool)),
            show_tooltips=bool(read_pref(
                "ui/show_tooltips", cls.show_tooltips, type=bool)),
            theme=str(read_pref(
                "ui/theme", cls.theme, type=str)),

            #input
            input_display_mode=str(read_pref(
                "input/display_mode", cls.input_display_mode, type=str)),
            zoom_mode=_normalize_input_zoom_mode(read_pref(
                "input/zoom_mode", cls.zoom_mode)),
            input_show_badges=bool(read_pref(
                "input/show_badges", cls.input_show_badges, type=bool)),
            input_preview_mode=str(read_pref(
                "input/input_preview_mode", cls.input_preview_mode, type=str)),
            input_speedy_preview=bool(read_pref(
                "input/input_speedy_preview", cls.input_speedy_preview, type=bool)),
            default_alpha_policy=str(read_pref(
                "input/default_alpha_policy", cls.default_alpha_policy, type=str)),
            allow_mouse_wheel=bool(read_pref(
                "input/allow_mouse_wheel", cls.allow_mouse_wheel, type=bool)),

            # stitch
            feature_cache_enabled=bool(read_pref(
                "stitch/feature_cache_enabled", cls.feature_cache_enabled, type=bool)),
            default_profile=str(read_pref(
                "stitch/default_profile", cls.default_profile, type=str)),
            default_stitch_mode=str(read_pref(
                "stitch/default_stitch_mode", cls.default_stitch_mode, type=str)),
            default_grid_cols=int(str(read_pref(
                "stitch/default_grid_cols", cls.default_grid_cols, type=int))),
            stitch_interpolator=str(read_pref(
                "stitch/stitch_interpolator", cls.stitch_interpolator, type=str)),
            stitch_precision_mode=str(read_pref(
                "stitch/stitch_precision_mode", cls.stitch_precision_mode, type=str)),
            default_resolve_for=_normalize_speed_preset(str(read_pref(
                "stitch/default_resolve_for", cls.default_resolve_for, type=str))),
            retain_overlap_diagnostics=bool(read_pref(
                "stitch/retain_overlap_diagnostics", cls.retain_overlap_diagnostics, type=bool)),
            retain_seam_diagnostics=bool(read_pref(
                "stitch/retain_seam_diagnostics", cls.retain_seam_diagnostics, type=bool)),
            enable_candidate_debug=_normalize_hidden_bool_pref(
                read_pref("stitch/enable_candidate_debug", cls.enable_candidate_debug),
                cls.enable_candidate_debug,
            ),
            #output
            coverage_dim_mode=str(read_pref(
                "output/coverage_dim_mode", cls.coverage_dim_mode, type=str)),
            coverage_dim_percent=int(str(read_pref(
                "output/coverage_dim_percent", cls.coverage_dim_percent, type=int))),
            output_preview_mode=str(read_pref(
                "output/output_preview_mode", cls.output_preview_mode, type=str)),
            output_speedy_preview=bool(read_pref(
                "output/output_speedy_preview", cls.output_speedy_preview, type=bool)),
            output_interpolator=str(read_pref(
                "output/output_interpolator", cls.output_interpolator, type=str)),
            confirm_unsaved_mosaic=bool(read_pref(
                "output/confirm_unsaved_mosaic", cls.confirm_unsaved_mosaic, type=bool)),

            #export
            export_jpeg_quality=int(str(read_pref(
                "export/export_jpeg_quality", cls.export_jpeg_quality, type=int))),
            export_scale_percent=int(str(read_pref(
                "export/export_scale_percent", cls.export_scale_percent, type=int))),
            default_export_format=str(read_pref(
                "export/default_export_format", cls.default_export_format, type=str)),
            default_export_dir=str(read_pref(
                "export/default_export_dir", cls.default_export_dir, type=str)),
            export_interpolator=str(read_pref(
                "export/export_interpolator", cls.export_interpolator, type=str)),
            save_overlap_mask=bool(read_pref(
                "export/save_overlap_mask", cls.save_overlap_mask, type=bool)),
            save_seam_coverage=bool(read_pref(
                "export/save_seam_coverage", cls.save_seam_coverage, type=bool)),

            #extra
            enable_subpixel_refinement=_normalize_hidden_bool_pref(
                read_pref(
                    "stitch/enable_subpixel_refinement",
                    read_pref("extra/enable_subpixel_refinement", cls.enable_subpixel_refinement),
                ),
                cls.enable_subpixel_refinement,
            ),
            log_message_colors=_normalize_hidden_csv_pref(
                read_pref("extra/log_message_colors", cls.log_message_colors),
                cls.log_message_colors,
            ),
            stderr_log_level=_normalize_log_level(
                read_pref("extra/stderr_log_level", None),
                cls.stderr_log_level,
            ),
            check_for_updates_on_startup=_normalize_hidden_bool_pref(
                read_pref("extra/check_for_updates_on_startup", None),
                cls.check_for_updates_on_startup,
            ),
        )

    def save(self) -> None:
        #ui
        write_pref("ui/start_maximized", bool(self.start_maximized))
        write_pref("ui/show_advanced_controls", bool(self.show_advanced_controls))
        write_pref("ui/show_tooltips", bool(self.show_tooltips))
        write_pref("ui/theme", self.theme)
        write_pref("input/allow_mouse_wheel", bool(self.allow_mouse_wheel))

        #input
        write_pref("input/display_mode", str(self.input_display_mode))
        write_pref("input/zoom_mode", _normalize_input_zoom_mode(self.zoom_mode))
        write_pref("input/show_badges", bool(self.input_show_badges))
        write_pref("input/input_preview_mode", self.input_preview_mode)
        write_pref("input/input_speedy_preview", bool(self.input_speedy_preview))
        write_pref("input/default_alpha_policy", self.default_alpha_policy)

        #stitch
        write_pref("output/confirm_unsaved_mosaic", bool(self.confirm_unsaved_mosaic))
        write_pref("stitch/feature_cache_enabled", bool(self.feature_cache_enabled))
        write_pref("stitch/default_profile", self.default_profile)
        write_pref("stitch/default_stitch_mode", self.default_stitch_mode)
        write_pref("stitch/default_grid_cols", int(self.default_grid_cols))
        write_pref("stitch/stitch_interpolator", self.stitch_interpolator)
        write_pref("stitch/stitch_precision_mode", self.stitch_precision_mode)
        write_pref("stitch/default_resolve_for", _normalize_speed_preset(self.default_resolve_for))
        write_pref("stitch/enable_subpixel_refinement", bool(self.enable_subpixel_refinement))
        write_pref("stitch/retain_overlap_diagnostics", bool(self.retain_overlap_diagnostics))
        write_pref("stitch/retain_seam_diagnostics", bool(self.retain_seam_diagnostics))
        write_pref("stitch/enable_candidate_debug", bool(self.enable_candidate_debug))

        #output
        write_pref("output/coverage_dim_mode", self.coverage_dim_mode)
        write_pref("output/coverage_dim_percent", int(self.coverage_dim_percent))
        write_pref("output/output_preview_mode", self.output_preview_mode)
        write_pref("output/output_speedy_preview", bool(self.output_speedy_preview))
        write_pref("output/output_interpolator", self.output_interpolator)

        #export
        write_pref("export/export_jpeg_quality", int(self.export_jpeg_quality))
        write_pref("export/export_scale_percent", int(self.export_scale_percent))
        write_pref("export/default_export_format", self.default_export_format)
        write_pref("export/default_export_dir", self.default_export_dir.strip())
        write_pref("export/export_interpolator", self.export_interpolator)
        write_pref("export/save_overlap_mask", bool(self.save_overlap_mask))
        write_pref("export/save_seam_coverage", bool(self.save_seam_coverage))

        #extra
        write_pref("extra/log_message_colors", _normalize_hidden_csv_pref(self.log_message_colors, self.log_message_colors))
        write_pref("extra/stderr_log_level", _normalize_log_level(self.stderr_log_level, self.stderr_log_level))
        write_pref("extra/check_for_updates_on_startup", bool(self.check_for_updates_on_startup))

        sync_prefs()

