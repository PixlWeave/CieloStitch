# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from typing import List
import logging
import time

from PySide6.QtCore import QObject, Signal, Slot
from ...config.config import cfg
from ...core.stitching.base_mode import BaseStitchMode
from ...core.stitching.grid_mode import GridMode
from ...core.stitching.free_mode import FreeMode
from ...utils.message import message_color_passed
from ...utils.image_io import (
    load_panel_images_from_paths,
    normalize_stitch_precision_mode,
)
from ...utils.image_manipulation import from_internal_float32
from ...state.profiles import (
    SolarProfile,
    LunarProfile,
    LandscapeProfile,
    GeneralProfile,
    NightscapeProfile,
    MilkyWayProfile,
    SolarHAlphaProfile,
    PanoramaProfile,
)

logger = logging.getLogger(__name__)


def _emit_stitching_summary(log_emit, *, elapsed: float, total: int, engine=None, color: str = "red") -> None:
    if engine is not None and hasattr(engine, "current_summary"):
        msg, summary_color = engine.current_summary(elapsed)
        log_emit(msg, summary_color)
        return

    log_emit(
        BaseStitchMode.format_stitching_summary(
            total=total,
            placed=0,
            skipped=0,
            elapsed=elapsed,
        ),
        color,
    )


def _new_profile_for_profile(profile: str):
    if profile == "solar":
        return SolarProfile()
    if profile == "solar-h-alpha":
        return SolarHAlphaProfile()
    if profile == "lunar":
        return LunarProfile()
    if profile == "landscape":
        return LandscapeProfile()
    if profile == "nightscape":
        return NightscapeProfile()
    if profile == "milky-way":
        return MilkyWayProfile()
    if profile == "panorama":
        return PanoramaProfile()
    if profile == "general":
        return GeneralProfile()
    raise ValueError(f"Unknown profile: {profile}")


def _order_paths_for_stitching(sorted_panel_paths: List[str]) -> List[str]:
    """Return panel paths in the index order expected by the stitch engine.

    Both freeform and grid modes receive paths in their original file-sort order
    (= camera capture order).  GridMode maps capture indices to physical slots
    internally via its ``_slot_of_index`` table, so no pre-reordering is needed
    here.
    """
    return list(sorted_panel_paths or [])


class StitchWorker(QObject):
    # log = Signal(str)
    log = Signal((str, str, object), (str, str), (str,)) # type: ignore
    images_read = Signal(int)
    # Progress update signal (current, total) for status bar
    progress_update = Signal(int, int)
    # Placement update signal (placed) for right-side status text
    placement_update = Signal(int)
    progress_finished = Signal()

    finished = Signal(str, object, object)
    failed = Signal(str)
    cancelled = Signal()
    done = Signal()

    def __init__(
        self,
        sorted_panel_paths: list,
        alpha_policy: str,
    ):
        super().__init__()
        # Snapshot paths at run creation time so UI-side list mutations do not affect this run.
        self.sorted_panel_paths = list(sorted_panel_paths or [])
        self.input_bit_depth = None
        self.output_bit_depth = None
        self.profile = cfg.state.profile
        self.stitch_mode = cfg.state.stitch_mode
        self.grid_cols = cfg.state.ssm.grid_cols
        self.scan_order = cfg.state.ssm.scan_order
        self.start_corner = cfg.state.ssm.start_corner
        self.alternating = bool(cfg.state.ssm.alternating=="yes")
        self.advanced_profile_settings = cfg.state.psm.get_values(self.profile)
        self.blend_type = self.advanced_profile_settings['blend_type']
        self.alpha_policy = alpha_policy
        self.show_pixel_stat = bool(cfg.state.show_pixel_stat)
        # self.last_overlap_mask = None
        self.last_canvas_system = None
        self.last_panel_source_by_index = None
        self.last_panel_evidence_by_index = None
        self.last_seam_heatmap = None
        self._cancel_requested = False
        self._t0 = None

    def request_cancel(self):
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested
    
    def _emit_cancelled(self):
        self.progress_finished.emit()
        self.cancelled.emit()


    @Slot()
    def run(self):
        try:
            self._t0 = time.perf_counter()
            engine = None  # keep a handle for partial/cancel summaries
            if self._is_cancelled():
                try:
                    elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
                    total = len(self.sorted_panel_paths)
                    _emit_stitching_summary(self.log[str, str].emit, elapsed=elapsed, total=total, color="red")
                except Exception:
                    logger.debug("Failed to emit cancellation summary at run start", exc_info=True)
                self._emit_cancelled()
                return

            profile = _new_profile_for_profile(self.profile)
            for attr, value in self.advanced_profile_settings.items():
                if hasattr(profile, attr):
                    setattr(profile, attr, value)
            profile.blend_type = self.blend_type
            # if cfg.state.ssm.is_grid_capture:
            #     profile.auto_scan_order = self.scan_order
            #     profile.scan_start_corner = self.start_corner
            #     profile.alternating_direction = self.alternating

            stitch_paths = _order_paths_for_stitching(self.sorted_panel_paths)
            stitch_precision_mode = normalize_stitch_precision_mode(
                str(getattr(cfg.prefs, "stitch_precision_mode", "auto") or "auto")
            )

            items, self.input_bit_depth, effective_stitch_precision_mode = load_panel_images_from_paths(
                stitch_paths,
                log_cb=self.log[str, str].emit,
                alpha_policy=self.alpha_policy,
                stitch_precision_mode=stitch_precision_mode,
            )
            if len(items) < 1:
                raise ValueError("No readable panel images.")

            self.images_read.emit(self.input_bit_depth)

            if effective_stitch_precision_mode == "fp16" and int(self.input_bit_depth or 0) >= 16:
                self.log[str, str].emit(
                    "Warning: Compact (16-bit float) mode with 16-bit source panels reduces internal stitch precision. "
                    "Use Standard (32-bit float) for best quality.",
                    "red",
                )

            if stitch_precision_mode == "auto":
                mode_label = "Compact (16-bit float)" if effective_stitch_precision_mode == "fp16" else "Standard (32-bit float)"
                self.log[str, str].emit(f"Stitch precision auto-selected: {mode_label}.", "")

            self.log[str, str].emit(
                f"Stitching started. profile={self.profile}, mode={self.stitch_mode}, grid_cols={self.grid_cols}, images={len(items)}",
                ""
            )
            self.log[str, str].emit(
                f"Subpixel alignment: {'on' if bool(getattr(cfg.prefs, 'enable_subpixel_refinement', True)) else 'off'}.",
                "",
            )

            if cfg.state.ssm.is_grid_capture: # GRID_MODES
                # actual difference is in _compute_global_transforms
                engine = GridMode(profile, grid_cols=self.grid_cols, show_pixel_stat=self.show_pixel_stat)
            else: # auto
                engine = FreeMode(profile, show_pixel_stat=self.show_pixel_stat)
            # Keep a reference on self too in case we need it in outer scopes
            try:
                total_loaded = int(len(items))

                def _compute_placed_display() -> int:
                    # Engine tracks successful non-anchor blends; include anchor for panel-level UI count.
                    placed = int(getattr(engine, "_progress_placed", 0) or 0)
                    if total_loaded > 0 and placed <= max(0, total_loaded - 1):
                        placed += 1
                    return max(0, min(placed, total_loaded))

                # Create a wrapper callback that intelligently routes signals
                # This handles both progress updates (int, int) and logging (str, str)
                last_progress_emit = 0.0
                progress_emit_interval = 0.1

                def unified_callback(arg1, arg2):
                    nonlocal last_progress_emit
                    now = time.perf_counter()
                    if isinstance(arg1, int) and isinstance(arg2, int):
                        if arg1 == arg2 or (now - last_progress_emit) >= progress_emit_interval:
                            last_progress_emit = now
                            self.progress_update.emit(arg1, arg2)
                            try:
                                self.placement_update.emit(_compute_placed_display())
                            except Exception:
                                pass
                    else:
                        # Only forward important or required messages
                        _color = arg2
                        if message_color_passed(_color):
                            self.log[str, str].emit(arg1, _color)

                mosaic = engine.stitch(
                    items,
                    progress_cb=unified_callback,
                    cancel_cb=self._is_cancelled,
                )
            except InterruptedError:
                # Graceful partial summary using engine's counters
                try:
                    elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
                    _emit_stitching_summary(self.log[str, str].emit, elapsed=elapsed, total=len(items), engine=engine, color="red")
                except Exception:
                    logger.debug("Failed to emit interrupted summary during stitch run", exc_info=True)
                self._emit_cancelled()
                return
            if mosaic is None or self._is_cancelled():
                try:
                    elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
                    _emit_stitching_summary(self.log[str, str].emit, elapsed=elapsed, total=len(items), engine=engine, color="red")
                except Exception:
                    logger.debug("Failed to emit cancelled summary after mosaic generation", exc_info=True)
                self._emit_cancelled()
                return

            # overlap_mask = None
            canvas_system = getattr(engine, "last_canvas_system", None)
            # if canvas_system is not None:
            #     coverage = getattr(canvas_system, "coverage_count", None)
            #     if coverage is not None:
            #         overlap_mask = coverage >= 2

            output_img = from_internal_float32(mosaic, self.input_bit_depth)
            if self._is_cancelled():
                try:
                    elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
                    _emit_stitching_summary(self.log[str, str].emit, elapsed=elapsed, total=len(items), engine=engine, color="red")
                except Exception:
                    pass
                self._emit_cancelled()
                return
        
            # self.last_overlap_mask = overlap_mask
            self.last_canvas_system = canvas_system
            # Capture seam heatmap from engine if available
            try:
                self.last_seam_heatmap = getattr(canvas_system, "seam_heatmap", None)
            except Exception:
                self.last_seam_heatmap = None
                logger.debug("Failed to capture seam heatmap from canvas system", exc_info=True)
            try:
                self.last_panel_source_by_index = dict(getattr(engine, "_panel_source_by_index", {}) or {})
            except Exception:
                self.last_panel_source_by_index = None
            try:
                self.last_panel_evidence_by_index = dict(getattr(engine, "_panel_evidence_by_index", {}) or {})
            except Exception:
                self.last_panel_evidence_by_index = None

            # Hide progress bar after successful completion
            try:
                self.placement_update.emit(_compute_placed_display())
            except Exception:
                pass
            self.progress_finished.emit()
            self.finished.emit("", output_img, mosaic.copy())
        except InterruptedError:
            try:
                elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
                _emit_stitching_summary(
                    self.log[str, str].emit,
                    elapsed=elapsed,
                    total=len(self.sorted_panel_paths),
                    engine=engine,
                    color="red",
                )
            except Exception:
                logger.debug("Failed to emit interrupted summary in outer handler", exc_info=True)
            # Hide progress bar on cancel
            self._emit_cancelled()
        except Exception as exc:
            # Hide progress bar on error
            self.progress_finished.emit()
            self.failed.emit(str(exc))
        finally:
            self.done.emit()
