# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

"""
Capture Session State Manager

Manages session-level parameters that describe the capture session.
These parameters are global and persistent across profile changes.

Session parameters include:
  - Capture Plan: how the images were physically captured (grid, scan order, etc.)
  - Session Characteristics: metadata about the capture session (mount precision, seeing, overlap)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import math
from contextlib import contextmanager
from PySide6.QtCore import Signal, QObject
from typing import Any, Dict, List

from .preferences import AppPreferences
from .profiles.grid_guided import GridGuidedStitching
from ..utils.data_types import as_bool, as_int, as_float, as_str
from ..config.constants import DEFAULT_OVERLAP_PCT, GRID_MODES, NON_GRID_MODES


class SessionModel(QObject):
    """
    Manages global session-level settings (not profile-specific).

    Session parameters describe the capture session:
      - capture_mode, grid settings, scan order, etc.
      - mount_precision, seeing_condition

    These settings are:
      - Global (not per-profile)
      - Persistent across profile changes
      - Immutable during a session (represent how images were captured)
    """

    gridSettingsChanged = Signal()
    sessionMassSettingsChanged = Signal()
    sessionDirtyChanged = Signal()
    sessionValueChanged = Signal()  # fires on every value write, regardless of dirty state
    _KNOWN_VALUE_KEYS = {
        "current_stitch_mode",
        "grid_cols",
        "scan_order",
        "start_corner",
        "alternating",
        "image_sort_order",
        "mount_precision",
        "seeing",
        "overlap_x_pct",
        "overlap_y_pct",
        "staged_matching_mode",
        "refs_per_stage",
        "enable_grid_phase_fallback",
        "enable_grid_nominal_fallback",
        "enable_grid_shift_guard",
        "grid_guide",
    }

    def __init__(self, prefs: AppPreferences | None = None) -> None:
        super().__init__()
        self._prefs = prefs or AppPreferences.load()

        # Dirty tracking
        self._dirty: bool = False
        self._suppress_dirty: int = 0

        # Session state attributes
        self.current_stitch_mode: str = self._prefs.default_stitch_mode
        self.grid_cols: int = self._prefs.default_grid_cols
        self.scan_order: str = "row-wise"
        self.start_corner: str = "top-left"
        self.alternating: str = "no"
        self.image_sort_order = "natural"
        self.mount_precision: str = "normal"
        self.seeing: str = "average"
        self.overlap_x_pct: float = DEFAULT_OVERLAP_PCT
        self.overlap_y_pct: float = DEFAULT_OVERLAP_PCT
        self.enable_grid_phase_fallback=True
        self.enable_grid_nominal_fallback=True
        self.enable_grid_shift_guard=True
        self.grid_guide: GridGuidedStitching = GridGuidedStitching()
        self.staged_matching_mode: str = "auto"
        self.refs_per_stage: int = 6


    # def __setattr__(self, name: str, value: Any) -> None:
    #     # Set the new value
    #     super().__setattr__(name, value)
    #     # Skip tracking for internal attributes and during initialization
    #     if name not in ['_dirty', '_suppress_dirty', 'gridSettingsChanged'] and hasattr(self, '_suppress_dirty'):
    #         self.mark_dirty()
    #         self.check_for_grid_changed(name)

    @property
    def scan_icon_label(self) -> str:
        return self.scan_icon_filename()

    @property
    def is_non_grid(self) -> bool:
        """Return True if stitch mode is not grid based"""
        return self.current_stitch_mode in NON_GRID_MODES

    @property
    def is_grid_capture(self) -> bool:
        """Return True if stitch mode is grid based"""
        return self.current_stitch_mode in GRID_MODES

    def check_for_grid_changed(self, name) -> None:
        if self._suppress_dirty == 0:
            if name in [
                'stitch_mode',
                'current_stitch_mode',
                'grid_cols',
                'scan_order',
                'start_corner',
                'alternating',
                'overlap_x_pct',
                'overlap_y_pct',
            ]:
                self.gridSettingsChanged.emit()

    def get_values(self) -> Dict[str, Any]:
        """Get the current session state as a dictionary."""
        return self.to_dict()

    def __eq__(self, other: Any) -> bool:
        """Compare semantic equality - two SessionModels are equal if their values are equal."""
        if not isinstance(other, SessionModel):
            return False
        return (
            # current_stitch_mode excluded from equality check by design
            # self.current_stitch_mode == other.current_stitch_mode
            self.grid_cols == other.grid_cols
            and self.scan_order == other.scan_order
            and self.start_corner == other.start_corner
            and self.alternating == other.alternating
            and self.image_sort_order == other.image_sort_order
            and self.mount_precision == other.mount_precision
            and self.seeing == other.seeing
            and self.overlap_x_pct == other.overlap_x_pct
            and self.overlap_y_pct == other.overlap_y_pct
            and self.staged_matching_mode == other.staged_matching_mode
            and self.refs_per_stage == other.refs_per_stage
            and self.enable_grid_phase_fallback == other.enable_grid_phase_fallback
            and self.enable_grid_nominal_fallback == other.enable_grid_nominal_fallback
            and self.enable_grid_shift_guard == other.enable_grid_shift_guard
            and self.grid_guide == other.grid_guide
        )

    def set_values(self, values: "SessionModel | Dict[str, Any]") -> None:
        """Replace all session values with provided state (only if changed).

        Args:
            values: Either a SessionModel instance or a dictionary of session values.
        """
        # Convert dict to SessionModel for comparison if needed
        if isinstance(values, dict):
            values_model = SessionModel(self._prefs)
            values_model.from_dict(values)
            values = values_model

        if values == self:
            return  # no change → don't mark dirty

        # Copy all attributes
        # self.current_stitch_mode = values.current_stitch_mode
        self.grid_cols = values.grid_cols
        self.scan_order = values.scan_order
        self.start_corner = values.start_corner
        self.alternating = values.alternating
        self.image_sort_order = values.image_sort_order
        self.mount_precision = values.mount_precision
        self.seeing = values.seeing
        self.overlap_x_pct = values.overlap_x_pct
        self.overlap_y_pct = values.overlap_y_pct
        self.enable_grid_phase_fallback = values.enable_grid_phase_fallback
        self.enable_grid_nominal_fallback = values.enable_grid_nominal_fallback
        self.enable_grid_shift_guard = values.enable_grid_shift_guard
        # Avoid aliasing mutable grid_guide objects across models.
        self.grid_guide = GridGuidedStitching.from_dict(values.grid_guide.to_dict())
        self.staged_matching_mode = values.staged_matching_mode
        self.refs_per_stage = values.refs_per_stage
        self.mark_dirty()
        self.gridSettingsChanged.emit()
        self.sessionMassSettingsChanged.emit()
        if self._suppress_dirty == 0:
            self.sessionValueChanged.emit()

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a single session value by key."""
        return getattr(self, key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set a single session value by key (only if changed)."""
        if key not in self._KNOWN_VALUE_KEYS:
            logger.warning(f"Ignoring unknown session key: {key}")
            return
        if getattr(self, key, None) != value:
            setattr(self, key, value)
            self.mark_dirty()
            self.check_for_grid_changed(key)
            if self._suppress_dirty == 0:
                self.sessionValueChanged.emit()

    def is_modified(self) -> bool:
        """Return True if session has been modified since last clear_dirty()."""
        return self._dirty

    def clear_dirty(self) -> None:
        """Mark session as clean (not modified)."""
        if self._dirty:
            self._dirty = False
            self.sessionDirtyChanged.emit()

    def mark_dirty(self) -> None:
        """Mark session as modified."""
        if self._suppress_dirty == 0 and not self._dirty:
            self._dirty = True
            self.sessionDirtyChanged.emit()

    @contextmanager
    def suppress_dirty(self):
        """Context manager to suppress dirty marking during programmatic updates."""
        self._suppress_dirty += 1
        try:
            yield
        finally:
            self._suppress_dirty -= 1

    def set_default_values(self) -> None:
        """Reset session values to standard defaults."""
        # stitch_mode_changed = self.stitch_mode != self._prefs.default_stitch_mode
        with self.suppress_dirty(): # type: ignore
            # self.current_stitch_mode = self._prefs.default_stitch_mode
            self.grid_cols = self._prefs.default_grid_cols
            self.scan_order = "row-wise"
            self.start_corner = "top-left"
            self.alternating = "no"
            self.image_sort_order = "natural"
            self.mount_precision = "normal"
            self.seeing = "average"
            self.overlap_x_pct = DEFAULT_OVERLAP_PCT
            self.overlap_y_pct = DEFAULT_OVERLAP_PCT
            self.staged_matching_mode = "auto"
            self.refs_per_stage = 6
            self.enable_grid_phase_fallback = True
            self.enable_grid_nominal_fallback = True
            self.enable_grid_shift_guard = True
            self.grid_guide = GridGuidedStitching()
            self.clear_dirty()
            # if stitch_mode_changed:
            #     self.stitchModeChanged.emit()
            self.gridSettingsChanged.emit()
            self.sessionMassSettingsChanged.emit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert current session state to dictionary."""
        return {
            # "current_stitch_mode": self.current_stitch_mode,
            "grid_cols": self.grid_cols,
            "scan_order": self.scan_order,
            "start_corner": self.start_corner,
            "alternating": self.alternating,
            "image_sort_order": self.image_sort_order,
            "mount_precision": self.mount_precision,
            "seeing": self.seeing,
            "overlap_x_pct": self.overlap_x_pct,
            "overlap_y_pct": self.overlap_y_pct,
            "staged_matching_mode": self.staged_matching_mode,
            "refs_per_stage": self.refs_per_stage,
            "enable_grid_phase_fallback": self.enable_grid_phase_fallback,
            "enable_grid_nominal_fallback": self.enable_grid_nominal_fallback,
            "enable_grid_shift_guard": self.enable_grid_shift_guard,
            "grid_guide": self.grid_guide.to_dict(),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load session state from dictionary."""
        # new_stitch_mode = as_str(data.get("stitch_mode"), self._prefs.default_stitch_mode)
        # stitch_mode_changed = self.stitch_mode != new_stitch_mode
        with self.suppress_dirty(): # type: ignore
            # current_stitch_mode is intentionally not loaded from dict to preserve existing stitch mode across sessions,
            # self.current_stitch_mode = new_stitch_mode
            self.grid_cols = max(1, as_int(data.get("grid_cols"), self._prefs.default_grid_cols))
            self.scan_order = as_str(data.get("scan_order"), "row-wise")
            self.start_corner = as_str(data.get("start_corner"), "top-left")
            self.alternating = as_str(data.get("alternating"), "no")
            self.image_sort_order = as_str(data.get("image_sort_order"), "natural")
            self.mount_precision = as_str(data.get("mount_precision"), "normal")
            self.seeing = as_str(data.get("seeing"), "average")
            self.overlap_x_pct = as_float(data.get("overlap_x_pct"), DEFAULT_OVERLAP_PCT)
            self.overlap_y_pct = as_float(data.get("overlap_y_pct"), DEFAULT_OVERLAP_PCT)
            self.staged_matching_mode = as_str(data.get("staged_matching_mode"), "auto")
            self.refs_per_stage = as_int(data.get("refs_per_stage"), 6)
            self.enable_grid_phase_fallback = as_bool(data.get("enable_grid_phase_fallback"), True)
            self.enable_grid_nominal_fallback = as_bool(data.get("enable_grid_nominal_fallback"), True)
            self.enable_grid_shift_guard = as_bool(data.get("enable_grid_shift_guard"), True)
            grid_guide_data = data.get("grid_guide", {})
            if not isinstance(grid_guide_data, dict):
                grid_guide_data = {}
            self.grid_guide = GridGuidedStitching.from_dict(grid_guide_data)
            self.clear_dirty()  # Loaded state shouldn't be marked dirty,
            # if stitch_mode_changed:
            #     self.stitchModeChanged.emit()
            self.gridSettingsChanged.emit()
            self.sessionMassSettingsChanged.emit()

    def scan_icon_filename(self) -> str:
        """Build the scan icon filename for the current scan settings, or empty when not in grid mode."""
        if self.is_non_grid:
            return ""
        try:
            alt = "alt" if self.alternating == "yes" else "noalt"
            return f"scan-{self.start_corner}-{self.scan_order}-{alt}.png"
        except Exception as e:
            logger.error(f"from scan_icon_filename: {e}", exc_info=True)
            return ""

    def get_capture_slots(self, n: int) -> List[int]:
        if n <= 0 or self.grid_cols <= 0:
            return []

        if not self.is_grid_capture:
            return list(range(n))

        _alternating = self.alternating == "yes"
        rows = int(math.ceil(n / float(self.grid_cols)))
        top_first = self.start_corner.startswith("top")
        left_first = self.start_corner.endswith("left")
        row_base = list(range(rows)) if top_first else list(range(rows - 1, -1, -1))
        col_base = list(range(self.grid_cols)) if left_first else list(range(self.grid_cols - 1, -1, -1))

        out: List[int] = []

        if self.scan_order == "column-wise":
            for i, c in enumerate(col_base):
                rr = row_base if (not _alternating or i % 2 == 0) else list(reversed(row_base))
                for r in rr:
                    idx = r * self.grid_cols + c
                    if 0 <= idx < n:
                        out.append(idx)
        else:
            for i, r in enumerate(row_base):
                cc = col_base if (not _alternating or i % 2 == 0) else list(reversed(col_base))
                for c in cc:
                    idx = r * self.grid_cols + c
                    if 0 <= idx < n:
                        out.append(idx)

        return out