# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

"""Centralized application states management.

This module provides a single source of truth for all application states,
eliminating scattered states variables across MainWindow and helper functions.

The AppStateManager class owns:
- Profile and stitch mode
- Panel paths and grid settings
- Stitching run states
- Output image states
- Settings file associations
- UI preference flags (dirty, running, etc.)

Changes to states emit signals that can be observed by UI coordinators.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from PySide6.QtCore import Signal
from PySide6.QtCore import QObject

from ..config.constants import SPEED_PRESETS
from ..state.preferences import AppPreferences
from .session_model import SessionModel
from .profile_model import ProfileModel

class AppStateManager(QObject):
    """Centralized application states with change signals.

    Emits signals when states changes, allowing UI coordinators to respond
    without scattered update logic throughout the codebase.

    State is organized into logical groups:
    - Profile: profile
    - Session: grid, panel paths, sorting
    - Stitching: running states, output image, overlays
    - Settings: dirty flags, file associations
    - UI: preview modes, zoom mode
    """

    profileChanged = Signal()  # profile name
    stitchModeChanged = Signal() # stitch_mode name
    settingsFileChanged = Signal()

    def __init__(self, prefs: AppPreferences | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._window = parent
        prefs = prefs or AppPreferences.load()

        self._profile = prefs.default_profile
        self._stitch_mode: str = prefs.default_stitch_mode
        # Profile states and Session states
        self.psm: ProfileModel = ProfileModel(self._profile)
        self.ssm: SessionModel = SessionModel(prefs)

        # Stitching states
        self._is_running: bool = False
        self._last_resolved_values: Dict[str, Any] = {}
        self._last_resolve_inputs: Dict[str, Any] = {}
        self._is_resolved: bool = False
        self._show_pixel_stat: bool = True
        self._cli_mode: bool = False
        preset = str(getattr(prefs, "default_resolve_for", "balanced") or "balanced").strip().lower()
        self._speed_preset: str = preset if preset in SPEED_PRESETS else "balanced"

        # Saved Settings states
        self._open_project_file: Optional[str] = None
        self._open_profile: Optional[str] = None

    def find(self, field_name) -> ProfileModel | SessionModel | AppStateManager | None:
        # get a handle of psm or ssm or ap_state that contains the field
        profile_fields = self.psm.get_values(self._profile)
        if field_name not in profile_fields:
            profile_fields = self.psm.get_default_values(self._profile)
        if field_name in profile_fields:
            return self.psm
        if field_name in self.ssm.get_values():
            return self.ssm
        if hasattr(self, field_name) and not callable(getattr(self, field_name)):
            return self
        return None

    def get_any_value(self, field_name) -> Any:
        found = self.find(field_name)
        if found is not None:
            if found is self:
                return getattr(self, field_name, None)
            return found.get_value(field_name, None)
        return None

    def set_any_value(self, field_name, value: Any) -> None:
        # set a profile or session field value
        found = self.find(field_name)
        if found is not None:
            if found is self:
                setattr(self, field_name, value)
                return
            found.set_value(field_name, value)
            return

    @property
    def profile(self) -> str:
        return self._profile

    @profile.setter
    def profile(self, value: str) -> None:
        if self._profile != value:
            self._profile = value
            # Keep the profile model's implicit target in sync with app state.
            self.psm.current_profile = value
            # Invalidate resolve snapshot — resolved values belong to the old profile.
            self._last_resolved_values = {}
            self._last_resolve_inputs = {}
            self.profileChanged.emit()

    # === Stitching State ===
    @property
    def stitch_mode(self) -> str:
        return self._stitch_mode

    @stitch_mode.setter
    def stitch_mode(self, value: str) -> None:
        if self._stitch_mode != value:
            self._stitch_mode = value
            # Important: update existing instance
            self.ssm.current_stitch_mode = value
            self.stitchModeChanged.emit()
            self.ssm.gridSettingsChanged.emit()

    @property
    def cli_mode(self) -> bool:
        return self._cli_mode

    @cli_mode.setter
    def cli_mode(self, value: bool) -> None:
        if self._cli_mode != value:
            self._cli_mode = value

    @property
    def is_running(self) -> bool:
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        if self._is_running != value:
            self._is_running = value

    @property
    def is_resolved(self) -> bool:
        return self._is_resolved

    @is_resolved.setter
    def is_resolved(self, value: bool) -> None:
        if self._is_resolved != value:
            self._is_resolved = value

    @property
    def open_project_file(self) -> Optional[str]:
        return self._open_project_file
            
    @property
    def open_profile(self) -> Optional[str]:
        return self._open_profile

    def set_open_settings_file(self, path: str | None, profile: str | None) -> None:
        changed = (self._open_project_file != path) or (self._open_profile != profile)
        if changed:
            self._open_project_file = path
            self._open_profile = profile
            self.settingsFileChanged.emit()

    @property
    def is_in_file_mode(self) -> bool:
        """Return True if a settings file is open for the active profile."""
        return bool(self._open_project_file and self._open_profile == self._profile)

    @property
    def show_pixel_stat(self) -> bool:
        return self._show_pixel_stat

    @show_pixel_stat.setter
    def show_pixel_stat(self, value: bool) -> None:
        self._show_pixel_stat = value

    @property
    def speed_preset(self) -> str:
        return self._speed_preset

    @speed_preset.setter
    def speed_preset(self, value: str) -> None:
        preset = str(value or "balanced").strip().lower()
        self._speed_preset = preset if preset in SPEED_PRESETS else "balanced"

    @property
    def last_resolved_values(self) -> Dict[str, Any]:
        return self._last_resolved_values

    @last_resolved_values.setter
    def last_resolved_values(self, value: Dict[str, Any]) -> None:
        self._last_resolved_values = value

    @property
    def last_resolve_inputs(self) -> Dict[str, Any]:
        return self._last_resolve_inputs

    @last_resolve_inputs.setter
    def last_resolve_inputs(self, value: Dict[str, Any]) -> None:
        self._last_resolve_inputs = value
