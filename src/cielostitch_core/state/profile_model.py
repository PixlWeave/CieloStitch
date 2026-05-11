# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from .profile_states import (
    AFieldVals,
    ProfileStates as ps,
)

from PySide6.QtCore import QObject, Signal

class ProfileModel(QObject):
    """Qt-agnostic manager for per-profile Advanced settings only.

    Responsibilities:
      - Store per-profile (profile, stitching mode) values (as plain dicts)
      - Track modified (dirty) profiles
      - Provide suppression profile for programmatic updates
      - Snapshot and apply snapshots for round-trip save/load

    NOTE: Session/Basic settings are now managed separately in SessionModel.
          This manager handles ONLY profile & stitching mode specific stitching tuning parameters.
    """

    profileMassSettingsChanged = Signal()
    profileDirtyChanged = Signal()
    # fires on every value write, regardless of dirty state
    # it will be fired .e.g. when resolver changes values
    profileValueChanged = Signal()


    def __init__(self, profile: str, /) -> None:
        super().__init__()
        self._profiles = ps.PROFILES
        self._values_by_profile = ps.PROFILE_MODIFIED_VALUES
        self._defaults_by_profile = ps.PROFILE_DEFAULT_VALUES
        self._dirty_profiles: set[str] = set()
        self._suppress_dirty: int = 0
        self._current_profile = profile

    @property
    def current_profile(self) -> str:
        return self._current_profile

    @current_profile.setter
    def current_profile(self, value: str) -> None:
        self._current_profile = value

    def __getattr__(self, name: str) -> Any:
        """Enable attribute-style access: psm.overlap_x_pct instead of psm.get_value('overlap_x_pct')"""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name in ps.PROFILE_FIELD_NAMES_ALL:
            return self.get_value(name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_values(self, profile: str = None) -> AFieldVals:
        profile = profile if profile is not None else self._current_profile
        # dict() will help to return a copy/snapshot which is not mutable
        return dict(self._values_by_profile.get(profile, {}) or {})

    def set_values(self, values: AFieldVals, profile: str = None) -> None:
        profile = profile if profile is not None else self._current_profile
        new_values = dict(values)
        current_values = dict(self._values_by_profile.get(profile, {}) or {})
        if current_values == new_values:
            return
        self._values_by_profile[profile] = new_values
        self.mark_dirty(profile)
        self.profileMassSettingsChanged.emit()
        if self._suppress_dirty == 0:
            self.profileValueChanged.emit()

    def set_default_values(self, profile: str = None) -> None:
        profile = profile if profile is not None else self._current_profile
        self._values_by_profile[profile] = self.get_default_values(profile)
        self.mark_dirty(profile)
        self.profileMassSettingsChanged.emit()

    def get_value(self, field_name: str, default: Any = None, profile: str = None) -> Any:
        profile = profile if profile is not None else self._current_profile
        bucket = self._values_by_profile.get(profile, {})
        val = bucket.get(field_name)

        # If value not found in modified values, fall back to defaults
        if val is None:
            val = self.get_default_value(field_name, default, profile)

        return val if val is not None else default

    def set_value(self, field_name: str = None, value: Any = None, profile: str = None) -> None:
        """Set a single profile value by field name.

        Args:
            field_name: Name of the field to set
            value: Value to set
            profile: Profile name (defaults to active profile)

        Note: Will create the bucket for the profile if it doesn't exist, ensuring
        new fields can be added.
        """
        profile = profile if profile is not None else self._current_profile
        # No-op writes should not dirty the profile or emit value-changed.
        if self.get_value(field_name, None, profile) == value:
            return
        # Create bucket if it doesn't exist (enables adding new fields)
        if profile not in self._values_by_profile:
            self._values_by_profile[profile] = {}
        bucket = self._values_by_profile[profile]
        # Always update the field (don't skip if it already exists or is None)
        bucket[field_name] = value
        self.mark_dirty(profile)
        if self._suppress_dirty == 0:
            self.profileValueChanged.emit()

    def get_default_values(self, profile: str = None) -> AFieldVals:
        profile = profile if profile is not None else self._current_profile
        # dict() will help to return a copy/snapshot which is not mutable
        return dict(self._defaults_by_profile.get(profile, {}) or {})

    def get_default_value(self, field_name: str = None, defa: Any = None, profile: str = None) -> Any:
        profile = profile if profile is not None else self._current_profile
        return self.get_default_values(profile).get(field_name, defa)

    # ----- Dirty tracking -----
    def is_modified(self, profile: str = None) -> bool:
        """Check if the given profile (or active profile) has been modified."""
        profile = profile if profile is not None else self._current_profile
        return profile in self._dirty_profiles

    def any_modified(self) -> bool:
        """Check if any profile has been modified."""
        return bool(self._dirty_profiles)

    def clear_dirty(self, profile: str = None) -> None:
        profile = profile if profile is not None else self._current_profile
        dirty_changed = False
        if profile == 'all':
            dirty_changed = bool(self._dirty_profiles)
            self._dirty_profiles.clear()
        else:
            if profile in self._dirty_profiles:
                self._dirty_profiles.discard(profile)
                dirty_changed = True
        if dirty_changed:
            self.profileDirtyChanged.emit()

    def mark_dirty(self, profile: str = None) -> None:
        profile = profile if profile is not None else self._current_profile
        if self._suppress_dirty == 0 and profile not in self._dirty_profiles:
            self._dirty_profiles.add(profile)
            self.profileDirtyChanged.emit()

    @contextmanager
    def suppress_dirty(self):
        self._suppress_dirty += 1
        try:
            yield
        finally:
            self._suppress_dirty -= 1
