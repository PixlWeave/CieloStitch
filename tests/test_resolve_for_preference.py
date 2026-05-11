# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

from cielostitch_core.state.preferences import AppPreferences
from cielostitch_core.state.state_manager import AppStateManager


def test_app_state_uses_default_resolve_for_from_preferences():
    prefs = AppPreferences(default_resolve_for="best")
    state = AppStateManager(prefs)
    assert state.speed_preset == "best"


def test_app_state_speed_preset_falls_back_to_balanced_for_invalid_values():
    prefs = AppPreferences(default_resolve_for="turbo")
    state = AppStateManager(prefs)
    assert state.speed_preset == "balanced"

    state.speed_preset = "ultra"
    assert state.speed_preset == "balanced"
