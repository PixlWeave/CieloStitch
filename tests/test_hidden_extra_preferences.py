# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

from cielostitch_core.state.preferences import AppPreferences
from cielostitch_core.utils.message import message_color_passed


def test_hidden_extra_preferences_load_new_section_and_legacy_fallback(monkeypatch):
    values = {
        "stitch/enable_subpixel_refinement": "false",
        "extra/enable_subpixel_refinement": "true",
        "extra/log_message_colors": "default, red, panel",
    }

    def fake_read_pref(key, default=None, value_type=None, type=None):
        return values.get(key, default)

    monkeypatch.setattr("cielostitch_core.state.preferences.read_pref", fake_read_pref)

    prefs = AppPreferences.load()

    assert prefs.enable_subpixel_refinement is False
    assert prefs.log_message_colors == "default,red,panel"


def test_hidden_subpixel_preference_uses_legacy_extra_as_fallback(monkeypatch):
    values = {
        "extra/enable_subpixel_refinement": "false",
    }

    def fake_read_pref(key, default=None, value_type=None, type=None):
        return values.get(key, default)

    monkeypatch.setattr("cielostitch_core.state.preferences.read_pref", fake_read_pref)

    prefs = AppPreferences.load()

    assert prefs.enable_subpixel_refinement is False


def test_message_color_passed_uses_hidden_allowlist(monkeypatch):
    prefs = AppPreferences(log_message_colors="default,panel")

    class _CfgStub:
        pass

    cfg_stub = _CfgStub()
    cfg_stub.prefs = prefs
    cfg_stub.allowed_message_colors = {"default", "panel"}

    monkeypatch.setattr("cielostitch_core.utils.message.cfg", cfg_stub)

    assert message_color_passed("") is True
    assert message_color_passed("panel") is True
    assert message_color_passed("debug") is False


def test_subpixel_preference_saves_to_stitch_namespace(monkeypatch):
    writes = {}

    def fake_write_pref(key, value):
        writes[key] = value

    monkeypatch.setattr("cielostitch_core.state.preferences.write_pref", fake_write_pref)
    monkeypatch.setattr("cielostitch_core.state.preferences.sync_prefs", lambda: None)

    prefs = AppPreferences(enable_subpixel_refinement=False)
    prefs.save()

    assert writes.get("stitch/enable_subpixel_refinement") is False
    assert "extra/enable_subpixel_refinement" not in writes


def test_candidate_debug_preference_loads_from_stitch_namespace(monkeypatch):
    values = {
        "stitch/enable_candidate_debug": "true",
    }

    def fake_read_pref(key, default=None, value_type=None, type=None):
        return values.get(key, default)

    monkeypatch.setattr("cielostitch_core.state.preferences.read_pref", fake_read_pref)

    prefs = AppPreferences.load()

    assert prefs.enable_candidate_debug is True


def test_candidate_debug_preference_saves_to_stitch_namespace(monkeypatch):
    writes = {}

    def fake_write_pref(key, value):
        writes[key] = value

    monkeypatch.setattr("cielostitch_core.state.preferences.write_pref", fake_write_pref)
    monkeypatch.setattr("cielostitch_core.state.preferences.sync_prefs", lambda: None)

    prefs = AppPreferences(enable_candidate_debug=True)
    prefs.save()

    assert writes.get("stitch/enable_candidate_debug") is True