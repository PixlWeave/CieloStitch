# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

"""
Structural invariant test: no SessionModel field should exist as a class attribute
in any xxxProfile class inside the profiles folder.

If this test fails, a field that belongs to SessionModel has been accidentally added
to a profile class, which causes silent routing bugs (app_state_manager.find() will
route the field to psm instead of ssm).
"""
from cielostitch_core.state.preferences import AppPreferences
from cielostitch_core.state.session_model import SessionModel
from cielostitch_core.state.profile_states import PROFILE_CLASSES


def _session_field_names() -> set:
    """Return all public instance attribute names defined by SessionModel."""
    prefs = AppPreferences()
    instance = SessionModel(prefs)
    return {
        name for name in vars(instance)
        if not name.startswith("_")
    }


def _profile_field_names(profile_cls) -> set:
    """Return all public non-callable class-level attributes of a profile class."""
    return {
        name for name in dir(profile_cls)
        if not name.startswith("_") and not callable(getattr(profile_cls, name))
    }


def test_no_session_fields_in_profiles():
    session_fields = _session_field_names()

    violations: dict[str, set] = {}
    for profile_name, profile_cls in PROFILE_CLASSES.items():
        profile_fields = _profile_field_names(profile_cls)
        overlap = session_fields & profile_fields
        if overlap:
            violations[profile_name] = overlap

    assert not violations, (
        "The following profile class(es) contain SessionModel field(s):\n" +
        "\n".join(f"  {name}: {sorted(fields)}" for name, fields in violations.items())
    )
