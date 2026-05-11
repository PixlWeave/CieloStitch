# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

from cielostitch_core.state.preferences import AppPreferences
from cielostitch_core.state.session_model import SessionModel


def make_prefs(
    default_stitch_mode: str = "grid-guided",
    default_grid_cols: int = 3,
) -> AppPreferences:
    return AppPreferences(
        default_stitch_mode=default_stitch_mode,
        default_grid_cols=default_grid_cols,
    )


def make_manager(
    default_stitch_mode: str = "grid-guided",
    default_grid_cols: int = 3,
) -> SessionModel:
    return SessionModel(make_prefs(default_stitch_mode, default_grid_cols))


def test_initialization_uses_preferences_defaults() -> None:
    manager = make_manager(default_stitch_mode="grid-guided", default_grid_cols=3)

    assert manager.current_stitch_mode == "grid-guided"
    assert manager.grid_cols == 3
    assert manager.is_modified() is False


def test_get_values_returns_copy() -> None:
    manager = make_manager()

    values = manager.get_values()
    values["grid_cols"] = 99

    assert manager.grid_cols == 3
    assert manager.get_values()["grid_cols"] == 3


def test_get_value() -> None:
    manager = make_manager()

    assert manager.get_value("grid_cols") == 3
    assert manager.get_value("scan_order") == "row-wise"
    assert manager.get_value("missing", "fallback") == "fallback"


def test_set_value_marks_dirty_only_on_change() -> None:
    manager = make_manager()
    manager.clear_dirty()

    manager.set_value("grid_cols", 5)
    assert manager.grid_cols == 5
    assert manager.is_modified() is True

    manager.clear_dirty()
    manager.set_value("grid_cols", 5)
    assert manager.is_modified() is False


def test_set_value_overlap_pct_emits_grid_settings_changed() -> None:
    manager = make_manager()
    grid_changed_count = {"count": 0}

    def _on_grid_changed() -> None:
        grid_changed_count["count"] += 1

    manager.gridSettingsChanged.connect(_on_grid_changed)

    manager.set_value("overlap_x_pct", 25.0)
    manager.set_value("overlap_y_pct", 30.0)

    assert grid_changed_count["count"] == 2


def test_set_value_ignores_unknown_keys() -> None:
    manager = make_manager()
    manager.clear_dirty()

    manager.set_value("not_a_real_field", 123)

    assert manager.is_modified() is False
    assert not hasattr(manager, "not_a_real_field")


def test_set_values_from_dict_replaces_state() -> None:
    manager = make_manager()
    manager.clear_dirty()

    manager.set_values(
        {
            "grid_cols": 5,
            "scan_order": "column-wise",
            "start_corner": "bottom-right",
            "alternating": "yes",
            "image_sort_order": "mtime",
            "mount_precision": "high",
            "seeing": "good",
            "overlap_x_pct": 25.0,
            "overlap_y_pct": 30.0,
            "staged_matching_mode": "manual",
            "refs_per_stage": 4,
            "enable_grid_phase_fallback": False,
            "enable_grid_nominal_fallback": False,
            "enable_grid_shift_guard": False,
            "grid_guide": {
                "grid_absolute_nominal_score": 2.5,
                "grid_expected_shift_tolerance": 0.2,
                "grid_orthogonal_shift_tolerance": 0.3,
                "grid_nominal_shift_bias": 1.5,
                "grid_phase_fallback_min_response": 0.05,
                "grid_phase_fallback_max_shift_ratio": 0.8,
            },
        }
    )

    assert manager.grid_cols == 5
    assert manager.scan_order == "column-wise"
    assert manager.start_corner == "bottom-right"
    assert manager.alternating == "yes"
    assert manager.mount_precision == "high"
    assert manager.seeing == "good"
    assert manager.overlap_x_pct == 25.0
    assert manager.overlap_y_pct == 30.0
    assert manager.staged_matching_mode == "manual"
    assert manager.enable_grid_phase_fallback is False
    assert manager.enable_grid_nominal_fallback is False
    assert manager.enable_grid_shift_guard is False
    assert manager.grid_guide.grid_expected_shift_tolerance == 0.2
    assert manager.is_modified() is True


def test_set_values_with_equivalent_state_does_not_mark_dirty() -> None:
    manager = make_manager()
    manager.clear_dirty()

    manager.set_values(manager.to_dict())

    assert manager.is_modified() is False


def test_set_values_from_other_session_model() -> None:
    manager = make_manager()
    other = make_manager(default_grid_cols=7)
    other.scan_order = "column-wise"

    manager.clear_dirty()
    manager.set_values(other)

    assert manager.grid_cols == 7
    assert manager.scan_order == "column-wise"
    assert manager.is_modified() is True


def test_set_values_from_other_session_model_copies_grid_guide() -> None:
    manager = make_manager()
    other = make_manager()
    other.grid_guide.grid_expected_shift_tolerance = 0.33

    manager.set_values(other)
    assert manager.grid_guide.grid_expected_shift_tolerance == 0.33

    # Mutating source model afterward must not change manager.
    other.grid_guide.grid_expected_shift_tolerance = 0.77
    assert manager.grid_guide.grid_expected_shift_tolerance == 0.33


def test_dirty_tracking() -> None:
    manager = make_manager()

    assert manager.is_modified() is False

    manager.mark_dirty()
    assert manager.is_modified() is True

    manager.clear_dirty()
    assert manager.is_modified() is False


def test_suppress_dirty() -> None:
    manager = make_manager()
    manager.clear_dirty()

    with manager.suppress_dirty():
        manager.set_value("grid_cols", 8)
        manager.set_value("scan_order", "column-wise")

    assert manager.grid_cols == 8
    assert manager.scan_order == "column-wise"
    assert manager.is_modified() is False

    manager.mark_dirty()
    assert manager.is_modified() is True


def test_set_default_values_resets_state_and_clears_dirty() -> None:
    manager = make_manager(default_stitch_mode="auto", default_grid_cols=6)

    manager.set_value("grid_cols", 10)
    manager.set_value("scan_order", "column-wise")
    assert manager.is_modified() is True

    manager.set_default_values()

    assert manager.grid_cols == 6
    assert manager.scan_order == "row-wise"
    assert manager.start_corner == "top-left"
    assert manager.alternating == "no"
    assert manager.image_sort_order == "natural"
    assert manager.mount_precision == "normal"
    assert manager.seeing == "average"
    assert manager.overlap_x_pct == 20.0
    assert manager.overlap_y_pct == 20.0
    assert manager.staged_matching_mode == "auto"
    assert manager.refs_per_stage == 6
    assert manager.enable_grid_phase_fallback is True
    assert manager.enable_grid_nominal_fallback is True
    assert manager.enable_grid_shift_guard is True
    assert manager.is_modified() is False


def test_to_dict_and_from_dict_round_trip() -> None:
    manager = make_manager()
    manager.set_value("grid_cols", 9)
    manager.set_value("scan_order", "column-wise")
    manager.grid_guide.grid_expected_shift_tolerance = 0.33

    snapshot = manager.to_dict()

    restored = make_manager()
    restored.from_dict(snapshot)

    assert restored.grid_cols == 9
    assert restored.scan_order == "column-wise"
    assert restored.grid_guide.grid_expected_shift_tolerance == 0.33
    assert restored.to_dict() == snapshot
    assert restored.is_modified() is False


def test_from_dict_invalid_data_uses_defaults_and_stays_clean() -> None:
    manager = make_manager()
    manager.clear_dirty()

    manager.from_dict(
        {
            "grid_cols": "invalid",
            "scan_order": None,
            "overlap_x_pct": "bad",
            "grid_guide": "not-a-dict",
        }
    )

    assert manager.grid_cols == 3
    assert manager.scan_order == "row-wise"
    assert manager.overlap_x_pct == 20.0
    assert manager.is_modified() is False


def test_get_capture_slots_non_grid_mode_returns_linear_order() -> None:
    manager = make_manager()
    manager.current_stitch_mode = "auto"
    manager.grid_cols = 3

    assert manager.get_capture_slots(5) == [0, 1, 2, 3, 4]


def test_get_capture_slots_grid_mode_row_wise() -> None:
    manager = make_manager()
    manager.current_stitch_mode = "grid-guided"
    manager.grid_cols = 3
    manager.scan_order = "row-wise"
    manager.start_corner = "top-left"
    manager.alternating = "no"

    assert manager.get_capture_slots(6) == [0, 1, 2, 3, 4, 5]


def test_get_capture_slots_grid_mode_column_wise_alternating() -> None:
    manager = make_manager()
    manager.current_stitch_mode = "grid-guided"
    manager.grid_cols = 3
    manager.scan_order = "column-wise"
    manager.start_corner = "top-left"
    manager.alternating = "yes"

    assert manager.get_capture_slots(6) == [0, 3, 4, 1, 2, 5]


def test_get_capture_slots_grid_mode_bottom_right_row_wise_alternating() -> None:
    manager = make_manager()
    manager.current_stitch_mode = "grid-guided"
    manager.grid_cols = 3
    manager.scan_order = "row-wise"
    manager.start_corner = "bottom-right"
    manager.alternating = "yes"

    assert manager.get_capture_slots(6) == [5, 4, 3, 0, 1, 2]


def test_scan_icon_filename() -> None:
    manager = make_manager()
    manager.current_stitch_mode = "grid-guided"
    manager.scan_order = "row-wise"
    manager.start_corner = "top-left"
    manager.alternating = "yes"

    assert manager.scan_icon_filename() == "scan-top-left-row-wise-alt.png"

    manager.current_stitch_mode = "freeform"
    assert manager.scan_icon_filename() == ""
