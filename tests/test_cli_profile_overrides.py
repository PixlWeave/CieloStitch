# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import json
import os

import numpy as np
import pytest

from cielostitch_core import cli
from cielostitch_core.config.config import cfg
from cielostitch_core.state.preferences import AppPreferences
from cielostitch_core.state.state_manager import AppStateManager

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_IMG = np.zeros((100, 120, 3), dtype=np.float32)
_FAKE_ITEMS = [("panel.tif", _FAKE_IMG)]
_FAKE_LOAD_RESULT = (_FAKE_ITEMS, 16, "fp32")


def _reset_state(monkeypatch):
    prefs = AppPreferences(default_profile="general", default_stitch_mode="auto")
    monkeypatch.setattr(cfg, "prefs", prefs)
    monkeypatch.setattr(cfg, "state", AppStateManager(prefs))
    return prefs


def _patch_io(monkeypatch, load_result=None):
    """Patch the image-scan / load helpers and output helpers."""
    result = load_result if load_result is not None else _FAKE_LOAD_RESULT
    monkeypatch.setattr(cli, "_scan_image_paths", lambda folder: ["panel.tif"])
    monkeypatch.setattr(cli, "load_panel_images_from_paths", lambda paths, **kw: result)
    monkeypatch.setattr(cli, "resolve_output_path", lambda **kwargs: "out.tif")
    monkeypatch.setattr(cli, "from_internal_float32", lambda mosaic, _bd: mosaic)
    monkeypatch.setattr(cli, "save_image", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# DummyEngine — captures what profile was constructed with
# ---------------------------------------------------------------------------

class _DummyEngine:
    captured: dict = {}

    def __init__(self, profile, grid_cols=None):
        type(self).captured = {
            "profile_blend_type": getattr(profile, "blend_type", None),
            "profile_gain_compensation": getattr(profile, "gain_compensation", None),
            "profile_transform_mode": getattr(profile, "transform_mode", None),
            "profile_photometric_min_overlap_px": getattr(profile, "photometric_min_overlap_px", None),
            "psm_blend_type": cfg.state.psm.blend_type,
            "psm_gain_compensation": cfg.state.psm.gain_compensation,
            "ssm_enable_grid_nominal_fallback": cfg.state.ssm.enable_grid_nominal_fallback,
            "ssm_staged_matching_mode": cfg.state.ssm.staged_matching_mode,
            "ssm_refs_per_stage": cfg.state.ssm.refs_per_stage,
            "grid_cols": grid_cols,
        }

    @staticmethod
    def stitch(items, **kwargs):
        return items[0][1]


# ---------------------------------------------------------------------------
# 1. Profile + blend/gain overrides sync to runtime state
# ---------------------------------------------------------------------------

def test_cli_profile_overrides_sync_to_runtime_state(monkeypatch):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = [
        "input_dir",
        "--profile", "solar",
        "--stitch-mode", "freeform",
        "--blend-type", "feather",
        "--gain-compensation", "uniform",
    ]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert _DummyEngine.captured["profile_blend_type"] == "feather"
    assert _DummyEngine.captured["profile_gain_compensation"] == "uniform"
    assert _DummyEngine.captured["psm_blend_type"] == "feather"
    assert _DummyEngine.captured["psm_gain_compensation"] == "uniform"


# ---------------------------------------------------------------------------
# 2. _parse_cli_args — new flags --resolve / --load-settings
# ---------------------------------------------------------------------------

def test_parse_cli_args_resolve_flag():
    parsed = cli._parse_cli_args(["input_dir", "--resolve"])
    assert parsed is not None
    (_, _, _, _, _, _, _, _, _, _, _, settings_file, resolve) = parsed
    assert resolve is True
    assert settings_file is None


def test_parse_cli_args_resolve_short_flag():
    parsed = cli._parse_cli_args(["input_dir", "-r"])
    assert parsed is not None
    resolve = parsed[-1]
    assert resolve is True


def test_parse_cli_args_load_settings_flag():
    parsed = cli._parse_cli_args(["input_dir", "--load-settings", "my_settings.json"])
    assert parsed is not None
    (_, _, _, _, _, _, _, _, _, _, _, settings_file, resolve) = parsed
    assert settings_file == "my_settings.json"
    assert resolve is False


def test_parse_cli_args_load_settings_short_flag():
    parsed = cli._parse_cli_args(["input_dir", "-l", "s.json"])
    assert parsed is not None
    settings_file = parsed[-2]
    assert settings_file == "s.json"


# ---------------------------------------------------------------------------
# 3. _parse_cli_args — sentinel None defaults for profile / cols / overlaps
# ---------------------------------------------------------------------------

def test_parse_cli_args_sentinel_defaults():
    parsed = cli._parse_cli_args(["input_dir"])
    assert parsed is not None
    (_, _, engine, profile_name, _, cols, overlap_x, overlap_y, _, _, _, _, _) = parsed
    assert profile_name is None
    assert engine is None
    assert cols is None
    assert overlap_x is None
    assert overlap_y is None


def test_parse_cli_args_explicit_cols_and_overlaps():
    parsed = cli._parse_cli_args(["input_dir", "-c", "4", "-x", "0", "-y", "0"])
    assert parsed is not None
    (_, _, _, _, _, cols, overlap_x, overlap_y, _, _, _, _, _) = parsed
    assert cols == 4
    assert overlap_x == 0.0
    assert overlap_y == 0.0


def test_parse_cli_args_engine_flag():
    parsed = cli._parse_cli_args(["input_dir", "--engine", "simple"])
    assert parsed is not None
    engine = parsed[2]
    assert engine == "simple"


def test_parse_cli_args_engine_short_flag():
    parsed = cli._parse_cli_args(["input_dir", "-e", "auto"])
    assert parsed is not None
    engine = parsed[2]
    assert engine == "auto"


# ---------------------------------------------------------------------------
# 4. _scan_image_paths
# ---------------------------------------------------------------------------

def test_scan_image_paths_missing_dir():
    with pytest.raises(ValueError, match="does not exist"):
        cli._scan_image_paths("/nonexistent/path/xyz123")


def test_scan_image_paths_filters_extensions(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    (tmp_path / "c.fits").write_bytes(b"")
    (tmp_path / "d.png").write_bytes(b"")

    paths = cli._scan_image_paths(str(tmp_path))
    basenames = [os.path.basename(p) for p in paths]
    assert "a.tif" in basenames
    assert "c.fits" in basenames
    assert "d.png" in basenames
    assert "b.txt" not in basenames


def test_scan_image_paths_sorted(tmp_path):
    for name in ("z.tif", "a.tif", "m.tif"):
        (tmp_path / name).write_bytes(b"")
    paths = cli._scan_image_paths(str(tmp_path))
    basenames = [os.path.basename(p) for p in paths]
    assert basenames == sorted(basenames)


# ---------------------------------------------------------------------------
# 5. _load_and_validate_settings_file
# ---------------------------------------------------------------------------

def _valid_snap(**overrides):
    snap = {"kind": "cielostitch_settings", "version": 1, "profile": "solar"}
    snap.update(overrides)
    return snap


def test_load_settings_file_missing():
    with pytest.raises(ValueError, match="not found"):
        cli._load_and_validate_settings_file("/nonexistent/file.json")


def test_load_settings_file_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("not json {{{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        cli._load_and_validate_settings_file(str(f))


def test_load_settings_file_not_a_dict(tmp_path):
    f = tmp_path / "arr.json"
    f.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        cli._load_and_validate_settings_file(str(f))


def test_load_settings_file_wrong_kind(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"kind": "other", "version": 1, "profile": "solar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        cli._load_and_validate_settings_file(str(f))


def test_load_settings_file_wrong_version(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"kind": "cielostitch_settings", "version": 99, "profile": "solar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        cli._load_and_validate_settings_file(str(f))


def test_load_settings_file_missing_profile_field(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"kind": "cielostitch_settings", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="profile"):
        cli._load_and_validate_settings_file(str(f))


def test_load_settings_file_valid(tmp_path):
    snap = _valid_snap(profile_data={"blend_type": "feather"})
    f = tmp_path / "s.json"
    f.write_text(json.dumps(snap), encoding="utf-8")
    result = cli._load_and_validate_settings_file(str(f))
    assert result["profile"] == "solar"
    assert result["profile_data"]["blend_type"] == "feather"


# ---------------------------------------------------------------------------
# 6. Settings file wiring in main()
# ---------------------------------------------------------------------------

def test_cli_settings_file_profile_used_when_no_cli_profile(monkeypatch, tmp_path):
    """When --profile is not given, profile from settings file is used."""
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    snap = _valid_snap(profile="lunar")
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps(snap), encoding="utf-8")

    captured = {}

    class _CapEngine:
        def __init__(self, profile, grid_cols=None):
            captured["profile_type"] = type(profile).__name__

        @staticmethod
        def stitch(items, **kwargs):
            return items[0][1]

    monkeypatch.setattr(cli, "FreeMode", _CapEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = ["input_dir", "--stitch-mode", "freeform", "--load-settings", str(sf)]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert captured["profile_type"] == "LunarProfile"


def test_cli_engine_flag_updates_runtime_state(monkeypatch):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = ["input_dir", "--engine", "simple", "--stitch-mode", "freeform"]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert cfg.state.engine == "simple"


def test_cli_explicit_profile_overrides_settings_file(monkeypatch, tmp_path):
    """Explicit --profile CLI flag beats the settings file profile."""
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    snap = _valid_snap(profile="lunar")
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps(snap), encoding="utf-8")

    captured = {}

    class _CapEngine:
        def __init__(self, profile, grid_cols=None):
            captured["profile_type"] = type(profile).__name__

        @staticmethod
        def stitch(items, **kwargs):
            return items[0][1]

    monkeypatch.setattr(cli, "FreeMode", _CapEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = ["input_dir", "--stitch-mode", "freeform", "--profile", "solar", "--load-settings", str(sf)]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert captured["profile_type"] == "SolarProfile"


def test_cli_settings_file_profile_data_applied(monkeypatch, tmp_path):
    """profile_data from the settings file is applied to the profile object."""
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    snap = _valid_snap(profile="general", profile_data={"blend_type": "seamless"})
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps(snap), encoding="utf-8")

    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = ["input_dir", "--stitch-mode", "freeform", "--load-settings", str(sf)]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert _DummyEngine.captured["profile_blend_type"] == "seamless"


def test_cli_settings_file_session_data_applied_to_ssm(monkeypatch, tmp_path):
    """session_data from the settings file is applied to ssm before sentinel reads."""
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    snap = _valid_snap(profile="general", session_data={"overlap_x_pct": 33.0, "overlap_y_pct": 27.0})
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps(snap), encoding="utf-8")

    ssm_calls = {}

    original_from_dict = cfg.state.ssm.__class__.from_dict

    def _capture_from_dict(self, d):
        ssm_calls["called"] = True
        ssm_calls["data"] = d
        original_from_dict(self, d)

    monkeypatch.setattr(cfg.state.ssm.__class__, "from_dict", _capture_from_dict)
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = ["input_dir", "--stitch-mode", "freeform", "--load-settings", str(sf)]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert ssm_calls.get("called") is True
    assert ssm_calls["data"].get("overlap_x_pct") == 33.0


def test_cli_settings_file_cli_overlaps_override_file(monkeypatch, tmp_path):
    """Explicit --overlap-x-pct CLI value overrides the value from settings file."""
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    snap = _valid_snap(profile="general", session_data={"overlap_x_pct": 33.0})
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps(snap), encoding="utf-8")

    ssm_state = {}

    class _CapEngine:
        def __init__(self, profile, grid_cols=None):
            ssm_state["overlap_x_pct"] = cfg.state.ssm.overlap_x_pct

        @staticmethod
        def stitch(items, **kwargs):
            return items[0][1]

    monkeypatch.setattr(cli, "FreeMode", _CapEngine)
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: {})

    argv = ["input_dir", "--stitch-mode", "freeform", "--load-settings", str(sf), "-x", "10"]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert ssm_state["overlap_x_pct"] == 10.0


# ---------------------------------------------------------------------------
# 7. --resolve / auto mode invokes compute_resolved_params
# ---------------------------------------------------------------------------

def test_cli_resolve_flag_invokes_resolver(monkeypatch):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    resolver_calls = []

    def _fake_resolver(ri):
        resolver_calls.append(ri)
        return {}

    monkeypatch.setattr(cli, "compute_resolved_params", _fake_resolver)
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)

    argv = ["input_dir", "--stitch-mode", "freeform", "--resolve"]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert len(resolver_calls) == 1
    ri = resolver_calls[0]
    assert ri.panel_w == 120
    assert ri.panel_h == 100


def test_cli_auto_stitch_mode_invokes_resolver(monkeypatch):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    resolver_calls = []

    def _fake_resolver(ri):
        resolver_calls.append(ri)
        return {}

    monkeypatch.setattr(cli, "compute_resolved_params", _fake_resolver)
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)

    argv = ["input_dir", "--stitch-mode", "auto"]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert len(resolver_calls) == 1


def test_cli_resolved_session_fields_apply_to_runtime_state(monkeypatch):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    monkeypatch.setattr(
        cli,
        "compute_resolved_params",
        lambda _ri: {
            "transform_mode": "homography",
            "photometric_min_overlap_px": 4321,
            "photometric_full_confidence_px": 54321,
            "enable_grid_nominal_fallback": False,
            "staged_matching_mode": "manual",
            "refs_per_stage": 7,
        },
    )
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)

    argv = ["input_dir", "--stitch-mode", "freeform", "--resolve"]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert _DummyEngine.captured["profile_transform_mode"] == "homography"
    assert _DummyEngine.captured["profile_photometric_min_overlap_px"] == 4321
    assert _DummyEngine.captured["ssm_enable_grid_nominal_fallback"] is False
    assert _DummyEngine.captured["ssm_staged_matching_mode"] == "manual"
    assert _DummyEngine.captured["ssm_refs_per_stage"] == 7


def test_cli_freeform_no_resolve_skips_resolver(monkeypatch):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    resolver_calls = []
    monkeypatch.setattr(cli, "compute_resolved_params", lambda ri: resolver_calls.append(ri) or {})
    monkeypatch.setattr(cli, "FreeMode", _DummyEngine)

    argv = ["input_dir", "--stitch-mode", "freeform"]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert len(resolver_calls) == 0


# ---------------------------------------------------------------------------
# 8. Invalid settings file aborts main()
# ---------------------------------------------------------------------------

def test_cli_invalid_settings_file_aborts(monkeypatch, tmp_path):
    _reset_state(monkeypatch)
    _patch_io(monkeypatch)

    sf = tmp_path / "bad.json"
    sf.write_text("not json", encoding="utf-8")

    engine_created = []
    monkeypatch.setattr(cli, "FreeMode", lambda *a, **kw: engine_created.append(1))

    argv = ["input_dir", "--stitch-mode", "freeform", "--load-settings", str(sf)]
    monkeypatch.setattr(cli.sys, "argv", ["cielostitch-cli", *argv])
    cli.main()

    assert len(engine_created) == 0
