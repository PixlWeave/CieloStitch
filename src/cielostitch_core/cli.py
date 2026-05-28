# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import json
import os
import sys

from cielostitch_core.config.config import cfg
from cielostitch_core.config.constants import (
    BLEND_TYPES,
    ENGINES,
    GAIN_COMPENSATION_OPTIONS,
    NON_GRID_MODES,
    SPEED_PRESETS,
    STITCH_MODES,
    DEFAULT_MESSAGE_COLORS_STR,
)
from cielostitch_core.core.session_resolver import ResolverInputs, compute_resolved_params
from cielostitch_core.stitching.free_mode import FreeMode
from cielostitch_core.stitching.grid_mode import GridMode
from cielostitch_core.state.profiles import (
    GeneralProfile,
    LandscapeProfile,
    LunarProfile,
    MilkyWayProfile,
    NightscapeProfile,
    PanoramaProfile,
    SolarHAlphaProfile,
    SolarProfile,
)
from cielostitch_core.utils.file_systems import resolve_output_path
from cielostitch_core.utils.image_io import load_panel_images_from_paths, SUPPORTED_EXTS, save_image
from cielostitch_core.utils.image_manipulation import from_internal_float32


def _scan_image_paths(image_folder: str) -> list:
    """Return a sorted list of supported image file paths in image_folder."""
    if not os.path.isdir(image_folder):
        raise ValueError(f"Input folder does not exist or is not a directory: {image_folder}")
    try:
        files = sorted(os.listdir(image_folder))
    except OSError as exc:
        raise ValueError(f"Cannot list input folder '{image_folder}': {exc}") from exc
    return [
        os.path.join(image_folder, f)
        for f in files
        if f.lower().endswith(tuple(SUPPORTED_EXTS))
    ]


def _print_usage():
    print("Usage:")
    print(
        "cielostitch-cli <image_folder>\n"
        "[--output|-o <output.(tif|tiff|png|jpg|jpeg|bmp|fit|fits|webp|xisf)>]\n"
        "[--engine|-e auto|simple|cielo] [--profile|-p <name>] [--stitch-mode|-m auto|freeform|grid-guided|fixed-overlap|zero-overlap]\n"
        "[--cols|-c <int>] [--overlap-x-pct|-x <0..100>] [--overlap-y-pct|-y <0..100>]\n"
        "[--blend-type|-b <type>] [--alpha-policy|-a auto|keep|drop|forbid]\n"
        "[--gain-compensation|-g <mode>]\n"
        "[--load-settings|-l <path/to/settings.json>] [--resolve|-r]\n"
    )
    print(
        "Profiles: solar, solar-h-alpha, lunar, landscape, nightscape,\n"
        "milky-way, panorama, general\n"
    )
    print(
        "Only image_folder is positional, all other arguments are optional.\n"
        "If omitted, output is auto-saved to image_folder/output with a generated file name.\n"
        "If not specified, profile='general' and stitch_mode='auto' are the defaults.\n"
        "Engine defaults to preferences when available, otherwise 'cielo'.\n"
        "All other arguments default to preferences.\n"
        "\n"
        "Parameter resolution order (lowest to highest priority):\n"
        "  1. Profile built-in defaults\n"
        "  2. Settings file (-l): session_data -> SSM, profile_data -> profile\n"
        "  3. Explicit CLI overrides (-b, -g)\n"
        "  4. Resolver (-r, or implicitly in auto mode) -- runs last, same as GUI\n"
        "Session-level args (-c, -x, -y, -p) are not touched by the resolver:\n"
        "  settings file fills unspecified slots; explicit CLI flag always wins.\n"
    )


def _parse_cli_args(argv):
    if len(argv) < 1:
        return None

    image_folder = argv[0]
    output_path = None

    profile_arg = None  # None = not explicitly given; resolved in main()
    engine = None
    stitch_mode = "auto"
    cols = None          # None = not explicitly given; resolved in main()
    overlap_x_pct = None # None = not explicitly given; resolved in main()
    overlap_y_pct = None # None = not explicitly given; resolved in main()
    alpha_policy = str(getattr(cfg.prefs, "default_alpha_policy", "auto") or "auto").strip().lower()
    blend_type = None
    gain_compensation = None
    settings_file = None
    resolve = False
    alias_map = {
        "-o": "--output",
        "-p": "--profile",
        "-e": "--engine",
        "-m": "--stitch-mode",
        "-c": "--cols",
        "-x": "--overlap-x-pct",
        "-y": "--overlap-y-pct",
        "-b": "--blend-type",
        "-a": "--alpha-policy",
        "-g": "--gain-compensation",
        "-l": "--load-settings",
        "-r": "--resolve",
    }

    i = 1
    while i < len(argv):
        tok_raw = argv[i]
        tok_lower = tok_raw.lower()
        tok_norm = alias_map.get(tok_lower, tok_lower)
        if not tok_norm.startswith("--"):
            print(f"Unknown positional argument: {argv[i]}")
            return None

        if tok_norm == "--resolve":
            resolve = True
            i += 1
            continue

        i += 1
        if i >= len(argv):
            print(f"{tok_raw} requires a value.")
            return None
        val = argv[i]
        val_lower = val.lower()

        if tok_norm == "--output":
            output_path = val
            i += 1
            continue

        if tok_norm == "--profile":
            profile_arg = val_lower
            i += 1
            continue

        if tok_norm == "--engine":
            if val_lower not in ENGINES:
                print(f"--engine must be one of: {', '.join(ENGINES)}")
                return None
            engine = val_lower
            i += 1
            continue

        if tok_norm == "--stitch-mode":
            if val_lower not in STITCH_MODES:
                print("--stitch-mode must be one of: auto, freeform, grid-guided, fixed-overlap, zero-overlap.")
                return None
            stitch_mode = val_lower
            i += 1
            continue

        if tok_norm == "--cols":
            try:
                cols = int(val)
            except ValueError:
                print("--cols must be an integer.")
                return None
            if cols < 1:
                print("--cols must be >= 1.")
                return None
            i += 1
            continue

        if tok_norm == "--overlap-x-pct":
            try:
                overlap_x_pct = float(val)
            except ValueError:
                print("--overlap-x-pct must be a number.")
                return None
            if not 0.0 <= overlap_x_pct <= 100.0:
                print("--overlap-x-pct must be in [0, 100].")
                return None
            i += 1
            continue

        if tok_norm == "--overlap-y-pct":
            try:
                overlap_y_pct = float(val)
            except ValueError:
                print("--overlap-y-pct must be a number.")
                return None
            if not 0.0 <= overlap_y_pct <= 100.0:
                print("--overlap-y-pct must be in [0, 100].")
                return None
            i += 1
            continue

        if tok_norm == "--blend-type":
            if val_lower not in BLEND_TYPES:
                print(f"--blend-type must be one of: {', '.join(BLEND_TYPES)}")
                return None
            blend_type = val_lower
            i += 1
            continue

        if tok_norm == "--alpha-policy":
            if val_lower not in ("auto", "keep", "drop", "forbid"):
                print("--alpha-policy must be one of: auto, keep, drop, forbid.")
                return None
            alpha_policy = val_lower
            i += 1
            continue

        if tok_norm == "--gain-compensation":
            if val_lower not in GAIN_COMPENSATION_OPTIONS:
                print(f"--gain-compensation must be one of: {', '.join(GAIN_COMPENSATION_OPTIONS)}")
                return None
            gain_compensation = val_lower
            i += 1
            continue

        if tok_norm == "--load-settings":
            settings_file = val
            i += 1
            continue

        print(f"Unknown argument: {tok_lower}")
        return None


    return (
        image_folder,
        output_path,
        engine,
        profile_arg,
        stitch_mode,
        cols,
        overlap_x_pct,
        overlap_y_pct,
        blend_type,
        alpha_policy,
        gain_compensation,
        settings_file,
        resolve,
    )


def _load_and_validate_settings_file(path: str) -> dict:
    """Load and validate an CieloStitch settings JSON file.

    Args:
        path: Path to the settings JSON file.

    Returns:
        Validated settings snapshot dict.

    Raises:
        ValueError: If the file is missing, unreadable, or has an invalid format.
    """
    if not os.path.isfile(path):
        raise ValueError(f"Settings file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in settings file: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read settings file '{path}': {exc}") from exc

    if not isinstance(snap, dict):
        raise ValueError("Settings file must contain a JSON object.")
    if snap.get("kind") != "cielostitch_settings":
        raise ValueError("Not a valid CieloStitch settings file (missing or wrong 'kind').")
    if snap.get("version") != 1:
        raise ValueError(f"Unsupported settings file version: {snap.get('version')}.")
    if "profile" not in snap:
        raise ValueError("Settings file is missing required 'profile' field.")
    return snap


def main():
    parsed = _parse_cli_args(sys.argv[1:])
    if parsed is None:
        _print_usage()
        return
    (
        image_folder,
        output_path_arg,
        engine,
        profile_name,
        stitch_mode,
        cols,
        overlap_x_pct,
        overlap_y_pct,
        blend_type,
        alpha_policy,
        gain_compensation,
        settings_file,
        resolve,
    ) = parsed

    try:
        paths = _scan_image_paths(image_folder)
        items, input_bit_depth, _ = load_panel_images_from_paths(paths, alpha_policy=alpha_policy)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return
    if items:
        print(f"Detected input depth from {os.path.basename(items[0][0])}: {input_bit_depth}-bit")
    print(f"Loaded images: {len(items)}")

    # --- Resolve sentinel CLI args and apply settings file ---
    settings_snap = None
    if settings_file is not None:
        try:
            settings_snap = _load_and_validate_settings_file(settings_file)
            print(f"Loaded settings: {settings_file}")
        except ValueError as exc:
            print(f"Settings file error: {exc}")
            return

    _file_session_data = (settings_snap.get("session_data") or {}) if settings_snap is not None else {}
    _file_profile_data = (settings_snap.get("profile_data") or {}) if settings_snap is not None else {}

    # Apply full session_data to ssm so all fields (mount_precision, scan_order, etc.)
    # are populated before sentinel reads below. CLI-explicit values override afterward.
    if _file_session_data:
        cfg.state.ssm.from_dict(_file_session_data)

    # stitch_mode: CLI explicit >  "auto"
    # profile: CLI explicit > Settings file > "general"
    # ---------------------------------------------------
    if profile_name is None:
        if settings_snap:
            profile_name = str(settings_snap.get("profile", "general")).strip().lower()
        else:
            profile_name = "general"

    if engine is None:
        engine = str(getattr(cfg.prefs, "default_stitching_engine", "cielo") or "cielo").strip().lower() or "cielo"
    if engine not in ENGINES:
        engine = "cielo"

    profile_mapping = {
        "solar": SolarProfile,
        "solar-h-alpha": SolarHAlphaProfile,
        "lunar": LunarProfile,
        "landscape": LandscapeProfile,
        "nightscape": NightscapeProfile,
        "milky-way": MilkyWayProfile,
        "panorama": PanoramaProfile,
        "general": GeneralProfile,
    }
    if profile_name not in profile_mapping:
        print(
            "Profile must be one of: solar, solar-h-alpha, lunar, landscape, nightscape,\n"
            "milky-way, panorama, or general\n"
        )
        return
    profile = profile_mapping[profile_name]()

    if _file_profile_data:
        for _attr, _val in _file_profile_data.items():
            if hasattr(profile, _attr):
                setattr(profile, _attr, _val)
        # Sync profile_data into psm so subsequent psm.set_value() overrides are consistent.
        cfg.state.psm.set_values(_file_profile_data, profile_name)

    # cols, overlaps: CLI explicit > ssm (already has file values if settings loaded, else defaults).
    # Reading from ssm avoids the falsy-zero bug that `val or default` would have for 0.0.
    if cols is None:
        cols = int(cfg.state.ssm.grid_cols)
    if overlap_x_pct is None:
        overlap_x_pct = float(cfg.state.ssm.overlap_x_pct)
    if overlap_y_pct is None:
        overlap_y_pct = float(cfg.state.ssm.overlap_y_pct)

    if len(items) < 1:
        print("No readable images found in the image folder.")
        return
    if input_bit_depth is None:
        print("No readable images found")
        return

    default_ext = str(getattr(cfg.prefs, "default_export_format", ".tif") or ".tif").strip().lower()
    if default_ext not in SUPPORTED_EXTS:
        default_ext = ".tif"
    output_path = resolve_output_path(
        image_folder=image_folder,
        output_path_arg=output_path_arg,
        profile=profile_name,
        stitch_mode=stitch_mode,
        default_ext=default_ext,
        output_subdir="output",
    )
    print(f"Output path: {output_path}")

    cfg.state.cli_mode = True
    cfg.state.engine = engine
    cfg.state.profile = profile_name
    cfg.state.stitch_mode = stitch_mode
    cfg.state.ssm.grid_cols = cols
    cfg.state.ssm.overlap_x_pct = overlap_x_pct
    cfg.state.ssm.overlap_y_pct = overlap_y_pct

    if blend_type is not None:
        profile.blend_type = blend_type
        cfg.state.psm.set_value("blend_type", blend_type, profile_name)
    if gain_compensation is not None:
        profile.gain_compensation = gain_compensation
        cfg.state.psm.set_value("gain_compensation", gain_compensation, profile_name)

    print(f"\nProfile: {profile_name}")
    print(f"Engine: {engine}")
    print(f"Stitch mode: {stitch_mode}")
    print("Resolve parameters: ", "yes" if resolve or stitch_mode == "auto" else "no")

    # Parameters: Resolver > CLI explicit > Settings file
    # ---------------------------------------------------
    if resolve or stitch_mode == "auto":
        _, first_img = items[0]
        panel_h, panel_w = first_img.shape[:2]
        speed_preset = str(getattr(cfg.state, "speed_preset", "balanced") or "balanced").strip().lower()
        if speed_preset not in SPEED_PRESETS:
            speed_preset = "balanced"
        resolver_inputs = ResolverInputs(
            subject=profile_name,
            panel_w=panel_w,
            panel_h=panel_h,
            panel_count=len(items),
            fpx_mode=str(getattr(cfg.state.ssm, "fpx_mode", "") or ""),
            focal_length_mm=float(getattr(cfg.state.ssm, "focal_length_mm", 0.0) or 0.0),
            sensor_width_mm=float(getattr(cfg.state.ssm, "sensor_width_mm", 0.0) or 0.0),
            sensor_height_mm=float(getattr(cfg.state.ssm, "sensor_height_mm", 0.0) or 0.0),
            hfov_deg=float(getattr(cfg.state.ssm, "hfov_deg", 0.0) or 0.0),
            camera_angle_deg=float(getattr(cfg.state.ssm, "camera_angle_deg", 0.0) or 0.0),
            stitch_mode=stitch_mode,
            # projection_mode=None,
            mount_precision=str(getattr(cfg.state.ssm, "mount_precision", "normal") or "normal"),
            bit_depth=input_bit_depth,
            overlap_x_pct=overlap_x_pct,
            overlap_y_pct=overlap_y_pct,
            speed_preset=speed_preset,
        )
        resolved = compute_resolved_params(resolver_inputs)
        print("\nResolved parameters:")
        for k, v in resolved.items():
            print(f"  {k}: {v}")
        for attr, val in resolved.items():
            if hasattr(profile, attr):
                setattr(profile, attr, val)
                cfg.state.psm.set_value(attr, val, profile_name)
            elif hasattr(cfg.state.ssm, attr):
                cfg.state.ssm.set_value(attr, val)

        print("\nSettings parameters:")
        for k in vars(type(profile)):
            if not k.startswith("_") and not callable(getattr(type(profile), k)) and k not in resolved:
                print(f"  {k}: {getattr(profile, k)}")
    else:
        print("\nSettings parameters:")
        for k in vars(type(profile)):
            if not k.startswith("_") and not callable(getattr(type(profile), k)):
                print(f"  {k}: {getattr(profile, k)}")

    if stitch_mode in NON_GRID_MODES:
        engine = FreeMode(profile)
    else:
        engine = GridMode(profile, grid_cols=cols)

    # Enable panel-placement messages for the CLI progress callback.
    cfg.allowed_message_colors.add("panel")

    def _cli_progress_cb(msg: str, color: str = "") -> None:
        c = str(color or "").strip().lower()
        if c not in DEFAULT_MESSAGE_COLORS_STR:
            return
        prefix = {
            "red": "[!] ",
            "yellow": "[~] ",
            "green": "[+] ",
            "panel": "  -> ",
        }.get(c, "")
        print(f"{prefix}{msg}")

    print("Stitching...")
    mosaic = engine.stitch(items, progress_cb=_cli_progress_cb)

    print("Saving...")
    mosaic_out = from_internal_float32(mosaic, input_bit_depth)
    save_image(mosaic_out, output_path, jpeg_quality_pct=95)

    print("Done.")


if __name__ == "__main__":
    main()