# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import os
from pathlib import Path
from datetime import datetime
from cielostitch_core.config.constants import SUPPORTED_EXTS

def fullpath(path, file)->str:
    return str(os.path.join(path, file))


def add_suffix(path, suffix):
    p = Path(path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def default_output_filename(profile: str, stitch_mode: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{profile}_{stitch_mode}_mosaic_{ts}{ext}"


def normalize_save_path_with_ext(path: str, selected_ext: str) -> str:
    if not path:
        return ""
    ext = os.path.splitext(path)[1].lower()
    selected_ext = str(selected_ext or ".tif").strip().lower()
    if not selected_ext.startswith("."):
        selected_ext = f".{selected_ext}"

    supported = {str(e).lower() for e in SUPPORTED_EXTS}

    if not ext:
        return f"{path}{selected_ext}"
    if ext not in supported:
        return f"{os.path.splitext(path)[0]}{selected_ext}"
    return path


def resolve_output_path(
    image_folder: str,
    output_path_arg: str | None,
    profile: str,
    stitch_mode: str,
    default_ext: str,
    output_subdir: str = "output",
) -> str:
    if output_path_arg:
        return normalize_save_path_with_ext(output_path_arg, default_ext)

    output_dir = os.path.join(image_folder, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    filename = default_output_filename(profile, stitch_mode, default_ext)
    return os.path.join(output_dir, filename)