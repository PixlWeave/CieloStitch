# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import os
import re
from typing import List, Callable


def natural_sort_key(name: str):
    """Return a key for natural sorting (numbers in strings sorted numerically)."""
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            # Prefix token type to keep mixed numeric/text keys comparable.
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def sort_panel_paths(paths: List[str], option: str) -> List[str]:
    """Sort a list of file paths according to the given option.

    Supported options (match existing UI labels):
    - "Name": case-insensitive basename sort
    - "Modified ...": sort by mtime, breaking ties by natural name
    - Anything else: natural sort by basename
    """

    def natural(path: str):
        return natural_sort_key(os.path.basename(path))

    normalized_option = (option or "Natural").strip().lower()

    if normalized_option == "name":
        key: Callable[[str], object] = lambda p: os.path.basename(p).lower()
    elif normalized_option.startswith("modified"):
        def key(p: str):
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = 0.0
            return mtime, natural(p)
    else:
        key = natural

    return sorted(paths, key=key)
