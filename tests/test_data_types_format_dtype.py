# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.utils.data_types import format_dtype


def test_format_dtype_long_default_from_numpy_dtype():
    assert format_dtype(np.dtype(np.float32)) == "float32"


def test_format_dtype_short_mode():
    assert format_dtype("uint16", fmt="short") == "u16"


def test_format_dtype_int_mode():
    assert format_dtype("float64", fmt="int") == "64"


def test_format_dtype_parses_dtype_wrapper_string():
    assert format_dtype("dtype('float16')", fmt="long") == "float16"


def test_format_dtype_unknown_value_returns_unknown():
    assert format_dtype("not-a-dtype") == "unknown"
