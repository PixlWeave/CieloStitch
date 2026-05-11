# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from typing import Any

import numpy as np

import math

def is_val_equal(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return a == b

def as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return default


def format_dtype(value: Any, fmt: str = "long", unknown: str = "unknown") -> str:
    """Return a normalized dtype label.

    Supported formats:
    - long: float32 / uint16 / int32 / bool
    - short: f32 / u16 / i32 / b1
    - int: 32 / 16 / 8 / 1
    """
    mode = str(fmt or "long").strip().lower()
    if mode not in {"long", "short", "int"}:
        mode = "long"

    def _format(kind: str, bits: int) -> str:
        b = int(bits)
        if mode == "int":
            return str(b)
        if kind == "bool":
            return "bool" if mode == "long" else f"b{b}"
        if mode == "long":
            return f"{kind}{b}"
        prefix = {"float": "f", "uint": "u", "int": "i"}.get(kind, "")
        return f"{prefix}{b}" if prefix else unknown

    if value is None:
        return unknown

    # Normalize from numpy dtype first when possible.
    try:
        dt = np.dtype(value)
    except Exception:
        dt = None

    if dt is not None:
        if dt.kind == "b":
            return _format("bool", 1)
        bits = int(dt.itemsize * 8)
        if dt.kind == "f":
            return _format("float", bits)
        if dt.kind == "u":
            return _format("uint", bits)
        if dt.kind == "i":
            return _format("int", bits)

    raw = str(value).strip().lower()
    if not raw:
        return unknown

    if raw.isdigit():
        return str(int(raw)) if mode == "int" else str(int(raw))

    # Parse strings like "float32", "uint16", "dtype('float32')".
    tokens = ("float", "uint", "int")
    for token in tokens:
        idx = raw.find(token)
        if idx >= 0:
            tail = raw[idx + len(token):]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                return _format(token, int(digits))

    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        return str(int(digits)) if mode == "int" else str(int(digits))

    if "bool" in raw:
        return _format("bool", 1)

    return unknown