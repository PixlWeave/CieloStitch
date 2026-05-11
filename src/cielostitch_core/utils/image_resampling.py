# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple, Optional
# from cielostitch_core.config.constants import RESAMPLERS

# Public names exposed in preferences/UI
# Note: "nearest" is excluded because INTER_NEAREST/INTER_NEAREST_EXACT are not reliably
# supported in cv2.warpPerspective across OpenCV versions. Users requesting "nearest"
# will automatically fall back to "linear" in choose_warp_interpolator().

# Map name -> OpenCV flag
CV2_FLAGS = {
    "nearest": getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST),
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,  # reliable for cv2.resize only
}


def _svd_scale_from_h(H: np.ndarray) -> Tuple[float, float]:
    """Return (smin, smax) singular values (principal stretches) of 2x2 part of H."""
    if H is None:
        return 1.0, 1.0
    try:
        A = H[:2, :2].astype(np.float64, copy=False)
        s = np.linalg.svd(A, compute_uv=False)
        smin = float(min(s)) if np.isfinite(s).all() else 1.0
        smax = float(max(s)) if np.isfinite(s).all() else 1.0
    except Exception:
        smin = 1.0
        smax = 1.0
    if smin <= 0 or not np.isfinite(smin):
        smin = 1.0
    if smax <= 0 or not np.isfinite(smax):
        smax = 1.0
    return smin, smax


def is_integer_translation(H: np.ndarray, eps: float = 1e-6) -> bool:
    """True if H is identity except integer tx, ty (no rotation/scale/shear)."""
    if H is None or H.shape != (3, 3):
        return False
    a, b, tx = H[0]
    c, d, ty = H[1]
    if abs(a - 1.0) > eps or abs(d - 1.0) > eps or abs(b) > eps or abs(c) > eps:
        return False
    return abs(tx - round(tx)) < eps and abs(ty - round(ty)) < eps

def choose_warp_interpolator(H: np.ndarray, prefer: str = "auto") -> Tuple[int, Optional[float]]:
    """
    Select interpolation for warpPerspective/Affine.

    Policy is seam-oriented for stitching:
    - prefer linear for near-identity and normal mosaic warps,
    - use light preblur for minification,
    - use cubic for moderate upscale,
    - reserve Lanczos for strong upscale only.
    """
    try:
        prefer = (prefer or "auto").strip().lower()
    except Exception:
        prefer = "auto"

    if prefer != "auto":
        if prefer == "area":
            return cv2.INTER_LINEAR, 0.5
        if prefer == "nearest":
            return cv2.INTER_LINEAR, None
        return int(CV2_FLAGS.get(prefer, cv2.INTER_LINEAR)), None

    try:
        if H is not None and is_integer_translation(H):
            return cv2.INTER_LINEAR, None
    except Exception:
        pass

    try:
        smin, smax = _svd_scale_from_h(H)
    except Exception:
        smin, smax = 1.0, 1.0

    if smin < 0.5:
        return cv2.INTER_LINEAR, 1.0
    if smin < 0.8:
        return cv2.INTER_LINEAR, 0.5

    if smax <= 1.15:
        return cv2.INTER_LINEAR, None

    if smax <= 2.0:
        return cv2.INTER_CUBIC, None

    return cv2.INTER_LANCZOS4, 0.3   


def choose_resize_interpolator(scale: float, prefer: str = "auto") -> int:
    """Flag for pure resizing (cv2.resize). scale < 1 → AREA in Auto, else Lanczos."""
    prefer = (prefer or "auto").strip().lower()
    if prefer != "auto":
        return CV2_FLAGS.get(prefer, cv2.INTER_LINEAR)
    return CV2_FLAGS["area"] if scale < 1.0 else CV2_FLAGS["lanczos"]


def resize_for_export(img: np.ndarray, scale: float, mode: str = "auto") -> np.ndarray:
    """Resize helper for export. scale=0.5 halves size, 2.0 doubles. Works for mono/color."""
    if img is None:
        return img
    h, w = img.shape[:2]
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    interp = choose_resize_interpolator(scale, mode)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    src = np.asarray(img)
    original_dtype = src.dtype

    # OpenCV resize does not support float16 input; upcast for resize and restore dtype.
    if original_dtype == np.float16:
        resized = cv2.resize(src.astype(np.float32), (new_w, new_h), interpolation=interp)
        return resized.astype(np.float16)

    # bool is not a valid cv2.resize depth; treat as binary mask and threshold back.
    if original_dtype == np.bool_:
        resized = cv2.resize(src.astype(np.uint8), (new_w, new_h), interpolation=interp)
        return resized > 0

    return cv2.resize(src, (new_w, new_h), interpolation=interp)
