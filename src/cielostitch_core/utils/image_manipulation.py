# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import cv2
import numpy as np

def ensure_mono_2d(img: np.ndarray, filename: str) -> np.ndarray|None:
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] in (3, 4):
        if img.dtype in (np.uint8, np.uint16):
            code = cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(img, code)
        # Keep float paths in float arithmetic to avoid quantizing [0,1] images to 0/1.
        if np.issubdtype(img.dtype, np.floating):
            b = img[..., 0].astype(np.float32, copy=False)
            g = img[..., 1].astype(np.float32, copy=False)
            r = img[..., 2].astype(np.float32, copy=False)
            gray = 0.114 * b + 0.587 * g + 0.299 * r
            return gray.astype(img.dtype, copy=False)

        b = img[..., 0].astype(np.uint64)
        g = img[..., 1].astype(np.uint64)
        r = img[..., 2].astype(np.uint64)
        gray = (114 * b + 587 * g + 299 * r + 500) // 1000
        return gray.astype(img.dtype)
    raise ValueError(f"{filename}: unsupported image shape {img.shape}")

def storage_bit_depth(img: np.ndarray, filename: str) -> int:
    """Return container/storage depth directly from dtype (e.g. float32 -> 32)."""
    if img is None:
        raise ValueError(f"{filename}: image is None")
    dt = np.asarray(img).dtype
    if np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.floating):
        return int(dt.itemsize * 8)
    raise ValueError(f"{filename}: unsupported dtype {dt}")


def integer_content_bit_depth(img: np.ndarray, filename: str) -> int:
    """Infer effective integer content depth for integer normalization/restoration paths."""
    if img is None:
        raise ValueError(f"{filename}: image is None")
    arr = np.asarray(img)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{filename}: integer content depth requires integer dtype, got {arr.dtype}")

    max_val = int(arr.max()) if arr.size else 0
    min_val = int(arr.min()) if arr.size else 0

    if arr.dtype == np.uint8:
        return 8
    if arr.dtype == np.uint16:
        if max_val <= 1023:
            return 10
        if max_val <= 4095:
            return 12
        return 16
    if arr.dtype == np.uint32:
        if max_val <= 16_777_215:
            return 24
        return 32
    if arr.dtype in (np.int16, np.int32, np.int64):
        # Consider both positive and negative ranges for signed integer inputs.
        signed_abs_max = max(abs(min_val), abs(max_val))
        if signed_abs_max <= 255:
            return 8
        if signed_abs_max <= 4095:
            return 12
        if signed_abs_max <= 65535:
            return 16
        if signed_abs_max <= 16_777_215:
            return 24
        raise ValueError(f"{filename}: signed integer abs max {signed_abs_max} exceeds 24-bit mono range")
    raise ValueError(f"{filename}: unsupported integer dtype {arr.dtype}")


def to_internal_float32(
    img: np.ndarray,
    bit_depth: int,
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    # , filename: str
    raw = img
    out_dtype = np.dtype(output_dtype)
    if out_dtype not in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Unsupported output dtype: {out_dtype}")

    if raw.dtype in (np.float32, np.float64):
        # Keep float inputs in their native photometric scale and only clamp.
        # Per-image min/max normalization changes relative panel brightness,
        # which hurts matching/gain estimation and introduces seam artifacts.
        out = np.nan_to_num(raw.astype(out_dtype, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out, 0.0, 1.0).astype(out_dtype, copy=False)

    if np.issubdtype(raw.dtype, np.signedinteger):
        raw = np.clip(raw, 0, None)

    # Always normalize through float32 for integer sources.
    # Casting integer types (e.g. uint16 max=65535) directly to float16
    # (max=65504) produces inf for the top values, corrupting the pipeline.
    # Normalizing to [0,1] via float32 first is safe; the final cast to
    # float16 then works without overflow because values are already in [0,1].
    arr = raw.astype(np.float32, copy=False)
    max_in = float((1 << bit_depth) - 1)
    if max_in <= 0.0:
        return np.zeros_like(arr, dtype=out_dtype)
    return np.clip(arr / max_in, 0.0, 1.0).astype(out_dtype, copy=False)


def from_internal_float32(img: np.ndarray, bit_depth: int) -> np.ndarray:
    max_out = float((1 << bit_depth) - 1)
    clipped = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    if bit_depth in (32, 64):
        # Float-source sessions should stay in float output domain.
        return clipped

    # Upcast to float32 before scaling: float16 * 65535 overflows float16 (max ~65504).
    out = clipped * max_out
    out = np.rint(out)
    if bit_depth == 8:
        return out.astype(np.uint8)
    if bit_depth in (10, 12, 16):
        return out.astype(np.uint16)
    if bit_depth == 24:
        return out.astype(np.uint32)
    raise ValueError(f"Unsupported output bit depth: {bit_depth}")

def fast_percentile(a: np.ndarray, q, *, bins: int = 2048) -> np.ndarray:
    """Approximate percentile(s) using a histogram for speed on large arrays.

    - Works for arbitrary numeric ranges (not only [0,1]).
    - Supports scalar or sequence `q` in [0,100].
    - Ignores NaNs.

    This is typically 5–20x faster than np.percentile on multi-megapixel frames
    while being visually indistinguishable for preview normalization.
    """
    a = np.asarray(a)
    # Flatten and drop NaNs early
    a_flat = a.ravel()
    if a_flat.size == 0:
        # Return NaN for each requested percentile on empty input
        return np.full_like(np.asarray(q, dtype=np.float32), np.nan, dtype=np.float32)
    mask = np.isfinite(a_flat)
    if not np.any(mask):
        # Return NaN for each requested percentile if all values are invalid
        return np.full_like(np.asarray(q, dtype=np.float32), np.nan, dtype=np.float32)
    v = a_flat[mask].astype(np.float32, copy=False)

    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.asarray(q, dtype=np.float32)
    if vmax <= vmin:
        # All values equal
        return np.full_like(np.asarray(q, dtype=np.float32), vmin, dtype=np.float32)

    # Compute histogram and CDF
    bins = int(max(16, bins))
    hist, edges = np.histogram(v, bins=bins, range=(vmin, vmax))
    cdf = np.cumsum(hist).astype(np.float64)
    total = cdf[-1]
    if total <= 0:
        return np.full_like(np.asarray(q, dtype=np.float32), vmin, dtype=np.float32)

    def _interp_percent(pct: float) -> float:
        # Convert 0–100 pct to target rank
        target = (pct / 100.0) * total
        # Find first bin where CDF >= target
        idx = int(np.searchsorted(cdf, target, side="left"))
        if idx <= 0:
            return float(edges[0])
        if idx >= len(cdf):
            return float(edges[-1])
        # Linear interpolation within the bin
        cdf_prev = cdf[idx - 1]
        bin_count = max(cdf[idx] - cdf_prev, 1.0)
        t = (target - cdf_prev) / bin_count
        # Bin span
        lo = edges[idx]
        hi = edges[idx + 1]
        return float(lo + (hi - lo) * t)

    if np.isscalar(q):
        return np.float32(_interp_percent(float(q)))
    q_arr = np.asarray(q, dtype=np.float32)
    out = np.empty_like(q_arr, dtype=np.float32)
    it = np.nditer(q_arr, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        out[it.multi_index] = np.float32(_interp_percent(float(it[0])))
        it.iternext()
    return out

def robust_minmax(a: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0, speedy = False) -> tuple[float, float]:
    """Percentile-based min/max with a small guard to avoid zero division."""
    if speedy:
        lo = float(fast_percentile(a, p_lo))
        hi = float(fast_percentile(a, p_hi))
    else:
        # np.nanpercentile ignores NaN (e.g. from FITS border pixels)
        lo, hi = (float(v) for v in np.nanpercentile(a, (p_lo, p_hi)))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi

def apply_normalization(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0, speedy = False) -> np.ndarray:
    arr = arr.astype(np.float32, copy=True)
    lo, hi = robust_minmax(arr, p_lo, p_hi, speedy)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

def apply_smooth_s_curve(view_arr: np.ndarray, strength: float = 0.25) -> np.ndarray:
    x = np.clip(view_arr, 0.0, 1.0)
    sc = x * x * x * (10.0 + x * (-15.0 + 6.0 * x))
    return np.clip(x * (1 - strength) + sc * strength, 0.0, 1.0)

# Solar images benefit from midtone lift slightly shifted upward,
# because most pixels sit around ~0.6–0.8
# boosts solar surface contrast
# avoids crushing prominences
# better for chromosphere mosaics
def apply_subtle_s_curve(view_arr: np.ndarray, strength: float = 0.15) -> np.ndarray:
    pivot = 0.55
    x = np.clip(view_arr, 0.0, 1.0)
    s = np.float32(strength)
    t = x - pivot
    sc = x + s * (x * (1 - x)) * t * 2.0
    return np.clip(sc, 0.0, 1.0)

# Helper: apply sRGB gamma for display
# This improves mid-tone rendering noticeably
def apply_srgb_gamma(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    out = np.empty_like(v)

    mask = v <= 0.0031308
    out[mask] = 12.92 * v[mask]
    out[~mask] = 1.055 * np.power(v[~mask], 1 / 2.4) - 0.055

    return out

# Add local contrast, very important.
# This gives the “nice pop” look most viewers have
def apply_local_contrast(view_arr: np.ndarray) -> np.ndarray:
    # Convert to uint8 for CLAHE/OpenCV operations
    u8 = (np.clip(view_arr, 0, 1) * 255).astype(np.uint8)

    # Grayscale path
    if u8.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        u8 = clahe.apply(u8)
        return u8.astype(np.float32) / 255.0

    # Color path (handle optional alpha)
    has_alpha = (u8.ndim == 3 and u8.shape[2] == 4)
    if has_alpha:
        bgr = u8[..., :3]
        alpha = u8[..., 3:4]  # keep as uint8 for reattachment
    else:
        bgr = u8
        alpha = None

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, c1, c2 = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, c1, c2])
    out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    out = out_bgr.astype(np.float32) / 255.0

    # Reattach alpha if present, preserving original alpha unchanged
    if has_alpha:
        out = np.concatenate([out, alpha.astype(np.float32) / 255.0], axis=2)

    return out

# Add mild sharpening. Very subtle but makes the image feel crisp
def apply_preview_sharpen(view_arr: np.ndarray) -> np.ndarray:
    import cv2

    blur = cv2.GaussianBlur(view_arr, (0, 0), 0.8)
    return np.clip(view_arr * 1.25 - blur * 0.25, 0, 1)

# Apply vibrance (better than saturation)
# Avoids oversaturating already strong colors
def apply_vibrance(view_arr: np.ndarray, amount=0.08):
    import cv2

    # Ensure input is in [0,1] float32 range for processing
    x = np.clip(view_arr, 0.0, 1.0).astype(np.float32, copy=False)

    # Mono images: vibrance is a no-op (keep single-channel for grayscale pipeline)
    if x.ndim == 2 or (x.ndim == 3 and x.shape[2] == 1):
        return x

    # Handle alpha if present by operating on BGR and preserving A unchanged
    has_alpha = (x.ndim == 3 and x.shape[2] == 4)
    if has_alpha:
        bgr = x[..., :3]
        alpha = x[..., 3:4]
    else:
        bgr = x
        alpha = None

    # Convert to HSV, boost saturation moderately, and convert back
    hsv = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = np.clip(s.astype(np.float32) * (1 + amount), 0, 255).astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    out_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out = out_bgr.astype(np.float32) / 255.0

    if has_alpha:
        # Reattach original alpha channel
        out = np.concatenate([out, alpha], axis=2)

    return out

def apply_brightness(x: np.ndarray, factor: float = 0.9) -> np.ndarray:
    return np.clip(x * factor, 0.0, 1.0)

