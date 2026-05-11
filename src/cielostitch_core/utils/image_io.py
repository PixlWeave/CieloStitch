# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import os
from typing import Iterable, List, Optional
import cv2
import numpy as np
import tifffile
import logging
from ..config.constants import (
    ALPHA_POLICY_SCAN_LIMIT,
    UINT16_TO_UINT8_SCALE,
    UINT8_TO_UINT16_SCALE,
)

try:
    from astropy.io import fits
except Exception:  # pragma: no cover - optional dependency
    fits = None

try:
    from xisf import XISF
except Exception:  # pragma: no cover - optional dependency
    XISF = None

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".fit", ".fits", ".xisf")

# OpenCV TIFF decoder logs a hard error when a decoded strip/tile buffer exceeds 1 GiB.
# In that case we prefer tifffile to avoid noisy stderr while still loading successfully.
TIFF_OPENCV_TILE_LIMIT_BYTES = 1 << 30
TIFF_SAMPLEFORMAT_FLOAT = 3


def _estimate_tiff_payload_bytes(path: str) -> Optional[int]:
    """Best-effort estimate of decoded TIFF payload size for first series."""
    try:
        with tifffile.TiffFile(path) as tf:
            if not tf.series:
                return None
            series = tf.series[0]
            shape = tuple(int(v) for v in series.shape)
            dtype = np.dtype(series.dtype)
            if not shape:
                return None
            return int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    except Exception:
        return None


def _should_prefer_tifffile_for_tiff(path: str) -> bool:
    payload_bytes = _estimate_tiff_payload_bytes(path)
    if payload_bytes is not None and payload_bytes >= TIFF_OPENCV_TILE_LIMIT_BYTES:
        return True
    return _tiff_uses_float_sample_format(path)


def _iter_sample_formats(sample_format) -> Iterable[int]:
    if sample_format is None:
        return ()
    if isinstance(sample_format, (list, tuple, np.ndarray)):
        seq = sample_format
    else:
        seq = (sample_format,)
    out = []
    for item in seq:
        try:
            value = getattr(item, "value", item)
            out.append(int(value))
        except Exception:
            continue
    return tuple(out)


def _tiff_uses_float_sample_format(path: str) -> bool:
    """Return True when TIFF SampleFormat indicates IEEE float payload.

    OpenCV may emit hard errors for these TIFF variants even when tifffile can read them.
    """
    try:
        with tifffile.TiffFile(path) as tf:
            if not tf.pages:
                return False
            page0 = tf.pages[0]
            sample_format = getattr(page0, "sampleformat", None)
            if sample_format is None:
                tag = page0.tags.get("SampleFormat")
                sample_format = getattr(tag, "value", None)
            return any(v == TIFF_SAMPLEFORMAT_FLOAT for v in _iter_sample_formats(sample_format))
    except Exception:
        return False


def _read_tiff_with_tifffile(path: str):
    """Read TIFF via tifffile and normalize byte order for downstream processing."""
    try:
        arr = tifffile.imread(path)
    except Exception:
        return None
    if arr is None:
        return None
    arr = np.asarray(arr)
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    return arr

def ext_to_selected_filter(ext: str) -> str:
    """Map a file extension to the corresponding name filter used in the Save dialog.

    Keeps the filter label strings in sync with IMAGE_FILTER_LST.
    """
    e = (ext or "").lower()
    if e in (".tif", ".tiff"):
        return "TIFF (*.tif *.tiff)"
    if e == ".png":
        return "PNG (*.png)"
    if e in (".jpg", ".jpeg"):
        return "JPEG (*.jpg *.jpeg)"
    if e == ".bmp":
        return "BMP (*.bmp)"
    if e == ".webp":
        return "WebP (*.webp)"
    if e in (".fit", ".fits"):
        return "FITS (*.fit *.fits)"
    if e == ".xisf":
        return "XISF (*.xisf)"
    # Fallback
    return "TIFF (*.tif *.tiff)"

def is_supported_image(path: str) -> bool:
    """Check if path has a supported image extension."""
    if not path or '.' not in path:
        return False
    ext = '.' + path.lower().rsplit('.', 1)[-1]
    return ext in SUPPORTED_EXTS

def extract_valid_image_paths(mime_data) -> list:
    """Extract local file paths from mime data that have supported extensions."""
    if not mime_data or not mime_data.hasUrls():
        return []
    paths = []
    for url in mime_data.urls():
        try:
            if url.isLocalFile():
                path = url.toLocalFile()
                if path and is_supported_image(path):
                    paths.append(path)
        except Exception:
            continue
    return paths


class AlphaPolicyError(RuntimeError):
    """Raised when non-opaque alpha is present and current policy forbids auto-dropping."""
    pass

def to_u8(a: np.ndarray) -> np.ndarray:
    """Convert float/integer arrays to uint8 in a consistent, safe way.

    - Floats are NaN/Inf-safe, clamped to [0,1], then scaled to [0,255] with rounding.
    - uint16 is downscaled with rounding.
    - Other integer types are scaled based on their dtype range.
    - uint8 is returned unchanged.
    """
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        return np.round(a * 255.0).astype(np.uint8)
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        # 16-bit to 8-bit: scale 0-65535 -> 0-255 with proper rounding.
        return np.round(a.astype(np.float32) / UINT16_TO_UINT8_SCALE).astype(np.uint8)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        rng = max(1, info.max - info.min)
        return np.round((a.astype(np.float32) - info.min) * 255.0 / rng).astype(np.uint8)
    # Fallback: treat as float
    a = a.astype(np.float32, copy=False)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    return np.round(a * 255.0).astype(np.uint8)


def to_u16(a: np.ndarray) -> np.ndarray:
    """Convert float/integer arrays to uint16 in a consistent, safe way.

    - Floats are NaN/Inf-safe, clamped to [0,1], then scaled to [0,65535] with rounding.
    - uint8 is upscaled proportionally.
    - Other integer types are scaled based on their dtype range.
    - uint16 is returned unchanged.
    """
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        return np.round(a * 65535.0).astype(np.uint16)
    if a.dtype == np.uint16:
        return a
    if a.dtype == np.uint8:
        return (a.astype(np.uint16) * UINT8_TO_UINT16_SCALE).astype(np.uint16)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        rng = max(1, info.max - info.min)
        return np.round((a.astype(np.float32) - info.min) * 65535.0 / rng).astype(np.uint16)
    # Fallback: treat as float
    a = a.astype(np.float32, copy=False)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    return np.round(a * 65535.0).astype(np.uint16)

def read_image(path: str, alpha_policy: str = "auto"):
    ext = os.path.splitext(path)[1].lower()
    # Policies:
    #  - auto: drop fully-opaque alpha; raise on non-opaque alpha
    #  - drop: always drop alpha
    #  - forbid: error if any alpha channel is present
    #  - keep: always keep alpha (pass through 4-channel arrays)
    if alpha_policy not in {"auto", "drop", "forbid", "keep"}:
        raise ValueError(f"Invalid alpha_policy: {alpha_policy}")
    if ext == ".xisf":
        if XISF is None:
            raise RuntimeError("XISF support requires package 'xisf'.")
        reader = XISF(path)
        data = np.asarray(reader.read_image(0, data_format="channels_last"))
        if data is None:
            return None
        if data.ndim == 3 and data.shape[2] == 1:
            return data[..., 0]
        if data.ndim == 3 and data.shape[2] == 3:
            return data[..., [2, 1, 0]]
        if data.ndim == 3 and data.shape[2] == 4:
            arr = data[..., [2, 1, 0, 3]]
            if alpha_policy == "keep":
                return arr
            alpha = arr[..., 3]
            if alpha_policy == "drop":
                return arr[..., :3]
            if np.issubdtype(alpha.dtype, np.floating):
                fully_opaque = np.all(alpha >= 0.999)
            elif np.issubdtype(alpha.dtype, np.integer):
                fully_opaque = np.all(alpha == np.iinfo(alpha.dtype).max)
            else:
                fully_opaque = False
            if fully_opaque and alpha_policy == "auto":
                return arr[..., :3]
            if fully_opaque:
                raise AlphaPolicyError(
                    f"{os.path.basename(path)} contains an alpha channel (opaque).\nPolicy 'forbid' prohibits alpha."
                )
            if alpha_policy == "auto":
                raise AlphaPolicyError(
                    f"{os.path.basename(path)} has a non-opaque alpha channel.\nChoose 'keep' to preserve, or 'drop' to discard alpha."
                )
            raise AlphaPolicyError(
                f"{os.path.basename(path)} has a non-opaque alpha channel.\nPolicy 'forbid' prohibits alpha."
            )
        return data
    if ext in (".fit", ".fits"):
        if fits is None:
            raise RuntimeError("FITS support requires astropy. Install package 'astropy'.")
        data = fits.getdata(path)
        if data is None:
            return None
        data = np.asarray(data)
        data = np.squeeze(data)
        # Normalize to native byte order (FITS is big-endian >f4; downstream
        # processing expects native-endian dtypes like float32)
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder('='), copy=False)
        # Sanitize NaN/Inf from unsampled border pixels so the preview and
        # stitching pipelines receive clean finite data
        if np.issubdtype(data.dtype, np.floating):
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

        # Mono FITS
        if data.ndim == 2:
            return data

        # Color FITS: accept HxWxC or CxHxW where C in {3,4}
        if data.ndim == 3:
            # h, w, c = None, None, None
            arr = data
            # Channels-first (C,H,W) -> move channels to last
            if arr.shape[0] in (3, 4) and arr.shape[1] > 1 and arr.shape[2] > 1:
                arr = np.moveaxis(arr, 0, -1)
            # Now expect channels-last
            if arr.shape[-1] in (3, 4):
                # Standardize to OpenCV BGR(A) order from common RGB(A) in FITS
                if arr.shape[-1] == 3:
                    arr = arr[..., [2, 1, 0]]  # RGB -> BGR
                else:  # 4 channels
                    arr = arr[..., [2, 1, 0, 3]]  # RGBA -> BGRA
                    # Alpha policy handling
                    if alpha_policy == "keep":
                        return arr
                    alpha = arr[..., 3]
                    if alpha_policy == "drop":
                        arr = arr[..., :3]
                    else:
                        if np.issubdtype(alpha.dtype, np.floating):
                            fully_opaque = np.all(alpha >= 0.999)
                        elif np.issubdtype(alpha.dtype, np.integer):
                            maxv = np.iinfo(alpha.dtype).max
                            fully_opaque = np.all(alpha == maxv)
                        else:
                            fully_opaque = False

                        if fully_opaque:
                            # In auto and forbid modes, treat fully opaque as safe to drop in auto; in forbid, still error
                            if alpha_policy == "auto":
                                arr = arr[..., :3]
                            else:  # forbid
                                raise AlphaPolicyError(
                                    f"{os.path.basename(path)} contains an alpha channel (opaque).\nPolicy 'forbid' prohibits alpha."
                                )
                        else:
                            # Non-opaque alpha
                            if alpha_policy == "auto":
                                raise AlphaPolicyError(
                                    f"{os.path.basename(path)} has a non-opaque alpha channel.\nChoose 'keep' to preserve, or 'drop' to discard alpha."
                                )
                            else:  # forbid
                                raise AlphaPolicyError(
                                    f"{os.path.basename(path)} has a non-opaque alpha channel.\nPolicy 'forbid' prohibits alpha."
                                )
                return arr

        raise ValueError(f"{os.path.basename(path)}: unsupported FITS shape {data.shape}")

    # Non-FITS: read with OpenCV unchanged to preserve bit-depth/channels.
    # For large TIFF payloads, skip OpenCV first to avoid hardcoded 1 GiB tile-buffer error logs.
    if ext in (".tif", ".tiff") and _should_prefer_tifffile_for_tiff(path):
        arr = _read_tiff_with_tifffile(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None and ext in (".tif", ".tiff"):
            # OpenCV cannot decode all TIFF subtypes (e.g. float16 TIFFs written by tifffile).
            arr = _read_tiff_with_tifffile(path)
    if arr is None:
        return None
    # If 4-channel (e.g., BGRA), handle according to policy
    if arr.ndim == 3 and arr.shape[2] == 4:
        if alpha_policy == "keep":
            return arr
        if alpha_policy == "drop":
            return arr[..., :3]
        alpha = arr[..., 3]
        if np.issubdtype(alpha.dtype, np.floating):
            fully_opaque = np.all(alpha >= 0.999)
        elif np.issubdtype(alpha.dtype, np.integer):
            maxv = np.iinfo(alpha.dtype).max
            fully_opaque = np.all(alpha == maxv)
        else:
            fully_opaque = False

        if fully_opaque:
            if alpha_policy == "auto":
                return arr[..., :3]
            # forbid
            raise AlphaPolicyError(
                f"{os.path.basename(path)} contains an alpha channel (opaque).\nPolicy 'forbid' prohibits alpha."
            )
        else:
            if alpha_policy == "auto":
                raise AlphaPolicyError(
                    f"{os.path.basename(path)} has a non-opaque alpha channel.\nChoose 'keep' to preserve, or 'drop' to discard alpha."
                )
            else:  # forbid
                raise AlphaPolicyError(
                    f"{os.path.basename(path)} has a non-opaque alpha channel.\nPolicy 'forbid' prohibits alpha."
                )
    return arr


def save_image(image, output_path: str, jpeg_quality_pct: int=95):
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".xisf":
        if XISF is None:
            raise RuntimeError("XISF support requires package 'xisf'.")
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[..., [2, 1, 0]]
        elif arr.ndim == 3 and arr.shape[2] == 4:
            raise RuntimeError("XISF export does not currently support 4-channel images.")
        XISF.write(output_path, np.ascontiguousarray(arr), creator_app="CieloStitch")
        return

    if ext in (".fit", ".fits"):
        if fits is None:
            raise RuntimeError("FITS support requires astropy. Install package 'astropy'.")
        arr = np.asarray(image)
        # FITS image HDUs do not support float16 payloads; promote to float32.
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        # If saving color (HxWxC in BGR(A) as used by OpenCV), convert to RGB(A)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            if arr.shape[2] == 3:
                rgb = arr[..., [2, 1, 0]]  # BGR -> RGB
            else:
                rgb = arr[..., [2, 1, 0, 3]]  # BGRA -> RGBA
            # Many FITS consumers store color as (C,H,W); keep that convention
            arr = np.moveaxis(rgb, -1, 0)
        hdu = fits.PrimaryHDU(arr)
        try:
            hdu.writeto(output_path, overwrite=True)
        except Exception as e:
            logger.error(f"Failed to write FITS file: {e}")
            raise
        return

    if ext in (".tif", ".tiff"):
        # TIFF: preserve dtype exactly as provided (supports uint8/uint16/float)
        arr = np.asarray(image)
        tifffile.imwrite(output_path, arr, dtype=arr.dtype)
        return

    # For formats handled by OpenCV (PNG/JPEG/BMP), sanitize dtype/range/channels
    arr = np.asarray(image)

    # is_color = arr.ndim == 3 and arr.shape[2] in (3, 4)

    if ext in (".jpg", ".jpeg", ".bmp", ".webp"):
        # These formats should be 8-bit without alpha
        arr8 = to_u8(arr)

        # Drop alpha if present for formats that do not support it reliably.
        if ext in (".jpg", ".jpeg", ".bmp") and arr8.ndim == 3 and arr8.shape[2] == 4:
            arr8 = arr8[..., :3]

        # Ensure C-contiguous memory for OpenCV
        arr8 = np.ascontiguousarray(arr8)
        # Control JPEG quality via global settings if available
        params = None
        if ext in (".jpg", ".jpeg"):
            try:
                q = int(jpeg_quality_pct)
                # Clamp to OpenCV accepted range 0..100, use 95 default
                q = max(0, min(100, q))
                params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            except Exception:
                params = None
        elif ext == ".webp":
            try:
                q = int(jpeg_quality_pct)
                q = max(0, min(100, q))
                params = [int(cv2.IMWRITE_WEBP_QUALITY), q]
            except Exception:
                params = None
        ok = cv2.imwrite(output_path, arr8, params or [])
        if not ok:
            raise RuntimeError(f"Failed to save image: {output_path}")
        return

    if ext == ".png":
        # PNG supports 8-bit and 16-bit and optional alpha; prefer preserving depth
        if np.issubdtype(arr.dtype, np.floating):
            # For float, use 16-bit to preserve detail
            arr16 = to_u16(arr)
            arr_to_write = np.ascontiguousarray(arr16)
        elif arr.dtype == np.uint16 or arr.dtype == np.uint8:
            arr_to_write = np.ascontiguousarray(arr)
        else:
            # Fallback: convert integers to 16-bit proportionally; others to 8-bit
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                scaled = np.round((arr.astype(np.float32) - info.min) * 65535.0 / max(1, info.max - info.min)).astype(np.uint16)
                arr_to_write = np.ascontiguousarray(scaled)
            else:
                arr_to_write = np.ascontiguousarray(to_u8(arr.astype(np.float32)))

        ok = cv2.imwrite(output_path, arr_to_write)
        if not ok:
            raise RuntimeError(f"Failed to save image: {output_path}")
        return

    # Default: attempt OpenCV write with contiguous array
    ok = cv2.imwrite(output_path, np.ascontiguousarray(arr))
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def resolve_alpha_policy(msg_parent, paths: Iterable[str], current: str) -> Optional[str]:
    """When policy is 'auto', scan images and (if needed) prompt the user once.

    Returns one of 'keep'/'drop'/'auto' if a decision is made, or None if the user cancels.
    """
    try:
        if current != "auto":
            return current
        offending_path = None
        for p in list(paths)[:ALPHA_POLICY_SCAN_LIMIT]:  # defensive limit
            try:
                _ = read_image(p, alpha_policy="auto")
            except Exception as _exc:
                if isinstance(_exc, AlphaPolicyError):
                    offending_path = p
                    break
        if offending_path is None:
            return current

        import os
        from PySide6.QtWidgets import QMessageBox

        base = os.path.basename(offending_path)
        msg = QMessageBox(msg_parent)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Alpha Channel Detected")
        msg.setText(
            f"{base} has a non-opaque alpha channel.\n\n"
            "How would you like to handle alpha for this stitch run?"
        )
        keep_btn = msg.addButton("Keep alpha", QMessageBox.AcceptRole)
        drop_btn = msg.addButton("Drop alpha", QMessageBox.DestructiveRole)
        # cancel_btn = msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(keep_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is keep_btn:
            return "keep"
        if clicked is drop_btn:
            return "drop"
        return None
    except Exception:
        # Fail open: let the run proceed with the current policy
        return current


# ---------------------------------------------------------------------------
# Panel image loader (shared between CLI and GUI stitching worker)
# ---------------------------------------------------------------------------
from .image_manipulation import integer_content_bit_depth, storage_bit_depth, to_internal_float32  # noqa: E402


def normalize_stitch_precision_mode(mode: str) -> str:
    """Normalize a stitch-precision-mode string to one of 'auto', 'fp16', 'fp32'."""
    m = str(mode or "auto").strip().lower()
    return m if m in ("auto", "fp16", "fp32") else "auto"


def _resolve_stitch_precision_mode(mode: str, panel_bit_depth: int) -> str:
    m = normalize_stitch_precision_mode(mode)
    if m == "auto":
        return "fp32" if int(panel_bit_depth or 0) >= 16 else "fp16"
    return m


def load_panel_images_from_paths(
    sorted_panel_paths: List[str],
    log_cb=None,
    alpha_policy: str = "auto",
    stitch_precision_mode: str = "auto",
):
    """Load, validate and convert panel images to internal float format.

    Args:
        sorted_panel_paths: Ordered list of image file paths.
        log_cb: Optional callable(message, color) for skip/warning messages.
        alpha_policy: One of 'auto', 'keep', 'drop', 'forbid'.
        stitch_precision_mode: One of 'auto', 'fp16', 'fp32'.

    Returns:
        Tuple of (items, bit_depth, effective_stitch_precision_mode) where
        items is a list of (path, float_array) pairs.

    Raises:
        ValueError: On alpha policy conflicts, mixed bit depths, or unsupported shapes.
    """
    items = []
    bit_depth = None
    effective_stitch_precision_mode = None
    for p in sorted_panel_paths:
        try:
            arr = read_image(p, alpha_policy=alpha_policy)
        except AlphaPolicyError as exc:
            raise ValueError(
                f"{p}: {exc}. To proceed by discarding alpha, run with --alpha-policy drop"
            ) from exc
        except Exception:
            if log_cb:
                log_cb(f"Skip: cannot read {p}", "red")
            continue
        if arr is None:
            if log_cb:
                log_cb(f"Skip: cannot read {p}", "red")
            continue
        if arr.ndim == 4:
            raise ValueError(
                f"{p}: unsupported 4D image shape {arr.shape}. "
                "Only 2D mono or 3D color/RGBA panels are supported."
            )
        if arr.ndim == 3 and arr.shape[2] == 4 and alpha_policy != "keep":
            arr = arr[..., :3]

        if arr.dtype == np.uint8:
            panel_bit_depth = 8
        elif arr.dtype == np.uint16:
            panel_bit_depth = 16
        elif np.issubdtype(arr.dtype, np.floating):
            panel_bit_depth = storage_bit_depth(arr, p)
        else:
            panel_bit_depth = integer_content_bit_depth(arr, p)

        if bit_depth is None:
            bit_depth = panel_bit_depth
        elif bit_depth != panel_bit_depth:
            raise ValueError(
                f"Mixed panel storage bit depths are not supported: "
                f"expected {bit_depth}-bit but {p} is {panel_bit_depth}-bit."
            )

        if effective_stitch_precision_mode is None:
            effective_stitch_precision_mode = _resolve_stitch_precision_mode(
                stitch_precision_mode, panel_bit_depth
            )

        use_fp16 = effective_stitch_precision_mode == "fp16"
        arr_f = to_internal_float32(
            arr,
            panel_bit_depth,
            output_dtype=np.float16 if use_fp16 else np.float32,
        )
        if arr_f.ndim == 4:
            raise ValueError(
                f"{p}: unsupported normalized 4D image shape {arr_f.shape}. "
                "Only 2D mono or 3D color/RGBA panels are supported."
            )
        if arr_f.ndim == 3 and arr_f.shape[2] == 4:
            alpha = arr_f[..., 3:4].astype(arr_f.dtype, copy=False)
            arr_f[..., :3] = arr_f[..., :3] * alpha

        items.append((p, arr_f))

    eff = effective_stitch_precision_mode or normalize_stitch_precision_mode(stitch_precision_mode)
    return items, (bit_depth or 8), eff
