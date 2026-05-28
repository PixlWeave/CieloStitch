# Simple engine core algorithm for stitching
# Extracted from app-side stitch_worker.py for reuse in both app and headless process

import cv2
import numpy as np
from typing import List

from cielostitch_core.utils.image_io import (
    read_image,
    to_u8,
)
from cielostitch_core.utils.image_manipulation import integer_content_bit_depth, storage_bit_depth


def _storage_depth_for_array(arr: np.ndarray, path: str) -> int:
    if arr.dtype == np.uint8:
        return 8
    if arr.dtype == np.uint16:
        return 16
    if np.issubdtype(arr.dtype, np.floating):
        return storage_bit_depth(arr, path)
    return integer_content_bit_depth(arr, path)


def _prepare_simple_engine_image(path: str, alpha_policy: str) -> tuple[np.ndarray, int]:
    arr = read_image(path, alpha_policy=alpha_policy)
    if arr is None:
        raise ValueError(f"Cannot read {path}")

    panel_bit_depth = _storage_depth_for_array(arr, path)

    if arr.ndim == 4:
        raise ValueError(f"{path}: unsupported 4D image shape {arr.shape}")

    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[..., 3:4]
        if np.issubdtype(arr.dtype, np.floating):
            rgb = np.nan_to_num(arr[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
            alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
            arr = (rgb * alpha_f).astype(np.float32, copy=False)
        else:
            rgb = arr[..., :3].astype(np.float32, copy=False)
            alpha_scale = alpha.astype(np.float32, copy=False)
            if alpha.dtype == np.uint8:
                alpha_scale /= 255.0
            elif alpha.dtype == np.uint16:
                alpha_scale /= 65535.0
            else:
                max_alpha = float(np.max(alpha_scale)) if alpha_scale.size else 1.0
                alpha_scale /= max(1.0, max_alpha)
            arr = np.round(rgb * alpha_scale).astype(arr.dtype if arr.dtype in (np.uint8, np.uint16) else np.uint8)

    arr_u8 = to_u8(arr)
    if arr_u8.ndim == 2:
        arr_u8 = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2BGR)
    elif arr_u8.ndim == 3 and arr_u8.shape[2] == 1:
        arr_u8 = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2BGR)
    elif arr_u8.ndim == 3 and arr_u8.shape[2] != 3:
        raise ValueError(f"{path}: unsupported channel shape {arr_u8.shape}")

    return np.ascontiguousarray(arr_u8), panel_bit_depth


def run_simple_engine(
    sorted_panel_paths: List[str],
    *,
    alpha_policy: str,
    log_emit,
    cancel_cb=None,
    simple_settings: dict = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    images: list[np.ndarray] = []
    input_bit_depth: int | None = None
    settings = dict(simple_settings or {})
    simple_transform_mode = str(settings.get("simple_transform_mode", "auto") or "auto").strip().lower()
    simple_confidence_threshold = float(settings.get("simple_confidence_threshold", 0.2) or 0.2)
    simple_registration_resol_mp = float(settings.get("simple_registration_resol_mp", 0.6) or 0.6)
    simple_seam_estimation_resol_mp = float(settings.get("simple_seam_estimation_resol_mp", 0.1) or 0.1)
    simple_compositing_resol_mp = float(settings.get("simple_compositing_resol_mp", -1.0) or -1.0)
    simple_wave_correction = bool(settings.get("simple_wave_correction", True))
    simple_scans_retry = bool(settings.get("simple_scans_retry", True))

    for path in sorted_panel_paths:
        if cancel_cb and cancel_cb():
            raise InterruptedError()
        try:
            arr_u8, panel_bit_depth = _prepare_simple_engine_image(path, alpha_policy)
        except Exception as exc:
            log_emit(f"Simple engine skip: {exc}", "red")
            continue
        if input_bit_depth is None:
            input_bit_depth = panel_bit_depth
        images.append(arr_u8)

    if len(images) < 2:
        raise ValueError("Simple engine needs at least two readable images.")

    def _make_stitcher(mode=None):
        try:
            return cv2.Stitcher_create() if mode is None else cv2.Stitcher_create(mode)
        except TypeError:
            return cv2.Stitcher_create()

    def _run_attempt(mode=None):
        stitcher = _make_stitcher(mode)
        try:
            stitcher.setPanoConfidenceThresh(simple_confidence_threshold)
        except Exception:
            pass
        try:
            stitcher.setRegistrationResol(simple_registration_resol_mp)
        except Exception:
            pass
        try:
            stitcher.setSeamEstimationResol(simple_seam_estimation_resol_mp)
        except Exception:
            pass
        try:
            stitcher.setCompositingResol(simple_compositing_resol_mp)
        except Exception:
            pass
        try:
            stitcher.setWaveCorrection(simple_wave_correction)
        except Exception:
            pass
        return stitcher.stitch(images)

    initial_mode = None
    if simple_transform_mode == "scans" and hasattr(cv2, "Stitcher_SCANS"):
        initial_mode = getattr(cv2, "Stitcher_SCANS")

    status, pano = _run_attempt(initial_mode)
    if (
        simple_scans_retry
        and simple_transform_mode != "scans"
        and int(status) in {cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL, cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL}
        and pano is None
        and hasattr(cv2, "Stitcher_SCANS")
    ):
        log_emit(
            "Simple engine: retrying with SCANS mode after initial camera-model failure.",
            "yellow",
        )
        status, pano = _run_attempt(getattr(cv2, "Stitcher_SCANS"))

    if cancel_cb and cancel_cb():
        raise InterruptedError()
    if int(status) != int(cv2.Stitcher_OK) or pano is None:
        reason = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images or more usable overlap for the Simple engine.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed in the Simple engine.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed in the Simple engine.",
        }.get(int(status), f"Simple engine failed with status {status}.")
        raise RuntimeError(reason)

    pano_u8 = np.ascontiguousarray(to_u8(pano))
    pano_float = pano_u8.astype(np.float32) / 255.0
    return pano_u8, pano_float, int(input_bit_depth or 8)
