# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

# Image warping

import numpy as np
import cv2
import logging
from ..utils.image_resampling import choose_warp_interpolator, is_integer_translation

logger = logging.getLogger(__name__)


class Warper:

    def __init__(self, interpolation=cv2.INTER_LINEAR, mask_valid_thresh=1e-4, mask_close_px=1):
        # interpolation may be a cv2 flag or a string mode ("auto", "lanczos", ...)
        # Validate and normalize the interpolation parameter
        if interpolation is None:
            self.interpolation = cv2.INTER_LINEAR
        elif isinstance(interpolation, str):
            # Keep as string, _resolve_interp will convert it
            self.interpolation = interpolation.strip().lower()
        else:
            # Assume it's already an OpenCV flag (int)
            self.interpolation = int(interpolation)
        self.mask_valid_thresh = float(mask_valid_thresh)
        self.mask_close_px = int(mask_close_px)

    def _resolve_interp(self, H):
        """Return (cv2_flag, preblur_sigma_or_None) for this warp based on user setting."""
        try:
            if isinstance(self.interpolation, str):
                flag, pre_sigma = choose_warp_interpolator(H, self.interpolation)
                # Ensure flag is an integer, never a string
                if not isinstance(flag, int):
                    raise ValueError(f"choose_warp_interpolator returned non-int flag: {flag} (type: {type(flag)})")
                return flag, (float(pre_sigma) if pre_sigma else None)
            # If it's already an int (OpenCV flag), use it directly
            interp_int = int(self.interpolation)
            return interp_int, None
        except Exception as e:
            # Fallback to safe default if anything goes wrong, but log the error
            logger.warning(f"Interpolation resolution failed: {e}. Falling back to INTER_LINEAR.")
            return cv2.INTER_LINEAR, None

    def warp_into_roi(self, image, H, roi_bounds):
        """Warp an image into a destination ROI instead of the full canvas."""
        x0, y0, x1, y1 = roi_bounds
        roi_w = int(x1 - x0)
        roi_h = int(y1 - y0)
        if roi_w <= 0 or roi_h <= 0:
            raise ValueError(f"Invalid ROI bounds: {roi_bounds}")

        translate = np.eye(3, dtype=np.float64)
        translate[0, 2] = -float(x0)
        translate[1, 2] = -float(y0)
        return self.warp(image, translate @ H, (roi_h, roi_w))

    def warp(self, image, H, output_shape):
        """
        Warp image into given output shape.

        Parameters:
            image: uint16 2D numpy array
            H: 3x3 homography matrix
            output_shape: (height, width)

        Returns:
            warped_image: uint16 array
            warped_mask: uint8 array
        """
        height, width = output_shape

        # Validate H matrix
        if H is None:
            raise ValueError("Homography matrix H cannot be None")
        if not isinstance(H, np.ndarray):
            H = np.array(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {H.shape}")

        # Check for NaN or Inf in H
        if not np.isfinite(H).all():
            raise ValueError("Homography matrix contains NaN or Inf values")

        # Resolve interpolation and possible prefilter based on transform
        flag, pre_sigma = self._resolve_interp(H)
        exact_integer_translation = is_integer_translation(H)

        # Validate flag is a valid integer, not a string
        if isinstance(flag, str):
            raise TypeError(f"ERROR: Interpolation flag is a string '{flag}', expected integer.")
        flag = int(flag)

        image_f = image.astype(np.float32)
        if image_f.ndim not in (2, 3):
            raise ValueError(
                f"warp expects a 2D grayscale image or 3D color/RGBA image, got shape {image_f.shape}"
            )

        if pre_sigma and pre_sigma > 1e-6:
            # very small anti-aliasing Gaussian; ksize=(0,0) lets OpenCV derive kernel size
            image_f = cv2.GaussianBlur(image_f, ksize=(0, 0), sigmaX=pre_sigma, sigmaY=pre_sigma)
        # If premultiplied RGBA is provided, warp color (premultiplied) and alpha separately
        # Check for 3D with 4 channels (should always be 3D now)
        has_alpha = (image_f.ndim == 3 and image_f.shape[2] == 4)
        if has_alpha:
            # Warp the full 4-channel RGBA array in one pass (RGB + alpha together)
            # avoids a second warpPerspective call and an intermediate np.dstack.
            warped_image_f = cv2.warpPerspective(
                image_f,
                H,
                (width, height),
                flags=flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float32)

            # Validity from warped alpha channel
            valid = warped_image_f[..., 3] > self.mask_valid_thresh
            warped_image_f[~valid] = 0.0

            warped_image = warped_image_f if image.dtype == np.float32 else warped_image_f.astype(image.dtype)

            warped_mask = valid.astype(np.uint8)
            if self.mask_close_px > 0:
                k = 2 * self.mask_close_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)
            return warped_image, warped_mask

        # Non-alpha path (legacy): use weight normalization as before
        if exact_integer_translation:
            warped_signal = cv2.warpPerspective(
                image_f,
                H,
                (width, height),
                flags=flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            if warped_signal.ndim > 2:
                valid = np.any(np.abs(warped_signal) > self.mask_valid_thresh, axis=2)
            else:
                valid = np.abs(warped_signal) > self.mask_valid_thresh

            warped_image_f = warped_signal
            if warped_image_f.ndim > 2:
                warped_image_f[~valid] = 0.0
            else:
                warped_image_f[~valid] = 0.0

            if np.issubdtype(image.dtype, np.integer):
                info = np.iinfo(image.dtype)
                warped_image_f = np.clip(warped_image_f, info.min, info.max)
            warped_image = warped_image_f.astype(image.dtype)

            warped_mask = valid.astype(np.uint8)
            if self.mask_close_px > 0:
                k = 2 * self.mask_close_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)

            return warped_image, warped_mask

        # Use a single-channel weight map so the validity mask is 2-D even for color images.
        # Warp signal and weight with identical interpolation.
        # This compensates edge darkening from border mixing.
        weight = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

        warped_signal = cv2.warpPerspective(
            image_f,
            H,
            (width, height),
            flags=flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_weight = cv2.warpPerspective(
            weight,
            H,
            (width, height),
            flags=flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Normalize by warped_weight
        valid = warped_weight > self.mask_valid_thresh

        # Avoid boolean-indexed division that collapses dimensions
        if warped_signal.ndim > 2:
            # Expand denominator to (H,W,1) so it broadcasts to (H,W,C) for any number of channels
            denom = np.where(valid, warped_weight, 1.0)[..., None]
            warped_image_f = warped_signal / denom
            # Zero out invalid pixels across all channels
            warped_image_f[~valid] = 0.0
        else:
            denom = np.where(valid, warped_weight, 1.0)
            # Expand denominator to broadcast correctly with multichannel signal
            # if warped_signal.ndim > 2:
            #     denom = denom[..., None]
            warped_image_f = warped_signal / denom
            warped_image_f[~valid] = 0.0

        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            warped_image_f = np.clip(warped_image_f, info.min, info.max)
        warped_image = warped_image_f.astype(image.dtype)

        # Conservative valid-region mask from warped support.
        warped_mask = valid.astype(np.uint8)
        if self.mask_close_px > 0:
            k = 2 * self.mask_close_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)

        return warped_image, warped_mask
