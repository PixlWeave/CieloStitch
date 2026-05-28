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

import math
import logging
from ..utils.image_resampling import is_integer_translation, choose_warp_interpolator
from cielostitch_core.projection.apap_warper import warp_apap

logger = logging.getLogger(__name__)


def normalize_projection_mode(mode) -> str:
    normalized = str(mode or "native").strip().lower()
    if normalized == "none":
        return "native"
    return normalized or "native"


class Warper:
    def project_image(self, image):
        """Project the image using the configured projection mode. Returns the projected image."""
        if not self.is_projection_enabled():
            return image
        if self.projection_mode == "cylindrical":
            intrinsics = self._projection_intrinsics_px(image)
            if intrinsics is None:
                return image
            fx, fy = intrinsics
            h, w = image.shape[:2]
            cx = w / 2.0
            cy = h / 2.0
            # Import here to avoid circular import
            from cielostitch_core.projection.cylindrical import cylindrical_project_image
            return cylindrical_project_image(image, fx, fy, cx=cx, cy=cy)
        # Add more projection modes here as needed
        return image

    def warp_into_roi(self, image, H, roi_bounds):
        """Warp an image into a destination ROI instead of the full canvas.
        
        Args:
            image: Source image to warp
            H: 3x3 homography mapping source → destination canvas coordinates
            roi_bounds: (x0, y0, x1, y1) ROI in destination canvas coordinates
            
        Returns:
            Tuple of (warped_image, warped_mask) cropped to ROI dimensions
        """
        x0, y0, x1, y1 = roi_bounds
        output_shape = (y1 - y0, x1 - x0)
        
        # Compose H with ROI translation to map source → ROI-local coordinates.
        # Must use matrix multiplication for projective transforms (h20≠0 or h21≠0)
        # to correctly handle perspective division: x' = (h·x)/w, where w depends on h20,h21.
        T_roi = np.eye(3, dtype=np.float64)
        T_roi[0, 2] = -x0
        T_roi[1, 2] = -y0
        H_roi = T_roi @ H
        
        return self.warp(image, H_roi, output_shape)

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

    @staticmethod
    def fill_closed_mask_holes(image_f: np.ndarray, valid_mask: np.ndarray, closed_mask: np.ndarray) -> np.ndarray:
        """Fill mask-added pixels from the nearest originally valid sample.

        APAP remaps can contain tiny unsupported holes. Closing the mask helps the
        overlap geometry stay contiguous, but any newly-added mask pixels must also
        receive image content or ghost-guard/seam logic will see false black overlap.
        """
        original_valid = np.asarray(valid_mask, dtype=bool)
        target_valid = np.asarray(closed_mask, dtype=bool)
        added = target_valid & (~original_valid)
        if not np.any(added) or not np.any(original_valid):
            return image_f

        valid_u8 = original_valid.astype(np.uint8, copy=False)
        invalid_u8 = (1 - valid_u8).astype(np.uint8, copy=False)
        _dist, labels = cv2.distanceTransformWithLabels(
            invalid_u8,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )

        valid_coords = np.column_stack(np.nonzero(original_valid))
        if valid_coords.size == 0:
            return image_f

        label_to_coord = np.zeros((valid_coords.shape[0] + 1, 2), dtype=np.int32)
        label_to_coord[1:, :] = valid_coords
        nearest = label_to_coord[np.clip(labels[added], 0, valid_coords.shape[0])]
        if image_f.ndim == 2:
            image_f[added] = image_f[nearest[:, 0], nearest[:, 1]]
        else:
            image_f[added] = image_f[nearest[:, 0], nearest[:, 1], :]
        return image_f

    def __init__(
        self,
        interpolation=cv2.INTER_LINEAR,
        mask_valid_thresh=1e-4,
        mask_close_px=1,
        projection_mode="native",
        focal_length_mm=0.0,
        sensor_width_mm=0.0,
        sensor_height_mm=0.0,
        camera_angle_deg=0.0,
        fpx_mode="factor",
        hfov_deg=0.0,
        fpx_factor=1.0,
    ):
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
        self.projection_mode = normalize_projection_mode(projection_mode)
        self.focal_length_mm = float(focal_length_mm or 0.0)
        self.sensor_width_mm = float(sensor_width_mm or 0.0)
        self.sensor_height_mm = float(sensor_height_mm or 0.0)
        self.camera_angle_deg = float(camera_angle_deg or 0.0)
        self.fpx_mode = str(fpx_mode or "factor").strip().lower()
        self.hfov_deg = float(hfov_deg or 0.0)
        self.fpx_factor = float(fpx_factor if fpx_factor is not None else 1.5)
        self._projection_warning_emitted = False

    def is_projection_enabled(self) -> bool:
        return self.projection_mode != "native"


    def _projection_intrinsics_px(self, image):
        if self.projection_mode != "cylindrical":
            return None

        h, w = image.shape[:2]
        if h <= 1 or w <= 1:
            return None

        # Apply portrait/landscape rotation by swapping sensor axes.
        quarter_turn = int(round(self.camera_angle_deg / 90.0)) % 4

        if self.fpx_mode == "fov":
            # Mode 2: Scene FOV supplied — derive fpx from horizontal angle.
            if self.hfov_deg <= 0.0 or self.hfov_deg >= 180.0:
                return None
            half_angle = math.radians(self.hfov_deg * 0.5)
            fx = (w * 0.5) / math.tan(half_angle)
            fy = fx
            if not np.isfinite(fx) or not np.isfinite(fy) or fx <= 1e-6 or fy <= 1e-6:
                return None
            return float(fx), float(fy)

        if self.fpx_mode == "factor":
            # Mode 3: Unknown — estimate fpx as image_width * factor.
            factor = max(0.1, self.fpx_factor)
            fx = float(w) * factor
            fy = fx
            return float(fx), float(fy)

        # Mode 1 (default): derive fpx from physical camera/lens dimensions.
        if self.focal_length_mm <= 0.0 or self.sensor_width_mm <= 0.0 or self.sensor_height_mm <= 0.0:
            return None

        sensor_x = self.sensor_width_mm
        sensor_y = self.sensor_height_mm
        if quarter_turn % 2 == 1:
            sensor_x, sensor_y = sensor_y, sensor_x

        if sensor_x <= 0.0 or sensor_y <= 0.0:
            return None

        fx = (self.focal_length_mm / sensor_x) * float(w)
        fy = (self.focal_length_mm / sensor_y) * float(h)
        if not np.isfinite(fx) or not np.isfinite(fy) or fx <= 1e-6 or fy <= 1e-6:
            return None
        return float(fx), float(fy)

    def warp_apap(self, image, registration, output_shape, roi_origin=(0, 0)):
        """Warp an image using an APAP registration result (delegated to apap_warper)."""
        return warp_apap(
            image,
            registration,
            output_shape,
            roi_origin=roi_origin,
            interpolation=self.interpolation,
            mask_valid_thresh=self.mask_valid_thresh,
            mask_close_px=self.mask_close_px,
            projection_enabled=self.is_projection_enabled(),
            projection_mode=self.projection_mode,
            projection_intrinsics_fn=(lambda img: self._projection_intrinsics_px(img)),
        )

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
                weight = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
                warped_weight = cv2.warpPerspective(
                    weight,
                    H,
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                valid = warped_weight > self.mask_valid_thresh
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
