# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

# SIFT CPU / GPU switch


import cv2
import numpy as np
import logging

from ..utils.image_manipulation import robust_minmax

logger = logging.getLogger(__name__)


class FeatureDetector:

    def __init__(self,
                 downscale_factor: float = 0.5,
                 max_features: int | None = None,
                 feature_sensitivity: float | None = None):
        """
        downscale_factor: scale for feature detection (0.5 recommended)
        max_features: cap on number of strongest keypoints to keep (maps to SIFT nfeatures)
        feature_sensitivity: higher = fewer, stronger points (maps to SIFT contrastThreshold)
        """
        self.downscale_factor = float(downscale_factor)
        self.max_features = None if max_features is None else int(max(0, max_features))
        self.feature_sensitivity = 0.01 if feature_sensitivity is None else float(feature_sensitivity)

        # CPU SIFT
        # Map friendly params to SIFT args. Keep edgeThreshold same as before.
        sift_kwargs = {
            "contrastThreshold": self.feature_sensitivity,
            "edgeThreshold": 10,
        }
        if self.max_features and self.max_features > 0:
            sift_kwargs["nfeatures"] = self.max_features

        self.sift = cv2.SIFT_create(**sift_kwargs)  # noinspection PyUnresolvedReference

    @staticmethod
    def _prepare_image(image):
        # Ensure grayscale for SIFT; accept color inputs by converting to luma
        img = image
        if img.ndim == 3 and img.shape[2] >= 3:
            if img.dtype in (np.uint8, np.uint16):
                # OpenCV expects BGR order
                code = cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
                img = cv2.cvtColor(img, code)
            else:
                img = img.astype(np.float32, copy=False)
                b = img[..., 0]
                g = img[..., 1]
                r = img[..., 2]
                img = 0.114 * b + 0.587 * g + 0.299 * r
        else:
            if img.dtype != np.uint8:
                img = img.astype(np.float32, copy=False)

        # Fast path: when uint8 already spans full dynamic range, avoid robust stretch.
        if img.dtype == np.uint8:
            min_v = int(np.min(img))
            max_v = int(np.max(img))
            if max_v <= min_v:
                return np.zeros_like(img, dtype=np.uint8)
            if min_v == 0 and max_v == 255:
                return img
            scale = 255.0 / float(max_v - min_v)
            return np.clip((img.astype(np.float32) - float(min_v)) * scale, 0.0, 255.0).astype(np.uint8)

        # Robust contrast stretch using the existing fast histogram-based path.
        p1, p99 = robust_minmax(img, 1.0, 99.0, speedy=True)
        img = np.clip(img, p1, p99)

        # Normalize to 0–255 uint8
        img = (img - p1) / (p99 - p1 + 1e-6)
        img = (img * 255.0).astype(np.uint8)
        return img

    def _downscale(self, image):
        if self.downscale_factor == 1.0:
            return image

        return cv2.resize(
            image,
            None,
            fx=self.downscale_factor,
            fy=self.downscale_factor,
            interpolation=cv2.INTER_AREA
        )

    def detect(self, image):
        """
        Returns:
            keypoints
            descriptors
            scale_factor (to rescale homography later)
        """

        # Prepare image
        img_8 = self._prepare_image(image)

        # Downscale for speed
        img_small = self._downscale(img_8)

        try:
            keypoints, descriptors = self.sift.detectAndCompute(
                img_small,
                None
            )
        except Exception as e:
            logger.warning(f"SIFT detectAndCompute failed: {e}. Recreating detector once.")
            sift_kwargs = {
                "contrastThreshold": self.feature_sensitivity,
                "edgeThreshold": 10,
            }
            if self.max_features and self.max_features > 0:
                sift_kwargs["nfeatures"] = self.max_features
            self.sift = cv2.SIFT_create(**sift_kwargs)
            try:
                keypoints, descriptors = self.sift.detectAndCompute(img_small, None)
            except Exception as e2:
                logger.error(f"SIFT retry also failed: {e2}")
                raise

        return keypoints, descriptors, self.downscale_factor