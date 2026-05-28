# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import numpy as np
import cv2

class IlluminationNormalizer:

    @staticmethod
    def compute_linear_correction(
            canvas,
            warped,
            overlap_mask,
            gain_clamp=(0.7, 1.3),
            min_overlap_px=2_000,
            full_confidence_px=250_000
    ):
        # Shape check: all inputs must have the same shape in the first two dimensions
        if canvas.shape[:2] != warped.shape[:2] or canvas.shape[:2] != overlap_mask.shape[:2]:
            raise ValueError("canvas, warped, and overlap_mask must have the same spatial shape")

        overlap_mask = overlap_mask.astype(bool)
        count = np.count_nonzero(overlap_mask)
        if count < min_overlap_px:
            return 1.0, 0.0  # not enough overlap

        # Use a single photometric channel for robustness:
        # - Mono: use as-is
        # - RGB: use luminance
        # - RGBA: ignore alpha, use RGB luminance
        def _to_luma(img):
            if img.ndim == 2:
                return img.astype(np.float32, copy=False)
            # Handle 4D by reshaping to 3D - this ensures rgb is always 3D max
            while img.ndim > 3:
                img = img.reshape(img.shape[0], img.shape[1], -1)
            # Drop alpha if present; keep first 3 channels
            rgb = img[..., :min(img.shape[2], 3)].astype(np.float32, copy=False)
            # Broadcast-safe luminance using Rec.601-like weights (works fine for linear-ish stacks)
            # Shape handling: rgb can be (H,W,1) or (H,W,2/3) - should always be 3D now
            if rgb.shape[2] == 1:
                return rgb[..., 0]
            elif rgb.shape[2] == 2:
                # If only 2 channels, average them
                return 0.5 * (rgb[..., 0] + rgb[..., 1])
            else:
                # 3 channels
                r, g, b = rgb[..., 2], rgb[..., 1], rgb[..., 0]
                # Use BGR order if images are OpenCV-loaded (B,G,R)
                # Coeffs for luminance from BGR
                return 0.114 * b + 0.587 * g + 0.299 * r

        canvas_luma = _to_luma(canvas)
        warped_luma = _to_luma(warped)

        canvas_vals = canvas_luma[overlap_mask].astype(np.float32, copy=False)
        warped_vals = warped_luma[overlap_mask].astype(np.float32, copy=False)

        # Reject NaN/Inf samples before robust statistics and regression.
        finite = np.isfinite(canvas_vals) & np.isfinite(warped_vals)
        if np.count_nonzero(finite) < min_overlap_px:
            return 1.0, 0.0
        if not np.all(finite):
            canvas_vals = canvas_vals[finite]
            warped_vals = warped_vals[finite]
            count = int(np.count_nonzero(finite))

        # Large overlaps are expensive and may over-weight one region.
        if count > 2_000_000:
            step = max(1, count // 500_000)
            canvas_vals = canvas_vals[::step]
            warped_vals = warped_vals[::step]

        # Robust outlier rejection in overlap.
        med = np.median(warped_vals)
        mad = np.median(np.abs(warped_vals - med)) + 1e-6
        keep = np.abs(warped_vals - med) < (4.0 * mad)
        if np.count_nonzero(keep) > 200:
            warped_vals = warped_vals[keep]
            canvas_vals = canvas_vals[keep]

        x_mean = warped_vals.mean()
        y_mean = canvas_vals.mean()
        x_var = ((warped_vals - x_mean) ** 2).mean()

        if x_var < 1e-6:
            gain = 1.0
            offset = y_mean - x_mean
        else:
            cov = ((warped_vals - x_mean) * (canvas_vals - y_mean)).mean()
            gain = cov / x_var
            offset = y_mean - gain * x_mean

        gain = float(np.clip(gain, gain_clamp[0], gain_clamp[1]))

        # Keep additive correction bounded to avoid over-correction.
        if np.issubdtype(canvas.dtype, np.integer):
            info = np.iinfo(canvas.dtype)
            offset_clip = 0.1 * float(info.max - info.min)
        else:
            offset_clip = 0.1
        offset = float(np.clip(offset, -offset_clip, offset_clip))

        # Confidence weighting: weak overlaps should not force large correction.
        if full_confidence_px <= min_overlap_px:
            confidence = 1.0
        else:
            confidence = (count - min_overlap_px) / float(full_confidence_px - min_overlap_px)
            confidence = float(np.clip(confidence, 0.0, 1.0))

        gain = 1.0 + confidence * (gain - 1.0)
        offset = confidence * offset
        return gain, offset

    @staticmethod
    def apply_linear_correction(image, gain, offset=0.0):
        """
        Apply scalar gain/offset to photometric channels only.
        - Mono: apply to the single channel
        - RGB: apply to all 3 channels
        - RGBA: apply to RGB only; leave alpha unchanged
        """
        g = float(gain)
        o = float(offset)
        img_f = image.astype(np.float32, copy=False).copy()
        # original_shape = img_f.shape

        if img_f.ndim == 2:
            img_f = img_f * g + o
        elif img_f.ndim == 3:
            # Apply to first 3 channels, preserve any extras (e.g., alpha)
            c_eff = min(img_f.shape[2], 3)
            img_f[..., :c_eff] = img_f[..., :c_eff] * g + o
        else:
            # Unexpected dimensionality; apply globally as a fallback
            img_f = img_f * g + o

        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            img_f = np.clip(img_f, info.min, info.max)
        return img_f.astype(image.dtype, copy=False)

    @staticmethod
    def flatten_low_frequency(image, mask, sigma_px=140, strength=0.6):

        mask_bool = mask.astype(bool)
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return image

        # Process only around the valid warped panel region.
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        roi_img = image[y0:y1, x0:x1].astype(np.float32)
        roi_mask = mask_bool[y0:y1, x0:x1]
        roi_mask_f = roi_mask.astype(np.float32)

        # Remember if input is multichannel for proper expansion later
        is_multichannel = roi_img.ndim > 2
        if is_multichannel:
            photo_channels = min(roi_img.shape[2], 3)
            roi_photo = roi_img[..., :photo_channels]
        else:
            photo_channels = 1
            roi_photo = roi_img

        # Downsample ROI for faster low-frequency estimation on large canvases.
        # Use spatial dimensions only to avoid counting color channels.
        max_dim = max(roi_photo.shape[:2])
        ds = max(1, int(np.ceil(max_dim / 1024)))
        if ds > 1:
            small_size = (max(1, roi_photo.shape[1] // ds), max(1, roi_photo.shape[0] // ds))

            roi_photo_s = cv2.resize(roi_photo, small_size, interpolation=cv2.INTER_AREA)
            roi_mask_s = cv2.resize(roi_mask_f, small_size, interpolation=cv2.INTER_AREA)

            sigma_s = float(max(1.0, sigma_px / ds))

            # Ensure mask can broadcast with image (add channel dimension if needed)
            if roi_photo_s.ndim > 2 and roi_mask_s.ndim == 2:
                roi_mask_s_expanded = roi_mask_s[..., None]
            else:
                roi_mask_s_expanded = roi_mask_s

            local_mean_s = cv2.GaussianBlur(roi_photo_s * roi_mask_s_expanded, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s)
            local_norm_s = cv2.GaussianBlur(roi_mask_s, (0, 0), sigmaX=sigma_s, sigmaY=sigma_s)

            # Ensure denominator broadcasts correctly
            if local_mean_s.ndim > 2 and local_norm_s.ndim == 2:
                local_norm_s_expanded = local_norm_s[..., None]
            else:
                local_norm_s_expanded = local_norm_s

            low_freq_s = local_mean_s / (local_norm_s_expanded + 1e-6)

            # Resize back to original spatial size
            # For multi-channel, cv2.resize expects (H,W,C) and preserves C dimension
            low_freq = cv2.resize(low_freq_s, (roi_photo.shape[1], roi_photo.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            sigma = float(max(1.0, sigma_px))

            # Ensure mask can broadcast with image (add channel dimension if needed)
            if roi_photo.ndim > 2 and roi_mask_f.ndim == 2:
                roi_mask_f_expanded = roi_mask_f[..., None]
            else:
                roi_mask_f_expanded = roi_mask_f

            local_mean = cv2.GaussianBlur(roi_photo * roi_mask_f_expanded, (0, 0), sigmaX=sigma, sigmaY=sigma)
            local_norm = cv2.GaussianBlur(roi_mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)

            # Ensure denominator broadcasts correctly
            if local_mean.ndim > 2 and local_norm.ndim == 2:
                local_norm_expanded = local_norm[..., None]
            else:
                local_norm_expanded = local_norm

            low_freq = local_mean / (local_norm_expanded + 1e-6)

        # Ensure low_freq has proper channel dimension for multi-channel images
        if is_multichannel:
            if low_freq.ndim == 2:
                # Expand from (H,W) to (H,W,C)
                low_freq = np.repeat(low_freq[..., None], photo_channels, axis=2)
            elif low_freq.ndim == 3 and low_freq.shape[2] == 1 and photo_channels > 1:
                # If low_freq has 1 channel, but we need multiple, repeat it
                low_freq = np.repeat(low_freq, photo_channels, axis=2)
            # Compute global_mean per-channel
            global_mean = (roi_photo * roi_mask_f[..., None]).sum(axis=(0, 1)) / (roi_mask_f.sum() + 1e-6)
        else:
            # Grayscale image (2D)
            global_mean = float((roi_photo * roi_mask_f).sum() / (roi_mask_f.sum() + 1e-6))

        corrected_roi = roi_img.copy()

        # Safely handle boolean indexing for both 2D and multichannel images
        if is_multichannel:
            photo_view = corrected_roi[..., :photo_channels]
            photo_view[roi_mask] = (
                roi_photo[roi_mask] - float(strength) * (low_freq[roi_mask] - global_mean)
            )
        else:
            # For 2D, simple element-wise subtraction
            corrected_roi[roi_mask] = (
                roi_photo[roi_mask] - float(strength) * (low_freq[roi_mask] - global_mean)
            )

        corrected = image.astype(np.float32).copy()
        corrected[y0:y1, x0:x1] = corrected_roi

        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            corrected = np.clip(corrected, info.min, info.max)
        return corrected.astype(image.dtype)

    def compute_gain(self, canvas, warped, overlap_mask):
        gain, _ = self.compute_linear_correction(canvas, warped, overlap_mask)
        return gain

    def apply_gain(self, image, gain):
        return self.apply_linear_correction(image, gain, 0.0)
