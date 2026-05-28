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

class LocalGainNormalizer:
    """
    Computes spatially-varying gain + offset correction via sub-tiling.

    Uses the model: I' = gain(x,y) * I + offset(x,y)

    This handles:
    - Multiplicative differences (exposure, limb darkening) → gain
    - Additive differences (sky glow, scattered light, bias) → offset

    Suitable for solar, lunar, and deep-sky mosaics.
    """

    def __init__(
        self,
        tile_size=128,
        stride=None,
        eps=1e-6,
        lum_only_gain=True,
        min_overlap_pixels=100,
        gain_clamp=(0.7, 1.3),
        offset_clamp=None,
        smooth_sigma=None,
        use_offset=True,
        lowfreq_sigma=None,
    ):
        """
        Args:
            tile_size: Size of sub-tiles for local gain computation
            stride: Step size for tile grid (default: max(tile_size // 2, 12) for dense sampling)
            eps: Small value to avoid division by zero
            lum_only_gain: Compute single scalar gain from luminance (prevents color shifts)
            min_overlap_pixels: Minimum overlap pixels required per tile
            gain_clamp: (min, max) tuple to clamp gain values
            offset_clamp: (min, max) tuple to clamp offset values (None = auto based on dtype)
            smooth_sigma: Gaussian blur sigma for gain/offset map smoothing (default: tile_size/4)
            use_offset: Enable offset correction (critical for solar/MW)
            lowfreq_sigma: Blur sigma for low-frequency extraction before solving
        """
        self.tile_size = tile_size
        self.stride = stride or max(tile_size // 2, 12)
        self.eps = eps
        self.lum_only_gain = lum_only_gain
        self.min_overlap_pixels = min_overlap_pixels
        self.gain_clamp = gain_clamp
        self.offset_clamp = offset_clamp
        # Reduced from /4 to /5 for better lunar edge preservation
        self.smooth_sigma = smooth_sigma or (tile_size / 5.0)
        if self.smooth_sigma < tile_size / 10.0:
            self.smooth_sigma = tile_size / 10.0
        self.use_offset = use_offset
        self.lowfreq_sigma = (lowfreq_sigma or float(np.clip(self.tile_size * 0.8, 8.0, 35.0)))

    # ---------------------------
    # Public API
    # ---------------------------
    @staticmethod
    def _resize_image(image, width, height):
        interpolation = cv2.INTER_AREA if width <= image.shape[1] and height <= image.shape[0] else cv2.INTER_LINEAR
        src = np.asarray(image)
        original_dtype = src.dtype

        # OpenCV resize does not support float16 directly.
        if original_dtype == np.float16:
            src = src.astype(np.float32)

        # bool is not a valid OpenCV depth; convert through uint8 and threshold back.
        if original_dtype == np.bool_:
            src = src.astype(np.uint8)

        # Some OpenCV builds only resize up to 4 channels directly.
        if src.ndim == 3 and src.shape[2] > 4:
            resized_channels = [
                cv2.resize(src[..., c], (width, height), interpolation=interpolation)
                for c in range(src.shape[2])
            ]
            resized = np.stack(resized_channels, axis=2)
        else:
            resized = cv2.resize(src, (width, height), interpolation=interpolation)

        if original_dtype == np.bool_:
            return resized > 0
        if original_dtype == np.float16:
            return resized.astype(np.float16)
        return resized

    @staticmethod
    def _choose_solve_scale(height, width):
        max_dim = max(height, width)
        if max_dim >= 2048:
            return 4
        if max_dim >= 1024:
            return 2
        return 1

    def compute_gain_map(self, base, new, overlap, tile_size=None):
        """
        Compute spatially-varying gain + offset maps from overlap region.

        Args:
            base: Canvas image (reference)
            new: New panel image (to be corrected)
            overlap: Boolean mask of overlap region
            tile_size: Override instance tile_size (optional)

        Returns:
            gain_map: (H, W) or (H, W, C) gain map
            offset_map: (H, W) or (H, W, C) offset map (or None if, use_offset=False)
        """
        if tile_size is not None:
            eff_tile_size = tile_size
            eff_stride = max(tile_size // 2, 12)
        else:
            eff_tile_size = self.tile_size
            eff_stride = self.stride

        # Check if we have any overlap
        if not np.any(overlap):
            gain_map = np.ones_like(new, dtype=np.float32)
            offset_map = np.zeros_like(new, dtype=np.float32) if self.use_offset else None
            return gain_map, offset_map

        # Determine number of channels
        if new.ndim == 2:
            C = 1
        else:
            C = new.shape[2]

        # Work only in overlap region for efficiency
        ys, xs = np.where(overlap)
        y0 = max(0, int(ys.min()))
        y1 = min(overlap.shape[0], int(ys.max()) + 1)
        x0 = max(0, int(xs.min()))
        x1 = min(overlap.shape[1], int(xs.max()) + 1)

        # Extract overlap ROI
        base_roi = base[y0:y1, x0:x1]
        new_roi = new[y0:y1, x0:x1]
        overlap_roi = overlap[y0:y1, x0:x1]

        H, W = overlap_roi.shape[:2]
        overlap_pixels = int(np.count_nonzero(overlap_roi))

        # Skip if ROI is too small
        if H < eff_tile_size or W < eff_tile_size:
            # Fall back to uniform gain
            return self._uniform_gain_fallback(base, new, overlap)

        # Experimental local mode: be conservative when overlap support is weak.
        min_required_overlap = max(self.min_overlap_pixels * 8, eff_tile_size * eff_tile_size // 4)
        if overlap_pixels < min_required_overlap:
            return self._uniform_gain_fallback(base, new, overlap)

        solve_scale = self._choose_solve_scale(H, W)
        if solve_scale > 1:
            solve_h = max(H // solve_scale, eff_tile_size)
            solve_w = max(W // solve_scale, eff_tile_size)
            base_solve = self._resize_image(base_roi, solve_w, solve_h)
            new_solve = self._resize_image(new_roi, solve_w, solve_h)
            overlap_solve = cv2.resize(
                overlap_roi.astype(np.uint8),
                (solve_w, solve_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            solve_tile_size = max(24, int(round(eff_tile_size / solve_scale)))
            solve_stride = max(12, int(round(eff_stride / solve_scale)))
            solve_min_overlap = max(32, int(round(self.min_overlap_pixels / float(solve_scale * solve_scale))))
        else:
            base_solve = base_roi
            new_solve = new_roi
            overlap_solve = overlap_roi
            solve_h = H
            solve_w = W
            solve_tile_size = eff_tile_size
            solve_stride = eff_stride
            solve_min_overlap = self.min_overlap_pixels

        overlap_solve_bool = overlap_solve.astype(bool, copy=False)

        # UNIVERSAL ADAPTIVE: Compute global statistics for adaptive behavior
        # This makes the normalizer work optimally across solar, lunar, and MW
        # Pre-convert once before the tile loop to avoid repeated per-tile allocations
        new_solve_f32 = new_solve.astype(np.float32)
        base_solve_f32 = base_solve.astype(np.float32)
        if new_solve_f32.ndim == 2:
            new_solve_signal = new_solve_f32
        else:
            new_solve_signal = self._rgb_to_luminance(new_solve_f32)
        global_signal = new_solve_signal[overlap_solve_bool]

        global_var = np.var(global_signal) if global_signal.size > 0 else 1.0
        global_mean = np.mean(global_signal) if global_signal.size > 0 else 1.0

        # SMOOTH ADAPTIVE GAIN CLAMP: Continuous transition from solar/lunar to MW
        # Eliminates hard switches and discontinuities between panels
        ratio = global_var / (global_mean**2 + 1e-6)
        # Smooth interpolation factor: 0 (solar/lunar) → 1 (MW)
        t = np.clip((ratio - 0.01) / 0.05, 0.0, 1.0)

        # Interpolate between tight (solar) and wide (MW) clamps
        gain_min = 0.75 * (1 - t) + 0.6 * t
        gain_max = 1.25 * (1 - t) + 1.5 * t
        effective_gain_clamp = (float(gain_min), float(gain_max))

        # SMOOTH ADAPTIVE SMOOTHING: Varies with data characteristics
        # Solar (low variance) → smoother, Lunar (medium) → sharper, MW (high) → balanced
        smooth_factor = 0.15 + 0.2 * np.clip(ratio / 0.05, 0.0, 1.0)
        effective_smooth_sigma = eff_tile_size * smooth_factor

        # CRITICAL FIX: Adaptive lowfreq_sigma tied to TILE SIZE, not ROI size
        # Increased scale and upper limit for MW large-scale gradients
        # Solar (48px) → ~38, Lunar (64px) → ~51, MW (128px) → ~102 (clipped to 35)
        adaptive_lowfreq = np.clip(eff_tile_size * 0.8, 8.0, 35.0)
        # Use configured value if provided, otherwise adaptive
        if self.lowfreq_sigma is None:
            effective_lowfreq = adaptive_lowfreq
        else:
            effective_lowfreq = self.lowfreq_sigma

        # PRODUCTION-GRADE FIX: Accumulate predicted signal, then derive maps
        # This preserves the full linear model and avoids parameter inconsistency
        predicted_roi = np.zeros((solve_h, solve_w), dtype=np.float32)
        weight_map_roi = np.zeros((solve_h, solve_w), dtype=np.float32)

        # Sample tiles densely across overlap
        for y in range(0, solve_h - solve_tile_size + 1, solve_stride):
            for x in range(0, solve_w - solve_tile_size + 1, solve_stride):
                y2 = y + solve_tile_size
                x2 = x + solve_tile_size

                mask_tile = overlap_solve_bool[y:y2, x:x2]
                overlap_count = np.count_nonzero(mask_tile)

                if overlap_count < solve_min_overlap:
                    continue

                base_tile = base_solve_f32[y:y2, x:x2]
                new_tile = new_solve_f32[y:y2, x:x2]

                # Compute gain + offset for this tile
                if self.use_offset:
                    gain, offset = self._compute_tile_correction(
                        base_tile, new_tile, mask_tile,
                        effective_lowfreq, global_var, effective_gain_clamp
                    )
                else:
                    gain = self._compute_tile_gain(
                        base_tile, new_tile, mask_tile,
                        effective_lowfreq, effective_gain_clamp
                    )
                    offset = 0.0

                # Skip bad tiles
                if not np.isfinite(gain) or gain < 0.1 or gain > 10.0:
                    continue
                if self.use_offset and not np.isfinite(offset):
                    continue

                # Weight by variance to reduce noise from flat tiles (softer weighting)
                # new_tile is already float32 (pre-converted above the loop)
                new_tile_signal = new_solve_signal[y:y2, x:x2]
                tile_vals = new_tile_signal[mask_tile]

                tile_var = np.var(tile_vals) if tile_vals.size > 0 else 0.0
                # UNIVERSAL: sqrt-based variance weighting (smoother across all targets)
                # Linear weighting biases against flat tiles too much (hurts solar/lunar)
                # FINAL REFINEMENT: Floor raised 0.5→0.6 for extreme cases (flat solar, noisy MW)
                var_weight = np.sqrt(tile_var / (tile_var + 1e-4))
                var_weight = np.clip(var_weight, 0.6, 1.0)

                weight = mask_tile.astype(np.float32) * var_weight

                # Accumulate PREDICTED signal (preserves linear model consistency)
                predicted_tile = gain * new_tile_signal + offset
                predicted_roi[y:y2, x:x2] += predicted_tile * weight
                weight_map_roi[y:y2, x:x2] += weight

        # Normalize accumulated prediction by weight
        valid_mask = weight_map_roi > 1e-6

        # GUARD: If sparse overlap, fall back to uniform gain
        if np.count_nonzero(valid_mask) < max(50, solve_min_overlap):
            return self._uniform_gain_fallback(base, new, overlap)

        predicted_roi[valid_mask] /= weight_map_roi[valid_mask]

        # Now derive gain/offset maps from the averaged prediction
        # Get luminance of new_roi (reuse already-converted new_solve_f32)
        new_roi_signal = new_solve_signal

        # UNIVERSAL FIX: Use smoothed denominator for stable gain (helps all targets)
        # This avoids sensitivity to noise, star edges, and small variations
        # FINAL REFINEMENT: Adaptive blur sigma (solar→smooth, lunar→sharp, MW→stable)
        blur_sigma = np.clip(eff_tile_size * 0.08, 2.0, 10.0)
        new_roi_smooth = cv2.GaussianBlur(
            new_roi_signal, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma
        )

        # Derive gain map: gain = predicted / smooth_new (stabilized with adaptive clamp)
        gain_map_roi = np.ones((solve_h, solve_w), dtype=np.float32)
        safe_new = new_roi_smooth + self.eps
        gain_map_roi[valid_mask] = np.clip(
            predicted_roi[valid_mask] / safe_new[valid_mask],
            effective_gain_clamp[0], effective_gain_clamp[1]
        )

        # Derive offset map: offset = predicted - gain * new
        # Use original (not smoothed) signal for offset computation
        offset_map_roi = np.zeros((solve_h, solve_w), dtype=np.float32) if self.use_offset else None
        if self.use_offset:
            offset_map_roi[valid_mask] = (
                predicted_roi[valid_mask] - gain_map_roi[valid_mask] * new_roi_signal[valid_mask]
            )

        # Apply adaptive Gaussian smoothing (varies by target type)
        gain_map_roi = cv2.GaussianBlur(
            gain_map_roi, (0, 0),
            sigmaX=effective_smooth_sigma, sigmaY=effective_smooth_sigma,
            borderType=cv2.BORDER_REPLICATE
        )

        if self.use_offset:
            # FINAL REFINEMENT: Smooth offset slightly less than gain
            # Preserves lunar edges and MW contrast better
            offset_sigma = effective_smooth_sigma * 0.8
            offset_map_roi = cv2.GaussianBlur(
                offset_map_roi, (0, 0),
                sigmaX=offset_sigma, sigmaY=offset_sigma,
                borderType=cv2.BORDER_REPLICATE
            )

        # Clamp the smoothed maps (using adaptive clamp)
        gain_map_roi = np.clip(gain_map_roi, effective_gain_clamp[0], effective_gain_clamp[1])

        if self.use_offset:
            if self.offset_clamp is not None:
                offset_map_roi = np.clip(offset_map_roi, self.offset_clamp[0], self.offset_clamp[1])
            else:
                # FINAL REFINEMENT: Signal-based clamp (more physically meaningful)
                # Uses actual signal level instead of dtype range
                signal_scale = np.mean(new_roi_signal[valid_mask]) if np.any(valid_mask) else 1.0
                max_offset = 0.1 * signal_scale
                offset_map_roi = np.clip(offset_map_roi, -max_offset, max_offset)

        if solve_scale > 1:
            gain_map_roi = cv2.resize(gain_map_roi, (W, H), interpolation=cv2.INTER_LINEAR)
            if self.use_offset:
                offset_map_roi = cv2.resize(offset_map_roi, (W, H), interpolation=cv2.INTER_LINEAR)
            new_roi_lum_full = self._rgb_to_luminance(new_roi.astype(np.float32)) if new_roi.ndim != 2 else new_roi.astype(np.float32)
        else:
            new_roi_lum_full = self._rgb_to_luminance(new_roi.astype(np.float32)) if new_roi.ndim != 2 else new_roi.astype(np.float32)

        # DISABLED: Feathering was suppressing correction too much for solar
        # The Gaussian smoothing above is sufficient for smooth transitions

        # Create full-size maps (identity everywhere, correction only in overlap)
        if C == 1:
            gain_map = np.ones((base.shape[0], base.shape[1]), dtype=np.float32)
            gain_map[y0:y1, x0:x1] = gain_map_roi

            if self.use_offset:
                offset_map = np.zeros((base.shape[0], base.shape[1]), dtype=np.float32)
                offset_map[y0:y1, x0:x1] = offset_map_roi
            else:
                offset_map = None
        else:
            # UNIVERSAL FIX: Channel-aware offset scaling (critical for color gradients)
            # Gain applies uniformly, but offset scales per channel to preserve color balance
            gain_map = np.ones((base.shape[0], base.shape[1], C), dtype=np.float32)
            for c in range(C):
                gain_map[y0:y1, x0:x1, c] = gain_map_roi

            if self.use_offset:
                offset_map = np.zeros((base.shape[0], base.shape[1], C), dtype=np.float32)

                # Compute per-channel scaling factors (stabilized and normalized)
                new_roi_lum = self._rgb_to_luminance(new_roi.astype(np.float32))

                # Collect all channel ratios first
                channel_ratios = []
                for c in range(C):
                    # Scale offset proportionally to channel contribution
                    # Use larger eps to prevent explosion in dark regions
                    channel_ratio = new_roi[..., c].astype(np.float32) / (new_roi_lum_full + 1e-3)
                    # CRITICAL: Tighter clip to prevent color artifacts
                    channel_ratio = np.clip(channel_ratio, 0.3, 3.0)
                    channel_ratios.append(channel_ratio)

                # CRITICAL: Normalize ratios so average = 1 (prevents global color drift)
                valid_mask_rgb = (new_roi_lum > 1e-3)
                if np.any(valid_mask_rgb):
                    for c in range(C):
                        mean_ratio = np.mean(channel_ratios[c][valid_mask_rgb])
                        if mean_ratio > 1e-6:
                            channel_ratios[c] /= mean_ratio

                # Apply normalized ratios
                for c in range(C):
                    offset_scaled = offset_map_roi * channel_ratios[c]
                    offset_map[y0:y1, x0:x1, c] = offset_scaled
            else:
                offset_map = None

        return gain_map, offset_map

    def _uniform_gain_fallback(self, base, new, overlap):
        """Fallback to uniform gain+offset when local gain fails."""
        overlap_bool = overlap.astype(bool)

        if base.ndim == 2:
            base_vals = base[overlap_bool].astype(np.float32)
            new_vals = new[overlap_bool].astype(np.float32)
        else:
            # Use luminance
            base_lum = self._rgb_to_luminance(base.astype(np.float32))
            new_lum = self._rgb_to_luminance(new.astype(np.float32))
            base_vals = base_lum[overlap_bool]
            new_vals = new_lum[overlap_bool]

        if base_vals.size > 100 and new_vals.size > 100:
            if self.use_offset:
                gain, offset = self._compute_linear_correction(base_vals, new_vals)
            else:
                gain = self._compute_robust_gain(base_vals, new_vals)
                offset = 0.0
        else:
            gain = 1.0
            offset = 0.0

        gain_map = np.full_like(new, gain, dtype=np.float32)
        offset_map = np.full_like(new, offset, dtype=np.float32) if self.use_offset else None

        return gain_map, offset_map

    @staticmethod
    def apply_gain(image, gain_map, offset_map=None):
        """
        Apply gain + offset maps to image.

        Args:
            image: Input image
            gain_map: Gain map from compute_gain_map
            offset_map: Offset map (optional)

        Returns:
            Corrected image: gain_map * image + offset_map
        """
        # Ensure compatible shapes
        if image.ndim != gain_map.ndim:
            if image.ndim == 3 and gain_map.ndim == 2:
                gain_map = gain_map[..., None]
            elif image.ndim == 2 and gain_map.ndim == 3:
                gain_map = gain_map[..., 0]

        result = image.astype(np.float32) * gain_map

        if offset_map is not None:
            if image.ndim != offset_map.ndim:
                if image.ndim == 3 and offset_map.ndim == 2:
                    offset_map = offset_map[..., None]
                elif image.ndim == 2 and offset_map.ndim == 3:
                    offset_map = offset_map[..., 0]
            result += offset_map

        # Clip to valid range
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            result = np.clip(result, info.min, info.max)

        return result.astype(image.dtype)

    # ---------------------------
    # Core logic
    # ---------------------------
    def _compute_tile_correction(self, base, new, mask, lowfreq_sigma=None, global_var=None, gain_clamp=None):
        """Compute gain + offset for a tile using low-frequency data with adaptive parameters."""
        base = base.astype(np.float32, copy=False)
        new = new.astype(np.float32, copy=False)
        mask_bool = mask.astype(bool)

        # Use provided or default lowfreq_sigma
        sigma = lowfreq_sigma if lowfreq_sigma is not None else self.lowfreq_sigma

        # Extract low-frequency signal (critical for stability)
        base_lf = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
        new_lf = cv2.GaussianBlur(new, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Use luminance for scalar correction
        if base.ndim == 2:
            base_vals = base_lf[mask_bool]
            new_vals = new_lf[mask_bool]
        else:
            base_lum = self._rgb_to_luminance(base_lf)
            new_lum = self._rgb_to_luminance(new_lf)
            base_vals = base_lum[mask_bool]
            new_vals = new_lum[mask_bool]

        return self._compute_linear_correction(base_vals, new_vals, global_var, gain_clamp)

    def _compute_tile_gain(self, base, new, mask, lowfreq_sigma=None, gain_clamp=None):
        """Compute gain only (no offset) for a tile using low-frequency data with adaptive sigma."""
        base = base.astype(np.float32, copy=False)
        new = new.astype(np.float32, copy=False)
        mask_bool = mask.astype(bool)

        # Use provided or default lowfreq_sigma
        sigma = lowfreq_sigma if lowfreq_sigma is not None else self.lowfreq_sigma

        # Extract low-frequency signal
        base_lf = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
        new_lf = cv2.GaussianBlur(new, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Use luminance
        if base.ndim == 2:
            base_vals = base_lf[mask_bool]
            new_vals = new_lf[mask_bool]
        else:
            base_lum = self._rgb_to_luminance(base_lf)
            new_lum = self._rgb_to_luminance(new_lf)
            base_vals = base_lum[mask_bool]
            new_vals = new_lum[mask_bool]

        return self._compute_robust_gain(base_vals, new_vals, gain_clamp)

    def _compute_linear_correction(self, y, x, global_var=None, gain_clamp=None):
        """
        Compute gain + offset using linear regression with adaptive confidence and clamp.

        Uses covariance method (more robust than the least squares for this use case).
        Adaptive offset confidence makes this universal for solar, lunar, and MW.
        """
        # Use provided or default gain clamp
        if gain_clamp is None:
            gain_clamp = self.gain_clamp
        if x.size < 20:
            return 1.0, 0.0

        # Filter valid values
        valid = (x > self.eps) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if x.size < 20:
            return 1.0, 0.0

        # Robust outlier rejection (10% trimmed)
        x_med = np.median(x)
        y_med = np.median(y)
        x_mad = np.median(np.abs(x - x_med))
        y_mad = np.median(np.abs(y - y_med))

        inliers = (np.abs(x - x_med) < 5 * (x_mad + 1e-6)) & (np.abs(y - y_med) < 5 * (y_mad + 1e-6))
        x = x[inliers]
        y = y[inliers]

        if x.size < 10:
            return 1.0, 0.0

        # Compute gain and offset via covariance
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))

        cov = np.mean((x - x_mean) * (y - y_mean))
        var = np.mean((x - x_mean) ** 2)

        if var < 1e-4:
            # UNIVERSAL ADAPTIVE: Low variance tiles with data-driven offset confidence
            gain = 1.0

            # Adaptive confidence based on global variance
            if global_var is not None and global_var > 1e-6:
                # Compare local variance to global variance
                var_ratio = var / global_var
                # Solar (low global_var) → moderate conf, MW (high global_var) → strong conf
                conf = np.clip(var_ratio * 2.0, 0.1, 1.0)
            else:
                # Fallback: fixed confidence with higher floor
                conf = np.clip(var / 5e-5, 0.2, 1.0)

            offset_est = y_mean - x_mean
            offset = conf * offset_est
        else:
            gain = cov / var
            offset = y_mean - gain * x_mean

            # SMOOTH ADAPTIVE OFFSET CLAMP: Continuous scaling from solar to MW
            # Eliminates rigid boundaries
            if global_var is not None and y_mean > 1e-6:
                ratio = global_var / (y_mean**2 + 1e-6)
                # Smooth interpolation: 0 (solar) → 1 (MW)
                t = np.clip(ratio / 0.05, 0.0, 1.0)
                # Solar ±5% → MW ±15%
                offset_scale = 0.05 * (1 - t) + 0.15 * t
            else:
                offset_scale = 0.1  # Default 10%

            # Apply smooth adaptive clamp
            offset = np.clip(offset, -offset_scale * y_mean, offset_scale * y_mean)

        # Clamp (using adaptive clamp)
        gain = float(np.clip(gain, gain_clamp[0], gain_clamp[1]))

        return gain, offset

    def _compute_robust_gain(self, a, b, gain_clamp=None):
        """Compute robust gain from pixel arrays with outlier rejection."""
        # Use provided or default gain clamp
        if gain_clamp is None:
            gain_clamp = self.gain_clamp
        valid = (b > self.eps) & (a > 0) & np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]

        if a.size < 20:
            return 1.0

        # Compute ratio
        ratio = a / (b + self.eps)

        # Trimmed mean (remove 10% outliers from each tail)
        lo = np.percentile(ratio, 10)
        hi = np.percentile(ratio, 90)
        mask = (ratio >= lo) & (ratio <= hi)
        ratio_trimmed = ratio[mask]

        if ratio_trimmed.size < 10:
            gain = float(np.median(ratio))
        else:
            gain = float(np.mean(ratio_trimmed))

        # Clamp gain (using adaptive clamp)
        gain = float(np.clip(gain, gain_clamp[0], gain_clamp[1]))

        return gain

    @staticmethod
    def _rgb_to_luminance(img):
        """Convert RGB to luminance using ITU-R BT.709 coefficients."""
        if img.ndim == 2:
            return img
        # Handle different channel counts
        if img.shape[2] >= 3:
            # The pipeline uses BGR channel order (confirmed by COLOR_BGR2GRAY in detector.py and image_manipulation.py). 
            # Channel 0 is Blue, not Red. 
            # BT.709	R=0.2126	G=0.7152	B=0.0722
            return 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        elif img.shape[2] == 2:
            return 0.5 * (img[..., 0] + img[..., 1])
        else:
            return img[..., 0]
