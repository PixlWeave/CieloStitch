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
import warnings
import logging
from ..config.constants import MAX_MULTIBAND_LEVELS


logger = logging.getLogger(__name__)

class Blender:
    # Class constants for magic numbers
    EPSILON = np.finfo(np.float32).eps
    NARROW_OVERLAP_THRESHOLD = 80
    NARROW_OVERLAP_FEATHER_RATIO = 0.25
    MIN_FEATHER_PX = 5
    DEFAULT_FEATHER_PX = 40
    SEAMLESS_PAD_PX = 16
    MULTIBAND_PAD_PX = 8
    MAX_MULTIBAND_LEVELS = 7
    WEIGHT_BLUR_SIGMA = 2.0
    # Balanced seamless sigma keeps legacy behavior: sqrt(3^2 + 3^2 + 2^2) ~= 4.69.
    SEAMLESS_BALANCED_SIGMA = (3.0 ** 2 + 3.0 ** 2 + 2.0 ** 2) ** 0.5
    SEAMLESS_QUALITY_PARAMS = {
        "fast": {"multiband_levels": 3, "weight_sigma": 2.4},
        "balanced": {"multiband_levels": 4, "weight_sigma": SEAMLESS_BALANCED_SIGMA},
        "best": {"multiband_levels": 5, "weight_sigma": 6.0},
    }
    PHOTOMETRIC_MIN_OVERLAP_PX = 2_000
    PHOTOMETRIC_FULL_CONFIDENCE_PX = 250_000
    GAIN_SAMPLE_THRESHOLD = 5_000_000
    GAIN_SAMPLE_SIZE = 1_000_000
    DIAGNOSTIC_SAMPLE_THRESHOLD = 1_000_000
    DIAGNOSTIC_SAMPLE_SIZE = 250_000
    TRIMMED_MEAN_PERCENTILE = 10  # Trim 10% from each tail
    # Luminance coefficients (ITU-R BT.601) - for BGR channel order (as used by OpenCV)
    LUM_COEFFS = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    # Adaptive multiband thresholds
    ADAPTIVE_MB_SMALL_OVERLAP = 100   # pixels - use 3 levels
    ADAPTIVE_MB_MEDIUM_OVERLAP = 300  # pixels - use 4 levels, else 5 levels
    ADAPTIVE_MB_MAX_BOOST = 1
    EDGE_AWARE_MIN_PIXELS = 4_096
    EDGE_AWARE_MIN_RATIO = 0.02

    @staticmethod
    def _require_float(value, field_name):
        """Parse a required finite float or raise a descriptive error."""
        try:
            parsed = float(value)
            if not np.isfinite(parsed):
                raise ValueError("non-finite")
            return parsed
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid value for '{field_name}': expected a finite number, got {value!r}."
            ) from exc

    @staticmethod
    def _require_int(value, field_name):
        """Parse a required integer or raise a descriptive error."""
        try:
            # Reject non-integral numeric inputs (e.g. 1.9) under strict policy.
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value) or not float(value).is_integer():
                    raise ValueError("not-integral")
            return int(value)
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = float(value)
                if not np.isfinite(parsed) or not parsed.is_integer():
                    raise ValueError("not-integral")
                return int(parsed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid value for '{field_name}': expected an integer, got {value!r}."
                ) from exc

    def __init__(
            self,
            feather_px, #40
            gain_method, #"median",
            gain_clamp, #(0.85, 1.15),
            blend_type, #"multiband",
            multiband_levels, #=5,
            seamless_quality, #="balanced",
            lum_only_gain, #=False,
            histogram_matching, #=False
            edge_aware_smoothing=True, #=True
            adaptive_mb_risk_boost_threshold=0.45,
            adaptive_mb_low_risk_threshold=0.10,
            adaptive_mb_max_boost=1,
                photometric_min_overlap_px=2_000,
                photometric_full_confidence_px=250_000,
    ):
        self.feather_px = feather_px
        self.gain_method = gain_method
        self.gain_clamp = gain_clamp
        self.blend_type = blend_type
        self.multiband_levels = int(multiband_levels)
        self.seamless_quality = seamless_quality
        self.lum_only_gain = lum_only_gain
        self.histogram_matching = histogram_matching
        self.edge_aware_smoothing = bool(edge_aware_smoothing)
        risk_boost = self._require_float(
            adaptive_mb_risk_boost_threshold,
            "adaptive_mb_risk_boost_threshold",
        )
        low_risk = self._require_float(
            adaptive_mb_low_risk_threshold,
            "adaptive_mb_low_risk_threshold",
        )
        max_boost = self._require_int(adaptive_mb_max_boost, "adaptive_mb_max_boost")
        min_overlap = self._require_int(photometric_min_overlap_px, "photometric_min_overlap_px")
        full_confidence = self._require_int(
            photometric_full_confidence_px,
            "photometric_full_confidence_px",
        )

        self.adaptive_mb_risk_boost_threshold = float(np.clip(risk_boost, 0.0, 1.0))
        self.adaptive_mb_low_risk_threshold = float(np.clip(low_risk, 0.0, 1.0))
        self.adaptive_mb_max_boost = max(0, max_boost)
        self.photometric_min_overlap_px = max(0, min_overlap)
        self.photometric_full_confidence_px = max(
            self.photometric_min_overlap_px,
            full_confidence,
        )

        # Map seamless quality to parameters
        self._setup_seamless_params(seamless_quality)

    # --------------------------------------------------
    # Setup Methods
    # --------------------------------------------------
    @staticmethod
    def _align_array_dims(arr1, arr2):
        """Align two arrays to a shared channel layout for blending math."""
        if arr1.ndim == 2 and arr2.ndim > 2:
            arr1 = np.repeat(arr1[..., None], arr2.shape[2], axis=2)
        elif arr1.ndim > 2 and arr2.ndim == 2:
            arr2 = np.repeat(arr2[..., None], arr1.shape[2], axis=2)
        elif arr1.ndim > 2 and arr2.ndim > 2 and arr1.shape[2] != arr2.shape[2]:
            common_channels = min(arr1.shape[2], arr2.shape[2])
            arr1 = arr1[..., :common_channels]
            arr2 = arr2[..., :common_channels]

        return arr1, arr2

    @staticmethod
    def _align_to_reference(arr, reference):
        """Align an array's channel layout to match a reference array."""
        if reference.ndim > 2 and arr.ndim == 2:
            arr = np.repeat(arr[..., None], reference.shape[2], axis=2)
        elif reference.ndim == 2 and arr.ndim > 2:
            arr = arr[..., 0]
        elif reference.ndim > 2 and arr.ndim > 2 and arr.shape[2] != reference.shape[2]:
            arr = arr[..., :min(arr.shape[2], reference.shape[2])]

        return arr

    def _setup_seamless_params(self, quality):
        """Return seamless blending parameters for the requested quality."""
        quality_key = str(quality or "balanced").strip().lower()
        if quality_key not in self.SEAMLESS_QUALITY_PARAMS:
            quality_key = "balanced"
        params = self.SEAMLESS_QUALITY_PARAMS[quality_key]

        self.seamless_quality = quality_key
        self._seamless_multiband_levels = int(params["multiband_levels"])
        self._seamless_weight_sigma = float(params["weight_sigma"])
        return {
            "multiband_levels": self._seamless_multiband_levels,
            "weight_sigma": self._seamless_weight_sigma,
        }

    def _get_seamless_weight_sigma(self):
        """Return the configured seamless-mode weight smoothing sigma."""
        sigma = getattr(self, "_seamless_weight_sigma", None)
        if sigma is None:
            sigma = float(self._setup_seamless_params(self.seamless_quality)["weight_sigma"])
        return float(sigma)

    def _photometric_overlap_confidence(self, overlap):
        """Map overlap support to [0,1] confidence for photometric corrections."""
        count = int(np.count_nonzero(overlap))
        min_overlap = int(self.photometric_min_overlap_px)
        full_confidence = int(self.photometric_full_confidence_px)

        if count <= min_overlap:
            return 0.0
        if count >= full_confidence:
            return 1.0

        span = float(full_confidence - min_overlap)
        if span <= 0.0:
            return 1.0
        return float(np.clip((count - min_overlap) / span, 0.0, 1.0))

    @staticmethod
    def _attenuate_gain(gain, confidence):
        """Blend gain toward identity using overlap confidence."""
        conf = float(np.clip(confidence, 0.0, 1.0))
        gain_arr = np.asarray(gain, dtype=np.float32)
        adjusted = 1.0 + conf * (gain_arr - 1.0)
        if np.ndim(gain) == 0:
            return float(adjusted)
        return adjusted.astype(np.float32, copy=False)

    @staticmethod
    def _attenuate_offset(offset, confidence):
        """Scale additive offset by overlap confidence."""
        conf = float(np.clip(confidence, 0.0, 1.0))
        offset_arr = np.asarray(offset, dtype=np.float32)
        adjusted = conf * offset_arr
        if np.ndim(offset) == 0:
            return float(adjusted)
        return adjusted.astype(np.float32, copy=False)

    @staticmethod
    def _mask_bbox(mask):
        """Return bounding box of non-zero mask as (y0, y1, x0, x1)."""
        mask_arr = np.asarray(mask) > 0
        if not np.any(mask_arr):
            return None

        rows = np.any(mask_arr, axis=1)
        cols = np.any(mask_arr, axis=0)
        ys = np.flatnonzero(rows)
        xs = np.flatnonzero(cols)
        return (
            int(ys[0]),
            int(ys[-1]) + 1,
            int(xs[0]),
            int(xs[-1]) + 1,
        )

    @staticmethod
    def _union_bbox(bbox_a, bbox_b):
        """Return the union of two bboxes, tolerating missing inputs."""
        if bbox_a is None:
            return bbox_b
        if bbox_b is None:
            return bbox_a

        ay0, ay1, ax0, ax1 = bbox_a
        by0, by1, bx0, bx1 = bbox_b
        return (
            min(ay0, by0),
            max(ay1, by1),
            min(ax0, bx0),
            max(ax1, bx1),
        )

    def _compute_blend_roi_padding(self):
        """Return a conservative context margin for ROI-limited blend operations."""
        pad = 8

        if self.blend_type in ("feather", "adaptive-feather"):
            feather_px = int(self.feather_px or 0)
            pad = max(pad, max(32, feather_px * 2))
        elif self.blend_type in ("multiband", "adaptive-multiband"):
            pad = max(pad, max(96, 1 << max(4, min(self.multiband_levels + 2, MAX_MULTIBAND_LEVELS))))
        elif self.blend_type == "seamless":
            pad = max(pad, max(160, 1 << max(5, min(self.multiband_levels + 2, MAX_MULTIBAND_LEVELS))))

        return int(pad)

    def _compute_shared_blend_bbox(self, canvas_mask, img_mask, min_pad=0):
        """Return a padded ROI covering the new panel and any existing overlap."""
        img_bbox = self._mask_bbox(img_mask)
        if img_bbox is None:
            return None

        overlap_mask = np.asarray(canvas_mask, dtype=bool) & np.asarray(img_mask, dtype=bool)
        overlap_bbox = self._mask_bbox(overlap_mask)
        blend_bbox = self._union_bbox(img_bbox, overlap_bbox)
        pad = max(int(min_pad), int(self._compute_blend_roi_padding()))
        return self._expand_bbox(blend_bbox, img_mask.shape, pad)

    def _compute_blend_regions(self, canvas_mask, img_mask, min_pad=0):
        """Return the image bbox and shared blend ROI for the current input mask."""
        img_bbox = self._mask_bbox(img_mask)
        if img_bbox is None:
            return None, None

        blend_bbox = self._compute_shared_blend_bbox(canvas_mask, img_mask, min_pad=min_pad)
        return img_bbox, blend_bbox

    @staticmethod
    def _expand_bbox(bbox, shape, pad):
        """Expand bbox by pad pixels and clamp it to image bounds."""
        if bbox is None:
            return None

        y0, y1, x0, x1 = bbox
        return (
            max(0, y0 - pad),
            min(shape[0], y1 + pad),
            max(0, x0 - pad),
            min(shape[1], x1 + pad),
        )

    # --------------------------------------------------
    # Helper Methods
    # --------------------------------------------------
    @staticmethod
    def _distance_transform(mask_uint8):
        """Compute distance transform."""
        mask_bin = (np.asarray(mask_uint8) > 0).astype(np.uint8, copy=False)
        return cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)

    @staticmethod
    def _gaussian_blur(img_f32, sigma=1.0):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(img_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def _pyr_down(img):
        """Downsample."""
        return cv2.pyrDown(img)

    @staticmethod
    def _pyr_up(img, dstsize=None):
        """Upsample."""
        return cv2.pyrUp(img, dstsize=dstsize)

    @staticmethod
    def _compute_pyramid_levels(min_side, requested_levels):
        """Return a safe Laplacian pyramid depth for the current ROI."""
        max_levels = int(np.floor(np.log2(max(2, min_side)))) - 1
        return max(0, min(int(requested_levels), max_levels))

    @staticmethod
    def _single_scale_blend(canvas_roi, img_roi, weight):
        """Blend a ROI without pyramid decomposition."""
        if canvas_roi.ndim > 2:
            weight = weight[..., None]
        return canvas_roi * (1.0 - weight) + img_roi * weight

    @staticmethod
    def _finalize_weight_map(weight, only_canvas, only_img, union):
        """Clamp blend weights and restore hard support outside overlap."""
        weight = np.clip(weight, 0.0, 1.0)
        weight[only_canvas] = 0.0
        weight[only_img] = 1.0
        weight[~union] = 0.0
        return weight

    def _smooth_weight_adaptive(self, w, guide_image, gaussian_sigma=1.0, use_edge_aware=True, overlap_mask=None):
        """
        Smooth weight map with optional edge-aware filtering.
        
        Applies bilateral filtering (edge-preserving) before final Gaussian blur.
        This prevents seams from crossing object boundaries while maintaining smoothness.
        Falls back to pure Gaussian if edge-aware step fails.
        
        Args:
            w: Weight map [0, 1] to smooth.
            guide_image: Image to use for edge detection (source image in overlap region).
            gaussian_sigma: Std dev for final Gaussian smoothing.
            use_edge_aware: If False, skip bilateral and use only Gaussian.
        
        Returns:
            Smoothed weight map.
        """
        if not use_edge_aware:
            return self._gaussian_blur(w, sigma=gaussian_sigma)

        if overlap_mask is not None:
            overlap_bool = np.asarray(overlap_mask, dtype=bool)
            overlap_count = int(np.count_nonzero(overlap_bool))
            total = int(overlap_bool.size)
            if total == 0:
                return self._gaussian_blur(w, sigma=gaussian_sigma)
            overlap_ratio = overlap_count / float(total)
            if overlap_count < self.EDGE_AWARE_MIN_PIXELS or overlap_ratio < self.EDGE_AWARE_MIN_RATIO:
                return self._gaussian_blur(w, sigma=gaussian_sigma)
        
        try:
            # Extract grayscale guide for edge detection
            if guide_image.ndim > 2:
                guide_gray = self._rgb_to_luminance(guide_image.astype(np.float32))
            else:
                guide_gray = guide_image.astype(np.float32)
            
            # Normalize guide to [0, 255] for bilateral filter
            g_min, g_max = float(np.min(guide_gray)), float(np.max(guide_gray))
            if g_max <= g_min:
                return self._gaussian_blur(w, sigma=gaussian_sigma)
            guide_norm = np.clip((guide_gray - g_min) / (g_max - g_min) * 255.0, 0, 255).astype(np.uint8)

            bilateral_d, sigma_color, sigma_space = self._compute_edge_aware_params(guide_norm)
            
            # Normalize weights to [0, 255] for bilateral filter
            w_norm = np.clip(w * 255.0, 0, 255).astype(np.uint8)
            
            # Edge-aware smoothing: bilateral filter the weight map guided by image edges.
            # Strategy: filter guide_norm to get an edge-smoothed version, compute the
            # per-pixel "edge strength" difference, and use it to suppress smoothing
            # of weights that cross strong image edges.
            # d=7: neighborhood diameter; sigmaColor/Space control smoothness and edge sensitivity.
            # We filter guide_norm to identify edge regions, then filter w_norm constrained
            # by those edges so blend weights do not bleed across object boundaries.
            # Compute edge strength via Sobel gradient magnitude on the guide image.
            # Sobel is the correct way to detect step edges (object boundaries):
            # high gradient → strong edge → preserve weight boundary there.
            # (Bilateral-vs-original diff detects fine texture, not step edges — wrong approach.)
            guide_f32 = guide_norm.astype(np.float32)
            sobel_x = cv2.Sobel(guide_f32, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(guide_f32, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            # Normalize gradient to [0, 1]; max theoretical gradient for uint8 is ~360
            grad_max = float(np.max(grad_mag))
            if grad_max > 0:
                edge_strength = np.clip(grad_mag / grad_max, 0.0, 1.0).astype(np.float32)
            else:
                edge_strength = np.zeros_like(grad_mag, dtype=np.float32)

            w_filtered = cv2.bilateralFilter(
                w_norm,
                d=bilateral_d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space,
            )
            # Blend: where edge_strength is high (real object boundary), preserve original w_norm;
            # elsewhere (flat regions) use the smoothed weight map.
            w_blended = (w_filtered.astype(np.float32) * (1.0 - edge_strength) +
                         w_norm.astype(np.float32) * edge_strength)

            # Convert back to [0, 1]
            w_smooth_float = np.clip(w_blended / 255.0, 0.0, 1.0).astype(np.float32)
            
            # Apply final Gaussian for consistent smoothness
            w_smooth = self._gaussian_blur(w_smooth_float, sigma=gaussian_sigma)
            
            return w_smooth
        except Exception as e:
            # Graceful fallback to Gaussian if edge-aware step fails
            logger.warning(
                f"Edge-aware weight smoothing failed ({e.__class__.__name__}: {e}); "
                f"falling back to Gaussian smoothing"
            )
            return self._gaussian_blur(w, sigma=gaussian_sigma)

    @staticmethod
    def _estimate_noise_level(gray_u8):
        """Estimate noise/texture level from normalized grayscale guide in [0,255]."""
        gray_f = np.asarray(gray_u8, dtype=np.float32)
        if gray_f.size == 0:
            return 0.0
        lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
        median = float(np.median(lap))
        mad = float(np.median(np.abs(lap - median)))
        sigma = 1.4826 * mad
        return float(np.clip(sigma / 255.0, 0.0, 1.0))

    @classmethod
    def _compute_edge_aware_params(cls, guide_u8):
        """Compute bilateral parameters adaptively from guide scale and noise."""
        guide = np.asarray(guide_u8)
        h, w = guide.shape[:2]
        min_side = max(1, min(h, w))
        noise = cls._estimate_noise_level(guide)

        # Larger ROIs need wider spatial support; noisy regions need more color smoothing.
        d = int(np.clip(round(min_side / 128.0), 5, 15))
        if d % 2 == 0:
            d += 1
        sigma_space = float(np.clip(12.0 + min_side / 32.0, 12.0, 36.0))
        sigma_color = float(np.clip(12.0 + (noise * 80.0), 12.0, 64.0))
        return d, sigma_color, sigma_space

    @classmethod
    def _sample_1d_values(cls, values):
        """Deterministically downsample large 1D vectors for robust statistics."""
        arr = np.asarray(values)
        n = int(arr.size)
        if n == 0 or n <= cls.DIAGNOSTIC_SAMPLE_THRESHOLD:
            return arr
        step = max(1, n // cls.DIAGNOSTIC_SAMPLE_SIZE)
        return arr[::step]

    def _compute_overlap_seam_risk(self, canvas, img, overlap):
        """Estimate seam risk in [0,1] from luminance and gradient mismatch over overlap."""
        overlap_bool = np.asarray(overlap, dtype=bool)
        overlap_bbox = self._mask_bbox(overlap_bool)
        if overlap_bbox is None:
            return 0.0

        y0, y1, x0, x1 = overlap_bbox
        overlap_local = overlap_bool[y0:y1, x0:x1]

        canvas_f = np.asarray(canvas, dtype=np.float32)
        img_f = np.asarray(img, dtype=np.float32)

        if canvas_f.ndim > 2:
            canvas_lum = self._rgb_to_luminance(canvas_f)
        else:
            canvas_lum = canvas_f

        if img_f.ndim > 2:
            img_lum = self._rgb_to_luminance(img_f)
        else:
            img_lum = img_f

        canvas_lum = canvas_lum[y0:y1, x0:x1]
        img_lum = img_lum[y0:y1, x0:x1]

        c_vals = canvas_lum[overlap_local]
        i_vals = img_lum[overlap_local]
        if c_vals.size == 0 or i_vals.size == 0:
            return 0.0

        c_vals = self._sample_1d_values(c_vals)
        i_vals = self._sample_1d_values(i_vals)

        # Robust contrast-normalized luminance mismatch.
        mismatch = np.abs(c_vals - i_vals)
        diff_p90 = float(np.percentile(mismatch, 90))
        dyn = float(max(1.0, np.percentile(c_vals, 95) - np.percentile(c_vals, 5)))
        lum_risk = float(np.clip(diff_p90 / dyn, 0.0, 1.0))

        # Gradient mismatch catches local structural inconsistency near seams.
        gx_c = cv2.Sobel(canvas_lum, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(canvas_lum, cv2.CV_32F, 0, 1, ksize=3)
        gx_i = cv2.Sobel(img_lum, cv2.CV_32F, 1, 0, ksize=3)
        gy_i = cv2.Sobel(img_lum, cv2.CV_32F, 0, 1, ksize=3)
        grad_c = np.sqrt(gx_c ** 2 + gy_c ** 2)
        grad_i = np.sqrt(gx_i ** 2 + gy_i ** 2)
        grad_diff = np.abs(grad_c - grad_i)[overlap_local]
        grad_base = (grad_c + grad_i)[overlap_local] * 0.5
        grad_norm = grad_diff / (grad_base + 1.0)
        grad_norm = self._sample_1d_values(grad_norm)
        grad_risk = float(np.clip(np.percentile(grad_norm, 90), 0.0, 1.0))

        risk = (0.65 * lum_risk) + (0.35 * grad_risk)
        return float(np.clip(risk, 0.0, 1.0))

    @staticmethod
    def _write_blend_result(roi, out, union, c_alpha=None, i_alpha=None):
        """Write a blended ROI result back into the destination ROI."""
        
        # shape[0] -> height
        # shape[1] -> width
        # shape[2] -> channels

        if roi.ndim == 2 and out.ndim > 2:
            if out.shape[2] == 0:
                raise ValueError("out has no channels")
            elif out.shape[2] == 1:
                out2 = out[..., 0]
            else:
                coeffs = np.array([0.114, 0.587, 0.299], dtype=np.float32) # BGR weights (OpenCV style)
                n_color = min(out.shape[2], 3)  # ignore alpha channel if present
                out2 = np.tensordot(out[..., :n_color], coeffs[:n_color], axes=([2], [0]))
            roi[union] = out2[union]
        elif roi.ndim > 2 and out.ndim == 2:
            # out2 = np.repeat(out[..., None], roi.shape[2], axis=2) # memory heavy
            # roi[union] = out2[union]
            roi[union] = out[union][..., None] # faster, low memory, cleaner
        elif roi.ndim > 2 and out.ndim > 2 and roi.shape[2] != out.shape[2]:
            common_channels = min(roi.shape[2], out.shape[2])
            roi_part = roi[..., :common_channels]
            out_part = out[..., :common_channels]
            roi_part[union] = out_part[union]
        else:
            roi[union] = out[union]

        if (
            roi.ndim == 3 and roi.shape[2] == 4 and
            c_alpha is not None and i_alpha is not None
        ):
            out_a = np.maximum(c_alpha, i_alpha)
            roi_alpha = roi[..., 3]
            roi_alpha[union] = out_a[union]

    # --------------------------------------------------
    # Histogram Matching
    # --------------------------------------------------
    @staticmethod
    def _match_histogram_channel(source, reference, mask=None):
        """Match histogram of source to reference for a single channel.

        Uses cumulative distribution function (CDF) matching.
        """
        if mask is not None:
            src_vals = source[mask]
            ref_vals = reference[mask]
        else:
            src_vals = source.ravel()
            ref_vals = reference.ravel()

        if src_vals.size == 0 or ref_vals.size == 0:
            return source

        # Build the LUT in a shared 8-bit intensity domain.
        if source.dtype == np.uint8 and reference.dtype == np.uint8:
            src_domain = source
            ref_domain = reference
            src_hist, _ = np.histogram(src_vals, bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(ref_vals, bins=256, range=(0, 256))
            domain_min = 0.0
            domain_max = 255.0
        else:
            src_vals_f32 = src_vals.astype(np.float32, copy=False)
            ref_vals_f32 = ref_vals.astype(np.float32, copy=False)

            domain_min = float(min(np.min(src_vals_f32), np.min(ref_vals_f32)))
            domain_max = float(max(np.max(src_vals_f32), np.max(ref_vals_f32)))
            if not np.isfinite(domain_min) or not np.isfinite(domain_max) or domain_max <= domain_min:
                return source

            scale = 255.0 / (domain_max - domain_min)
            src_domain = np.clip((source.astype(np.float32) - domain_min) * scale, 0.0, 255.0).astype(np.uint8)
            ref_domain = np.clip((reference.astype(np.float32) - domain_min) * scale, 0.0, 255.0).astype(np.uint8)

            src_hist, _ = np.histogram(
                np.clip((src_vals_f32 - domain_min) * scale, 0.0, 255.0),
                bins=256,
                range=(0, 256),
            )
            ref_hist, _ = np.histogram(
                np.clip((ref_vals_f32 - domain_min) * scale, 0.0, 255.0),
                bins=256,
                range=(0, 256),
            )

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1] + 1e-10
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1] + 1e-10

        # Build lookup table: for each source intensity, find matching reference intensity
        lut = np.searchsorted(ref_cdf, src_cdf, side='left').clip(0, 255).astype(np.uint8)

        # Apply lookup table
        matched_domain = lut[src_domain]
        if source.dtype == np.uint8 and reference.dtype == np.uint8:
            return matched_domain

        scale_back = (domain_max - domain_min) / 255.0
        result = matched_domain.astype(np.float32) * scale_back + domain_min
        return result.astype(source.dtype, copy=False)

    def _apply_histogram_matching(self, canvas, img, overlap):
        """Apply histogram matching to img to match canvas in overlap region."""
        if not np.any(overlap):
            return img

        img_matched = img.copy()

        if img.ndim == 2:
            # Grayscale
            img_matched = self._match_histogram_channel(img, canvas, overlap)
        else:
            # Per-channel matching
            for c in range(min(img.shape[2], canvas.shape[2], 3)):
                img_matched[..., c] = self._match_histogram_channel(
                    img[..., c], canvas[..., c], overlap
                )

        return img_matched

    # --------------------------------------------------
    # Adaptive Multiband: Auto pyramid levels based on overlap size
    # --------------------------------------------------
    def _compute_adaptive_levels(self, overlap, canvas=None, img=None):
        """Compute optimal pyramid levels from overlap geometry and seam-risk cues."""
        effective_overlap_thickness = self.estimate_overlap_thickness(overlap)
        if effective_overlap_thickness <= 0:
            return self.multiband_levels

        # Adaptive levels based on overlap size
        if effective_overlap_thickness < self.ADAPTIVE_MB_SMALL_OVERLAP:
            levels = 3  # Small overlap: fewer levels for sharper blending
        elif effective_overlap_thickness < self.ADAPTIVE_MB_MEDIUM_OVERLAP:
            levels = 4  # Medium overlap: balanced
        else:
            levels = 5  # Large overlap: more levels for smoother blending

        if canvas is None or img is None:
            return int(levels)

        risk = self._compute_overlap_seam_risk(canvas, img, overlap)
        if risk >= self.adaptive_mb_risk_boost_threshold:
            levels += self.adaptive_mb_max_boost
        elif risk <= self.adaptive_mb_low_risk_threshold and levels > 3:
            levels -= 1

        return int(np.clip(levels, 3, min(self.MAX_MULTIBAND_LEVELS, self.multiband_levels + self.adaptive_mb_max_boost)))

    def _get_seamless_multiband_levels(self):
        """Return the multiband depth configured for the seamless mode."""
        levels = getattr(self, "_seamless_multiband_levels", None)
        if levels is None:
            levels = int(self._setup_seamless_params(self.seamless_quality)["multiband_levels"])
        return int(levels)

    def estimate_overlap_thickness(self, overlap):
        """Estimate effective overlap thickness from the overlap mask geometry."""
        overlap_u8 = (np.asarray(overlap) > 0).astype(np.uint8, copy=False)
        if not np.any(overlap_u8):
            return 0

        dist = self._distance_transform(overlap_u8).astype(np.float32, copy=False)
        inside = dist[overlap_u8 > 0]
        if inside.size == 0:
            return 0

        inside = inside[np.isfinite(inside)]
        if inside.size == 0:
            return 0

        inside = self._sample_1d_values(inside)

        effective_radius = float(np.percentile(inside, 75))
        if not np.isfinite(effective_radius) or effective_radius <= 0.0:
            return 0

        max_thickness = int(min(overlap_u8.shape[:2]))
        return max(1, min(max_thickness, int(round(2.0 * effective_radius))))

    def get_effective_feather_radius(self, overlap=None):
        """Return the feather radius that feather-based blending would use."""
        blend_type = str(self.blend_type or "").strip().lower()
        if blend_type not in ("feather", "adaptive-feather"):
            return None

        base_feather = int(self.feather_px or self.DEFAULT_FEATHER_PX)
        if blend_type != "adaptive-feather" or overlap is None or not np.any(overlap):
            return max(self.MIN_FEATHER_PX, base_feather)

        effective_overlap_thickness = self.estimate_overlap_thickness(overlap)
        if effective_overlap_thickness <= 0:
            return max(self.MIN_FEATHER_PX, base_feather)

        if effective_overlap_thickness < self.NARROW_OVERLAP_THRESHOLD:
            return max(
                self.MIN_FEATHER_PX,
                int(effective_overlap_thickness * self.NARROW_OVERLAP_FEATHER_RATIO),
            )

        return max(self.MIN_FEATHER_PX, base_feather)

    def _adaptive_multiband_blend(self, canvas, img, canvas_mask, img_mask, seam_heatmap=None):
        """Multiband blending with auto-computed pyramid levels based on overlap size."""
        overlap = canvas_mask.astype(bool) & img_mask.astype(bool)
        if not np.any(overlap):
            return canvas

        levels = self._compute_adaptive_levels(overlap, canvas=canvas, img=img)
        return self._multiband_blend_roi(
            canvas,
            img,
            canvas_mask,
            img_mask,
            seam_heatmap=seam_heatmap,
            multiband_levels=levels,
        )

    # --------------------------------------------------
    # Adaptive Feathering: Intelligent feather width based on overlap size
    # --------------------------------------------------
    def _adaptive_feather_blend(self, canvas, img, canvas_mask, img_mask):
        """
        Adaptive feathering: uses intelligent feather width based on overlap thickness.
        - Narrow overlaps (<80px): feather = 25% of overlap thickness (min 5px) → sharper seams
        - Wide overlaps (≥80px): feather = standard feather_px → soft seams
        """
        overlap = canvas_mask.astype(bool) & img_mask.astype(bool)
        if not np.any(overlap):
            return canvas

        # Ensure compatible dimensions
        canvas, img = self._align_array_dims(canvas, img)

        effective_overlap_thickness = self.estimate_overlap_thickness(overlap)
        if effective_overlap_thickness <= 0:
            return canvas

        roi_bbox = self._compute_shared_blend_bbox(canvas_mask, img_mask)
        if roi_bbox is None:
            return canvas

        y0, y1, x0, x1 = roi_bbox

        feather_px = self.get_effective_feather_radius(overlap)

        # Compute distance-weighted blend weights
        canvas_u8 = canvas_mask[y0:y1, x0:x1].astype(np.uint8)
        img_u8 = img_mask[y0:y1, x0:x1].astype(np.uint8)
        d_canvas = self._distance_transform(canvas_u8).astype(np.float32)
        d_img = self._distance_transform(img_u8).astype(np.float32)

        # Cap distances at feather width
        d_canvas = np.minimum(d_canvas, float(feather_px))
        d_img = np.minimum(d_img, float(feather_px))

        # Compute blend weights
        w = d_img / (d_canvas + d_img + self.EPSILON)
        w = np.clip(w, 0.0, 1.0)

        # Apply Gaussian blur for smooth transitions
        w = self._gaussian_blur(w, sigma=self.WEIGHT_BLUR_SIGMA)

        # Blend
        canvas_f = canvas.astype(np.float32, copy=False)
        img_f = img.astype(np.float32, copy=False)
        overlap_roi = overlap[y0:y1, x0:x1]
        canvas_roi = canvas_f[y0:y1, x0:x1]
        img_roi = img_f[y0:y1, x0:x1]

        if canvas_roi.ndim > 2 and img_roi.ndim > 2:
            w_exp = w[..., None]
            blended = canvas_roi * (1.0 - w_exp) + img_roi * w_exp
            canvas_roi[overlap_roi] = blended[overlap_roi]
        else:
            blended = canvas_roi * (1.0 - w) + img_roi * w
            canvas_roi[overlap_roi] = blended[overlap_roi]

        return canvas_f.astype(canvas.dtype, copy=False)

    # --------------------------------------------------
    # Seamless Blending: Enhanced Multiband with Extra Smoothing
    # --------------------------------------------------
    def _seamless_blend(self, canvas, img, canvas_mask, img_mask, seam_heatmap=None):
        """
        Seamless blending using enhanced multiband approach with extra smoothing.

        This produces better results than Poisson blending for mosaics by:
        - Using FEWER pyramid levels (3-5 vs 5-7) for smoother local blending
        - Applying TRIPLE Gaussian smoothing (sigma=3,3,2) to blend weights
        - Using wider padding (16px vs 8px) for better boundary handling
        - Always applying gain correction for color matching

        Counter-intuitive finding: Fewer pyramid levels produce LESS visible seams
        because blending happens at finer scales, creating smoother local transitions.

        Much faster and better quality than iterative Poisson solvers.
        """
        overlap = canvas_mask.astype(bool) & img_mask.astype(bool)
        if not np.any(overlap):
            return canvas

        # Ensure compatible dimensions
        canvas, img = self._align_array_dims(canvas, img)

        canvas_f = canvas.astype(np.float32, copy=False)
        img_f = img.astype(np.float32, copy=False)

        # Use multiband blending with FEWER levels for ultra-smooth results.
        # Counter-intuitive but proven by user data: fewer levels = smoother seams.
        levels = self._get_seamless_multiband_levels()

        canvas_f = self._multiband_blend_roi_seamless(
            canvas_f,
            img_f,
            canvas_mask,
            img_mask,
            seam_heatmap=seam_heatmap,
            multiband_levels=levels,
        )

        return canvas_f.astype(canvas.dtype, copy=False)

    def _multiband_blend_roi_seamless(
        self,
        canvas,
        img,
        canvas_mask,
        img_mask,
        seam_heatmap=None,
        multiband_levels=None,
    ):
        """Enhanced multiband blending with extra smoothing for seamless results."""
        return self._multiband_blend_roi(
            canvas,
            img,
            canvas_mask,
            img_mask,
            seam_heatmap=seam_heatmap,
            multiband_levels=multiband_levels,
            roi_min_pad_px=self.SEAMLESS_PAD_PX,
            roi_pad_level_cap=7,
            weight_sigma=self._get_seamless_weight_sigma(),
        )

    # --------------------------------------------------
    # Gain Matching
    # --------------------------------------------------
    @staticmethod
    def _trimmed_mean(values, percentile=10):
        """Compute trimmed mean, excluding percentile% from each tail."""
        if values.size == 0:
            return 0.0
        lo = np.percentile(values, percentile)
        hi = np.percentile(values, 100 - percentile)
        mask = (values >= lo) & (values <= hi)
        trimmed = values[mask]
        return trimmed.mean() if trimmed.size > 0 else values.mean()

    def _compute_gain_for_channel(self, base_values, new_values):
        """Compute gain factor for a single channel."""
        if self.gain_method == "mean":
            g = base_values.mean() / (new_values.mean() + self.EPSILON)
        elif self.gain_method == "trimmed":
            base_tm = self._trimmed_mean(base_values, self.TRIMMED_MEAN_PERCENTILE)
            new_tm = self._trimmed_mean(new_values, self.TRIMMED_MEAN_PERCENTILE)
            g = base_tm / (new_tm + self.EPSILON)
        else:  # median (default)
            g = np.median(base_values) / (np.median(new_values) + self.EPSILON)
        return float(np.clip(g, self.gain_clamp[0], self.gain_clamp[1]))

    def _rgb_to_luminance(self, img):
        """Convert RGB image to luminance using ITU-R BT.601 coefficients."""
        if img.ndim == 2:
            return img
        channels = min(img.shape[2], 3)
        coeffs = self.LUM_COEFFS[:channels]
        # Normalize coefficients if fewer than 3 channels
        coeffs = coeffs / coeffs.sum()
        return np.tensordot(img[..., :channels], coeffs, axes=([2], [0]))

    @staticmethod
    def _sample_overlap_indices(overlap: np.ndarray, *, threshold: int, sample_size: int) -> np.ndarray | None:
        """Return deterministic sampled overlap indices for large masks, else None."""
        count = int(np.count_nonzero(overlap))
        if count <= int(threshold):
            return None
        idx = np.flatnonzero(overlap)
        if idx.size <= int(sample_size):
            return idx
        step = max(1, idx.size // int(sample_size))
        return idx[::step]

    def _compute_gain(self, base, new, overlap, sampled_idx: np.ndarray | None = None):
        """Compute gain factor(s) between base and new over the overlap region.

        If lum_only_gain is True, computes a single scalar gain from luminance
        and applies it uniformly to all channels (prevents color shifts).
        """
        overlap = overlap.astype(bool)
        count = int(np.count_nonzero(overlap))
        if count == 0:
            return 1.0
        use_idx = sampled_idx is not None and sampled_idx.size > 0

        # Luminance-only gain: compute scalar gain from luminance channel
        if self.lum_only_gain and base.ndim > 2 and new.ndim > 2:
            base_lum = self._rgb_to_luminance(base.astype(np.float32))
            new_lum = self._rgb_to_luminance(new.astype(np.float32))

            # Sample overlap pixels for performance (deterministic stride)
            if use_idx:
                a = base_lum.flat[sampled_idx]
                b = new_lum.flat[sampled_idx]
            elif count > self.GAIN_SAMPLE_THRESHOLD:
                idx = np.flatnonzero(overlap)
                step = max(1, len(idx) // self.GAIN_SAMPLE_SIZE)
                idx = idx[::step]
                a = base_lum.flat[idx]
                b = new_lum.flat[idx]
            else:
                a = base_lum[overlap]
                b = new_lum[overlap]

            # Return scalar gain (will be applied uniformly to all channels)
            return self._compute_gain_for_channel(a, b)

        # Grayscale case
        if base.ndim == 2:
            # Sample overlap pixels for performance (deterministic stride)
            if use_idx:
                a = base.flat[sampled_idx]
                b = new.flat[sampled_idx]
            elif count > self.GAIN_SAMPLE_THRESHOLD:
                idx = np.flatnonzero(overlap)
                step = max(1, len(idx) // self.GAIN_SAMPLE_SIZE)
                idx = idx[::step]
                a = base.flat[idx]
                b = new.flat[idx]
            else:
                a = base[overlap]
                b = new[overlap]

            return self._compute_gain_for_channel(a, b)

        # Multi-channel case (per-channel gain)
        if base.ndim <= 2 or new.ndim <= 2:
            # Fallback to scalar if one is 2D (shouldn't happen after alignment, but be safe)
            return 1.0

        c_eff = min(base.shape[2], new.shape[2], 3)
        gains = []
        # Pre-compute sampled indices once; reuse across channels
        if use_idx:
            idx = sampled_idx
            use_idx = True
        elif count > self.GAIN_SAMPLE_THRESHOLD:
            idx = np.flatnonzero(overlap)
            step = max(1, len(idx) // self.GAIN_SAMPLE_SIZE)
            idx = idx[::step]
            use_idx = True
        else:
            use_idx = False
        for c in range(c_eff):
            a = base[..., c].flat[idx] if use_idx else base[..., c][overlap]
            b = new[..., c].flat[idx] if use_idx else new[..., c][overlap]
            if a.size == 0:
                gains.append(1.0)
                continue
            gains.append(self._compute_gain_for_channel(a, b))
        if base.shape[2] > c_eff or new.shape[2] > c_eff:
            gains.append(1.0)
        return np.asarray(gains, dtype=np.float32)

    @staticmethod
    def _compute_offset(base, new, overlap, offset_clamp=800.0, sampled_idx: np.ndarray | None = None):
        """Compute additive offset(s) to match new to base over the overlap."""
        overlap = overlap.astype(bool)
        use_idx = sampled_idx is not None and sampled_idx.size > 0

        if base.ndim == 2:
            count = int(np.count_nonzero(overlap))
            if count == 0:
                return 0.0

            # Sample overlap pixels for performance (deterministic stride)
            if use_idx:
                a = base.flat[sampled_idx].astype(np.float32)
                b = new.flat[sampled_idx].astype(np.float32)
                off = np.median(a - b)
            elif count > Blender.GAIN_SAMPLE_THRESHOLD:
                idx = np.flatnonzero(overlap)
                step = max(1, len(idx) // Blender.GAIN_SAMPLE_SIZE)
                idx = idx[::step]
                a = base.flat[idx].astype(np.float32)
                b = new.flat[idx].astype(np.float32)
                off = np.median(a - b)
            else:
                a = base[overlap].astype(np.float32)
                b = new[overlap].astype(np.float32)
                off = np.median(a - b)

            return float(np.clip(off, -float(offset_clamp), float(offset_clamp)))

        # Multi-channel case
        if base.ndim <= 2 or new.ndim <= 2:
            # Fallback to scalar if one is 2D (shouldn't happen after alignment, but be safe)
            return 0.0

        c_eff = min(base.shape[2], new.shape[2], 3)
        offsets = []
        # Pre-compute sampled indices once; reuse across channels
        count_mc = int(np.count_nonzero(overlap))
        if use_idx:
            idx = sampled_idx
            use_idx = True
        elif count_mc > Blender.GAIN_SAMPLE_THRESHOLD:
            idx = np.flatnonzero(overlap)
            step = max(1, len(idx) // Blender.GAIN_SAMPLE_SIZE)
            idx = idx[::step]
            use_idx = True
        else:
            use_idx = False
        for c in range(c_eff):
            a = base[..., c].flat[idx].astype(np.float32) if use_idx else base[..., c][overlap].astype(np.float32)
            b = new[..., c].flat[idx].astype(np.float32) if use_idx else new[..., c][overlap].astype(np.float32)
            if a.size == 0:
                offsets.append(0.0)
                continue
            off = float(np.median(a - b))
            offsets.append(float(np.clip(off, -float(offset_clamp), float(offset_clamp))))
        if base.shape[2] > c_eff or new.shape[2] > c_eff:
            offsets.append(0.0)
        return np.asarray(offsets, dtype=np.float32)

    # --------------------------------------------------
    # Feather Blending
    # --------------------------------------------------
    def _fallback_feather_blend(self, canvas, img, canvas_mask, img_mask, overlap, blended_mask=None):
        """Fallback to standard feathering blend when advanced blending fails."""
        roi_bbox = self._compute_shared_blend_bbox(canvas_mask, img_mask)
        if roi_bbox is None:
            return canvas

        y0, y1, x0, x1 = roi_bbox
        alpha_w = self._distance_weight_alpha(
            canvas_mask[y0:y1, x0:x1], img_mask[y0:y1, x0:x1]
        ).astype(np.float32, copy=False)
        canvas_f = canvas.astype(np.float32, copy=False)
        img_f = img.astype(np.float32, copy=False)
        canvas_roi = canvas_f[y0:y1, x0:x1]
        img_roi = img_f[y0:y1, x0:x1]
        overlap_roi = overlap[y0:y1, x0:x1]
        alpha_exp = alpha_w[..., None] if canvas_roi.ndim > 2 else alpha_w
        blended_full = canvas_roi * (1.0 - alpha_exp) + img_roi * alpha_exp
        canvas_roi[overlap_roi] = blended_full[overlap_roi]
        if np.issubdtype(canvas.dtype, np.integer):
            info = np.iinfo(canvas.dtype)
            canvas_f = np.clip(canvas_f, info.min, info.max)
        canvas = canvas_f.astype(canvas.dtype, copy=False)
        if blended_mask is not None:
            blended_mask[overlap] = True
        return canvas

    def _distance_weight_alpha(self, canvas_mask, img_mask):
        """Compute per-pixel blend weight for the new image in overlap regions."""
        canvas_u8 = canvas_mask.astype(np.uint8)
        img_u8 = img_mask.astype(np.uint8)

        d_canvas = self._distance_transform(canvas_u8)
        d_img = self._distance_transform(img_u8)

        if self.feather_px is not None and self.feather_px > 0:
            d_canvas = np.minimum(d_canvas, float(self.feather_px))
            d_img = np.minimum(d_img, float(self.feather_px))

        denom = d_canvas + d_img + self.EPSILON
        return np.clip(d_img / denom, 0.0, 1.0)

    def _multiband_blend_roi(
        self,
        canvas,
        img,
        canvas_mask,
        img_mask,
        seam_heatmap=None,
        multiband_levels=None,
        roi_min_pad_px=None,
        roi_pad_level_cap=6,
        weight_sigma=1.0,
    ):
        """Multiband blending in ROI."""

        levels_requested = int(self.multiband_levels if multiband_levels is None else multiband_levels)

        roi_bbox = self._compute_shared_blend_bbox(
            canvas_mask,
            img_mask,
            min_pad=max(
                self.MULTIBAND_PAD_PX if roi_min_pad_px is None else int(roi_min_pad_px),
                1 << max(1, min(levels_requested, int(roi_pad_level_cap))),
            ),
        )
        if roi_bbox is None:
            return canvas

        y0, y1, x0, x1 = roi_bbox

        c = canvas[y0:y1, x0:x1]
        i = img[y0:y1, x0:x1]

        c, i = self._align_array_dims(c, i)
        c = c.astype(np.float32, copy=False)
        i = i.astype(np.float32, copy=False)
        c_alpha = None
        i_alpha = None
        if c.ndim == 3 and i.ndim == 3 and c.shape[2] == 4 and i.shape[2] == 4:
            c_alpha = c[..., 3].copy()
            i_alpha = i[..., 3].copy()
        cm = canvas_mask[y0:y1, x0:x1].astype(np.uint8)
        im = img_mask[y0:y1, x0:x1].astype(np.uint8)
        union = (cm > 0) | (im > 0)
        overlap = (cm > 0) & (im > 0)

        if not np.any(union):
            return canvas

        w = np.zeros((c.shape[0], c.shape[1]), dtype=np.float32)
        only_img = (im > 0) & (cm == 0)
        only_canvas = (cm > 0) & (im == 0)
        w[only_img] = 1.0
        w[only_canvas] = 0.0
        if np.any(overlap):
            d_c = self._distance_transform(cm)
            d_i = self._distance_transform(im)
            w[overlap] = d_i[overlap] / (d_i[overlap] + d_c[overlap] + self.EPSILON)
        w = self._smooth_weight_adaptive(
            w,
            i,
            gaussian_sigma=float(weight_sigma),
            use_edge_aware=self.edge_aware_smoothing,
            overlap_mask=overlap,
        )
        w = self._finalize_weight_map(w, only_canvas, only_img, union)
        if seam_heatmap is not None:
            s = (4.0 * w * (1.0 - w)).astype(np.float32, copy=False)
            if np.any(overlap):
                roi_seam = seam_heatmap[y0:y1, x0:x1]
                if roi_seam.ndim == 2 and s.ndim == 2:
                    # np.maximum(roi_seam, s, out=roi_seam)
                    roi_seam[overlap] = np.maximum(roi_seam[overlap], s[overlap])

        min_side = min(c.shape[0], c.shape[1])
        levels = self._compute_pyramid_levels(min_side, levels_requested)

        if levels == 0:
            out = self._single_scale_blend(c, i, w)
            canvas_f = canvas.astype(np.float32, copy=False)
            roi = canvas_f[y0:y1, x0:x1]
            roi[union] = out[union]

            if (
                roi.ndim == 3 and roi.shape[2] == 4 and
                c_alpha is not None and i_alpha is not None
            ):
                out_a = np.maximum(c_alpha, i_alpha)
                roi_alpha = roi[..., 3]
                roi_alpha[union] = out_a[union]

            if np.issubdtype(canvas.dtype, np.integer):
                info = np.iinfo(canvas.dtype)
                canvas_f = np.clip(canvas_f, info.min, info.max)
            return canvas_f.astype(canvas.dtype, copy=False)

        gp_c = [c]
        gp_i = [i]
        gp_w = [w]

        for level_idx in range(levels):
            dc = self._pyr_down(gp_c[-1])
            di = self._pyr_down(gp_i[-1])
            dw = self._pyr_down(gp_w[-1])

            gp_c.append(dc)
            gp_i.append(di)
            gp_w.append(dw)

        lp_c = [gp_c[-1]]
        lp_i = [gp_i[-1]]

        for l in range(levels, 0, -1):
            size = (gp_c[l - 1].shape[1], gp_c[l - 1].shape[0])

            up_c = self._pyr_up(gp_c[l], dstsize=size)
            up_i = self._pyr_up(gp_i[l], dstsize=size)

            # Ensure upsampled images match the exact shape of original levels
            base_c = gp_c[l - 1]
            base_i = gp_i[l - 1]

            up_c = self._align_to_reference(up_c, base_c)
            up_i = self._align_to_reference(up_i, base_i)

            # Also handle spatial size mismatches from pyrUp rounding
            if up_c.shape[:2] != base_c.shape[:2]:
                h, w = base_c.shape[:2]
                up_c = up_c[:h, :w]
            if up_i.shape[:2] != base_i.shape[:2]:
                h, w = base_i.shape[:2]
                up_i = up_i[:h, :w]

            diff_c = base_c - up_c
            diff_i = base_i - up_i

            lp_c.append(diff_c)
            lp_i.append(diff_i)

        lp_c = lp_c[::-1]
        lp_i = lp_i[::-1]

        blend_levels = []
        for l in range(levels + 1):
            wl = gp_w[l]
            lpc = lp_c[l]
            lpi = lp_i[l]

            lpc, lpi = self._align_array_dims(lpc, lpi)
            # Prepare weight for broadcasting
            if lpc.ndim > 2:
                wl_used = wl[..., None]
            else:
                wl_used = wl

            result = lpc * (1.0 - wl_used) + lpi * wl_used
            blend_levels.append(result)

        out = blend_levels[-1]

        for l in range(levels - 1, -1, -1):
            size = (blend_levels[l].shape[1], blend_levels[l].shape[0])

            out = self._pyr_up(out, dstsize=size)
            # Ensure dimensions match before addition (handle 2D, 3D, 4D)
            if out.ndim > 2 and blend_levels[l].ndim == 2:
                addend = blend_levels[l][..., None]
            elif out.ndim == 2 and blend_levels[l].ndim > 2:
                addend = blend_levels[l][..., 0]
            else:
                addend = blend_levels[l]
            # Final safety check: ensure shapes are fully compatible
            if out.shape != addend.shape:

                if out.ndim > 2 and addend.ndim == 2:
                    addend = np.repeat(addend[..., None], out.shape[2], axis=2)
                elif out.ndim == 2 and addend.ndim > 2:
                    addend = addend[..., 0]
                elif out.ndim > 2 and addend.ndim > 2 and out.shape[2] != addend.shape[2]:
                    # Handle mismatched channel counts
                    common_c = min(out.shape[2], addend.shape[2])
                    out = out[..., :common_c]
                    addend = addend[..., :common_c]

            out = out + addend

        canvas_f = canvas.astype(np.float32, copy=False)
        roi = canvas_f[y0:y1, x0:x1]

        self._write_blend_result(roi, out, union, c_alpha=c_alpha, i_alpha=i_alpha)

        if np.issubdtype(canvas.dtype, np.integer):
            info = np.iinfo(canvas.dtype)
            canvas_f = np.clip(canvas_f, info.min, info.max)
        return canvas_f.astype(canvas.dtype, copy=False)

    # --------------------------------------------------
    # Main Blend Function
    # --------------------------------------------------
    def blend(
        self,
        canvas,
        canvas_mask,
        img,
        img_mask,
        use_gain=True,
        use_offset=False,
        offset_clamp=800.0,
        coverage_count=None,
        blended_mask=None,
        seam_heatmap=None,
    ):
        """Blend new image into canvas."""

        # Ensure canvas and img have compatible dimensions
        canvas, img = self._align_array_dims(canvas, img)

        canvas_mask = canvas_mask.astype(bool)
        img_mask = img_mask.astype(bool)

        img_bbox, blend_roi_bbox = self._compute_blend_regions(canvas_mask, img_mask)
        if img_bbox is None:
            if coverage_count is not None:
                coverage_count[img_mask] += 1
            return canvas, canvas_mask

        y0, y1, x0, x1 = img_bbox
        by0, by1, bx0, bx1 = blend_roi_bbox

        overlap = canvas_mask & img_mask
        overlap_roi = overlap[y0:y1, x0:x1]
        has_overlap = np.any(overlap_roi)
        overlap_confidence = self._photometric_overlap_confidence(overlap_roi) if has_overlap else 0.0
        img_roi = img[y0:y1, x0:x1]
        canvas_roi = canvas[y0:y1, x0:x1]
        img_mask_roi = img_mask[y0:y1, x0:x1]

        # Apply histogram matching before gain (if enabled)
        if self.histogram_matching and has_overlap and overlap_confidence > 0.0:
            matched = self._apply_histogram_matching(canvas_roi, img_roi, overlap_roi)
            if overlap_confidence >= 1.0:
                img[y0:y1, x0:x1] = matched
            else:
                alpha = float(overlap_confidence)
                mixed = img_roi.astype(np.float32, copy=False) * (1.0 - alpha) + matched.astype(np.float32, copy=False) * alpha
                if np.issubdtype(img.dtype, np.integer):
                    info = np.iinfo(img.dtype)
                    mixed = np.clip(mixed, info.min, info.max)
                img[y0:y1, x0:x1] = mixed.astype(img.dtype, copy=False)

        if has_overlap and (use_gain or use_offset):
            # Fuse gain + offset math into a single float ROI pass to reduce allocations.
            img_roi_f32 = img_roi.astype(np.float32, copy=True)
            integer_dtype = np.issubdtype(img.dtype, np.integer)
            int_info = np.iinfo(img.dtype) if integer_dtype else None
            sampled_idx = self._sample_overlap_indices(
                overlap_roi,
                threshold=self.GAIN_SAMPLE_THRESHOLD,
                sample_size=self.GAIN_SAMPLE_SIZE,
            )

            if use_gain:
                gain = self._compute_gain(canvas_roi, img_roi_f32, overlap_roi, sampled_idx=sampled_idx)
                gain = self._attenuate_gain(gain, overlap_confidence)
                if np.ndim(gain) == 0:
                    img_roi_f32[img_mask_roi] *= float(gain)
                else:
                    gain_channels = min(len(gain), img.shape[2]) if img.ndim > 2 else 1
                    img_roi_f32[img_mask_roi] *= gain[:gain_channels]

            if use_offset:
                offset_input = img_roi_f32
                if use_gain and integer_dtype and int_info is not None:
                    # Preserve legacy behavior: offset was historically computed after
                    # gain quantization back to image dtype when both stages were enabled.
                    offset_input = np.clip(img_roi_f32, int_info.min, int_info.max).astype(img.dtype).astype(np.float32)
                offset = self._compute_offset(
                    canvas_roi,
                    offset_input,
                    overlap_roi,
                    offset_clamp=offset_clamp,
                    sampled_idx=sampled_idx,
                )
                offset = self._attenuate_offset(offset, overlap_confidence)
                if np.ndim(offset) == 0:
                    img_roi_f32[img_mask_roi] += float(offset)
                else:
                    offset_channels = min(len(offset), img.shape[2]) if img.ndim > 2 else 1
                    img_roi_f32[img_mask_roi] += offset[:offset_channels]

            if integer_dtype and int_info is not None:
                img_roi_f32 = np.clip(img_roi_f32, int_info.min, int_info.max)
            img[y0:y1, x0:x1] = img_roi_f32.astype(img.dtype, copy=False)

        if has_overlap:
            if self.blend_type == "none":
                pass
            elif self.blend_type == "adaptive-feather":
                # Adaptive feathering: intelligent feather width based on overlap size
                try:
                    canvas = self._adaptive_feather_blend(canvas, img, canvas_mask, img_mask)
                    if blended_mask is not None:
                        blended_mask[overlap] = True
                except Exception as e:
                    # Fallback to standard feathering if adaptive fails
                    logger.warning("Adaptive feather blending failed; falling back to feathering: %s", e, exc_info=True)
                    warnings.warn(f"Adaptive feather blending failed: {e}. Falling back to feathering.")
                    canvas = self._fallback_feather_blend(canvas, img, canvas_mask, img_mask, overlap, blended_mask)
            elif self.blend_type == "adaptive-multiband":
                # Adaptive multiband: auto pyramid levels based on overlap size
                try:
                    canvas = self._adaptive_multiband_blend(canvas, img, canvas_mask, img_mask, seam_heatmap)
                    if blended_mask is not None:
                        blended_mask[overlap] = True
                except Exception as e:
                    logger.warning("Adaptive multiband blending failed; falling back to feathering: %s", e, exc_info=True)
                    warnings.warn(f"Adaptive multiband blending failed: {e}. Falling back to feathering.")
                    canvas = self._fallback_feather_blend(canvas, img, canvas_mask, img_mask, overlap, blended_mask)
            elif self.blend_type == "seamless":
                # Seamless blending: enhanced multiband with extra smoothing
                try:
                    canvas = self._seamless_blend(canvas, img, canvas_mask, img_mask, seam_heatmap=seam_heatmap)
                    if blended_mask is not None:
                        blended_mask[overlap] = True
                except Exception as e:
                    # Fallback to standard feathering if seamless fails
                    logger.warning("Seamless blending failed; falling back to feathering: %s", e, exc_info=True)
                    warnings.warn(f"Seamless blending failed: {e}. Falling back to feathering.")
                    canvas = self._fallback_feather_blend(canvas, img, canvas_mask, img_mask, overlap, blended_mask)
            elif self.blend_type == "multiband":
                canvas = self._multiband_blend_roi(canvas, img, canvas_mask, img_mask, seam_heatmap)
                if blended_mask is not None:
                    blended_mask[overlap] = True
            else:
                alpha_w = self._distance_weight_alpha(
                    canvas_mask[by0:by1, bx0:bx1], img_mask[by0:by1, bx0:bx1]
                ).astype(np.float32, copy=False)
                canvas_f = canvas.astype(np.float32, copy=False)
                img_f = img.astype(np.float32, copy=False)
                canvas_blend_roi = canvas_f[by0:by1, bx0:bx1]
                img_blend_roi = img_f[by0:by1, bx0:bx1]
                overlap_blend_roi = overlap[by0:by1, bx0:bx1]

                if canvas_blend_roi.ndim > 2 and canvas_blend_roi.shape[2] == 4 and img_blend_roi.ndim > 2 and img_blend_roi.shape[2] == 4:
                    alpha_w_exp = alpha_w[..., None]
                    blended_rgb = canvas_blend_roi[..., :3] * (1.0 - alpha_w_exp) + img_blend_roi[..., :3] * alpha_w_exp
                    out_alpha = np.maximum(canvas_blend_roi[..., 3], img_blend_roi[..., 3])
                    blended_full = np.dstack((blended_rgb, out_alpha))
                else:
                    alpha_exp = alpha_w[..., None] if canvas_blend_roi.ndim > 2 else alpha_w
                    blended_full = canvas_blend_roi * (1.0 - alpha_exp) + img_blend_roi * alpha_exp

                canvas_blend_roi[overlap_blend_roi] = blended_full[overlap_blend_roi]

                if np.issubdtype(canvas.dtype, np.integer):
                    info = np.iinfo(canvas.dtype)
                    canvas_f = np.clip(canvas_f, info.min, info.max)
                canvas = canvas_f.astype(canvas.dtype, copy=False)
                if blended_mask is not None:
                    blended_mask[overlap] = True
                if seam_heatmap is not None:
                    s = (4.0 * alpha_w * (1.0 - alpha_w)).astype(np.float32, copy=False)
                    seam_roi = seam_heatmap[by0:by1, bx0:bx1]
                    seam_roi[overlap_blend_roi] = np.maximum(seam_roi[overlap_blend_roi], s[overlap_blend_roi])

        new_pixels = img_mask & (~canvas_mask)
        if np.any(new_pixels):
            canvas[new_pixels] = img[new_pixels]

        if coverage_count is not None:
            coverage_count[img_mask] += 1
        canvas_mask |= img_mask
        return canvas, canvas_mask

