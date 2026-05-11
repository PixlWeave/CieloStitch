# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import os
import threading
from typing import Optional
import numpy as np
import cv2
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...config.config import cfg
from ...core.detector import FeatureDetector
from ...core.matcher import FeatureMatcher
from ...core.warper import Warper
from ...core.blender import Blender
from ...core.stitching.illumination import IlluminationNormalizer
from ...core.stitching.local_gain_normalizer import LocalGainNormalizer
from ...core.feature_cache import (
    detect_with_cache,
    format_feature_cache_stats,
    get_feature_cache_stats,
)
from ...utils.message import emit_msg
from ...state.preferences import read_pref
from ...config.constants import MAX_MULTIBAND_LEVELS

logger = logging.getLogger(__name__)

SUBPIXEL_REFINEMENT_MAX_ABS_SHIFT_PX = 2.5
SUBPIXEL_REFINEMENT_MIN_RESPONSE = 0.02
SUBPIXEL_REFINEMENT_MIN_OVERLAP_PX = 256


class BaseStitchMode:
    """Common utilities shared by different stitching modes.

    Subclasses should focus on placement/global transform estimation and may
    override default parameters via protected methods.
    """

    #
    def __init__(self, profile, show_pixel_stat=False):
        self.profile = profile
        self.show_pixel_stat = bool(show_pixel_stat)
        self.last_canvas_system = None

        # Detector
        detector_downscale = cfg.state.psm.detector_downscale  # 0.5
        detector_max_points = cfg.state.psm.detector_max_points
        detector_feature_sensitivity = cfg.state.psm.detector_feature_sensitivity
        min_inlier_ratio = cfg.state.psm.min_inlier_ratio
        min_inliers = cfg.state.psm.min_inliers  # 12

        self.detector = FeatureDetector(
            downscale_factor=detector_downscale,
            max_features=detector_max_points,
            feature_sensitivity=detector_feature_sensitivity,
        )

        # Matcher
        self.matcher = FeatureMatcher(
            ratio_test=profile.ratio_test,
            ransac_thresh=profile.ransac_thresh,
            min_inlier_ratio=min_inlier_ratio,
            min_inliers=min_inliers,
        )

        # Warper
        try:
            stitch_interpolator = read_pref("stitch/stitch_interpolator", "auto", type=str)
        except Exception:
            stitch_interpolator = "auto"

        self.warper = Warper(interpolation=stitch_interpolator)

        # Blender
        self.blender = Blender(
            feather_px=cfg.state.psm.blend_feather_px,
            gain_method=cfg.state.psm.gain_method,
            gain_clamp=cfg.state.psm.gain_clamp,
            blend_type=cfg.state.psm.blend_type,
            multiband_levels=cfg.state.psm.multiband_levels,
            seamless_quality=cfg.state.psm.seamless_quality or "balanced",
            lum_only_gain=cfg.state.psm.lum_only_gain or False,
            histogram_matching=cfg.state.psm.histogram_matching or False,
            edge_aware_smoothing=getattr(cfg.state.psm, "edge_aware_smoothing", True),
            adaptive_mb_risk_boost_threshold=getattr(cfg.state.psm, "adaptive_mb_risk_boost_threshold", 0.45),
            adaptive_mb_low_risk_threshold=getattr(cfg.state.psm, "adaptive_mb_low_risk_threshold", 0.10),
            adaptive_mb_max_boost=getattr(cfg.state.psm, "adaptive_mb_max_boost", 1),
            photometric_min_overlap_px=getattr(cfg.state.psm, "photometric_min_overlap_px", 2_000),
            photometric_full_confidence_px=getattr(cfg.state.psm, "photometric_full_confidence_px", 250_000),
        )

        # Illumination Normalizer
        self.normalizer = IlluminationNormalizer()

        # Configure LocalGainNormalizer with profile-aware defaults
        # profile_name = getattr(profile, '__class__.__name__', 'unknown').lower()
        profile_name = type(profile).__name__.lower()

        # Profile-specific local normalization parameters
        if 'solar' in profile_name:
            # Solar: fine tiles, adaptive lowfreq, reduced smoothing for sharpness
            local_params = dict(
                tile_size=48,
                smooth_sigma=9.6,  # tile_size/5 for better edge preservation
                gain_clamp=(0.7, 1.3),
                use_offset=True,
                lowfreq_sigma=None  # Adaptive: will auto-scale to ~15-20
            )
        elif 'lunar' in profile_name:
            # Lunar: medium tiles, adaptive lowfreq, reduced smoothing for craters
            local_params = dict(
                tile_size=64,
                smooth_sigma=12.8,  # tile_size/5 (critical for crater sharpness)
                gain_clamp=(0.8, 1.2),
                use_offset=True,
                lowfreq_sigma=None  # Adaptive: will auto-scale to ~8-12
            )
        elif 'milky-way' in profile_name or 'nightscape' in profile_name:
            # Deep-sky: larger tiles, adaptive lowfreq for sky gradients
            local_params = dict(
                tile_size=128,
                smooth_sigma=25.6,  # tile_size/5
                gain_clamp=(0.7, 1.3),
                use_offset=True,
                lowfreq_sigma=None  # Adaptive: will auto-scale to ~20-30
            )
        else:
            # Default: universal adaptive settings
            local_params = dict(
                tile_size=128,
                smooth_sigma=25.6,  # tile_size/5
                gain_clamp=(0.8, 1.2),
                use_offset=True,
                lowfreq_sigma=None  # Adaptive based on image scale
            )

        # Illumination Normalizer2
        self.local_normalizer = LocalGainNormalizer(**local_params)

        # Unified progress counters (used by current_summary)
        self._progress_total = 0
        self._progress_placed = 0
        self._progress_skipped = 0
        self._subpixel_refinement_keys: set = set()
        self._placement_source_counts: dict = {
            "anchor": 0,
            "feature": 0,
            "phase": 0,
            "nominal": 0,
            "deterministic": 0,
        }
        self._panel_source_by_index: dict[int, str] = {}
        self._panel_evidence_by_index: dict[int, str] = {}
        self._phase_timing_secs: dict[str, float] = {
            "detect": 0.0,
            "match": 0.0,
            "transform": 0.0,
            "subpixel": 0.0,
            "warp": 0.0,
            "blend": 0.0,
        }
    #

    # ---- Shared helpers ----
    @staticmethod
    def check_cancel(cancel_cb=None):
        """Check if cancellation was requested and raise InterruptedError if so."""
        if cancel_cb is not None and cancel_cb():
            raise InterruptedError("stitch cancelled")

    @staticmethod
    def _recommended_feature_workers(panel_count: int) -> int:
        """Return a bounded worker count for parallel feature detection."""
        n = max(1, int(panel_count or 1))
        cpu = max(1, int(os.cpu_count() or 1))
        # Approximate physical cores to avoid overcommitting SMT-heavy systems.
        physical_est = max(1, (3 * cpu + 4) // 5)  # ceil(cpu * 0.6)
        return max(1, min(n, 12, physical_est))

    def _record_subpixel_refinement_applied(self, image_key) -> None:
        """Record that subpixel refinement was used for the panel with this key.

        image_key must be unique per placed panel (e.g. its scan/image index).
        Uses a set so repeated calls for the same panel never over-count.
        """
        keys = getattr(self, "_subpixel_refinement_keys", None)
        if keys is None:
            keys = set()
            self._subpixel_refinement_keys = keys
        keys.add(image_key)

    def _record_panel_source(self, image_index: int, source: str) -> None:
        try:
            self._panel_source_by_index[int(image_index)] = str(source)
        except Exception:
            pass

    def _record_panel_evidence(self, image_index: int, evidence) -> None:
        try:
            idx = int(image_index)
            if evidence:
                self._panel_evidence_by_index[idx] = evidence
            else:
                self._panel_evidence_by_index.pop(idx, None)
        except Exception:
            pass

    def _record_phase_time(self, key: str, delta: float) -> None:
        try:
            timings = getattr(self, "_phase_timing_secs", None)
            if timings is None:
                timings = {}
                self._phase_timing_secs = timings
            timings[str(key)] = float(timings.get(str(key), 0.0) or 0.0) + max(0.0, float(delta))
        except Exception:
            pass

    @staticmethod
    def format_stitching_summary(
            *,
            total: int,
            placed: int,
            skipped: int,
            elapsed: float,
            subpixel_refinement_count: int = 0,
            source_counts: dict | None = None,
    ) -> str:
        source_counts = source_counts or {}
        source_order = ["anchor", "deterministic", "feature", "phase", "nominal"]
        source_parts = [
            f"{key}:{source_counts[key]}"
            for key in source_order
            if source_counts.get(key, 0) > 0
        ]
        placed_detail = f" ({', '.join(source_parts)})" if source_parts else ""
        return (
            f"Stitching summary: total={total}, placed={placed}{placed_detail}, "
            f"skipped={skipped}, subpixel alignment count={subpixel_refinement_count}, time={elapsed:.2f}s"
        )

    def _match_feature_pair(self, ref_keypoints, ref_descriptors, ref_scale, current_keypoints,
                            current_descriptors, current_scale, allow_rotation=True):
        """Match features between a reference and current image pair.

        Args:
            ref_keypoints, ref_descriptors, ref_scale: Reference image features
            current_keypoints, current_descriptors, current_scale: Current image features
            allow_rotation: Whether to allow rotation in homography (default True)

        Returns:
            `(local_transform, inlier_mask)` or `(None, None)` if matching fails.
        """
        local_transform, inlier_mask, _metrics = self._match_feature_pair_with_metrics(
            ref_keypoints, ref_descriptors, ref_scale,
            current_keypoints, current_descriptors, current_scale,
            allow_rotation=allow_rotation,
        )
        return local_transform, inlier_mask

    @staticmethod
    def _subpixel_refinement_enabled() -> bool:
        return bool(getattr(cfg.prefs, "enable_subpixel_refinement", True))

    @staticmethod
    def _refinement_support_mask(image):
        if image.ndim == 3 and image.shape[2] == 4:
            return (image[..., 3] > 1e-6).astype(np.uint8)
        return np.ones(image.shape[:2], dtype=np.uint8)

    @classmethod
    def _warp_for_refinement(cls, image, transform, output_shape):
        height, width = output_shape
        warped = cv2.warpPerspective(
            image.astype(np.float32, copy=False),
            transform,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        support = cls._refinement_support_mask(image)
        valid = cv2.warpPerspective(
            support,
            transform,
            (int(width), int(height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        return warped, valid.astype(np.uint8)

    def _refine_transform_with_phase_correlation(self, ref_image, current_image, local_transform):
        if not self._subpixel_refinement_enabled():
            return local_transform, None
        if ref_image is None or current_image is None or local_transform is None:
            return local_transform, None

        ref_h, ref_w = ref_image.shape[:2]
        if ref_h < 32 or ref_w < 32:
            return local_transform, None

        try:
            warped_current, warped_mask = self._warp_for_refinement(current_image, local_transform, (ref_h, ref_w))
            overlap = warped_mask.astype(bool)
            if np.count_nonzero(overlap) < SUBPIXEL_REFINEMENT_MIN_OVERLAP_PX:
                return local_transform, None

            ref_valid = self._refinement_support_mask(ref_image).astype(bool)
            overlap &= ref_valid
            if np.count_nonzero(overlap) < SUBPIXEL_REFINEMENT_MIN_OVERLAP_PX:
                return local_transform, None

            ys, xs = np.where(overlap)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            ref_roi = ref_image[y0:y1, x0:x1]
            warped_roi = warped_current[y0:y1, x0:x1]

            residual_transform, response = self._estimate_translation_fallback(
                ref_roi,
                warped_roi,
                min_response=SUBPIXEL_REFINEMENT_MIN_RESPONSE,
                max_shift_ratio=0.08,
            )
            if residual_transform is None:
                return local_transform, {
                    "applied": False,
                    "response": float(response),
                    "dx": 0.0,
                    "dy": 0.0,
                }

            dx = float(residual_transform[0, 2])
            dy = float(residual_transform[1, 2])
            if (
                    abs(dx) > SUBPIXEL_REFINEMENT_MAX_ABS_SHIFT_PX
                    or abs(dy) > SUBPIXEL_REFINEMENT_MAX_ABS_SHIFT_PX
            ):
                return local_transform, {
                    "applied": False,
                    "response": float(response),
                    "dx": dx,
                    "dy": dy,
                }

            refined_transform = residual_transform @ local_transform
            return refined_transform, {
                "applied": True,
                "response": float(response),
                "dx": dx,
                "dy": dy,
            }
        except Exception:
            logger.debug("Subpixel refinement failed; keeping feature transform", exc_info=True)
            return local_transform, None

    def _match_feature_pair_with_metrics(self, ref_keypoints, ref_descriptors, ref_scale, current_keypoints,
                                         current_descriptors, current_scale, allow_rotation=True,
                                         ref_image=None, current_image=None):
        """Match features and return lightweight evidence metrics.

        The metrics are intended for registration-policy decisions, especially
        when deciding whether pair looks weak-texture enough to try
        translation fallback earlier.
        """
        ref_keypoint_count = len(ref_keypoints) if ref_keypoints is not None else 0
        current_keypoint_count = len(current_keypoints) if current_keypoints is not None else 0
        minimum_keypoint_count = min(ref_keypoint_count, current_keypoint_count)
        feature_metrics = {
            "kp_ref": int(ref_keypoint_count),
            "kp_curr": int(current_keypoint_count),
            "min_kp": int(minimum_keypoint_count),
            "match_count": 0,
            "inlier_count": 0,
            "has_descriptors": bool(ref_descriptors is not None and current_descriptors is not None),
            "transform_found": False,
        }

        if ref_descriptors is None or current_descriptors is None:
            return None, None, self.matcher.describe_pair_evidence(feature_metrics)

        t_match_start = time.perf_counter()
        good_matches = self.matcher.match(ref_keypoints, ref_descriptors, current_keypoints, current_descriptors)
        self._record_phase_time("match", time.perf_counter() - t_match_start)
        feature_metrics["match_count"] = int(len(good_matches))
        if len(good_matches) == 0:
            return None, None, self.matcher.describe_pair_evidence(feature_metrics)

        t_transform_start = time.perf_counter()
        local_transform, inlier_mask = self.matcher.compute_transform(
            ref_keypoints, current_keypoints, good_matches, ref_scale, current_scale, allow_rotation=allow_rotation
        )
        self._record_phase_time("transform", time.perf_counter() - t_transform_start)
        if inlier_mask is not None:
            feature_metrics["inlier_count"] = int(np.sum(inlier_mask))
        if local_transform is not None and inlier_mask is not None:
            t_subpixel_start = time.perf_counter()
            local_transform, refinement_metrics = self._refine_transform_with_phase_correlation(
                ref_image,
                current_image,
                local_transform,
            )
            self._record_phase_time("subpixel", time.perf_counter() - t_subpixel_start)
            if refinement_metrics is not None:
                feature_metrics["subpixel_refinement_applied"] = bool(refinement_metrics.get("applied", False))
                feature_metrics["subpixel_refinement_response"] = refinement_metrics.get("response", 0.0)
                feature_metrics["subpixel_refinement_dx"] = refinement_metrics.get("dx", 0.0)
                feature_metrics["subpixel_refinement_dy"] = refinement_metrics.get("dy", 0.0)
        feature_metrics["transform_found"] = bool(local_transform is not None and inlier_mask is not None)
        feature_metrics = self.matcher.describe_pair_evidence(feature_metrics)
        return local_transform, inlier_mask, feature_metrics

    def _is_weak_feature_evidence(self, feature_metrics):
        """Return True when feature evidence is weak enough to favor fallback.

        These thresholds are intentionally internal and conservative for the
        Phase 1 weak-texture path. They should steer clearly weak pairs toward
        phase correlation without changing the normal behavior for textured
        overlaps.
        """
        if not feature_metrics.get("has_descriptors", False):
            return True

        minimum_keypoint_count = int(feature_metrics.get("min_kp", 0))
        match_count = int(feature_metrics.get("match_count", 0))
        minimum_matches_for_transform = int(getattr(self.matcher, "min_matches_for_transform", 6) or 6)

        if minimum_keypoint_count < max(24, minimum_matches_for_transform * 4):
            return True

        if match_count < max(minimum_matches_for_transform, 8):
            return True

        return False

    def _find_best_match_among_references(self, current_keypoints, current_descriptors, current_scale,
                                          references, allow_rotation=True, cancel_cb=None):
        """Find best feature match among multiple reference images.

        Args:
            current_keypoints, current_descriptors, current_scale: Current image features
            references: List of reference dicts, each with keys: 'kp', 'des', 'scale', 'name'
            allow_rotation: Whether to allow rotation (default True)
            cancel_cb: Optional cancellation callback

        Returns:
            Tuple of (best_match_dict, best_match_count) where best_match_dict has keys:
            'score', 'ref_name', 'h_local', 'ref_index'
            Or (None, count) if no matches found
        """
        best_match = None
        best_match_count = 0

        # Iterate forward so equal scores continue to favor newer references.
        for ref_idx, reference in enumerate(references):
            if cancel_cb is not None:
                self.check_cancel(cancel_cb)

            good_matches = self.matcher.match(reference["kp"], reference["des"], current_keypoints, current_descriptors)
            best_match_count = max(best_match_count, len(good_matches))
            if len(good_matches) == 0:
                continue

            local_transform, inlier_mask = self.matcher.compute_transform(
                reference["kp"], current_keypoints, good_matches, reference["scale"], current_scale,
                allow_rotation=allow_rotation
            )
            if local_transform is None or inlier_mask is None:
                continue

            score = int(np.sum(inlier_mask))
            if best_match is None or score >= best_match["score"]:
                best_match = {
                    "score": score,
                    "ref_name": reference["name"],
                    "h_local": local_transform,
                    "ref_index": ref_idx
                }

        return best_match, best_match_count

    @staticmethod
    def _identity_transform(dtype=np.float64):
        """Create identity homography with consistent dtype.

        Args:
            dtype: Numpy dtype for the matrix (default np.float64)

        Returns:
            3x3 identity matrix with specified dtype
        """
        return np.eye(3, dtype=dtype)

    @staticmethod
    def _create_reference(name, image, kp, des, scale, global_h):
        """Create a reference dict with consistent structure.

        Args:
            name: Image name/identifier
            image: Image data (numpy array)
            kp: Keypoints from feature detection
            des: Descriptors from feature detection
            scale: Scale factor used during feature detection
            global_h: Global homography transform (3x3 matrix)

        Returns:
            Reference dict with standardized schema
        """
        return {
            "name": name,
            "image": image,
            "kp": kp,
            "des": des,
            "scale": scale,
            "global_h": global_h
        }

    def _create_anchor_reference(self, name, image, features):
        """Create initial anchor reference (first image in sequence).

        Args:
            name: Image name/identifier
            image: Image data (numpy array)
            features: Tuple of (keypoints, descriptors, scale) from detection

        Returns:
            Reference dict for anchor image with identity transform
        """
        kp, des, scale = features

        return self._create_reference(
            name=name,
            image=image,
            kp=kp,
            des=des,
            scale=scale,
            global_h=self._identity_transform()
        )

    def _init_stitch_session(self, image_items, cancel_cb=None, progress_cb=None,
                             detect_features=True):
        """Common initialization for both GridMode and FreeMode stitch sessions.

        Performs:
        - Input validation
        - Cancellation check
        - Progress counter initialization
        - Image list extraction
        - Parallel feature detection

        Args:
            image_items: List of (name, image) tuples
            cancel_cb: Optional cancellation callback
            progress_cb: Optional progress callback

        Returns:
            Tuple of (names, images, n, features, t_start) where:
            - names: List of image names
            - images: List of image data
            - n: Number of images
            - features: List of (kp, des, scale) tuples from detection
            - t_start: Start time for elapsed calculation
        """
        if len(image_items) < 1:
            raise ValueError("No images provided")

        self.check_cancel(cancel_cb)
        t_start = time.perf_counter()

        # Emit one concise threading summary so runtime parallelism is visible in the log panel.
        # Use the same formula as _detect_features_parallel so the count is accurate.
        _n_panels = len(image_items)
        feature_workers = self._recommended_feature_workers(_n_panels) if detect_features else 0
        try:
            cv_threads = int(cv2.getNumThreads())
        except Exception:
            cv_threads = -1
        cv_threads_text = str(cv_threads) if cv_threads >= 0 else "unknown"
        feature_text = str(feature_workers) if detect_features else "disabled"
        if detect_features:
            if cv_threads >= 0 and cv_threads == feature_workers:
                threading_line = (
                    f"Performance settings: using {feature_text} parallel CPU worker(s) "
                    "for panel analysis."
                )
            else:
                threading_line = (
                    f"Performance settings: using {cv_threads_text} CPU thread(s) and "
                    f"{feature_text} parallel worker(s) for panel analysis."
                )
            color = "green"
        else:
            threading_line = (
                f"Performance settings: using {cv_threads_text} CPU thread(s); "
                "parallel panel analysis is off in this mode."
            )
            color = "yellow"
        emit_msg(
            threading_line,
            color,
            progress_cb,
        )

        # Initialize progress counters
        try:
            self._progress_total = int(len(image_items))
            self._progress_placed = 0  # Start at 0; increment only on successful placement
            self._progress_skipped = 0
            self._placement_source_counts = {
                "anchor": 0,
                "feature": 0,
                "phase": 0,
                "nominal": 0,
                "deterministic": 0,
            }
            self._panel_source_by_index = {}
            self._panel_evidence_by_index = {}
            self._subpixel_refinement_keys = set()
            self._phase_timing_secs = {
                "detect": 0.0,
                "match": 0.0,
                "transform": 0.0,
                "subpixel": 0.0,
                "warp": 0.0,
                "blend": 0.0,
            }
        except Exception:
            pass

        # Extract names and images
        images = [img for _, img in image_items]
        names = [name for name, _ in image_items]
        n = len(images)

        if detect_features:
            # Detect features in parallel
            features = self._detect_features_parallel(
                names, images,
                check_cancel=lambda: self.check_cancel(cancel_cb),
                progress_cb=progress_cb
            )
        else:
            features = []

        return names, images, n, features, t_start

    def _detect_features_parallel(self, names, images, check_cancel=None, progress_cb=None):
        """Detect features for all images in parallel (with cache), with fallback to serial.

        Returns a list of (kp, des, scale) tuples matching the order of inputs.
        """
        n = len(images)
        features: list[Optional[tuple]] = [None] * n
        t_detect_start = time.perf_counter()
        try:
            cache_enabled = bool(read_pref("stitch/feature_cache_enabled", True, type=bool))
        except Exception:
            cache_enabled = True
        cache_before = get_feature_cache_stats(reset_counters=cache_enabled)
        try:
            # Keep worker count bounded by panel count and CPU capacity.
            # A ceiling of 12 avoids excessive memory pressure on large systems.
            max_workers = self._recommended_feature_workers(n)
            thread_local = threading.local()

            def _get_thread_detector() -> FeatureDetector:
                det = getattr(thread_local, "detector", None)
                if det is None:
                    det = FeatureDetector(
                        downscale_factor=self.detector.downscale_factor,
                        max_features=self.detector.max_features,
                        feature_sensitivity=self.detector.feature_sensitivity,
                    )
                    thread_local.detector = det
                return det

            def _detect_one(idx: int):
                detector = _get_thread_detector()
                return idx, detect_with_cache(detector, images[idx], names[idx], cache_enabled)

            # Avoid nested OpenCV + ThreadPool parallelism during feature extraction.
            # We run many detectAndCompute calls in parallel already, so forcing
            # OpenCV to single-thread mode for this stage reduces oversubscription.
            # cv2.setNumThreads is set inside this try/finally so it is always
            # restored even if emit_msg or the executor raises.
            prev_cv_threads = None
            cv_threads_forced = False
            try:
                try:
                    prev_cv_threads = int(cv2.getNumThreads())
                    if max_workers > 1 and prev_cv_threads > 1:
                        cv2.setNumThreads(1)
                        cv_threads_forced = True
                except Exception:
                    prev_cv_threads = None

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_detect_one, i): i for i in range(n)}
                    for fut in as_completed(futs):
                        if check_cancel is not None:
                            check_cancel()
                        i, (kp, des, scale) = fut.result()
                        features[i] = (kp, des, scale)
            finally:
                if cv_threads_forced and prev_cv_threads is not None:
                    try:
                        cv2.setNumThreads(max(1, int(prev_cv_threads)))
                    except Exception:
                        logger.debug("Could not restore OpenCV thread count after feature detection", exc_info=True)
        except InterruptedError:
            raise
        except Exception as _exc:
            logger.warning(
                "Parallel feature detection failed (%s: %s); falling back to serial.",
                type(_exc).__name__,
                _exc,
                exc_info=True,
            )
            emit_msg(
                f"Parallel detection unavailable ({type(_exc).__name__}); falling back to serial.",
                "yellow",
                progress_cb,
            )
            features = []
            for i in range(n):
                if check_cancel is not None:
                    check_cancel()
                kp, des, scale = detect_with_cache(self.detector, images[i], names[i], cache_enabled)
                features.append((kp, des, scale))
                # emit_msg(f"Detect features: {i + 1}/{n} ({names[i]})", "", progress_cb)

        total_keypoints = 0
        total_descriptor_bytes = 0
        for kp, des, _scale in features:
            total_keypoints += len(kp) if kp is not None else 0
            if des is not None:
                total_descriptor_bytes += int(getattr(des, "nbytes", 0))

        elapsed = time.perf_counter() - t_detect_start
        self._record_phase_time("detect", elapsed)
        detect_line = (
            f"Feature detection summary: images={n}, kp={total_keypoints}, "
            f"des={total_descriptor_bytes / (1024.0 * 1024.0):.1f} MB, time={elapsed:.2f}s"
        )
        emit_msg(detect_line, "" if n > 0 else "", progress_cb)

        if cache_enabled:
            cache_stats = get_feature_cache_stats(reset_counters=True)
            cache_delta_entries = int(cache_stats.get("entries", 0)) - int(cache_before.get("entries", 0))
            cache_prefix = "Feature cache reused" if int(cache_stats.get("hits", 0)) > 0 else "Feature cache warmup"
            cache_line = format_feature_cache_stats(cache_stats, prefix=cache_prefix)
            if cache_delta_entries > 0:
                cache_line = f"{cache_line}, new={cache_delta_entries}"
            emit_msg(cache_line, "green" if int(cache_stats.get("hits", 0)) > 0 else "", progress_cb)
        else:
            emit_msg("Feature cache: disabled", "", progress_cb)
        return features

    @staticmethod
    def _estimate_translation_fallback(
            ref_img,
            cur_img,
            min_response: float = 0.03,
            max_shift_ratio: float = 0.85,
    ):
        """Estimate pure translation mapping current -> reference via phase correlation.

        Attribute names and defaults are parameterized so subclasses can keep
        distinct thresholds without duplicating the algorithm.
        Returns (H, response) or (None, 0.0).
        """
        if ref_img is None or cur_img is None:
            return None, 0.0

        h = min(ref_img.shape[0], cur_img.shape[0])
        w = min(ref_img.shape[1], cur_img.shape[1])
        if h < 32 or w < 32:
            return None, 0.0

        ref = ref_img[:h, :w].astype(np.float32, copy=False)
        cur = cur_img[:h, :w].astype(np.float32, copy=False)

        # Convert multichannel images to grayscale for phase correlation.
        # Phase correlation requires 2D input; for RGB(A) use luminance
        if ref.ndim > 2:
            # Extract luminance from multi-channel image
            if ref.shape[2] == 4:
                # RGBA: use RGB channels only
                ref = ref[..., :3]
            if ref.shape[2] >= 3:
                # RGB or more: convert to luminance using standard weights
                ref = 0.114 * ref[..., 2] + 0.587 * ref[..., 1] + 0.299 * ref[..., 0]
            else:
                # Mono or 1-channel: just squeeze the dimension
                ref = ref[..., 0]

        if cur.ndim > 2:
            # Extract luminance from multi-channel image
            if cur.shape[2] == 4:
                # RGBA: use RGB channels only
                cur = cur[..., :3]
            if cur.shape[2] >= 3:
                # RGB or more: convert to luminance using standard weights
                cur = 0.114 * cur[..., 2] + 0.587 * cur[..., 1] + 0.299 * cur[..., 0]
            else:
                # Mono or 1-channel: just squeeze the dimension
                cur = cur[..., 0]

        # Downscale large images for faster/stabler phase correlation.
        max_dim = max(h, w)
        ds = max(1, int(np.ceil(max_dim / 1400.0)))
        if ds > 1:
            ref = cv2.resize(ref, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
            cur = cv2.resize(cur, (w // ds, h // ds), interpolation=cv2.INTER_AREA)

        # Normalize to reduce illumination sensitivity.
        ref = ref - float(ref.mean())
        cur = cur - float(cur.mean())
        ref_std = float(ref.std()) + 1e-6
        cur_std = float(cur.std()) + 1e-6
        ref = ref / ref_std
        cur = cur / cur_std

        win_y = np.hanning(ref.shape[0]).astype(np.float32)
        win_x = np.hanning(ref.shape[1]).astype(np.float32)
        window = np.outer(win_y, win_x)

        shift, response = cv2.phaseCorrelate(ref * window, cur * window)
        dx = float(shift[0]) * float(ds)
        dy = float(shift[1]) * float(ds)
        response = float(response)

        # max_shift_ratio = float(max_shift_ratio_attr, default_max_shift_ratio))
        if abs(dx) > max_shift_ratio * w or abs(dy) > max_shift_ratio * h:
            return None, response

        # min_response = float(min_response_attr, default_min_response))
        if response < min_response:
            return None, response

        h_t = np.eye(3, dtype=np.float64)
        h_t[0, 2] = dx
        h_t[1, 2] = dy
        return h_t, response

    def _try_nominal_fallback(self, ref_idx, relation, images, global_h):
        """Try nominal transform as fallback when feature matching fails."""
        enable_grid_nominal_fallback = cfg.state.ssm.enable_grid_nominal_fallback
        grid_nominal_score = cfg.state.ssm.grid_guide.grid_nominal_score  # 2.0

        if not enable_grid_nominal_fallback:
            return None
        h_nom = self._nominal_local_transform(images[ref_idx], relation)
        if h_nom is None:
            return None
        # Keep nominal placements intentionally weak so they do not outrank
        # feature- or phase-derived transforms and only blend into uncovered areas.
        score = int(round(grid_nominal_score))
        return global_h[ref_idx] @ h_nom, score

    def _nominal_local_transform(self, ref_img, relation):
        step_x, step_y = self._get_overlap_step(ref_img)

        # Local transform maps current -> reference frame.
        h_local = np.eye(3, dtype=np.float64)
        if relation == "left":
            h_local[0, 2] = step_x
        elif relation == "right":
            h_local[0, 2] = -step_x
        elif relation == "top":
            h_local[1, 2] = step_y
        elif relation == "bottom":
            h_local[1, 2] = -step_y
        else:
            return None
        return h_local

    @staticmethod
    def _get_overlap_step(ref_img):
        """Extract grid overlap parameters and compute step distances.

        Returns: (overlap_x_pct, overlap_y_pct, step_x, step_y)
        """
        if cfg.state.ssm.current_stitch_mode == 'zero-overlap':
            step_x = float(ref_img.shape[1])
            step_y = float(ref_img.shape[0])
        else:
            step_x = float(ref_img.shape[1]) * (1.0 - cfg.state.ssm.overlap_x_pct / 100.0)
            step_y = float(ref_img.shape[0]) * (1.0 - cfg.state.ssm.overlap_y_pct / 100.0)
        return step_x, step_y

    # Shared summary and pixel stats
    def current_summary(self, elapsed: float):
        """Return a (message, color) tuple for logs (works for partial runs)."""
        try:
            total = int(self._progress_total)
            placed = int(self._progress_placed)
            skipped = int(self._progress_skipped)
            subpixel_refinement_count = len(getattr(self, "_subpixel_refinement_keys", ()))
        except Exception:
            total, placed, skipped, subpixel_refinement_count = 0, 0, 0, 0
        source_counts = getattr(self, "_placement_source_counts", {})
        msg = self.format_stitching_summary(
            total=total,
            placed=placed,
            skipped=skipped,
            elapsed=elapsed,
            subpixel_refinement_count=subpixel_refinement_count,
            source_counts=source_counts,
        )
        if placed == total and skipped == 0:
            color = "green"
        elif skipped > 0:
            color = "red"
        else:
            color = ""
        return msg, color

    def _emit_pixel_stats(self, canvas_system, progress_cb):
        # gain_clamp = getattr(cfg.state.psm, 'gain_clamp', None) or (0.7, 1.3)
        coverage = canvas_system.coverage_count
        blended = canvas_system.blended_mask
        if coverage is None or blended is None:
            return
        try:
            out_h, out_w = canvas_system.canvas.shape[:2]
            emit_msg(f"Output image dimensions: {out_w}x{out_h}", "", progress_cb)
        except Exception as exc:
            logger.debug(f"Failed to compute output dimensions for pixel stats: {exc}")
        total = int(np.count_nonzero(coverage > 0))
        if total == 0:
            emit_msg("Pixel stats: no valid pixels to report.", "red", progress_cb)
            return
        single_no_blend = int(np.count_nonzero((coverage == 1) & (~blended)))
        overlap_no_blend = int(np.count_nonzero((coverage >= 2) & (~blended)))
        overlap_blend = int(np.count_nonzero((coverage >= 2) & blended))
        emit_msg(
            "Pixel stats (by final coverage): "
            f"single={single_no_blend} ({single_no_blend * 100.0 / total:.2f}%), "
            f"overlap_no_blend={overlap_no_blend} ({overlap_no_blend * 100.0 / total:.2f}%), "
            f"overlap_blend={overlap_blend} ({overlap_blend * 100.0 / total:.2f}%), "
            f"total={total}",
            "",
            progress_cb,
        )
        try:
            overlap_mask = coverage >= 2
            overlap_thickness = self.blender.estimate_overlap_thickness(overlap_mask)
            feather_radius = self.blender.get_effective_feather_radius(overlap_mask)
            if 0 < overlap_thickness < self.blender.NARROW_OVERLAP_THRESHOLD:
                emit_msg(
                    f"Warning: narrow overlap detected ({overlap_thickness}px < {self.blender.NARROW_OVERLAP_THRESHOLD}px). Seam visibility risk is higher.",
                    "yellow",
                    progress_cb,
                )

            msg = f"Effective overlap thickness: {overlap_thickness}px"
            if feather_radius is not None:
                msg = f"{msg}, feather radius used: {feather_radius}px"

            emit_msg(msg, "", progress_cb)
        except Exception as exc:
            logger.debug(f"Failed to compute overlap thickness diagnostics: {exc}")

    def _apply_illumination_and_blend(
            self,
            canvas_system,
            warped_img,
            warped_mask,
            roi_bounds=None,
            placement_score=None,
            min_blend_score=None
    ):
        """Apply gain compensation and blending to a warped image.

        Shared logic extracted from grid_mode and auto_mode to reduce duplication.

        Args:
            canvas_system: MosaicCanvas instance
            warped_img: Warped image
            warped_mask: Warped mask
            placement_score: (Optional) If < min_blend_score, preserve existing content
            min_blend_score: (Optional) Threshold for placement_score masking

        Returns:
            Updated (canvas, mask) tuple after blending
        """
        # Apply placement score mask if provided (grid mode fallback protection)
        enable_panel_flatten = cfg.state.psm.enable_panel_flatten  # False
        flatten_sigma_px = cfg.state.psm.flatten_sigma_px  # 140
        flatten_strength = cfg.state.psm.flatten_strength  # 0.6
        gain_clamp = getattr(cfg.state.psm, 'gain_clamp', None) or (0.7, 1.3)  # Default if None
        illumination_min_overlap_px = cfg.state.psm.illumination_min_overlap_px or 2_000  # Default if None
        illumination_full_confidence_px = cfg.state.psm.illumination_full_confidence_px or 250_000  # Default if None
        blend_offset_match = cfg.state.psm.blend_offset_match  # False
        blend_offset_clamp = cfg.state.psm.blend_offset_clamp  # 800.0

        if roi_bounds is None:
            canvas_view = canvas_system.canvas
            mask_view = canvas_system.mask
            coverage_view = canvas_system.coverage_count
            blended_view = canvas_system.blended_mask
            seam_view = canvas_system.seam_heatmap
        else:
            x0, y0, x1, y1 = roi_bounds
            canvas_view = canvas_system.canvas[y0:y1, x0:x1]
            mask_view = canvas_system.mask[y0:y1, x0:x1]
            coverage_view = (
                None
                if canvas_system.coverage_count is None
                else canvas_system.coverage_count[y0:y1, x0:x1]
            )
            blended_view = (
                None
                if canvas_system.blended_mask is None
                else canvas_system.blended_mask[y0:y1, x0:x1]
            )
            seam_view = (
                None
                if canvas_system.seam_heatmap is None
                else canvas_system.seam_heatmap[y0:y1, x0:x1]
            )

        if placement_score is not None and min_blend_score is not None:
            if placement_score < min_blend_score:
                warped_mask = warped_mask & (~mask_view.astype(bool))

        # Pre-blend advanced gain compensation.
        # "uniform" and "local" apply photometric correction before blending.
        if self.profile.gain_compensation not in ["none", "simple"]:
            if enable_panel_flatten:
                warped_img = self.normalizer.flatten_low_frequency(
                    warped_img,
                    warped_mask,
                    sigma_px=flatten_sigma_px,
                    strength=flatten_strength,
                )

            # Prefer alpha-aware overlap when both images carry alpha
            # All images from warper are guaranteed to be max 3D
            canvas_has_alpha = (canvas_view.ndim == 3 and canvas_view.shape[2] == 4)
            warped_has_alpha = (warped_img.ndim == 3 and warped_img.shape[2] == 4)

            if canvas_has_alpha and warped_has_alpha:
                ca = canvas_view[..., 3] > 0.9
                ia = warped_img[..., 3] > 0.9
                overlap = ca & ia
            else:
                overlap = (mask_view > 0) & (warped_mask > 0)

            if self.profile.gain_compensation == "local":
                # Local (sub-tile) gain + offset compensation
                tile_size = getattr(self.profile, "local_gain_tile_size", 128)
                gain_map, offset_map = self.local_normalizer.compute_gain_map(
                    canvas_view,
                    warped_img,
                    overlap,
                    tile_size=tile_size
                )
                warped_img = self.local_normalizer.apply_gain(warped_img, gain_map, offset_map)
            else:
                # Uniform gain+offset compensation
                gain, offset = self.normalizer.compute_linear_correction(
                    canvas_view,
                    warped_img,
                    overlap,
                    gain_clamp=gain_clamp,
                    min_overlap_px=illumination_min_overlap_px,
                    full_confidence_px=illumination_full_confidence_px
                )

                if gain <= gain_clamp[0] or gain >= gain_clamp[1]:
                    logger.warning(
                        "Gain correction hit clamp boundary (gain=%.3f, clamp=[%.2f, %.2f]). "
                        "This may indicate a calibration problem or strong illumination gradient.",
                        gain, gain_clamp[0], gain_clamp[1],
                    )

                warped_img = self.normalizer.apply_linear_correction(
                    warped_img,
                    gain,
                    offset
                )

        # Map gain-compensation mode to the blender's simple overlap correction.
        # none: no photometric correction in blender
        # simple: blender applies overlap-based gain, and optional offset matching
        # uniform/local: advanced correction already applied before blending

        if self.profile.gain_compensation not in ["none", "simple"]:
            use_simple_gain = False
            use_simple_offset = False
        elif self.profile.gain_compensation == "simple":
            use_simple_gain = True
            use_simple_offset = blend_offset_match
        else:
            use_simple_gain = False
            use_simple_offset = False

        canvas, mask = self.blender.blend(
            canvas_view,
            mask_view,
            warped_img,
            warped_mask,
            use_gain=use_simple_gain,
            use_offset=use_simple_offset,
            offset_clamp=blend_offset_clamp,
            coverage_count=coverage_view,
            blended_mask=blended_view,
            seam_heatmap=seam_view,
        )
        return canvas, mask

    def _finalize_stitch(self, canvas_system, t_start, progress_cb=None, heal_seams=False):
        """Common post-processing for stitching completion.

        Args:
            canvas_system: MosaicCanvas instance with final result
            t_start: Start time for elapsed calculation
            progress_cb: Progress callback function
            heal_seams: Whether to apply seam gap healing (GridMode only)

        Returns:
            Final stitched canvas image
        """
        if heal_seams:
            canvas_system.canvas = self._heal_internal_seam_gaps(
                canvas_system.canvas,
                canvas_system.mask
            )

        # Remove the safety pad/empty rim so preview/export have no border
        try:
            if hasattr(canvas_system, "crop_to_content"):
                canvas_system.crop_to_content()
        except Exception as exc:
            # Non-fatal: if anything goes wrong, keep uncropped result
            logger.warning(f"Content crop failed; keeping uncropped mosaic: {exc}")

        self.last_canvas_system = canvas_system

        if self.show_pixel_stat:
            self._emit_pixel_stats(canvas_system, progress_cb)

        # Emit final summary
        try:
            elapsed = time.perf_counter() - t_start
            msg, color = self.current_summary(elapsed)
            emit_msg(msg, color, progress_cb)
            timings = getattr(self, "_phase_timing_secs", {}) or {}
            ordered_keys = ["detect", "match", "transform", "subpixel", "warp", "blend"]
            parts = []
            for key in ordered_keys:
                sec = float(timings.get(key, 0.0) or 0.0)
                pct = (100.0 * sec / elapsed) if elapsed > 1e-9 else 0.0
                if sec <= 0.0005:
                    continue
                parts.append(f"{key}={sec:.2f}s ({pct:.1f}%)")
            if parts:
                emit_msg(
                    "Stage timing breakdown: " + ", ".join(parts),
                    "debug",
                    progress_cb,
                )
        except Exception as exc:
            logger.debug(f"Failed to emit final stitch summary: {exc}")

        return canvas_system.canvas

    def _compute_roi_padding(self) -> int:
        """Return a conservative ROI context margin for warp/blend operations.

        ROI-based placement must still include enough surrounding canvas context
        for feathering, multiband pyramids, and local gain estimation.
        """
        pad = 8

        try:
            blend_type = str(getattr(self.blender, "blend_type", "") or "").strip().lower()
        except Exception:
            blend_type = ""

        try:
            feather_px = int(getattr(self.blender, "feather_px", 0) or 0)
        except Exception:
            feather_px = 0

        try:
            multiband_levels = int(getattr(self.blender, "multiband_levels", 0) or 0)
        except Exception:
            multiband_levels = 0

        if blend_type in ("feather", "adaptive-feather"):
            pad = max(pad, max(32, feather_px * 2))
        elif blend_type in ("multiband", "adaptive-multiband"):
            pad = max(pad, max(96, 1 << max(4, min(multiband_levels + 2, MAX_MULTIBAND_LEVELS))))
        elif blend_type == "seamless":
            pad = max(pad, max(160, 1 << max(5, min(multiband_levels + 2, MAX_MULTIBAND_LEVELS))))

        try:
            gain_mode = str(getattr(self.profile, "gain_compensation", "none") or "none").strip().lower()
        except Exception:
            gain_mode = "none"

        if gain_mode == "local":
            try:
                tile_size = int(getattr(self.profile, "local_gain_tile_size", 128) or 128)
            except Exception:
                tile_size = 128
            pad = max(pad, max(128, tile_size * 2))

        return int(pad)

    def _warp_and_blend_single_image(self, canvas_system, image, global_h, image_name, index,
                                     placement_score=None, min_blend_score=None,
                                     max_allowed_size=None,
                                     cancel_cb=None, progress_cb=None) -> bool:
        """Shared logic for warping and blending a single image onto canvas.

        Args:
            canvas_system: MosaicCanvas instance
            image: Image to warp
            global_h: Global homography matrix
            image_name: Name of image for logging
            index: Current image index (for progress message)
            placement_score: (Optional) Score for grid mode fallback protection
            min_blend_score: (Optional) Threshold for placement_score masking
            max_allowed_size: (Optional) Maximum allowed warp bounds size (FreeMode validation)
            cancel_cb: (Optional) Cancellation callback
            progress_cb: (Optional) Progress callback

        Returns:
            True if warp/blend succeeded, False if validation failed
        """
        if cancel_cb is not None:
            self.check_cancel(cancel_cb)

        if cfg.state.cli_mode and index % 5 == 0:
            emit_msg(f"Processed {index} panels", "", progress_cb)

        # Compute warp bounds
        min_x, min_y, max_x, max_y = canvas_system.compute_warp_bounds(image, global_h)

        # Validate bounds (with optional max size limit for FreeMode)
        if max_allowed_size is not None:
            if abs(max_x - min_x) > max_allowed_size or abs(max_y - min_y) > max_allowed_size:
                emit_msg(f"Skipping {os.path.basename(image_name)} (warp bounds too large)", "red", progress_cb)
                try:
                    self._progress_skipped += 1
                except Exception:
                    pass
                return False

        # Expand canvas
        try:
            canvas_system.expand_canvas(min_x, min_y, max_x, max_y)
        except ValueError as exc:
            emit_msg(f"Skipping {os.path.basename(image_name)}: {exc}", "red", progress_cb)
            try:
                self._progress_skipped += 1
            except Exception:
                pass
            return False

        # Adjust homography for canvas offset
        h_adjusted = global_h.copy()
        h_adjusted[0, 2] += canvas_system.offset_x
        h_adjusted[1, 2] += canvas_system.offset_y

        roi_pad = self._compute_roi_padding()
        roi_x0 = max(0, int(np.floor(min_x + canvas_system.offset_x)) - roi_pad)
        roi_y0 = max(0, int(np.floor(min_y + canvas_system.offset_y)) - roi_pad)
        roi_x1 = min(canvas_system.canvas.shape[1], int(np.ceil(max_x + canvas_system.offset_x)) + roi_pad)
        roi_y1 = min(canvas_system.canvas.shape[0], int(np.ceil(max_y + canvas_system.offset_y)) + roi_pad)
        roi_bounds = (roi_x0, roi_y0, roi_x1, roi_y1)

        # Warp image
        try:
            t_warp_start = time.perf_counter()
            warped_img, warped_mask = self.warper.warp_into_roi(image, h_adjusted, roi_bounds)
            self._record_phase_time("warp", time.perf_counter() - t_warp_start)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

        # Apply illumination normalization and blending
        try:
            t_blend_start = time.perf_counter()
            blended_canvas, blended_mask = self._apply_illumination_and_blend(
                canvas_system, warped_img, warped_mask,
                roi_bounds=roi_bounds,
                placement_score=placement_score,
                min_blend_score=min_blend_score
            )
            self._record_phase_time("blend", time.perf_counter() - t_blend_start)
            canvas_system.canvas[roi_y0:roi_y1, roi_x0:roi_x1] = blended_canvas
            canvas_system.mask[roi_y0:roi_y1, roi_x0:roi_x1] = blended_mask

        except Exception:
            import traceback
            traceback.print_exc()
            raise

        # emit_msg(f"Blend panel: {index + 1}/{total} ({image_name})", "", progress_cb)
        try:
            self._progress_placed += 1
        except Exception:
            pass

        return True

    @staticmethod
    def _heal_internal_seam_gaps(canvas, mask, max_iters=14, min_neighbors=1):
        """Fill small gaps/holes inside seams using iterative neighbor averaging.

        This is shared utility for grid mode and potentially other modes.
        """
        known = mask.astype(bool).copy()
        holes = ~known
        if not np.any(holes):
            return canvas

        # Build a conservative "solid disk support" and target seam cracks inside it.
        disk = cv2.morphologyEx(
            known.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ).astype(bool)
        target_region = disk & (~known)
        if not np.any(target_region):
            return canvas

        filled = canvas.astype(np.float32, copy=False)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.float32)

        known_f = known.astype(np.float32)
        for _ in range(max_iters):
            target = target_region & (~known)
            if not np.any(target):
                break

            # For color images, expand known_f to (H,W,1) so multiplication broadcasts per-channel
            if filled.ndim == 3:
                known_exp = known_f[..., None]
            else:
                known_exp = known_f

            neigh_sum = cv2.filter2D(filled * known_exp, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            neigh_cnt = cv2.filter2D(known_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

            can_fill = target & (neigh_cnt >= float(min_neighbors))
            if not np.any(can_fill):
                break

            # Avoid (N,3)/(N,) broadcasting by expanding denominator to channels when needed
            if filled.ndim == 3:
                denom = np.maximum(neigh_cnt, 1e-8)[..., None]
            else:
                denom = np.maximum(neigh_cnt, 1e-8)
            new_vals = neigh_sum / denom
            filled[can_fill] = new_vals[can_fill]
            known[can_fill] = True
            known_f[can_fill] = 1.0

        if np.issubdtype(canvas.dtype, np.integer):
            info = np.iinfo(canvas.dtype)
            filled = np.clip(filled, info.min, info.max)
        return filled.astype(canvas.dtype, copy=False)

    def _sync_random_counters(self, total, skipped):
        """Synchronize progress counters for final summary."""
        try:
            placed = total - skipped
            self._progress_total = total
            self._progress_placed = placed
            self._progress_skipped = skipped
        except Exception:
            pass
