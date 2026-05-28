# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

# Feature matching logic

import cv2
import numpy as np
import threading
import logging

from .apap import APAPConfig, APAPEstimator, APAPRegistrationResult

logger = logging.getLogger(__name__)


class FeatureMatcher:

    # FLANN parameters for SIFT (float32 L2 descriptors)
    _FLANN_INDEX_KDTREE = 1
    _FLANN_INDEX_PARAMS = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
    _FLANN_SEARCH_PARAMS = {"checks": 50}
    _MAX_TRANSLATION_CONSENSUS_POINTS = 512
    _MAX_HOMOGRAPHY_MATCHES = 256
    _MAX_HOMOGRAPHY_AXIS_GROWTH = 8.0
    _MAX_HOMOGRAPHY_LINEAR_COND = 1e4

    def __init__(
        self,
        ratio_test=0.75,
        ransac_thresh=5.0,
        min_inlier_ratio=0.3,
        min_inliers=12,
        homography_max_matches=None,
        homography_axis_growth=None,
        homography_linear_cond_max=None,
        homography_allow_affine_fallback=False,
        apap_config=None,
    ):
        self.ratio_test = ratio_test
        self.ransac_thresh = ransac_thresh
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_inliers = int(min_inliers)
        self.min_matches_for_transform = max(6, self.min_inliers // 2)
        self.homography_max_matches = int(homography_max_matches or self._MAX_HOMOGRAPHY_MATCHES)
        self.homography_axis_growth = float(homography_axis_growth or self._MAX_HOMOGRAPHY_AXIS_GROWTH)
        self.homography_linear_cond_max = float(homography_linear_cond_max or self._MAX_HOMOGRAPHY_LINEAR_COND)
        self.homography_allow_affine_fallback = bool(homography_allow_affine_fallback)
        self.apap_estimator = APAPEstimator(config=apap_config if isinstance(apap_config, APAPConfig) else None)
        self._matcher_lock = threading.Lock()
        try:
            self._matcher = cv2.FlannBasedMatcher(self._FLANN_INDEX_PARAMS, self._FLANN_SEARCH_PARAMS)
            self._use_flann = True
        except Exception:
            # Fallback to brute-force if FLANN is unavailable
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
            self._use_flann = False
        self.last_transform_diagnostics = {}

    def _set_last_transform_diagnostics(self, **kwargs):
        self.last_transform_diagnostics = dict(kwargs)

    def match(self, kp1, des1, kp2, des2):
        """
        Perform knn matching + Lowe ratio test.
        """
        if des1 is None or des2 is None:
            return []

        d1 = np.asarray(des1)
        d2 = np.asarray(des2)

        # knnMatch expects 2D descriptor matrices and FLANN requires k <= train rows.
        if d1.ndim != 2 or d2.ndim != 2:
            return []
        if d1.shape[0] == 0 or d2.shape[0] < 2:
            return []

        # Always use float32 for L2 distance consistency across FLANN/BF backends.
        d1 = d1.astype(np.float32, copy=False)
        d2 = d2.astype(np.float32, copy=False)
        if not np.isfinite(d1).all() or not np.isfinite(d2).all():
            return []
        try:
            with self._matcher_lock:
                matches = self._matcher.knnMatch(d1, d2, k=2)
        except cv2.error:
            logger.debug("match rejected: reason=knn_match_failed", exc_info=True)
            return []

        good_matches = []
        for match_group in matches:
            if match_group is None or len(match_group) < 2:
                continue
            m, n = match_group[:2]
            if m is None or n is None:
                continue
            if m.distance < self.ratio_test * n.distance:
                good_matches.append(m)

        return good_matches

    def describe_pair_evidence(self, feature_metrics):
        """Classify pairwise registration evidence for developer diagnostics.

        The returned labels are heuristic and intended for debugging and policy
        decisions, not user-facing guarantees about image content.
        """
        diagnostics = dict(feature_metrics or {})

        min_kp = int(diagnostics.get("min_kp", 0) or 0)
        match_count = int(diagnostics.get("match_count", 0) or 0)
        inlier_count = int(diagnostics.get("inlier_count", 0) or 0)
        has_descriptors = bool(diagnostics.get("has_descriptors", False))
        transform_found = bool(diagnostics.get("transform_found", False))

        min_matches_for_transform = int(self.min_matches_for_transform or 6)
        sparse_keypoint_floor = max(24, min_matches_for_transform * 4)
        weak_match_floor = max(min_matches_for_transform, 8)
        ambiguous_match_floor = max(min_matches_for_transform + 2, 12)
        required_inliers = int(self.min_inliers or 0)
        required_ratio = float(self.min_inlier_ratio or 0.0)
        inlier_ratio = (float(inlier_count) / float(match_count)) if match_count > 0 else 0.0

        diagnostics["inlier_ratio"] = inlier_ratio

        if not has_descriptors:
            diagnostics["diagnostic_label"] = "descriptor_missing"
            diagnostics["failure_stage"] = "descriptors"
            return diagnostics

        if min_kp < sparse_keypoint_floor:
            diagnostics["diagnostic_label"] = "sparse_features"
            diagnostics["failure_stage"] = "detection"
            return diagnostics

        if match_count == 0:
            diagnostics["diagnostic_label"] = "no_distinctive_matches"
            diagnostics["failure_stage"] = "matching"
            return diagnostics

        if match_count < weak_match_floor:
            diagnostics["diagnostic_label"] = "weak_pair_evidence"
            diagnostics["failure_stage"] = "matching"
            return diagnostics

        if not transform_found:
            diagnostics["diagnostic_label"] = (
                "ambiguous_or_inconsistent"
                if match_count >= ambiguous_match_floor
                else "insufficient_geometric_support"
            )
            diagnostics["failure_stage"] = "geometry"
            return diagnostics

        if inlier_count < required_inliers or inlier_ratio < required_ratio:
            diagnostics["diagnostic_label"] = "partial_geometric_support"
            diagnostics["failure_stage"] = "geometry"
            return diagnostics

        diagnostics["diagnostic_label"] = "strong_pair"
        diagnostics["failure_stage"] = "none"
        return diagnostics

    def _estimate_translation_transform(self, pts1, pts2):
        if pts1 is None or pts2 is None or len(pts1) != len(pts2):
            return None, None

        displacements = pts1 - pts2
        if displacements.size == 0:
            return None, None

        total_points = int(displacements.shape[0])
        if total_points <= 0:
            return None, None

        sample_indices = np.arange(total_points, dtype=np.int32)
        max_points = int(self._MAX_TRANSLATION_CONSENSUS_POINTS)
        if total_points > max_points:
            sample_indices = np.linspace(0, total_points - 1, max_points, dtype=np.int32)
            sample_indices = np.unique(sample_indices)
        sampled_displacements = displacements[sample_indices]

        # Vectorised RANSAC: compute all pairwise residuals in one NumPy call.
        # Candidate search is bounded to sampled points to avoid O(N^2) memory.
        diff = sampled_displacements[:, None, :] - sampled_displacements[None, :, :]
        pairwise_residuals = np.linalg.norm(diff, axis=2)
        inlier_matrix = pairwise_residuals <= self.ransac_thresh
        inlier_counts = inlier_matrix.sum(axis=1)

        best_inlier_count = int(inlier_counts.max())
        if best_inlier_count <= 0:
            return None, None

        # Break ties by lowest median inlier error (preserves original selection logic)
        tied = np.where(inlier_counts == best_inlier_count)[0]
        best_error = np.inf
        best_idx = int(tied[0])
        for i in tied:
            err = float(np.median(pairwise_residuals[i, inlier_matrix[i]]))
            if err < best_error:
                best_error = err
                best_idx = int(i)

        consensus_center = sampled_displacements[best_idx]
        full_residuals = np.linalg.norm(displacements - consensus_center, axis=1)
        best_inlier_mask = full_residuals <= self.ransac_thresh
        best_inlier_count = int(np.count_nonzero(best_inlier_mask))

        inlier_ratio = best_inlier_count / float(total_points)
        if best_inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None, None

        dx = float(np.median(displacements[best_inlier_mask, 0]))
        dy = float(np.median(displacements[best_inlier_mask, 1]))

        h = np.eye(3, dtype=np.float64)
        h[0, 2] = dx
        h[1, 2] = dy
        return h, best_inlier_mask.astype(bool)

    def _meets_inlier_thresholds(self, inlier_mask):
        inlier_count = int(np.count_nonzero(inlier_mask))
        total_matches = int(inlier_mask.size)
        inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0
        return inlier_count >= self.min_inliers and inlier_ratio >= self.min_inlier_ratio

    def _is_homography_plausible(self, H, src_points):
        if H is None or src_points is None:
            return False
        if not np.isfinite(H).all():
            return False
        if src_points.ndim != 2 or src_points.shape[0] < 4 or src_points.shape[1] != 2:
            return False

        src_xy = np.asarray(src_points, dtype=np.float32)
        dst_xy = cv2.perspectiveTransform(src_xy.reshape(-1, 1, 2), H).reshape(-1, 2)
        if not np.isfinite(dst_xy).all():
            return False

        linear = np.asarray(H[:2, :2], dtype=np.float64)
        if not np.isfinite(linear).all():
            return False
        if abs(float(np.linalg.det(linear))) < 1e-8:
            return False
        try:
            cond = float(np.linalg.cond(linear))
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(cond) or cond > self.homography_linear_cond_max:
            return False

        src_w = float(np.max(src_xy[:, 0]) - np.min(src_xy[:, 0]))
        src_h = float(np.max(src_xy[:, 1]) - np.min(src_xy[:, 1]))
        dst_w = float(np.max(dst_xy[:, 0]) - np.min(dst_xy[:, 0]))
        dst_h = float(np.max(dst_xy[:, 1]) - np.min(dst_xy[:, 1]))

        # If one axis is narrow, still enforce plausibility on the other axis.
        # Only skip the area-growth gate when the feature spread is narrow on both axes.
        if src_w < 32.0 and src_h < 32.0:
            return True

        growth = float(self.homography_axis_growth)
        if dst_w > src_w * growth or dst_h > src_h * growth:
            return False

        if dst_w < src_w / growth or dst_h < src_h / growth:
            return False

        return True

    def compute_transform(self, kp1, kp2, matches,
                          scale1=1.0, scale2=1.0,
                          transform_mode="affine"):
        """
        Estimate geometric transform between two keypoint sets.
        transform_mode: "affine" (default), "homography", "apap", or "translation"
        """
        normalized_mode = str(transform_mode or "affine").strip().lower()
        self._set_last_transform_diagnostics(
            requested_mode=normalized_mode,
            resolved_mode=None,
            rejected_mode=None,
            rejection_reason=None,
            used_fallback=False,
            fallback_mode=None,
        )
        if len(matches) < self.min_matches_for_transform:
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejection_reason="insufficient_matches",
            )
            logger.debug(
                "compute_transform rejected: reason=insufficient_matches matches=%d required=%d mode=%s",
                len(matches),
                self.min_matches_for_transform,
                normalized_mode,
            )
            return None, None
        if not np.isfinite(scale1) or not np.isfinite(scale2):
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejection_reason="scale_nonfinite",
            )
            logger.debug(
                "compute_transform rejected: reason=scale_nonfinite scale1=%s scale2=%s mode=%s",
                scale1,
                scale2,
                normalized_mode,
            )
            return None, None
        if scale1 <= 0 or scale2 <= 0:
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejection_reason="scale_invalid",
            )
            logger.debug(
                "compute_transform rejected: reason=scale_invalid scale1=%s scale2=%s mode=%s",
                scale1,
                scale2,
                normalized_mode,
            )
            return None, None

        valid_pairs = []
        invalid_index_count = 0
        kp1_len = len(kp1)
        kp2_len = len(kp2)
        for match in matches:
            query_idx = int(getattr(match, "queryIdx", -1))
            train_idx = int(getattr(match, "trainIdx", -1))
            if query_idx < 0 or query_idx >= kp1_len or train_idx < 0 or train_idx >= kp2_len:
                invalid_index_count += 1
                continue
            valid_pairs.append((query_idx, train_idx, float(getattr(match, "distance", np.inf))))

        if invalid_index_count > 0:
            logger.debug(
                "compute_transform dropped invalid matches: dropped=%d total=%d mode=%s",
                invalid_index_count,
                len(matches),
                normalized_mode,
            )
        if len(valid_pairs) < self.min_matches_for_transform:
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejection_reason="insufficient_valid_matches",
            )
            logger.debug(
                "compute_transform rejected: reason=insufficient_valid_matches valid=%d required=%d mode=%s",
                len(valid_pairs),
                self.min_matches_for_transform,
                normalized_mode,
            )
            return None, None

        pts1 = np.float32([
            kp1[query_idx].pt for query_idx, _train_idx, _distance in valid_pairs
        ]).reshape(-1, 2)

        pts2 = np.float32([
            kp2[train_idx].pt for _query_idx, train_idx, _distance in valid_pairs
        ]).reshape(-1, 2)

        # Rescale back to full resolution
        pts1 /= scale1
        pts2 /= scale2
        if not np.isfinite(pts1).all() or not np.isfinite(pts2).all():
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejection_reason="point_nonfinite_after_scaling",
            )
            logger.debug("compute_transform rejected: reason=point_nonfinite_after_scaling mode=%s", normalized_mode)
            return None, None

        if normalized_mode == "translation":
            h, inliers = self._estimate_translation_transform(pts1, pts2)
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                resolved_mode="translation" if h is not None and inliers is not None else None,
                rejection_reason=None if h is not None and inliers is not None else "translation_estimation_failed",
            )
            return h, inliers

        if normalized_mode == "apap":
            registration = self.compute_apap_registration(
                kp1,
                kp2,
                valid_pairs,
                scale1=scale1,
                scale2=scale2,
                ransac_thresh=self.ransac_thresh,
                max_matches=self.homography_max_matches,
            )
            if registration.homography is not None and registration.inlier_mask is not None:
                if not self._meets_inlier_thresholds(registration.inlier_mask):
                    self._set_last_transform_diagnostics(
                        requested_mode=normalized_mode,
                        rejected_mode="apap",
                        rejection_reason="apap_bootstrap_inlier_threshold",
                    )
                    logger.debug("compute_transform rejected: reason=apap_bootstrap_inlier_threshold mode=%s", normalized_mode)
                    return None, None
                if not self._is_homography_plausible(registration.homography, pts2):
                    self._set_last_transform_diagnostics(
                        requested_mode=normalized_mode,
                        rejected_mode="apap",
                        rejection_reason="apap_bootstrap_implausible",
                    )
                    logger.debug("compute_transform rejected: reason=apap_bootstrap_implausible mode=%s", normalized_mode)
                    return None, None
                self._set_last_transform_diagnostics(
                    requested_mode=normalized_mode,
                    resolved_mode="apap",
                )
                return registration.homography, registration.inlier_mask

            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejected_mode="apap",
                rejection_reason="apap_bootstrap_failed",
            )
            logger.debug("compute_transform rejected: reason=apap_bootstrap_failed mode=%s", normalized_mode)
            if not self.homography_allow_affine_fallback:
                return None, None
            logger.debug("compute_transform degraded: %s->affine_fallback", normalized_mode)

        # Homography mode (projective)
        if normalized_mode == "homography":
            homography_pairs = valid_pairs
            if len(homography_pairs) > self.homography_max_matches:
                homography_pairs = sorted(homography_pairs, key=lambda pair: pair[2])[:self.homography_max_matches]
                pts1 = np.float32([
                    kp1[query_idx].pt for query_idx, _train_idx, _distance in homography_pairs
                ]).reshape(-1, 2)
                pts2 = np.float32([
                    kp2[train_idx].pt for _query_idx, train_idx, _distance in homography_pairs
                ]).reshape(-1, 2)
                pts1 /= scale1
                pts2 /= scale2

            support_points = []
            for keypoint in kp2:
                pt = getattr(keypoint, "pt", None)
                if pt is None or len(pt) != 2:
                    continue
                x = float(pt[0]) / float(scale2)
                y = float(pt[1]) / float(scale2)
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                support_points.append((x, y))
            if len(support_points) >= 4:
                plausibility_points = np.asarray(support_points, dtype=np.float32)
            else:
                plausibility_points = pts2

            # Use RANSAC for robust estimation
            H, inliers = cv2.findHomography(
                pts2,  # map current -> previous
                pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=1000,
                confidence=0.99,
            )
            if H is None or inliers is None:
                self._set_last_transform_diagnostics(
                    requested_mode=normalized_mode,
                    rejected_mode="homography",
                    rejection_reason="homography_estimation_failed",
                )
                logger.debug("compute_transform rejected: reason=homography_estimation_failed mode=%s", normalized_mode)
            else:
                inlier_mask = inliers.ravel().astype(bool)
                if not self._meets_inlier_thresholds(inlier_mask):
                    self._set_last_transform_diagnostics(
                        requested_mode=normalized_mode,
                        rejected_mode="homography",
                        rejection_reason="homography_inlier_threshold",
                    )
                    logger.debug("compute_transform rejected: reason=homography_inlier_threshold mode=%s", normalized_mode)
                elif not self._is_homography_plausible(H, plausibility_points):
                    self._set_last_transform_diagnostics(
                        requested_mode=normalized_mode,
                        rejected_mode="homography",
                        rejection_reason="homography_implausible",
                    )
                    logger.debug("compute_transform rejected: reason=homography_implausible mode=%s", normalized_mode)
                else:
                    self._set_last_transform_diagnostics(
                        requested_mode=normalized_mode,
                        resolved_mode="homography",
                    )
                    return H, inlier_mask

            if not self.homography_allow_affine_fallback:
                return None, None
            logger.debug("compute_transform degraded: %s->affine_fallback", normalized_mode)

        # Affine mode (default)
        m_affine, inliers = cv2.estimateAffinePartial2D(
            pts2,  # map current -> previous
            pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh
        )
        if m_affine is None or inliers is None:
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejected_mode="affine" if normalized_mode == "affine" else self.last_transform_diagnostics.get("rejected_mode"),
                rejection_reason="affine_estimation_failed",
                used_fallback=bool(normalized_mode in {"homography", "apap"}),
                fallback_mode="affine" if normalized_mode in {"homography", "apap"} else None,
            )
            logger.debug("compute_transform rejected: reason=affine_estimation_failed")
            return None, None
        if not np.isfinite(m_affine).all():
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejected_mode="affine" if normalized_mode == "affine" else self.last_transform_diagnostics.get("rejected_mode"),
                rejection_reason="affine_nonfinite",
                used_fallback=bool(normalized_mode in {"homography", "apap"}),
                fallback_mode="affine" if normalized_mode in {"homography", "apap"} else None,
            )
            logger.debug("compute_transform rejected: reason=affine_nonfinite")
            return None, None
        inlier_mask = inliers.ravel().astype(bool)
        if not self._meets_inlier_thresholds(inlier_mask):
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejected_mode="affine" if normalized_mode == "affine" else self.last_transform_diagnostics.get("rejected_mode"),
                rejection_reason="affine_inlier_threshold",
                used_fallback=bool(normalized_mode in {"homography", "apap"}),
                fallback_mode="affine" if normalized_mode in {"homography", "apap"} else None,
            )
            logger.debug("compute_transform rejected: reason=affine_inlier_threshold")
            return None, None
        # Convert 2x3 affine -> 3x3 matrix
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = m_affine
        # Guard against degenerate transforms while preserving estimated scale.
        if abs(np.linalg.det(h[:2, :2])) < 1e-8:
            self._set_last_transform_diagnostics(
                requested_mode=normalized_mode,
                rejected_mode="affine" if normalized_mode == "affine" else self.last_transform_diagnostics.get("rejected_mode"),
                rejection_reason="affine_degenerate",
                used_fallback=bool(normalized_mode in {"homography", "apap"}),
                fallback_mode="affine" if normalized_mode in {"homography", "apap"} else None,
            )
            logger.debug("compute_transform rejected: reason=affine_degenerate")
            return None, None
        self._set_last_transform_diagnostics(
            requested_mode=normalized_mode,
            resolved_mode="affine",
            rejected_mode=self.last_transform_diagnostics.get("rejected_mode"),
            rejection_reason=self.last_transform_diagnostics.get("rejection_reason"),
            used_fallback=bool(normalized_mode in {"homography", "apap"}),
            fallback_mode="affine" if normalized_mode in {"homography", "apap"} else None,
        )
        return h, inlier_mask

    def compute_apap_registration(
        self,
        kp1,
        kp2,
        valid_pairs,
        *,
        scale1=1.0,
        scale2=1.0,
        ransac_thresh=None,
        max_matches=None,
    ) -> APAPRegistrationResult:
        return self.apap_estimator.estimate_registration(
            kp1,
            kp2,
            valid_pairs,
            scale1=scale1,
            scale2=scale2,
            ransac_thresh=float(self.ransac_thresh if ransac_thresh is None else ransac_thresh),
            max_matches=int(self.homography_max_matches if max_matches is None else max_matches),
        )
