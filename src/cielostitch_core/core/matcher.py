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


class FeatureMatcher:

    # FLANN parameters for SIFT (float32 L2 descriptors)
    _FLANN_INDEX_KDTREE = 1
    _FLANN_INDEX_PARAMS = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
    _FLANN_SEARCH_PARAMS = {"checks": 50}
    _MAX_TRANSLATION_CONSENSUS_POINTS = 512

    def __init__(self, ratio_test=0.75, ransac_thresh=5.0, min_inlier_ratio=0.3, min_inliers=12):
        self.ratio_test = ratio_test
        self.ransac_thresh = ransac_thresh
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_inliers = int(min_inliers)
        self.min_matches_for_transform = max(6, self.min_inliers // 2)
        try:
            self._matcher = cv2.FlannBasedMatcher(self._FLANN_INDEX_PARAMS, self._FLANN_SEARCH_PARAMS)
            self._use_flann = True
        except Exception:
            # Fallback to brute-force if FLANN is unavailable
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
            self._use_flann = False

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

        d1 = d1.astype(np.float32, copy=False) if self._use_flann else d1
        d2 = d2.astype(np.float32, copy=False) if self._use_flann else d2
        matches = self._matcher.knnMatch(d1, d2, k=2)

        good_matches = []
        for match_group in matches:
            if len(match_group) == 2:
                m, n = match_group
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

    def compute_transform(self, kp1, kp2, matches,
                          scale1=1.0, scale2=1.0,
                          allow_rotation=True):

        if len(matches) < self.min_matches_for_transform:
            return None, None

        pts1 = np.float32([
            kp1[m.queryIdx].pt for m in matches
        ]).reshape(-1, 2)

        pts2 = np.float32([
            kp2[m.trainIdx].pt for m in matches
        ]).reshape(-1, 2)

        # Rescale back to full resolution
        pts1 /= scale1
        pts2 /= scale2

        if not allow_rotation:
            return self._estimate_translation_transform(pts1, pts2)

        # Estimate affine transform using RANSAC to filter out outliers
        # Impact: this is usually fine for tracked astro tiles and mild parallax, 
        # but it limits robustness for landscape panoramas with 
        # stronger perspective change or wide FOV.
        m_affine, inliers = cv2.estimateAffinePartial2D(
            pts2,  # map current -> previous
            pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh
        )

        if m_affine is None or inliers is None:
            return None, None

        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        total_matches = inlier_mask.size
        inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None, None

        # Convert 2x3 affine -> 3x3 matrix
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = m_affine

        sx = np.sqrt(h[0, 0] ** 2 + h[1, 0] ** 2)

        # Guard against degenerate transformations
        if sx < 1e-6:
            return None, None

        h[0, 0] /= sx
        h[0, 1] /= sx
        h[1, 0] /= sx
        h[1, 1] /= sx

        # sx = np.sqrt(h[0, 0] ** 2 + h[1, 0] ** 2)
        # h = homography matrix
        return h, inlier_mask
