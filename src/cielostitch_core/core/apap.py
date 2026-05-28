# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from dataclasses import dataclass, field
import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class APAPConfig:
    mesh_cols: int = 16
    mesh_rows: int = 16
    kernel_sigma_px: float = 64.0
    min_local_support: int = 12
    min_valid_mesh_node_ratio: float = 0.25
    use_hard_support_radius: bool = False
    support_radius_factor: float = 2.5
    regularization: float = 1e-3
    fallback_to_global: bool = True


@dataclass(frozen=True)
class APAPBootstrapResult:
    homography: np.ndarray | None
    inlier_mask: np.ndarray | None
    selected_pairs: tuple[tuple[int, int, float], ...] = ()
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class APAPRegistrationResult:
    homography: np.ndarray | None
    inlier_mask: np.ndarray | None
    remap_x: np.ndarray | None = None
    remap_y: np.ndarray | None = None
    support_source_points: np.ndarray | None = None
    support_destination_points: np.ndarray | None = None
    support_residuals: np.ndarray | None = None
    mesh_x: np.ndarray | None = None
    mesh_y: np.ndarray | None = None
    mesh_homographies: np.ndarray | None = None
    # Optional (R, C, 2) array giving the exact canvas-space centre of each mesh
    # node.  Set when the mesh has been composed through a non-separable transform
    # so that mesh_x / mesh_y no longer represent the true node positions.
    mesh_node_positions: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_local_warp(self) -> bool:
        return self.remap_x is not None and self.remap_y is not None

    @property
    def has_mesh(self) -> bool:
        return self.mesh_x is not None and self.mesh_y is not None and self.mesh_homographies is not None


class APAPEstimator:
    """Initial APAP estimator scaffold.

    The first implementation phase only computes a robust global projective
    bootstrap that later APAP-specific local warping can build on.
    """

    def __init__(self, config: APAPConfig | None = None):
        self.config = config or APAPConfig()

    def build_registration_result(self, bootstrap: APAPBootstrapResult) -> APAPRegistrationResult:
        diagnostics = dict(bootstrap.diagnostics or {})
        diagnostics.setdefault("mesh_cols", int(self.config.mesh_cols))
        diagnostics.setdefault("mesh_rows", int(self.config.mesh_rows))
        diagnostics.setdefault("kernel_sigma_px", float(self.config.kernel_sigma_px))
        diagnostics.setdefault("min_local_support", int(self.config.min_local_support))
        diagnostics.setdefault("min_valid_mesh_node_ratio", float(self.config.min_valid_mesh_node_ratio))
        diagnostics.setdefault("use_hard_support_radius", bool(self.config.use_hard_support_radius))
        diagnostics.setdefault("support_radius_factor", float(self.config.support_radius_factor))
        diagnostics.setdefault("local_warp_ready", False)
        diagnostics.setdefault("fallback_to_global", bool(self.config.fallback_to_global))
        return APAPRegistrationResult(
            homography=bootstrap.homography,
            inlier_mask=bootstrap.inlier_mask,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _normalization_transform(points: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            return None
        finite = np.isfinite(points).all(axis=1)
        if not np.any(finite):
            return None
        pts = points[finite]
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        if not np.isfinite(rms) or rms <= 1e-12:
            return None
        scale = float(np.sqrt(2.0) / rms)
        transform = np.array(
            [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return transform

    @staticmethod
    def _is_homography_reasonable(homography: np.ndarray) -> bool:
        homography = np.asarray(homography, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            return False
        if abs(float(homography[2, 2])) <= 1e-10:
            return False

        affine = homography[:2, :2]
        det = float(np.linalg.det(affine))
        if not np.isfinite(det) or abs(det) <= 1e-10:
            return False

        try:
            cond = float(np.linalg.cond(affine))
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(cond) or cond > 1e10:
            return False
        return True

    @staticmethod
    def _weighted_dlt_homography(src_points: np.ndarray, dst_points: np.ndarray, weights: np.ndarray, regularization: float):
        src_points = np.asarray(src_points, dtype=np.float64)
        dst_points = np.asarray(dst_points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if src_points.shape[0] < 4 or dst_points.shape[0] < 4 or weights.shape[0] != src_points.shape[0]:
            return None

        src_norm = APAPEstimator._normalization_transform(src_points)
        dst_norm = APAPEstimator._normalization_transform(dst_points)
        if src_norm is None or dst_norm is None:
            return None

        src_h = np.column_stack([src_points, np.ones((src_points.shape[0],), dtype=np.float64)])
        dst_h = np.column_stack([dst_points, np.ones((dst_points.shape[0],), dtype=np.float64)])
        src_normalized = (src_norm @ src_h.T).T
        dst_normalized = (dst_norm @ dst_h.T).T
        src_xy = src_normalized[:, :2]
        dst_xy = dst_normalized[:, :2]

        rows = []
        for (x, y), (u, v), w in zip(src_xy, dst_xy, weights):
            if not np.isfinite([x, y, u, v, w]).all() or w <= 0.0:
                continue
            sw = float(np.sqrt(w))
            rows.append(sw * np.array([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u], dtype=np.float64))
            rows.append(sw * np.array([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v], dtype=np.float64))
        if len(rows) < 8:
            return None

        matrix = np.stack(rows, axis=0)
        if regularization > 0.0:
            reg = np.sqrt(float(regularization)) * np.eye(9, dtype=np.float64)
            reg[-1, -1] = 0.0
            matrix = np.vstack([matrix, reg])

        try:
            _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        vector = vh[-1]
        if not np.isfinite(vector).all() or abs(float(vector[-1])) < 1e-10:
            return None
        homography = vector.reshape(3, 3)
        try:
            homography = np.linalg.inv(dst_norm) @ homography @ src_norm
        except np.linalg.LinAlgError:
            return None
        homography /= homography[2, 2]
        if not APAPEstimator._is_homography_reasonable(homography):
            return None
        return homography

    def _build_local_homography_mesh(
        self,
        support_source_points: np.ndarray,
        support_destination_points: np.ndarray,
        *,
        diagnostics: dict,
    ):
        support_source_points = np.asarray(support_source_points, dtype=np.float64)
        support_destination_points = np.asarray(support_destination_points, dtype=np.float64)
        if support_source_points.shape[0] < 4 or support_destination_points.shape[0] < 4:
            return None, None, None

        mesh_cols = max(2, int(diagnostics.get("mesh_cols", self.config.mesh_cols) or self.config.mesh_cols))
        mesh_rows = max(2, int(diagnostics.get("mesh_rows", self.config.mesh_rows) or self.config.mesh_rows))
        sigma = max(1.0, float(diagnostics.get("kernel_sigma_px", self.config.kernel_sigma_px) or self.config.kernel_sigma_px))
        min_local_support = max(4, int(diagnostics.get("min_local_support", self.config.min_local_support) or self.config.min_local_support))
        min_valid_mesh_node_ratio = float(
            diagnostics.get("min_valid_mesh_node_ratio", self.config.min_valid_mesh_node_ratio)
            or self.config.min_valid_mesh_node_ratio
        )
        min_valid_mesh_node_ratio = min(max(min_valid_mesh_node_ratio, 0.0), 1.0)
        use_hard_support_radius = bool(
            diagnostics.get("use_hard_support_radius", self.config.use_hard_support_radius)
            if "use_hard_support_radius" in diagnostics
            else self.config.use_hard_support_radius
        )
        support_radius_factor = float(
            diagnostics.get("support_radius_factor", self.config.support_radius_factor)
            or self.config.support_radius_factor
        )
        support_radius_factor = max(0.0, support_radius_factor)
        max_local_neighbors = max(min_local_support, min(48, int(max(min_local_support * 3, 16))))
        support_radius_px = max(sigma * support_radius_factor, 4.0)

        min_x = float(np.min(support_destination_points[:, 0]))
        max_x = float(np.max(support_destination_points[:, 0]))
        min_y = float(np.min(support_destination_points[:, 1]))
        max_y = float(np.max(support_destination_points[:, 1]))
        mesh_x = np.linspace(min_x, max_x, mesh_cols, dtype=np.float64)
        mesh_y = np.linspace(min_y, max_y, mesh_rows, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(mesh_x, mesh_y)
        # NaN-initialise so invalid nodes are unambiguously detectable downstream.
        mesh_h = np.full((mesh_rows, mesh_cols, 3, 3), np.nan, dtype=np.float64)
        valid_nodes = 0
        sigma2 = 2.0 * sigma * sigma

        support_dx = support_destination_points[:, 0]
        support_dy = support_destination_points[:, 1]
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                dx = support_dx - grid_x[row, col]
                dy = support_dy - grid_y[row, col]
                dist2 = (dx * dx) + (dy * dy)
                neighbor_order = np.argsort(dist2)
                if neighbor_order.size > max_local_neighbors:
                    neighbor_order = neighbor_order[:max_local_neighbors]

                if use_hard_support_radius:
                    clipped = neighbor_order[np.sqrt(dist2[neighbor_order]) <= support_radius_px]
                    # Keep locality preference without starving sparse regions.
                    if clipped.size >= min_local_support:
                        neighbor_order = clipped
                if neighbor_order.size < min_local_support:
                    continue
                local_dist2 = dist2[neighbor_order]
                weights = np.exp(-(local_dist2) / sigma2)
                homography = self._weighted_dlt_homography(
                    support_source_points[neighbor_order],
                    support_destination_points[neighbor_order],
                    weights,
                    self.config.regularization,
                )
                if homography is None:
                    continue
                mesh_h[row, col] = homography
                valid_nodes += 1

        total_nodes = mesh_rows * mesh_cols
        min_valid_nodes = max(4, int(math.ceil(total_nodes * min_valid_mesh_node_ratio)))
        diagnostics["mesh_min_valid_node_count"] = int(min_valid_nodes)
        if valid_nodes < min_valid_nodes:
            return None, None, None
        diagnostics["mesh_valid_node_count"] = int(valid_nodes)
        return mesh_x, mesh_y, mesh_h

    @staticmethod
    def _project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        homogeneous = np.concatenate([points, ones], axis=1)
        projected = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
        denom = projected[:, 2:3]
        safe = np.abs(denom) > 1e-8
        out = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
        out[safe[:, 0]] = projected[safe[:, 0], :2] / denom[safe[:, 0]]
        return out

    @staticmethod
    def _compose_mesh_axes(mesh_x: np.ndarray | None, mesh_y: np.ndarray | None, transform: np.ndarray):
        """Project mesh node destination coordinates through a transform.

        Returns the projected axis arrays when the transform is separable
        (pure translation / axis-aligned scale), otherwise returns (None, None)
        to signal that the caller must fall back to node-level composition.
        """
        if mesh_x is None or mesh_y is None:
            return mesh_x, mesh_y

        mesh_x = np.asarray(mesh_x, dtype=np.float64)
        mesh_y = np.asarray(mesh_y, dtype=np.float64)
        if mesh_x.ndim != 1 or mesh_y.ndim != 1 or mesh_x.size == 0 or mesh_y.size == 0:
            return None, None

        # Only the translation/scale components matter for axis-separability.
        # Allow a generous tolerance so small numerical noise in the reference
        # homography does not discard the mesh unnecessarily.
        grid_x, grid_y = np.meshgrid(mesh_x, mesh_y)
        points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
        projected = APAPEstimator._project_points(points, transform).reshape(mesh_y.size, mesh_x.size, 2)
        if not np.isfinite(projected).all():
            return None, None

        projected_x = projected[:, :, 0]
        projected_y = projected[:, :, 1]
        # Use a 1-pixel tolerance to survive sub-pixel residual rotation/shear.
        tolerance = 1.0
        if np.max(np.abs(projected_x - projected_x[:1, :])) > tolerance:
            return None, None
        if np.max(np.abs(projected_y - projected_y[:, :1])) > tolerance:
            return None, None
        return projected_x[0].copy(), projected_y[:, 0].copy()

    @staticmethod
    def _project_single_point(point: np.ndarray | tuple[float, float], homography: np.ndarray) -> np.ndarray | None:
        pt = np.asarray(point, dtype=np.float64).reshape(-1)
        if pt.size != 2:
            return None
        vec = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
        projected = np.asarray(homography, dtype=np.float64) @ vec
        denom = float(projected[2])
        if not np.isfinite(projected).all() or abs(denom) <= 1e-8:
            return None
        return np.array([float(projected[0]) / denom, float(projected[1]) / denom], dtype=np.float64)

    def compose_registration(self, registration: APAPRegistrationResult, transform: np.ndarray) -> APAPRegistrationResult:
        if registration.homography is None:
            return registration
        transform = np.asarray(transform, dtype=np.float64)
        diagnostics = dict(registration.diagnostics or {})
        support_destination_points = registration.support_destination_points
        support_residuals = registration.support_residuals
        if support_destination_points is not None:
            support_destination_points = self._project_points(support_destination_points, transform)
        if support_residuals is not None:
            linear = transform[:2, :2]
            support_residuals = np.asarray(support_residuals, dtype=np.float64) @ linear.T
        # Try to keep separable axes; fall back gracefully for non-separable transforms.
        mesh_x, mesh_y = self._compose_mesh_axes(registration.mesh_x, registration.mesh_y, transform)
        mesh_homographies = registration.mesh_homographies
        if mesh_homographies is not None:
            mesh_homographies = np.asarray(mesh_homographies, dtype=np.float64).copy()
            if mesh_homographies.ndim == 4 and mesh_homographies.shape[2:] == (3, 3):
                for row in range(mesh_homographies.shape[0]):
                    for col in range(mesh_homographies.shape[1]):
                        node_h = mesh_homographies[row, col]
                        if not np.isfinite(node_h).all() or abs(float(node_h[2, 2])) < 1e-10:
                            # Mark invalid nodes unambiguously so the warper skips them.
                            mesh_homographies[row, col] = np.nan
                            continue
                        mesh_homographies[row, col] = transform @ node_h
        # When axes could not be projected separably, derive new axes from the
        # extent of the still-valid composed node destinations.
        node_positions = None
        if (mesh_x is None or mesh_y is None) and mesh_homographies is not None:
            orig_x = registration.mesh_x
            orig_y = registration.mesh_y
            if orig_x is not None and orig_y is not None:
                orig_x = np.asarray(orig_x, dtype=np.float64)
                orig_y = np.asarray(orig_y, dtype=np.float64)
                # Collect destination (col, row) centres for each valid node after composition.
                dest_pts = []
                rows_n, cols_n = mesh_homographies.shape[:2]
                for r in range(rows_n):
                    for c in range(cols_n):
                        node_h = mesh_homographies[r, c]
                        if not np.isfinite(node_h).all() or abs(float(node_h[2, 2])) < 1e-10:
                            continue
                        # Project the original node centre through the composed homography.
                        x = float(orig_x[c])
                        y = float(orig_y[r])
                        dest_pt = self._project_single_point((x, y), transform)
                        if dest_pt is not None:
                            dest_pts.append((float(dest_pt[0]), float(dest_pt[1])))
                if len(dest_pts) >= 2:
                    dest_arr = np.array(dest_pts, dtype=np.float64)
                    mesh_x = np.linspace(float(np.min(dest_arr[:, 0])), float(np.max(dest_arr[:, 0])), orig_x.size, dtype=np.float64)
                    mesh_y = np.linspace(float(np.min(dest_arr[:, 1])), float(np.max(dest_arr[:, 1])), orig_y.size, dtype=np.float64)
                    # Build exact node positions (R, C, 2) in new canvas space so
                    # the remap can use the true centre of each node rather than
                    # the uniform bounding-box grid.
                    node_positions = np.full((orig_y.size, orig_x.size, 2), np.nan, dtype=np.float64)
                    rows_n2, cols_n2 = mesh_homographies.shape[:2]
                    for r2 in range(rows_n2):
                        for c2 in range(cols_n2):
                            if r2 >= orig_y.size or c2 >= orig_x.size:
                                continue
                            node_h2 = mesh_homographies[r2, c2]
                            if not np.isfinite(node_h2).all() or abs(float(node_h2[2, 2])) < 1e-10:
                                continue
                            ox = float(orig_x[c2])
                            oy = float(orig_y[r2])
                            dest_pt = self._project_single_point((ox, oy), transform)
                            if dest_pt is not None:
                                node_positions[r2, c2, 0] = float(dest_pt[0])
                                node_positions[r2, c2, 1] = float(dest_pt[1])
        return APAPRegistrationResult(
            homography=transform @ np.asarray(registration.homography, dtype=np.float64),
            inlier_mask=registration.inlier_mask,
            remap_x=None,
            remap_y=None,
            support_source_points=registration.support_source_points,
            support_destination_points=support_destination_points,
            support_residuals=support_residuals,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_homographies=mesh_homographies if (mesh_x is not None and mesh_y is not None) else None,
            mesh_node_positions=node_positions if (node_positions is not None and mesh_x is not None) else None,
            diagnostics=diagnostics,
        )

    def with_canvas_offset(self, registration: APAPRegistrationResult, offset_x: float, offset_y: float) -> APAPRegistrationResult:
        if registration.homography is None:
            return registration
        translate = np.eye(3, dtype=np.float64)
        translate[0, 2] = float(offset_x)
        translate[1, 2] = float(offset_y)
        translated_h = translate @ np.asarray(registration.homography, dtype=np.float64)
        support_destination_points = registration.support_destination_points
        if support_destination_points is not None:
            support_destination_points = np.asarray(support_destination_points, dtype=np.float64).copy()
            support_destination_points[:, 0] += float(offset_x)
            support_destination_points[:, 1] += float(offset_y)
        mesh_x = registration.mesh_x
        mesh_y = registration.mesh_y
        mesh_homographies = registration.mesh_homographies
        if mesh_x is not None:
            mesh_x = np.asarray(mesh_x, dtype=np.float64).copy() + float(offset_x)
        if mesh_y is not None:
            mesh_y = np.asarray(mesh_y, dtype=np.float64).copy() + float(offset_y)
        if mesh_homographies is not None:
            mesh_homographies = np.asarray(mesh_homographies, dtype=np.float64).copy()
            if mesh_homographies.ndim == 4 and mesh_homographies.shape[2:] == (3, 3):
                for row in range(mesh_homographies.shape[0]):
                    for col in range(mesh_homographies.shape[1]):
                        homography = mesh_homographies[row, col]
                        if not np.isfinite(homography).all() or abs(float(homography[2, 2])) < 1e-10:
                            continue
                        mesh_homographies[row, col] = translate @ homography
        node_positions = registration.mesh_node_positions
        if node_positions is not None:
            node_positions = np.asarray(node_positions, dtype=np.float64).copy()
            valid_pos = np.isfinite(node_positions).all(axis=2)
            node_positions[..., 0][valid_pos] += float(offset_x)
            node_positions[..., 1][valid_pos] += float(offset_y)
        return APAPRegistrationResult(
            homography=translated_h,
            inlier_mask=registration.inlier_mask,
            remap_x=None,
            remap_y=None,
            support_source_points=registration.support_source_points,
            support_destination_points=support_destination_points,
            support_residuals=registration.support_residuals,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_homographies=mesh_homographies,
            mesh_node_positions=node_positions,
            diagnostics=dict(registration.diagnostics or {}),
        )

    @staticmethod
    def _prepare_points(kp1, kp2, valid_pairs, scale1, scale2):
        pts1 = np.float32([
            kp1[query_idx].pt for query_idx, _train_idx, _distance in valid_pairs
        ]).reshape(-1, 2)
        pts2 = np.float32([
            kp2[train_idx].pt for _query_idx, train_idx, _distance in valid_pairs
        ]).reshape(-1, 2)
        pts1 /= scale1
        pts2 /= scale2
        return pts1, pts2

    def compute_bootstrap(
        self,
        kp1,
        kp2,
        valid_pairs,
        *,
        scale1: float,
        scale2: float,
        ransac_thresh: float,
        max_matches: int,
    ) -> APAPBootstrapResult:
        projective_pairs = list(valid_pairs)
        if len(projective_pairs) > max_matches:
            projective_pairs = sorted(projective_pairs, key=lambda pair: pair[2])[:max_matches]

        pts1, pts2 = self._prepare_points(kp1, kp2, projective_pairs, scale1, scale2)
        if not np.isfinite(pts1).all() or not np.isfinite(pts2).all():
            logger.debug("APAP bootstrap rejected: reason=point_nonfinite_after_scaling")
            return APAPBootstrapResult(
                homography=None,
                inlier_mask=None,
                selected_pairs=tuple(projective_pairs),
                diagnostics={"stage": "bootstrap", "failure_reason": "point_nonfinite_after_scaling"},
            )

        homography, inliers = cv2.findHomography(
            pts2,
            pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=1000,
            confidence=0.99,
        )
        if homography is None or inliers is None:
            logger.debug("APAP bootstrap rejected: reason=homography_estimation_failed")
            return APAPBootstrapResult(
                homography=None,
                inlier_mask=None,
                selected_pairs=tuple(projective_pairs),
                diagnostics={"stage": "bootstrap", "failure_reason": "homography_estimation_failed"},
            )

        return APAPBootstrapResult(
            homography=homography,
            inlier_mask=inliers.ravel().astype(bool),
            selected_pairs=tuple(projective_pairs),
            diagnostics={
                "stage": "bootstrap",
                "bootstrap_mode": "global_homography",
                "selected_match_count": int(len(projective_pairs)),
                "mesh_cols": int(self.config.mesh_cols),
                "mesh_rows": int(self.config.mesh_rows),
                "kernel_sigma_px": float(self.config.kernel_sigma_px),
                "min_local_support": int(self.config.min_local_support),
            },
        )

    def estimate_registration(
        self,
        kp1,
        kp2,
        valid_pairs,
        *,
        scale1: float,
        scale2: float,
        ransac_thresh: float,
        max_matches: int,
    ) -> APAPRegistrationResult:
        bootstrap = self.compute_bootstrap(
            kp1,
            kp2,
            valid_pairs,
            scale1=scale1,
            scale2=scale2,
            ransac_thresh=ransac_thresh,
            max_matches=max_matches,
        )
        result = self.build_registration_result(bootstrap)
        if bootstrap.homography is None or bootstrap.inlier_mask is None:
            return result

        selected_pairs = list(bootstrap.selected_pairs)
        if len(selected_pairs) == 0:
            return result

        pts1, pts2 = self._prepare_points(kp1, kp2, selected_pairs, scale1, scale2)
        projected = self._project_points(pts2, bootstrap.homography)
        residuals = pts1 - projected
        valid_mask = np.isfinite(projected).all(axis=1)
        valid_mask &= np.asarray(bootstrap.inlier_mask, dtype=bool)

        support_source_points = pts2[valid_mask]
        support_destination_points = pts1[valid_mask]
        support_residuals = residuals[valid_mask]

        diagnostics = dict(result.diagnostics or {})
        support_point_count = int(support_destination_points.shape[0])
        min_local_support = int(self.config.min_local_support)
        diagnostics["support_point_count"] = support_point_count
        diagnostics["selected_match_count"] = int(len(selected_pairs))
        diagnostics["local_warp_ready"] = bool(support_point_count >= min_local_support)
        diagnostics["fallback_reason"] = "insufficient_local_support" if support_point_count < min_local_support else ""
        mesh_x = mesh_y = mesh_homographies = None
        if diagnostics["local_warp_ready"]:
            mesh_x, mesh_y, mesh_homographies = self._build_local_homography_mesh(
                support_source_points,
                support_destination_points,
                diagnostics=diagnostics,
            )
            if mesh_homographies is None:
                diagnostics["fallback_reason"] = "insufficient_mesh_support"
        return APAPRegistrationResult(
            homography=result.homography,
            inlier_mask=result.inlier_mask,
            support_source_points=support_source_points,
            support_destination_points=support_destination_points,
            support_residuals=support_residuals,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_homographies=mesh_homographies,
            diagnostics=diagnostics,
        )