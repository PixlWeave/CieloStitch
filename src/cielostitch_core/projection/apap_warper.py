# SPDX-License-Identifier: MIT
#
# APAP-specific warping and remap logic for CieloStitch core.
#
# Copyright (c) 2026 Debasish Saha
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# --- APAP mesh and remap logic extracted from warper.py ---

def build_apap_mesh_remap(registration, output_shape, roi_origin):
    height, width = output_shape
    origin_x, origin_y = roi_origin
    support_threshold = 0.5
    mesh_x = registration.mesh_x
    mesh_y = registration.mesh_y
    mesh_h = registration.mesh_homographies
    if mesh_x is None or mesh_y is None or mesh_h is None:
        return None, None

    mesh_x = np.asarray(mesh_x, dtype=np.float64)
    mesh_y = np.asarray(mesh_y, dtype=np.float64)
    mesh_h = np.asarray(mesh_h, dtype=np.float64)
    if mesh_h.ndim != 4 or mesh_h.shape[2:] != (3, 3):
        return None, None
    if mesh_x.size < 2 or mesh_y.size < 2:
        return None, None

    node_src_x = np.full((mesh_y.size, mesh_x.size), np.nan, dtype=np.float64)
    node_src_y = np.full((mesh_y.size, mesh_x.size), np.nan, dtype=np.float64)
    node_positions = getattr(registration, "mesh_node_positions", None)
    if node_positions is not None:
        node_positions = np.asarray(node_positions, dtype=np.float64)
        if node_positions.shape != (mesh_y.size, mesh_x.size, 2):
            node_positions = None
    for row in range(mesh_y.size):
        for col in range(mesh_x.size):
            homography = mesh_h[row, col]
            if not np.isfinite(homography).all() or abs(float(homography[2, 2])) < 1e-10:
                continue
            try:
                inv_h = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                continue
            if node_positions is not None and np.isfinite(node_positions[row, col]).all():
                x = float(node_positions[row, col, 0])
                y = float(node_positions[row, col, 1])
            else:
                x = float(mesh_x[col])
                y = float(mesh_y[row])
            denom = (inv_h[2, 0] * x) + (inv_h[2, 1] * y) + inv_h[2, 2]
            if abs(float(denom)) <= 1e-8:
                continue
            node_src_x[row, col] = ((inv_h[0, 0] * x) + (inv_h[0, 1] * y) + inv_h[0, 2]) / denom
            node_src_y[row, col] = ((inv_h[1, 0] * x) + (inv_h[1, 1] * y) + inv_h[1, 2]) / denom

    valid_nodes = np.isfinite(node_src_x) & np.isfinite(node_src_y)
    if int(np.count_nonzero(valid_nodes)) < 4:
        return None, None
    if abs(float(mesh_x[-1] - mesh_x[0])) <= 1e-8 or abs(float(mesh_y[-1] - mesh_y[0])) <= 1e-8:
        return None, None

    abs_x, abs_y = np.meshgrid(
        np.arange(width, dtype=np.float32) + float(origin_x),
        np.arange(height, dtype=np.float32) + float(origin_y),
    )
    map_cols = ((abs_x - float(mesh_x[0])) / float(mesh_x[-1] - mesh_x[0])) * float(mesh_x.size - 1)
    map_rows = ((abs_y - float(mesh_y[0])) / float(mesh_y[-1] - mesh_y[0])) * float(mesh_y.size - 1)

    valid_col_indices = np.flatnonzero(np.any(valid_nodes, axis=0))
    valid_row_indices = np.flatnonzero(np.any(valid_nodes, axis=1))
    if valid_col_indices.size == 0 or valid_row_indices.size == 0:
        return None, None
    col_min = float(valid_col_indices[0]) - 0.5
    col_max = float(valid_col_indices[-1]) + 0.5
    row_min = float(valid_row_indices[0]) - 0.5
    row_max = float(valid_row_indices[-1]) + 0.5

    valid_weight = cv2.remap(
        valid_nodes.astype(np.float32),
        map_cols.astype(np.float32),
        map_rows.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    weighted_x = cv2.remap(
        np.where(valid_nodes, node_src_x, 0.0).astype(np.float32),
        map_cols.astype(np.float32),
        map_rows.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    weighted_y = cv2.remap(
        np.where(valid_nodes, node_src_y, 0.0).astype(np.float32),
        map_cols.astype(np.float32),
        map_rows.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    within_supported_domain = (
        (map_cols >= col_min)
        & (map_cols <= col_max)
        & (map_rows >= row_min)
        & (map_rows <= row_max)
    )
    dense_valid = (valid_weight >= support_threshold) & within_supported_domain
    dense_x = np.zeros((height, width), dtype=np.float32)
    dense_y = np.zeros((height, width), dtype=np.float32)
    dense_x[dense_valid] = weighted_x[dense_valid] / valid_weight[dense_valid]
    dense_y[dense_valid] = weighted_y[dense_valid] / valid_weight[dense_valid]
    map_x = np.where(dense_valid, dense_x, np.float32(-1.0))
    map_y = np.where(dense_valid, dense_y, np.float32(-1.0))
    return map_x.astype(np.float32), map_y.astype(np.float32)

def build_apap_remap(registration, output_shape, roi_origin):
    height, width = output_shape
    origin_x, origin_y = roi_origin
    support_threshold = 0.5
    if registration.homography is None:
        raise ValueError("APAP registration must provide a homography")
    if not bool(registration.diagnostics.get("local_warp_ready", False)):
        return None, None

    support_dest = registration.support_destination_points
    support_res = registration.support_residuals
    if support_dest is None or support_res is None or len(support_dest) == 0:
        return None, None

    support_dest = np.asarray(support_dest, dtype=np.float64)
    support_res = np.asarray(support_res, dtype=np.float64)
    valid_support = np.isfinite(support_dest).all(axis=1) & np.isfinite(support_res).all(axis=1)
    support_dest = support_dest[valid_support]
    support_res = support_res[valid_support]
    if support_dest.shape[0] == 0:
        return None, None

    mesh_cols = max(2, int(registration.diagnostics.get("mesh_cols", 16) or 16))
    mesh_rows = max(2, int(registration.diagnostics.get("mesh_rows", 16) or 16))
    sigma = float(registration.diagnostics.get("kernel_sigma_px", 64.0) or 64.0)
    sigma = max(1.0, sigma)
    min_local_support = max(1, int(registration.diagnostics.get("min_local_support", 1) or 1))
    max_local_neighbors = max(min_local_support, min(32, int(max(min_local_support * 2, 12))))
    support_radius_px = max(sigma * 2.5, 4.0)

    mesh_x = np.linspace(float(origin_x), float(origin_x + max(0, width - 1)), mesh_cols, dtype=np.float64)
    mesh_y = np.linspace(float(origin_y), float(origin_y + max(0, height - 1)), mesh_rows, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(mesh_x, mesh_y)

    residual_x = np.zeros((mesh_rows, mesh_cols), dtype=np.float32)
    residual_y = np.zeros((mesh_rows, mesh_cols), dtype=np.float32)
    node_valid = np.zeros((mesh_rows, mesh_cols), dtype=np.uint8)
    support_x = support_dest[:, 0]
    support_y = support_dest[:, 1]
    support_rx = support_res[:, 0]
    support_ry = support_res[:, 1]
    sigma2 = 2.0 * sigma * sigma

    for row in range(mesh_rows):
        for col in range(mesh_cols):
            dx = support_x - grid_x[row, col]
            dy = support_y - grid_y[row, col]
            dist2 = (dx * dx) + (dy * dy)
            neighbor_order = np.argsort(dist2)
            if neighbor_order.size > max_local_neighbors:
                neighbor_order = neighbor_order[:max_local_neighbors]
            neighbor_order = neighbor_order[np.sqrt(dist2[neighbor_order]) <= support_radius_px]
            if neighbor_order.size < min_local_support:
                continue
            local_dist2 = dist2[neighbor_order]
            weights = np.exp(-local_dist2 / sigma2)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-8:
                continue
            local_rx = support_rx[neighbor_order]
            local_ry = support_ry[neighbor_order]
            residual_x[row, col] = float(np.sum(weights * local_rx) / weight_sum)
            residual_y[row, col] = float(np.sum(weights * local_ry) / weight_sum)
            node_valid[row, col] = 1

    dense_rx = cv2.resize(residual_x, (width, height), interpolation=cv2.INTER_LINEAR)
    dense_ry = cv2.resize(residual_y, (width, height), interpolation=cv2.INTER_LINEAR)
    dense_valid = cv2.resize(node_valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) >= support_threshold
    dense_rx = np.where(dense_valid, dense_rx, 0.0)
    dense_ry = np.where(dense_valid, dense_ry, 0.0)

    abs_x, abs_y = np.meshgrid(
        np.arange(width, dtype=np.float64) + float(origin_x),
        np.arange(height, dtype=np.float64) + float(origin_y),
    )
    corrected_x = abs_x - dense_rx.astype(np.float64)
    corrected_y = abs_y - dense_ry.astype(np.float64)

    try:
        inv_h = np.linalg.inv(np.asarray(registration.homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None, None
    denom = (inv_h[2, 0] * corrected_x) + (inv_h[2, 1] * corrected_y) + inv_h[2, 2]
    valid = np.abs(denom) > 1e-8
    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    map_x[valid] = (((inv_h[0, 0] * corrected_x) + (inv_h[0, 1] * corrected_y) + inv_h[0, 2]) / denom)[valid]
    map_y[valid] = (((inv_h[1, 0] * corrected_x) + (inv_h[1, 1] * corrected_y) + inv_h[1, 2]) / denom)[valid]
    return map_x, map_y

def warp_apap(
    image,
    registration,
    output_shape,
    roi_origin=(0, 0),
    interpolation=cv2.INTER_LINEAR,
    mask_valid_thresh=1e-4,
    mask_close_px=1,
    projection_enabled=False,
    projection_mode="native",
    projection_intrinsics_fn=None,
):
    import math
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid output shape: {output_shape}")

    remap_x = registration.remap_x
    remap_y = registration.remap_y
    if remap_x is None or remap_y is None:
        remap_x, remap_y = build_apap_mesh_remap(registration, output_shape, roi_origin)
        if remap_x is not None and remap_y is not None and registration.homography is not None:
            unsupported = remap_x < 0.0
            if np.any(unsupported):
                try:
                    origin_x, origin_y = roi_origin
                    inv_h = np.linalg.inv(np.asarray(registration.homography, dtype=np.float64))
                    abs_x_grid, abs_y_grid = np.meshgrid(
                        np.arange(width, dtype=np.float64) + float(origin_x),
                        np.arange(height, dtype=np.float64) + float(origin_y),
                    )
                    denom = (inv_h[2, 0] * abs_x_grid) + (inv_h[2, 1] * abs_y_grid) + inv_h[2, 2]
                    safe = np.abs(denom) > 1e-8
                    fill_x = np.full((height, width), -1.0, dtype=np.float32)
                    fill_y = np.full((height, width), -1.0, dtype=np.float32)
                    fill_x[safe] = (((inv_h[0, 0] * abs_x_grid) + (inv_h[0, 1] * abs_y_grid) + inv_h[0, 2]) / denom)[safe].astype(np.float32)
                    fill_y[safe] = (((inv_h[1, 0] * abs_x_grid) + (inv_h[1, 1] * abs_y_grid) + inv_h[1, 2]) / denom)[safe].astype(np.float32)
                    remap_x = np.where(unsupported, fill_x, remap_x)
                    remap_y = np.where(unsupported, fill_y, remap_y)
                except np.linalg.LinAlgError:
                    pass
    if remap_x is None or remap_y is None:
        remap_x, remap_y = build_apap_remap(registration, output_shape, roi_origin)

    if remap_x is not None and remap_y is not None:
        remap_x = np.asarray(remap_x, dtype=np.float32)
        remap_y = np.asarray(remap_y, dtype=np.float32)
        if remap_x.shape != (height, width) or remap_y.shape != (height, width):
            raise ValueError(
                f"APAP remap shape mismatch: expected {(height, width)}, got {remap_x.shape} and {remap_y.shape}"
            )

        # Interpolation flag handling (copied from Warper._resolve_interp)
        flag = interpolation
        pre_sigma = None
        if isinstance(flag, str):
            from cielostitch_core.utils.image_resampling import choose_warp_interpolator
            flag, pre_sigma = choose_warp_interpolator(np.eye(3, dtype=np.float64), flag)
            if not isinstance(flag, int):
                raise ValueError(f"choose_warp_interpolator returned non-int flag: {flag} (type: {type(flag)})")
        else:
            flag = int(flag)

        image_f = image.astype(np.float32)
        if pre_sigma and pre_sigma > 1e-6:
            image_f = cv2.GaussianBlur(image_f, ksize=(0, 0), sigmaX=pre_sigma, sigmaY=pre_sigma)

        warped_image_f = cv2.remap(
            image_f,
            remap_x,
            remap_y,
            interpolation=flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        valid = np.isfinite(remap_x) & np.isfinite(remap_y)
        valid &= remap_x >= 0.0
        valid &= remap_y >= 0.0
        valid &= remap_x <= max(0.0, float(image.shape[1] - 1))
        valid &= remap_y <= max(0.0, float(image.shape[0] - 1))

        # Projection-specific valid region masking
        if projection_enabled and projection_mode == "cylindrical" and projection_intrinsics_fn is not None:
            intrinsics = projection_intrinsics_fn(image)
            if intrinsics is not None:
                fx_p, fy_p = intrinsics
                h_p, w_p = image.shape[:2]
                cx_p = (w_p - 1) * 0.5
                cy_p = (h_p - 1) * 0.5
                proj_x_min = cx_p + fx_p * math.atan(-cx_p / fx_p)
                proj_x_max = (w_p - 1) - proj_x_min
                proj_y_min = cy_p + fy_p * math.atan(-cy_p / fy_p)
                proj_y_max = (h_p - 1) - proj_y_min
                if proj_x_min > 0.5:
                    valid &= remap_x >= float(proj_x_min)
                if proj_x_max < (w_p - 1) - 0.5:
                    valid &= remap_x <= float(proj_x_max)
                if proj_y_min > 0.5:
                    valid &= remap_y >= float(proj_y_min)
                if proj_y_max < (h_p - 1) - 0.5:
                    valid &= remap_y <= float(proj_y_max)

        if warped_image_f.ndim > 2:
            warped_image_f[~valid] = 0.0
        else:
            warped_image_f[~valid] = 0.0

        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            warped_image_f = np.clip(warped_image_f, info.min, info.max)
        warped_image = warped_image_f.astype(image.dtype, copy=False)

        warped_mask = valid.astype(np.uint8)
        if mask_close_px > 0:
            k = 2 * mask_close_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            closed_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8, copy=False)
            if np.any(closed_mask > warped_mask):
                # Use the fill_closed_mask_holes logic from Warper
                from ..core.warper import Warper
                warped_image_f = Warper.fill_closed_mask_holes(warped_image_f, warped_mask, closed_mask)
                if np.issubdtype(image.dtype, np.integer):
                    info = np.iinfo(image.dtype)
                    warped_image_f = np.clip(warped_image_f, info.min, info.max)
                warped_image = warped_image_f.astype(image.dtype, copy=False)
            warped_mask = closed_mask

        return warped_image, warped_mask

    if registration.homography is None:
        raise ValueError("APAP registration must provide a bootstrap homography or dense remap")
    origin_x_f, origin_y_f = float(roi_origin[0]), float(roi_origin[1])
    translate_roi = np.eye(3, dtype=np.float64)
    translate_roi[0, 2] = -origin_x_f
    translate_roi[1, 2] = -origin_y_f
    from ..core.warper import Warper
    return Warper().warp(image, translate_roi @ np.asarray(registration.homography, dtype=np.float64), output_shape)
