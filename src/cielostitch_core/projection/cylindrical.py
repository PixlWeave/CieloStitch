# SPDX-License-Identifier: MIT
#
# Cylindrical projection logic for CieloStitch core.
#
# Copyright (c) 2026 Debasish Saha
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def cylindrical_project_image(image, fx, fy, cx, cy):
    """Project an image using cylindrical projection parameters."""
    h, w = image.shape[:2]
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(u, v)
    x_prime = (grid_x - cx) / fx
    y_prime = (grid_y - cy) / fy
    max_angle = np.nextafter(np.float32(np.pi * 0.5), np.float32(0.0))
    valid_x = np.abs(x_prime) < max_angle
    x = np.tan(np.clip(x_prime, -max_angle, max_angle))
    y = y_prime * np.sqrt((x * x) + 1.0)
    map_x = (fx * x + cx).astype(np.float32)
    map_y = (fy * y + cy).astype(np.float32)
    map_x[~valid_x] = -1.0
    map_y[~valid_x] = -1.0
    src_dtype = image.dtype
    src = image.astype(np.float32, copy=False)
    projected = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        projected = np.clip(projected, info.min, info.max)
    return projected.astype(src_dtype, copy=False)
