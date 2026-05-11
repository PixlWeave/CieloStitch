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


class MosaicCanvas:
    from ..config.constants import MAX_CANVAS_DIM, MAX_CANVAS_PIXELS
    def __init__(self, first_image, *, enable_overlap_diagnostics=True, enable_seam_diagnostics=True):
        # Support mono (H,W), color (H,W,C), including RGBA premultiplied
        h, w = first_image.shape[:2]

        self.canvas = first_image.copy()
        # Initialize 2-D validity mask: if RGBA, derive from alpha>0; else ones
        if self.canvas.ndim == 3 and self.canvas.shape[2] == 4:
            alpha = self.canvas[..., 3]
            self.mask = (alpha > 1e-6).astype(np.uint8)
        else:
            self.mask = np.ones((h, w), dtype=np.uint8)
        # Per-pixel diagnostics are optional to reduce stitch-time memory footprint.
        if bool(enable_overlap_diagnostics):
            # Per-pixel panel coverage count (tracks overlaps; uint8 supports up to 255 panels)
            self.coverage_count = self.mask.astype(np.uint8)
            # Per-pixel blended-overlap indicator
            self.blended_mask = np.zeros((h, w), dtype=bool)
        else:
            self.coverage_count = None
            self.blended_mask = None

        if bool(enable_seam_diagnostics):
            # Per-pixel seam strength map (float32 in [0,1]) accumulated across blends
            self.seam_heatmap = np.zeros((h, w), dtype=np.float32)
        else:
            self.seam_heatmap = None

        self.offset_x = 0
        self.offset_y = 0

    def get_canvas(self):
        return self.canvas

    def get_mask(self):
        return self.mask

    @staticmethod
    def compute_warp_bounds(image, H):
        # Use spatial dimensions only; ignore channels for color images
        h, w = image.shape[:2]

        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        corners = corners.reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)

        xs = warped_corners[:, 0, 0]
        ys = warped_corners[:, 0, 1]

        return xs.min(), ys.min(), xs.max(), ys.max()

    def expand_canvas(self, min_x, min_y, max_x, max_y):
        # Safety pad avoids occasional edge clipping from subpixel/rounding effects.
        pad = 3
        new_min_x = min(0, int(np.floor(min_x + self.offset_x - pad)))
        new_min_y = min(0, int(np.floor(min_y + self.offset_y - pad)))

        new_max_x = max(self.canvas.shape[1], int(np.ceil(max_x + self.offset_x + pad)))
        new_max_y = max(self.canvas.shape[0], int(np.ceil(max_y + self.offset_y + pad)))

        new_width = new_max_x - new_min_x
        new_height = new_max_y - new_min_y

        if new_width > self.MAX_CANVAS_DIM or new_height > self.MAX_CANVAS_DIM or new_width * new_height > self.MAX_CANVAS_PIXELS:
            raise ValueError(
                f"Canvas expansion would exceed safe limits: {new_width}x{new_height} "
                f"({new_width * new_height:,} pixels)"
            )

        # Preserve canvas numeric type and channel count (e.g., float32 internal pipeline).
        if self.canvas.ndim == 3:
            c = self.canvas.shape[2]
            new_canvas = np.zeros((new_height, new_width, c), dtype=self.canvas.dtype)
        else:
            new_canvas = np.zeros((new_height, new_width), dtype=self.canvas.dtype)
        new_mask = np.zeros((new_height, new_width), dtype=np.uint8)
        new_coverage = None
        if self.coverage_count is not None:
            new_coverage = np.zeros((new_height, new_width), dtype=np.uint8)

        new_blended = None
        if self.blended_mask is not None:
            new_blended = np.zeros((new_height, new_width), dtype=bool)

        new_seam = None
        if self.seam_heatmap is not None:
            new_seam = np.zeros((new_height, new_width), dtype=np.float32)

        # Copy old canvas into new position
        y_offset = -new_min_y
        x_offset = -new_min_x

        new_canvas[y_offset:y_offset + self.canvas.shape[0],
        x_offset:x_offset + self.canvas.shape[1]] = self.canvas

        new_mask[y_offset:y_offset + self.mask.shape[0],
        x_offset:x_offset + self.mask.shape[1]] = self.mask
        if new_coverage is not None and self.coverage_count is not None:
            new_coverage[y_offset:y_offset + self.coverage_count.shape[0],
            x_offset:x_offset + self.coverage_count.shape[1]] = self.coverage_count
        if new_blended is not None and self.blended_mask is not None:
            new_blended[y_offset:y_offset + self.blended_mask.shape[0],
            x_offset:x_offset + self.blended_mask.shape[1]] = self.blended_mask
        if new_seam is not None and self.seam_heatmap is not None:
            new_seam[y_offset:y_offset + self.seam_heatmap.shape[0],
                     x_offset:x_offset + self.seam_heatmap.shape[1]] = self.seam_heatmap

        self.canvas = new_canvas
        self.mask = new_mask
        self.coverage_count = new_coverage
        self.blended_mask = new_blended
        self.seam_heatmap = new_seam

        # Keep cumulative offset from the original global frame to current canvas.
        # offset tracks the top-left corner of the canvas in the original coordinate system
        self.offset_x += x_offset
        self.offset_y += y_offset

    def crop_to_content(self):
        """Crop canvas and all per-pixel maps to bounding box of valid mask.

        Removes the outer zero/transparent rim introduced by safety padding
        during stitching, so the final preview/export has no border.

        No-op if the mask is empty or already tightly bounded.
        """
        # Ensure mask exists and has any valid pixels
        if self.mask is None:
            return
        m = self.mask.astype(bool)
        if not np.any(m):
            return

        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        # If already tight, skip
        if y0 <= 0 and x0 <= 0 and y1 >= self.canvas.shape[0] and x1 >= self.canvas.shape[1]:
            return

        # Slice all arrays consistently (supports 2D and 3D canvas)
        self.canvas = self.canvas[y0:y1, x0:x1]
        self.mask = self.mask[y0:y1, x0:x1]
        if self.coverage_count is not None:
            self.coverage_count = self.coverage_count[y0:y1, x0:x1]
        if self.blended_mask is not None:
            self.blended_mask = self.blended_mask[y0:y1, x0:x1]
        if self.seam_heatmap is not None:
            self.seam_heatmap = self.seam_heatmap[y0:y1, x0:x1]

        # Update offsets so the new canvas origin is consistent with cropping
        # Subtract the crop boundaries to adjust offset since the canvas is now smaller
        try:
            self.offset_x -= x0
            self.offset_y -= y0
        except Exception:
            # Offsets are best-effort; safe to ignore if not used downstream
            pass
