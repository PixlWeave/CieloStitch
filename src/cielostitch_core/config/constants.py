# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

# Limit-Related Constants
# Practical RAM notes:
# - MAX_CANVAS_DIM / MAX_CANVAS_PIXELS are the main memory guards. Raising them
#   increases peak mosaic memory sharply because canvas, masks, coverage maps,
#   seam heatmaps, and temporary blend arrays all grow with pixel count.
# - AUTO_MODE_MAX_WARP_DIM / GRID_MODE_MAX_WARP_DIM mainly protect against bad
#   transforms creating oversized warped intermediates. Higher values can allow
#   larger temporary warp buffers and more sudden RAM spikes.
# - ALPHA_POLICY_SCAN_LIMIT affects only the pre-run alpha scan count. It has
#   negligible RAM impact compared with image decoding and stitching.
# - FEATURE_CACHE_CAPACITY controls how many feature/descriptor sets stay in
#   memory between runs. Higher values improve rerun speed but retain more RAM.
# - PREF_* / SESSION_* grid column limits are UI/session guards only. They do
#   not materially change stitch RAM usage by themselves.
# - EXPORT_SCALE_* can increase save-time RAM if exporting above 100%, because
#   the resized export image is allocated before writing.
# - JPEG_QUALITY_* affects file size and write time, not meaningful RAM usage.
MAX_CANVAS_DIM = 50_000
MAX_CANVAS_PIXELS = 1_000_000_000
AUTO_MODE_MAX_WARP_DIM = 25_000
GRID_MODE_MAX_WARP_DIM = 30_000
AUTO_MODE_MAX_WARP_PIXELS = 40_000_000
GRID_MODE_MAX_WARP_PIXELS = 50_000_000

ALPHA_POLICY_SCAN_LIMIT = 1_000
FEATURE_CACHE_CAPACITY = 512
PREF_GRID_COLS_MIN = 1
PREF_GRID_COLS_MAX = 100
SESSION_GRID_COLS_MIN = 1
SESSION_GRID_COLS_MAX = 100

EXPORT_SCALE_MIN_PCT = 10
EXPORT_SCALE_MAX_PCT = 400
JPEG_QUALITY_MIN = 50
JPEG_QUALITY_MAX = 100

# Image bit-depth conversion scales.
# 65535 / 255 is exactly 257.0, so uint8 <-> uint16 mapping can be symmetric.
UINT16_TO_UINT8_SCALE = 65535.0 / 255.0
UINT8_TO_UINT16_SCALE = int(UINT16_TO_UINT8_SCALE)

DEFAULT_COLS = 3
MAX_COLS = SESSION_GRID_COLS_MAX
DEFAULT_OVERLAP_PCT = 20.00

# Blending
MAX_MULTIBAND_LEVELS = 7

# all keys: default,red,yellow,green,panel,none,"",cli
DEFAULT_MESSAGE_COLORS_STR = "default,red,yellow,green,cli"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Exposed profiles in UI (order = how they appear in the combo box)
ENGINES = ["auto", "simple", "cielo"]
DEFAULT_ENGINE = 'cielo'
SIMPLE_ENGINE_MODES = ["auto", "panorama", "scans"]
IMAGE_PROFILES = [
    "solar",
    "solar-h-alpha",
    "lunar",
    "landscape",
    "nightscape",
    "milky-way",
    "panorama",
    "general",
]

RESAMPLERS = ["auto", "linear",  "cubic", "lanczos", "area"]
# ---------------------------------
STITCH_MODES = ["auto", "freeform", "grid-guided", "fixed-overlap", "zero-overlap"]
STITCH_PRECISIONS = {
    "auto": "auto",
    "fp32": "standard (32-bit float)",
    "fp16": "compact (16-bit float)",
}
GRID_MODES = ["grid-guided", "fixed-overlap", "zero-overlap"]
NON_GRID_MODES = ["auto", "freeform"]
FIXED_OVERLAP_MODES = ["fixed-overlap", "zero-overlap"]
UNKNOWN_OVERLAP_MODES = ["auto", "freeform", "grid-guided"]
# ---------------------------------
BLEND_TYPES = ["multiband", "adaptive-multiband", "feather", "adaptive-feather", "seamless", "none"]
EDGE_AWARE_TYPES = ["multiband", "adaptive-multiband", "seamless"]
GAIN_COMPENSATION_OPTIONS = ["none", "simple", "uniform", "local"]
SIMPLE_GAIN_METHODS = ["mean", "median", "trimmed"]
SEAMLESS_OPTIONS = ["fast", "balanced", "best"]
SPEED_PRESETS = ["fast", "balanced", "best"]
PROJECTION_MODES = ["native", "cylindrical"]
# Transform model options for feature matching
TRANSFORM_MODES = ["translation", "affine", "homography", "apap"]
BUNDLE_ADJUSTMENT_MODES = ["off", "translation", "affine"]
FPX_MODES = ["factor", "fov", "camera"]
MOUNT_PRECISIONS = ["tight", "normal", "sloppy", "manual", "handheld"]
SEEING_CONDITIONS = ["poor", "average", "good"]
SCAN_ORDERS = ["row-wise", "column-wise"]
START_CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"]
ALTERNATING = ["yes", "no"]
ALPHA_POLICIES = ["auto", "keep", "drop"]
SUPPORTED_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".fit", ".fits", ".xisf"]
IMAGE_FILTER_STR = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp *.fit *.fits *.xisf)"
IMAGE_FILTER_LST = (
    "TIFF (*.tif *.tiff);;"
    "PNG (*.png);;"
    "JPEG (*.jpg *.jpeg);;"
    "BMP (*.bmp);;"
    "WebP (*.webp);;"
    "FITS (*.fit *.fits);;"
    "XISF (*.xisf)"
    )

# Shared dtype label format for status/log/save text.
# Allowed values: "long" (float32), "short" (f32), "int" (32)
DTYPE_LABEL_FORMAT = "long"
