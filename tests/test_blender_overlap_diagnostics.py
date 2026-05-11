# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.core.blender import Blender


def _make_blender(blend_type: str = "seamless", multiband_levels: int = 5) -> Blender:
    return Blender(
        feather_px=40,
        gain_method="median",
        gain_clamp=(0.85, 1.15),
        blend_type=blend_type,
        multiband_levels=multiband_levels,
        seamless_quality="balanced",
        lum_only_gain=False,
        histogram_matching=False,
    )


def test_blend_tracks_overlap_geometry_and_diagnostics() -> None:
    blender = _make_blender(blend_type="seamless", multiband_levels=4)

    canvas = np.zeros((64, 64), dtype=np.float32)
    img = np.ones((64, 64), dtype=np.float32)

    canvas_mask = np.zeros((64, 64), dtype=bool)
    img_mask = np.zeros((64, 64), dtype=bool)
    canvas_mask[:, :40] = True
    img_mask[:, 24:] = True

    overlap = canvas_mask & img_mask
    only_canvas = canvas_mask & (~img_mask)
    only_img = img_mask & (~canvas_mask)

    coverage_count = canvas_mask.astype(np.uint8).copy()
    blended_mask = np.zeros((64, 64), dtype=bool)
    seam_heatmap = np.zeros((64, 64), dtype=np.float32)

    out_canvas, out_mask = blender.blend(
        canvas,
        canvas_mask,
        img,
        img_mask,
        coverage_count=coverage_count,
        blended_mask=blended_mask,
        seam_heatmap=seam_heatmap,
    )

    assert np.any(overlap)
    assert np.all(out_mask == (canvas_mask | img_mask))

    # Geometry truth from coverage count.
    assert np.all(coverage_count[overlap] == 2)
    assert np.all(coverage_count[only_canvas] == 1)
    assert np.all(coverage_count[only_img] == 1)

    # Blending/seam diagnostics should mark overlap only.
    assert np.all(blended_mask[overlap])
    assert not np.any(blended_mask[~overlap])
    assert np.max(seam_heatmap[overlap]) > 0.0
    assert np.all(seam_heatmap[~overlap] == 0.0)

    # Seamless blending can softly affect nearby non-overlap pixels.
    # Keep assertions bounded and mode-agnostic rather than forcing hard 0/1 writes.
    assert float(np.min(out_canvas[only_canvas])) >= 0.0
    assert float(np.max(out_canvas[only_canvas])) < 0.3
    assert float(np.min(out_canvas[only_img])) > 0.8



def test_blend_without_overlap_keeps_overlap_diagnostics_empty() -> None:
    blender = _make_blender(blend_type="seamless", multiband_levels=4)

    canvas = np.zeros((40, 80), dtype=np.float32)
    img = np.ones((40, 80), dtype=np.float32)

    canvas_mask = np.zeros((40, 80), dtype=bool)
    img_mask = np.zeros((40, 80), dtype=bool)
    canvas_mask[:, :30] = True
    img_mask[:, 40:] = True

    coverage_count = canvas_mask.astype(np.uint8).copy()
    blended_mask = np.zeros((40, 80), dtype=bool)
    seam_heatmap = np.zeros((40, 80), dtype=np.float32)

    out_canvas, out_mask = blender.blend(
        canvas,
        canvas_mask,
        img,
        img_mask,
        coverage_count=coverage_count,
        blended_mask=blended_mask,
        seam_heatmap=seam_heatmap,
    )

    assert np.all(out_mask == (canvas_mask | img_mask))
    assert not np.any(canvas_mask & img_mask)

    # With no overlap, diagnostics should remain empty.
    assert not np.any(blended_mask)
    assert np.all(seam_heatmap == 0.0)

    # Coverage still counts both contributors in their own regions.
    assert np.all(coverage_count[canvas_mask] == 1)
    assert np.all(coverage_count[img_mask] == 1)

    # New non-overlap pixels must be copied from the incoming panel.
    only_img = img_mask & (~canvas_mask)
    assert np.all(out_canvas[only_img] == 1.0)
