# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np

from cielostitch_core.utils import image_io


def test_read_image_prefers_tifffile_for_large_tiff(monkeypatch):
    expected = np.zeros((16, 16, 3), dtype=np.float16)

    monkeypatch.setattr(image_io, "_should_prefer_tifffile_for_tiff", lambda _path: True)
    monkeypatch.setattr(image_io, "_read_tiff_with_tifffile", lambda _path: expected)

    def _imread_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("cv2.imread should not be called for very large TIFF payloads")

    monkeypatch.setattr(image_io.cv2, "imread", _imread_should_not_be_called)

    out = image_io.read_image("S:/solar4.tif", alpha_policy="drop")

    assert out is expected


def test_read_image_falls_back_to_tifffile_when_opencv_fails(monkeypatch):
    expected = np.ones((8, 8), dtype=np.uint16)

    monkeypatch.setattr(image_io, "_should_prefer_tifffile_for_tiff", lambda _path: False)
    monkeypatch.setattr(image_io.cv2, "imread", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(image_io, "_read_tiff_with_tifffile", lambda _path: expected)

    out = image_io.read_image("S:/solar4.tiff", alpha_policy="drop")

    assert out is expected


def test_read_image_prefers_tifffile_for_float_sample_tiff(monkeypatch):
    expected = np.zeros((4, 4), dtype=np.float32)

    monkeypatch.setattr(image_io, "_estimate_tiff_payload_bytes", lambda _path: 1024)
    monkeypatch.setattr(image_io, "_tiff_uses_float_sample_format", lambda _path: True)
    monkeypatch.setattr(image_io, "_read_tiff_with_tifffile", lambda _path: expected)

    def _imread_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("cv2.imread should not be called for float-sample TIFF payloads")

    monkeypatch.setattr(image_io.cv2, "imread", _imread_should_not_be_called)

    out = image_io.read_image("S:/cox.tif", alpha_policy="drop")

    assert out is expected
