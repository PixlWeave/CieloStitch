# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

from pathlib import Path

import numpy as np
import pytest
import tifffile

from cielostitch_core.utils import image_meta


def test_tiff_protected_tag_ids_are_blocked():
    assert image_meta._is_tiff_protected_tag_id(256) is True  # width
    assert image_meta._is_tiff_protected_tag_id(257) is True  # height
    assert image_meta._is_tiff_protected_tag_id(258) is True  # bits-per-sample
    assert image_meta._is_tiff_protected_tag_id(262) is True  # photometric interpretation
    assert image_meta._is_tiff_protected_tag_id(270) is False  # description is allowed


def test_fits_protected_keywords_cover_structural_and_color_fields():
    assert image_meta._is_fits_protected_keyword("BITPIX") is True
    assert image_meta._is_fits_protected_keyword("NAXIS1") is True
    assert image_meta._is_fits_protected_keyword("WIDTH") is True
    assert image_meta._is_fits_protected_keyword("HEIGHT") is True
    assert image_meta._is_fits_protected_keyword("SHAPE") is True
    assert image_meta._is_fits_protected_keyword("CHANNELS") is True
    assert image_meta._is_fits_protected_keyword("BITDEPTH") is True
    assert image_meta._is_fits_protected_keyword("COLOR") is True
    assert image_meta._is_fits_protected_keyword("COLORSPC") is True
    assert image_meta._is_fits_protected_keyword("OBSERVER") is False


class _FakeXISF:
    last_written = None

    def __init__(self, arg):
        self._arg = arg

    def read_image(self, n=0, data_format="channels_last"):
        return np.zeros((2, 2, 1), dtype=np.uint16)

    def get_images_metadata(self):
        return [{"FITSKeywords": {}, "XISFProperties": {}}]

    def get_file_metadata(self):
        return {}

    @staticmethod
    def write(output_path, image_data, creator_app=None, image_metadata=None, xisf_metadata=None, codec=None, shuffle=False, level=None):
        _FakeXISF.last_written = {
            "output_path": str(output_path),
            "image_shape": tuple(np.asarray(image_data).shape),
            "creator_app": creator_app,
            "image_metadata": image_metadata or {},
            "xisf_metadata": xisf_metadata or {},
        }


def test_write_xisf_metadata_skips_protected_properties(monkeypatch, tmp_path):
    monkeypatch.setattr(image_meta, "HAS_XISF", True)
    monkeypatch.setattr(image_meta, "XISF", _FakeXISF)

    target = tmp_path / "target.xisf"
    out = tmp_path / "out.xisf"
    target.write_bytes(b"placeholder")

    image_meta.write_xisf_metadata(
        target,
        out,
        {
            "properties": {
                "NAXIS1": 101,
                "BITPIX": 16,
                "COLOR": "RGB",
                "WIDTH": 101,
                "HEIGHT": 57,
                "OBSERVER": "Unit Test",
            }
        },
        mode="overwrite",
    )

    written = _FakeXISF.last_written
    assert written is not None
    fits_keywords = written["image_metadata"]["FITSKeywords"]
    assert fits_keywords["OBSERVER"][0]["value"] == "Unit Test"
    assert "NAXIS1" not in fits_keywords
    assert "BITPIX" not in fits_keywords
    assert "COLOR" not in fits_keywords
    assert "WIDTH" not in fits_keywords
    assert "HEIGHT" not in fits_keywords


def test_can_transfer_metadata_allows_cross_family_formats():
    ok, reason = image_meta.can_transfer_metadata("source.fit", "target.tif")
    assert ok is True
    assert reason == ""


def test_convert_metadata_to_fits_filters_structural_fields():
    converted = image_meta._convert_metadata_for_target_family(
        {
            "format": "TIFF",
            "tags": {
                "ImageDescription": {"id": 270, "value": "obs-run-1"},
                "Software": {"id": 305, "value": "CieloStitch"},
                "ImageWidth": {"id": 256, "value": 2048},
            },
        },
        "fits",
    )

    header = converted["hdus"]["HDU_0"]["header"]
    assert "DESCRIPTION" in header
    assert "SOFTWARE" in header
    assert "WIDTH" not in header
    assert "NAXIS1" not in header


def test_convert_metadata_to_tiff_uses_description_payload_and_skips_bitpix():
    converted = image_meta._convert_metadata_for_target_family(
        {
            "format": "FITS",
            "hdus": {
                "HDU_0": {
                    "header": {
                        "OBJECT": {"value": "M31", "comment": ""},
                        "BITPIX": {"value": 16, "comment": ""},
                        "DATE-OBS": {"value": "2026-05-10T00:00:00", "comment": ""},
                    }
                }
            },
        },
        "tiff",
    )

    tags = converted["tags"]
    assert tags["ImageDescription"]["id"] == 270
    assert "OBJECT=M31" in tags["ImageDescription"]["value"]
    assert "BITPIX=16" not in tags["ImageDescription"]["value"]
    assert tags["DateTime"]["id"] == 306
    assert tags["DateTime"]["value"] == "2026:05:10 00:00:00"


def test_tiff_imwrite_kwargs_omits_invalid_datetime_value():
    kwargs = image_meta._tiff_imwrite_kwargs_from_tags({306: "not-a-datetime"})

    assert "datetime" not in kwargs


def test_convert_metadata_to_xisf_skips_protected_keys():
    converted = image_meta._convert_metadata_for_target_family(
        {
            "format": "FITS",
            "hdus": {
                "HDU_0": {
                    "header": {
                        "BITPIX": {"value": 16},
                        "NAXIS1": {"value": 100},
                        "OBSERVER": {"value": "Deb"},
                    }
                }
            },
        },
        "xisf",
    )

    props = converted["properties"]
    assert "OBSERVER" in props
    assert "BITPIX" not in props
    assert "NAXIS1" not in props


def test_write_fits_metadata_supports_inplace_output_path(tmp_path):
    if not image_meta.HAS_FITS:
        pytest.skip("astropy is required for FITS tests")

    target = tmp_path / "current_output.fit"
    original = np.arange(16, dtype=np.uint16).reshape(4, 4)
    image_meta.fits.writeto(target, original, overwrite=True)

    payload = {
        "hdus": {
            "HDU_0": {
                "header": {
                    "OBSERVER": {"value": "Unit Test", "comment": ""},
                    "BITPIX": {"value": 8, "comment": "must be ignored"},
                }
            }
        }
    }

    image_meta.write_fits_metadata(target, target, payload, mode="merge")

    with image_meta.fits.open(target) as hdul:
        got = np.asarray(hdul[0].data)
        assert np.array_equal(got, original)
        assert hdul[0].header.get("OBSERVER") == "Unit Test"
        # Protected structural keys should not be overwritten from transferred metadata.
        assert int(hdul[0].header.get("BITPIX")) == 16


def test_write_tiff_metadata_supports_inplace_output_path(tmp_path):
    target = tmp_path / "current_output.tif"
    original = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(target, original)

    payload = {
        "tags": {
            "ImageDescription": {"id": 270, "value": "Unit Test Description"},
            "ImageWidth": {"id": 256, "value": 9999},
        }
    }

    image_meta.write_tiff_metadata(target, target, payload, mode="merge")

    with tifffile.TiffFile(target) as tif:
        got = tif.asarray()
        assert np.array_equal(got, original)
        description = str(tif.pages[0].tags[270].value)
        assert "Unit Test Description" in description
        # Protected structural tags should not be overwritten by transferred metadata.
        assert int(tif.pages[0].tags[256].value) == original.shape[1]


def test_write_xisf_metadata_supports_inplace_output_path(tmp_path):
    if not image_meta.HAS_XISF:
        pytest.skip("xisf is required for XISF tests")

    target = tmp_path / "current_output.xisf"
    original = np.zeros((4, 4, 1), dtype=np.uint16)
    image_meta.XISF.write(target, original, creator_app="UnitTest")

    payload = {
        "properties": {
            "OBSERVER": "Unit Test",
            "NAXIS1": 123,
        },
        "metadata": {},
    }

    image_meta.write_xisf_metadata(target, target, payload, mode="merge")

    meta = image_meta.read_xisf_metadata(target)
    assert meta["properties"].get("OBSERVER") == "Unit Test"
    assert "NAXIS1" not in meta["properties"]
