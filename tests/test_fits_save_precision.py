# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# See the LICENSE file for details.

import numpy as np
import pytest

from cielostitch_core.utils.image_io import save_image

astropy_fits = pytest.importorskip("astropy.io.fits")


def test_save_image_fits_upcasts_float16_to_float32(tmp_path):
    arr = np.linspace(0.0, 1.0, num=64 * 64, dtype=np.float16).reshape(64, 64)
    out_path = tmp_path / "float16_export.fit"

    save_image(arr, str(out_path))

    loaded = astropy_fits.getdata(out_path)

    assert loaded.shape == arr.shape
    assert np.issubdtype(loaded.dtype, np.floating)
    assert loaded.dtype.itemsize == 4
    np.testing.assert_allclose(loaded.astype(np.float32), arr.astype(np.float32), rtol=1e-6, atol=1e-6)
