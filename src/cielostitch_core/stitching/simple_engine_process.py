# SPDX-License-Identifier: LicenseRef-Proprietary
#
# Copyright (c) 2026 Debasish Saha
# All rights reserved.
#
# Part of the CieloStitch proprietary application.
#
# This software is proprietary and confidential.
# Unauthorized copying, modification, distribution,
# reverse engineering, or resale is prohibited.
#
# See the LICENSE file for details.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from cielostitch_core.stitching.simple_engine import run_simple_engine


def _emit(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _configure_cv_threads(panel_count: int) -> None:
    try:
        cpu = max(1, int(os.cpu_count() or 1))
        physical_est = max(1, (3 * cpu + 4) // 5)
        panel_cap = max(1, min(int(panel_count or 1), 16))
        cv_threads = max(1, min(physical_est, panel_cap))
        cv2.setNumThreads(cv_threads)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("Expected request json path\n")
        return 2

    request_path = Path(args[0])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output_img_path = Path(str(request["output_img_path"]))
    float_mosaic_path = Path(str(request["float_mosaic_path"]))
    _configure_cv_threads(len(request.get("sorted_panel_paths") or []))

    try:
        output_img, float_mosaic, input_bit_depth = run_simple_engine(
            list(request.get("sorted_panel_paths") or []),
            alpha_policy=str(request.get("alpha_policy") or "auto"),
            log_emit=lambda message, color="": _emit({"type": "log", "message": str(message), "color": str(color or "")}),
            simple_settings=dict(request.get("simple_settings") or {}),
        )
        np.save(output_img_path, output_img, allow_pickle=False)
        np.save(float_mosaic_path, float_mosaic, allow_pickle=False)
        _emit(
            {
                "type": "result",
                "output_img_path": str(output_img_path),
                "float_mosaic_path": str(float_mosaic_path),
                "input_bit_depth": int(input_bit_depth),
            }
        )
        return 0
    except Exception as exc:
        _emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
