# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

__version__ = "0.5.00"
__author__ = "Debasish Saha"
__email__ = "dcsaha@gmail.com"
__status__ = "Development"
__website__ = "https://www.cielostitch.com"
__domain__ = "cielostitch.com"

def app_name() -> str:
    return "CieloStitch"

def _version() -> str:
    return __version__

def _author() -> str:
    return __author__

def _website() -> str:
    return  f"<a href='{__website__}'>{__domain__}</a>"

def _copy_right() -> str:
    return f"Copyright © 2026 {_author()}, {_website()}"

def _slogan() -> str:
    return (
        "Weave the heavens and the earth, frame by frame"
    )

def _license_html(app_version: str) -> str:
    core_version = _version()
    return (
        # "<br><p><b>Licensing</b><br>"
        f"CieloStitch App (version {app_version}): All Rights Reserved<br>"
        f"CieloStitch Core (version {core_version}): MIT License<br>"
        "<span style='font-size:9pt;'>"
        "Click on [view licenses] button for full terms."
        "</span>"
    )

def about(app_version: str) -> str:
    return (
        f"<span style='font-size:16pt; font-weight:600;'>{app_name()} {_version()}</span>"
        f"<p><span style='font-size:11pt;'>{_slogan()}</span></p>"
        f"{_copy_right()}"
        f"<p>{_license_html(app_version)}</p>"
        "<p>"
        "CieloStitch is a free software to help compose seamless<br>"
        "mosaics of solar, lunar, milky way and landscape photography.<br>"
        "<br>Clear skies! Cielos despejados!</p>"
    )
