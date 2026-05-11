# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations
from ..config.config import cfg

def emit_msg(msg, color="", log_cb=None):
    """Emit a message to console and optionally to log callback.

    Args:
        msg: The message text
        color: Color code ("red", "green", etc.)
        log_cb: Optional callback for UI logging (signature: log_cb(msg: str, color: str))
    """

    if log_cb is not None:
        try:
            if message_color_passed(color):
                log_cb(msg, color)
        except Exception:
            pass  # Silently ignore if log callback not available


def message_color_passed(color) -> bool:
    allowed = cfg.allowed_message_colors
    normalized = str(color or "").strip().lower()
    if normalized in {"", "default", "none"}:
        return "default" in allowed or "none" in allowed
    return normalized in allowed