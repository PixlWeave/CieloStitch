# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import logging
from dataclasses import dataclass

from ..state.preferences import AppPreferences
from ..state.state_manager import AppStateManager
from ..config.constants import DEFAULT_MESSAGE_COLORS_STR


def _build_allowed_message_colors(raw: str | None) -> set[str]:
    parts = {part.strip().lower() for part in str(raw or "").split(",") if part.strip()}
    return parts or {}

@dataclass
class Config:
    def __init__(self) -> None:
        self.prefs = AppPreferences.load()
        self.allowed_message_colors = _build_allowed_message_colors(
            getattr(self.prefs, "log_message_colors", DEFAULT_MESSAGE_COLORS_STR)
        )
        self.state = AppStateManager(self.prefs)
        self.apply_log_level()

    def apply_log_level(self) -> None:
        # Order: DEBUG,INFO,WARNING,ERROR,CRITICAL/FATAL
        level_name = getattr(self.prefs, "stderr_log_level", "WARNING")
        level = getattr(logging, level_name, logging.WARNING)
        logging.getLogger().setLevel(level)

cfg = Config()
