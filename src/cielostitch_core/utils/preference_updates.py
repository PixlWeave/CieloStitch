# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

from dataclasses import replace
import logging

from cielostitch_core.config.config import cfg
from cielostitch_core.state.preferences import write_pref

logger = logging.getLogger(__name__)


def update_cfg_prefs(**updates) -> None:
    try:
        cfg.prefs = replace(cfg.prefs, **updates)
    except Exception:
        logger.warning("Failed to update cfg.prefs with %s", list(updates.keys()), exc_info=True)


def safe_pref_value(name: str, fallback, *, missing=None, coalesce_empty: bool = False):
    try:
        value = getattr(cfg.prefs, name)
    except AttributeError:
        return fallback if missing is None else missing
    except Exception as exc:
        logger.warning(f"Unexpected error accessing {name}: {exc}")
        return fallback if missing is None else missing

    if coalesce_empty and not value:
        return fallback
    return value


def sync_runtime_pref(pref_key: str, pref_value, *, writer=write_pref, **cfg_updates) -> None:
    if cfg_updates:
        update_cfg_prefs(**cfg_updates)
    writer(pref_key, pref_value)