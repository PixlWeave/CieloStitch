# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

from typing import Dict, Any
from dataclasses import dataclass
from ...utils.data_types import as_float


@dataclass
class GridGuidedStitching:
    # --- Grid Fallback ---
    # Enable phase-correlation translation fallback when feature match fails.
    # enable_grid_phase_fallback: bool = True
    # Enable neighbor nominal-step fallback (uses expected grid step) when matches are weak.
    # enable_grid_nominal_fallback: bool = True
    # Guard against implausible neighbor shifts (prevents panel stacking/overlaps in wrong direction).
    # enable_grid_shift_guard: bool = True
    # If no neighbors are available, allow absolute nominal placement at grid slot.
    # enable_grid_absolute_nominal_fallback: bool = True

    # --- Additional grid guards & capture hints (not in Advanced group ordering) ---
    # Confidence score assigned to absolute nominal placements (low to avoid overwriting).
    grid_absolute_nominal_score: float = 1.0
    # Tolerance around expected neighbor shift along the primary axis.
    grid_expected_shift_tolerance: float = 0.12
    # Tolerance for orthogonal (unwanted) shift component relative to tile size.
    grid_orthogonal_shift_tolerance: float = 0.15
    # Bias strength toward nominal neighbor shift applied to solved transforms.
    grid_nominal_shift_bias: float = 1.0
    grid_phase_fallback_min_response: float = 0.018
    grid_phase_fallback_max_shift_ratio: float = 0.93
    grid_min_blend_score: float = 10.0
    grid_nominal_score: float = 2.0

    def to_dict(self) -> dict:
        """Convert the dataclass instance to a dictionary."""
        return {
            # 'enable_grid_phase_fallback': self.enable_grid_phase_fallback,
            # 'enable_grid_nominal_fallback': self.enable_grid_nominal_fallback,
            # 'enable_grid_shift_guard': self.enable_grid_shift_guard,
            # 'enable_grid_absolute_nominal_fallback': self.enable_grid_absolute_nominal_fallback,
            'grid_absolute_nominal_score': self.grid_absolute_nominal_score,
            'grid_expected_shift_tolerance': self.grid_expected_shift_tolerance,
            'grid_orthogonal_shift_tolerance': self.grid_orthogonal_shift_tolerance,
            'grid_nominal_shift_bias': self.grid_nominal_shift_bias,
            'grid_phase_fallback_min_response': self.grid_phase_fallback_min_response,
            'grid_phase_fallback_max_shift_ratio': self.grid_phase_fallback_max_shift_ratio,
            "grid_min_blend_score": self.grid_min_blend_score,
            "grid_nominal_score": self.grid_nominal_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GridGuidedStitching:
        """Create a dataclass instance from a dictionary."""
        return cls(
            # enable_grid_phase_fallback=data.get('enable_grid_phase_fallback', True),
            # enable_grid_nominal_fallback=data.get('enable_grid_nominal_fallback', True),
            # enable_grid_shift_guard=data.get('enable_grid_shift_guard', True),
            # enable_grid_absolute_nominal_fallback=data.get('enable_grid_absolute_nominal_fallback', True),
            grid_absolute_nominal_score=as_float(data.get('grid_absolute_nominal_score', 1.0), 1.0),
            grid_expected_shift_tolerance=as_float(data.get('grid_expected_shift_tolerance', 0.12), 0.12),
            grid_orthogonal_shift_tolerance=as_float(data.get('grid_orthogonal_shift_tolerance', 0.15), 0.15),
            grid_nominal_shift_bias=as_float(data.get('grid_nominal_shift_bias', 1.0), 1.0),
            grid_phase_fallback_min_response=as_float(data.get('grid_phase_fallback_min_response', 0.018), 0.018),
            grid_phase_fallback_max_shift_ratio=as_float(data.get('grid_phase_fallback_max_shift_ratio', 0.90), 0.90),
            grid_min_blend_score=as_float(data.get('grid_min_blend_score', 10.0), 10.0),
            grid_nominal_score=as_float(data.get('grid_nominal_score', 2.0), 2.0),
        )

