# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleAdjustmentCandidate:
    image_index: int
    transform: np.ndarray
    score: float
    source: str


@dataclass(frozen=True)
class BundleAdjustmentEdge:
    ref_index: int
    image_index: int
    relative_transform: np.ndarray
    score: float
    source: str


@dataclass(frozen=True)
class BundleAdjustmentDiagnostics:
    mode: str
    edge_count: int
    adjusted_panels: int
    mean_translation_shift_px: float
    max_translation_shift_px: float
    iterations: int
    status: str


def _normalize_mode(mode: str | None) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized in {"translation", "affine"}:
        return normalized
    return "off"


def _as_matrix(transform: np.ndarray | None) -> np.ndarray | None:
    if transform is None:
        return None
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    if abs(float(matrix[2, 2])) < 1e-10:
        return None
    return matrix / float(matrix[2, 2])


def _is_affine_like(transform: np.ndarray, *, perspective_tol: float = 1e-6) -> bool:
    return (
        abs(float(transform[2, 0])) <= perspective_tol
        and abs(float(transform[2, 1])) <= perspective_tol
        and abs(float(transform[2, 2]) - 1.0) <= perspective_tol
    )


def _invert_affine(transform: np.ndarray) -> np.ndarray | None:
    matrix = _as_matrix(transform)
    if matrix is None or not _is_affine_like(matrix):
        return None
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def _compose_affine(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    a = _as_matrix(left)
    b = _as_matrix(right)
    if a is None or b is None or not _is_affine_like(a) or not _is_affine_like(b):
        return None
    composed = a @ b
    return _as_matrix(composed)


def _translation_shift(before: np.ndarray | None, after: np.ndarray | None) -> float:
    if before is None or after is None:
        return 0.0
    return float(np.hypot(float(after[0, 2] - before[0, 2]), float(after[1, 2] - before[1, 2])))


def _solve_translation(
    initial: list[np.ndarray | None],
    edges: Sequence[BundleAdjustmentEdge],
    *,
    iterations: int = 6,
    prior_weight: float = 1.0,
) -> list[np.ndarray | None]:
    refined = [None if transform is None else np.asarray(transform, dtype=np.float64).copy() for transform in initial]
    if not refined or refined[0] is None:
        return refined

    base_positions = [None if transform is None else np.asarray(transform[:2, 2], dtype=np.float64).copy() for transform in refined]
    current_positions = [None if value is None else value.copy() for value in base_positions]

    for _ in range(iterations):
        next_positions = [None if value is None else value.copy() for value in current_positions]
        for image_index in range(1, len(refined)):
            if current_positions[image_index] is None or base_positions[image_index] is None:
                continue
            weighted_sum = np.asarray(base_positions[image_index], dtype=np.float64) * float(prior_weight)
            total_weight = float(prior_weight)
            for edge in edges:
                rel = _as_matrix(edge.relative_transform)
                if rel is None:
                    continue
                weight = max(1.0, float(edge.score))
                if edge.image_index == image_index and current_positions[edge.ref_index] is not None:
                    weighted_sum += (np.asarray(current_positions[edge.ref_index], dtype=np.float64) + rel[:2, 2]) * weight
                    total_weight += weight
                elif edge.ref_index == image_index and current_positions[edge.image_index] is not None:
                    weighted_sum += (np.asarray(current_positions[edge.image_index], dtype=np.float64) - rel[:2, 2]) * weight
                    total_weight += weight
            if total_weight > 0.0:
                next_positions[image_index] = weighted_sum / total_weight
        current_positions = next_positions

    for image_index in range(1, len(refined)):
        if refined[image_index] is None or current_positions[image_index] is None:
            continue
        refined[image_index][0, 2] = float(current_positions[image_index][0])
        refined[image_index][1, 2] = float(current_positions[image_index][1])
    return refined


def _solve_affine(
    initial: list[np.ndarray | None],
    edges: Sequence[BundleAdjustmentEdge],
    *,
    iterations: int = 6,
    prior_weight: float = 1.0,
) -> list[np.ndarray | None]:
    refined = [None if transform is None else np.asarray(transform, dtype=np.float64).copy() for transform in initial]
    if not refined or refined[0] is None:
        return refined

    base_affines = [None if transform is None else np.asarray(transform[:2, :], dtype=np.float64).copy() for transform in refined]
    current_affines = [None if value is None else value.copy() for value in base_affines]

    for _ in range(iterations):
        next_affines = [None if value is None else value.copy() for value in current_affines]
        for image_index in range(1, len(refined)):
            if current_affines[image_index] is None or base_affines[image_index] is None:
                continue
            weighted_sum = np.asarray(base_affines[image_index], dtype=np.float64) * float(prior_weight)
            total_weight = float(prior_weight)
            for edge in edges:
                rel = _as_matrix(edge.relative_transform)
                if rel is None or not _is_affine_like(rel):
                    continue
                weight = max(1.0, float(edge.score))
                if edge.image_index == image_index and current_affines[edge.ref_index] is not None:
                    ref_matrix = np.eye(3, dtype=np.float64)
                    ref_matrix[:2, :] = current_affines[edge.ref_index]
                    estimate = _compose_affine(ref_matrix, rel)
                    if estimate is not None:
                        weighted_sum += estimate[:2, :] * weight
                        total_weight += weight
                elif edge.ref_index == image_index and current_affines[edge.image_index] is not None:
                    inv_rel = _invert_affine(rel)
                    if inv_rel is None:
                        continue
                    cur_matrix = np.eye(3, dtype=np.float64)
                    cur_matrix[:2, :] = current_affines[edge.image_index]
                    estimate = _compose_affine(cur_matrix, inv_rel)
                    if estimate is not None:
                        weighted_sum += estimate[:2, :] * weight
                        total_weight += weight
            if total_weight <= 0.0:
                continue
            candidate = weighted_sum / total_weight
            determinant = float(np.linalg.det(candidate[:, :2]))
            if not np.isfinite(determinant) or abs(determinant) < 1e-8:
                continue
            next_affines[image_index] = candidate
        current_affines = next_affines

    for image_index in range(1, len(refined)):
        if refined[image_index] is None or current_affines[image_index] is None:
            continue
        refined[image_index][:2, :] = current_affines[image_index]
    return refined


def refine_global_transforms(
    global_transforms: Sequence[np.ndarray | None],
    candidate_groups: Sequence[Sequence[BundleAdjustmentEdge | BundleAdjustmentCandidate] | None],
    *,
    mode: str | None,
) -> tuple[list[np.ndarray | None], BundleAdjustmentDiagnostics]:
    normalized_mode = _normalize_mode(mode)
    refined = [None if transform is None else np.asarray(transform, dtype=np.float64).copy() for transform in global_transforms]
    if normalized_mode == "off":
        return refined, BundleAdjustmentDiagnostics(
            mode="off",
            edge_count=0,
            adjusted_panels=0,
            mean_translation_shift_px=0.0,
            max_translation_shift_px=0.0,
            iterations=0,
            status="disabled",
        )

    edges: list[BundleAdjustmentEdge] = []
    for group in candidate_groups:
        if not group:
            continue
        for item in group:
            if isinstance(item, BundleAdjustmentEdge):
                if item.ref_index < 0 or item.image_index < 0:
                    continue
                if item.ref_index >= len(refined) or item.image_index >= len(refined):
                    continue
                rel = _as_matrix(item.relative_transform)
                if rel is None:
                    continue
                if normalized_mode == "affine" and not _is_affine_like(rel):
                    continue
                edges.append(item)

    if not edges:
        return refined, BundleAdjustmentDiagnostics(
            mode=normalized_mode,
            edge_count=0,
            adjusted_panels=0,
            mean_translation_shift_px=0.0,
            max_translation_shift_px=0.0,
            iterations=0,
            status="no_edges",
        )

    iterations = 6
    if normalized_mode == "translation":
        refined = _solve_translation(refined, edges, iterations=iterations)
    else:
        refined = _solve_affine(refined, edges, iterations=iterations)

    shifts = [_translation_shift(before, after) for before, after in zip(global_transforms, refined)]
    adjusted_panels = sum(1 for shift in shifts[1:] if shift > 1e-6)
    non_anchor_shifts = [shift for shift in shifts[1:] if shift > 0.0]
    mean_shift = float(np.mean(non_anchor_shifts)) if non_anchor_shifts else 0.0
    max_shift = float(max(non_anchor_shifts)) if non_anchor_shifts else 0.0
    return refined, BundleAdjustmentDiagnostics(
        mode=normalized_mode,
        edge_count=len(edges),
        adjusted_panels=adjusted_panels,
        mean_translation_shift_px=mean_shift,
        max_translation_shift_px=max_shift,
        iterations=iterations,
        status="ok",
    )