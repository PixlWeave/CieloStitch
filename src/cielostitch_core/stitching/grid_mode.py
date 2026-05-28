# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import numpy as np
import os
from typing import Optional
import time
import logging
from cielostitch_core.config.config import cfg
from cielostitch_core.config.constants import GRID_MODE_MAX_WARP_DIM, GRID_MODE_MAX_WARP_PIXELS, FIXED_OVERLAP_MODES
from cielostitch_core.core.bundle import BundleAdjustmentDiagnostics, BundleAdjustmentEdge, refine_global_transforms
from cielostitch_core.core.mosaic_canvas import MosaicCanvas
from cielostitch_core.utils.message import emit_msg
from cielostitch_core.stitching.base_mode import BaseStitchMode
from cielostitch_core.utils.strings import first_and_last_part
from cielostitch_app.config.ui_constants import DISPLAY_NAME_SIZE

logger = logging.getLogger(__name__)

class GridMode(BaseStitchMode):

    def __init__(self, profile, grid_cols, show_pixel_stat=False):
        super().__init__(profile, show_pixel_stat=show_pixel_stat)
        self.grid_cols = grid_cols
        self._slot_of_index = None
        self._slot_to_index = None
        self._rows = None
        self._bundle_candidates = []
        self.last_bundle_adjustment_diagnostics = None

    def stitch(self, image_items, progress_cb=None, cancel_cb=None):
        deterministic_mode = cfg.state.stitch_mode in FIXED_OVERLAP_MODES
        # Phase 1: Common initialization (validation, progress counters, feature detection)
        names, images, image_count, features, start_time = self._init_stitch_session(
            image_items, cancel_cb=cancel_cb, progress_cb=progress_cb,
            detect_features=not deterministic_mode
        )

        if deterministic_mode:
            emit_msg("Deterministic mode: skipping feature detection.", "debug", progress_cb)

        # Phase 2: Initialize canvas and placed references
        t_setup = time.perf_counter()
        first_name, first_image = names[0], images[0]
        canvas_system = MosaicCanvas(
            first_image,
            enable_overlap_diagnostics=bool(getattr(cfg.prefs, "retain_overlap_diagnostics", True)),
            enable_seam_diagnostics=bool(getattr(cfg.prefs, "retain_seam_diagnostics", True)),
        )
        self._init_grid_layout(image_count)
        # emit_msg(f"Session stitch mode: {cfg.state.stitch_mode}", "debug", progress_cb)
        emit_msg(f"Completing post-detect setup", "", progress_cb)

        global_transforms, placement_scores, skipped = self._compute_global_transforms(
            image_count, images, names, features, cancel_cb, progress_cb
        )
        global_transforms = self._refine_global_transforms(global_transforms, progress_cb)

        self._preallocate_canvas_once(canvas_system, images, global_transforms)
        emit_msg(f"Post-detect setup {time.perf_counter() - t_setup:.2f}s", "", progress_cb)
        # emit_msg(f"Starting with {first_name}", "", progress_cb)

        self._warp_and_blend_panels(
            canvas_system, images, names, image_count,
            global_transforms, placement_scores,
            cancel_cb, progress_cb
        )

        self._sync_random_counters(image_count, skipped)
        return self._finalize_stitch(canvas_system, start_time, progress_cb=progress_cb, heal_seams=True)

    @staticmethod
    def _preallocate_canvas_once(canvas_system, images, global_transforms):
        min_x = 0.0
        min_y = 0.0
        max_x = float(canvas_system.canvas.shape[1])
        max_y = float(canvas_system.canvas.shape[0])

        for image, global_transform in zip(images, global_transforms):
            if global_transform is None:
                continue
            bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y = MosaicCanvas.compute_warp_bounds(image, global_transform)
            min_x = min(min_x, float(bounds_min_x))
            min_y = min(min_y, float(bounds_min_y))
            max_x = max(max_x, float(bounds_max_x))
            max_y = max(max_y, float(bounds_max_y))

        canvas_system.expand_canvas(min_x, min_y, max_x, max_y)

    # -------------------------
    # GRID SETUP
    # -------------------------

    def _init_grid_layout(self, image_count):
        slot_of_index = list(self._build_scan_order(image_count))
        if len(slot_of_index) != image_count:
            logger.warning(f"Scan order generation returned {len(slot_of_index)} slots instead of {image_count}. Falling back to sequential order.")
            slot_of_index = list(range(image_count))

        self._slot_of_index = slot_of_index
        self._slot_to_index = {slot: idx for idx, slot in enumerate(slot_of_index)}
        self._rows = int(np.ceil(image_count / float(self.grid_cols)))

    def _slot_row_col(self, slot):
        return slot // self.grid_cols, slot % self.grid_cols

    def _get_expected_reference_slot(self, cur_slot):
        idx = self._slot_to_index.get(cur_slot)
        if idx is None or idx == 0:
            return None
        return self._slot_of_index[idx - 1]

    # -------------------------
    # CORE TRANSFORMS
    # -------------------------

    def _compute_global_transforms(self, image_count, images, names, features, cancel_cb, progress_cb):
        global_transforms: list[Optional[np.ndarray]] = [None] * image_count
        placement_scores = [0.0] * image_count
        self._bundle_candidates = [[] for _ in range(image_count)]
        self.last_bundle_adjustment_diagnostics = None

        global_transforms[0] = np.eye(3)
        placement_scores[0] = 1e9
        skipped: int = 0
        counts = getattr(self, "_placement_source_counts", None)

        if cfg.state.stitch_mode in FIXED_OVERLAP_MODES:
            self._compute_deterministic_transforms(
                image_count, images, global_transforms, placement_scores, cancel_cb
            )
        else: # UNKNOWN_OVERLAP_MODES i.e. "grid-guided", "freeform"
            if counts is not None:
                counts["anchor"] = counts.get("anchor", 0) + 1
            self._record_panel_source(0, "anchor")
            skipped = self._compute_feature_transforms(
                image_count, images, names, features,
                global_transforms, placement_scores,
                cancel_cb, progress_cb
            )

        return global_transforms, placement_scores, skipped

    # -------------------------
    # DETERMINISTIC
    # -------------------------

    def _compute_deterministic_transforms(self, image_count, images, global_transforms, placement_scores, cancel_cb):
        
        # The first image already defines the canvas origin, so deterministic
        # placement must be expressed in that same reference frame.

        if not self._slot_of_index or len(self._slot_of_index) != image_count:
            raise ValueError(f"Grid initialization failed: expected {image_count} slots, got {len(self._slot_of_index) if self._slot_of_index else 0}")

        step_x, step_y = self._get_overlap_step(images[0])
        anchor_row, anchor_col = self._slot_row_col(self._slot_of_index[0])

        counts = getattr(self, "_placement_source_counts", None)
        for scan_index in range(image_count):
            self.check_cancel(cancel_cb)

            current_slot = self._slot_of_index[scan_index]
            row, col = self._slot_row_col(current_slot)

            global_transform = np.eye(3)
            # anchor_row and anchor_col affect every tile only by changing the
            # shared reference origin.
            global_transform[0, 2] = (col - anchor_col) * step_x
            global_transform[1, 2] = (row - anchor_row) * step_y

            # Transform arrays are keyed by image index (scan/capture order),
            # not by grid slot id.
            global_transforms[scan_index] = global_transform
            placement_scores[scan_index] = 1e9
            if counts is not None:
                counts["deterministic"] = counts.get("deterministic", 0) + 1
            self._record_panel_source(scan_index, "deterministic")

    # -------------------------
    # FEATURE MODE (ADAPTIVE)
    # -------------------------

    def _compute_feature_transforms(self, image_count, images, names, features,
                                    global_transforms, placement_scores,
                                    cancel_cb, progress_cb) -> int:

        skipped: int = 0

        def _normalize_candidate(candidate):
            if candidate is None:
                return None
            if len(candidate) >= 4:
                return candidate
            if len(candidate) == 3:
                return candidate[0], candidate[1], candidate[2], None
            return candidate[0], candidate[1], "feature", None

        # Image index 0 is the anchor, so placement starts at index 1.
        for scan_index in range(1, image_count):
            self.check_cancel(cancel_cb)

            current_index = scan_index
            current_slot = self._slot_of_index[current_index]
            current_row, current_col = self._slot_row_col(current_slot)

            best_candidate = None
            candidates = []
            step1_candidate = None
            step1_ref_idx = None
            step1_relation = None

            # Step 1: try the expected predecessor from scan order.
            expected_reference_slot = self._get_expected_reference_slot(current_slot)

            if expected_reference_slot is not None:
                ref_idx = self._slot_to_index.get(expected_reference_slot)

                if ref_idx is not None and ref_idx < current_index and global_transforms[ref_idx] is not None:

                    relation = self._scan_relation(expected_reference_slot, current_slot)

                    if relation:
                        step1_ref_idx = ref_idx
                        step1_relation = relation
                        candidate = self._pair_global(
                            ref_idx, current_index, relation,
                            global_transforms, features, images,
                            cancel_cb, progress_cb,
                            return_source=True,
                        )
                        candidate = _normalize_candidate(candidate)
                        step1_candidate = candidate

                        if candidate is not None and candidate[2] != "nominal":
                            best_candidate = candidate

            # Step 2: broaden to already-placed grid neighbors.
            if best_candidate is None:
                if step1_candidate is not None:
                    candidates.append(step1_candidate)

                neighbors = [
                    (current_row, current_col - 1, "left"),
                    (current_row, current_col + 1, "right"),
                    (current_row - 1, current_col, "top"),
                    (current_row + 1, current_col, "bottom"),
                ]

                for neighbor_row, neighbor_col, relation in neighbors:
                    if neighbor_row < 0 or neighbor_col < 0 or neighbor_col >= self.grid_cols or neighbor_row >= self._rows:
                        continue

                    neighbor_slot = neighbor_row * self.grid_cols + neighbor_col
                    ref_scan_idx = self._slot_to_index.get(neighbor_slot)

                    if ref_scan_idx is None or ref_scan_idx >= current_index or global_transforms[ref_scan_idx] is None:
                        continue

                    # Skip pair already tried in Step 1 to avoid duplicate processing.
                    if ref_scan_idx == step1_ref_idx and relation == step1_relation:
                        continue

                    candidate = self._pair_global(
                        ref_scan_idx, current_index, relation,
                        global_transforms, features, images,
                        cancel_cb, progress_cb,
                        return_source=True,
                    )
                    candidate = _normalize_candidate(candidate)

                    if candidate is not None:
                        candidates.append(candidate)

                if candidates:
                    self._record_bundle_candidates(current_index, candidates, global_transforms)
                    non_nominal_candidates = [candidate for candidate in candidates if candidate[2] != "nominal"]
                    ranked_candidates = non_nominal_candidates if non_nominal_candidates else candidates
                    ranked_candidates.sort(key=lambda x: float(x[1]), reverse=True)
                    best_candidate = ranked_candidates[0]

            if best_candidate is not None and not candidates:
                self._record_bundle_candidates(current_index, [best_candidate], global_transforms)

            # Finalize the chosen candidate for this image.
            if best_candidate is None:
                skipped += 1
                self._record_panel_source(current_index, "skipped")
                file_name = os.path.basename(names[current_index])
                emit_msg(f"Skipping {first_and_last_part(file_name, DISPLAY_NAME_SIZE)} (no valid transform)",
                         "red", progress_cb)
                continue

            chosen_transform, chosen_score, chosen_source, chosen_meta = best_candidate
            global_transforms[current_index] = chosen_transform
            placement_scores[current_index] = float(chosen_score)
            counts = getattr(self, "_placement_source_counts", None)
            if counts is not None:
                counts[chosen_source] = counts.get(chosen_source, 0) + 1
            self._record_panel_source(current_index, chosen_source)
            self._record_panel_evidence(
                current_index,
                chosen_meta if chosen_source == "feature" and chosen_meta else None,
            )
            if chosen_source == "feature" and chosen_meta and chosen_meta.get("subpixel_refinement_applied", False):
                self._record_subpixel_refinement_applied(current_index)
                emit_msg(
                    f"Subpixel alignment: ref={chosen_meta['ref_idx']}, cur={current_index}, dx={float(chosen_meta.get('subpixel_refinement_dx', 0.0)):.2f}, dy={float(chosen_meta.get('subpixel_refinement_dy', 0.0)):.2f}, response={float(chosen_meta.get('subpixel_refinement_response', 0.0)):.3f}",
                    "panel",
                    progress_cb,
                )

        return skipped

    def _record_bundle_candidates(self, image_index, candidates, global_transforms):
        if not hasattr(self, "_bundle_candidates") or image_index >= len(self._bundle_candidates):
            return
        bucket = self._bundle_candidates[image_index]
        seen_signatures = {
            (
                int(candidate.ref_index),
                round(float(candidate.score), 6),
                candidate.source,
                tuple(np.asarray(candidate.relative_transform, dtype=np.float64).round(6).ravel()),
            )
            for candidate in bucket
        }
        for candidate in candidates or []:
            if candidate is None or len(candidate) < 4:
                continue
            transform = candidate[0]
            score = candidate[1]
            source = candidate[2]
            meta = candidate[3] if isinstance(candidate[3], dict) else None
            if source == "nominal" or transform is None:
                continue
            ref_index = int(meta.get("ref_idx", -1)) if meta is not None else -1
            if ref_index < 0 or ref_index >= len(global_transforms):
                continue
            ref_transform = global_transforms[ref_index]
            if ref_transform is None:
                continue
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                continue
            try:
                relative_transform = np.linalg.inv(np.asarray(ref_transform, dtype=np.float64)) @ matrix
            except np.linalg.LinAlgError:
                continue
            signature = (
                ref_index,
                round(float(score), 6),
                str(source),
                tuple(np.asarray(relative_transform, dtype=np.float64).round(6).ravel()),
            )
            if signature in seen_signatures:
                continue
            bucket.append(
                BundleAdjustmentEdge(
                    ref_index=ref_index,
                    image_index=image_index,
                    relative_transform=np.asarray(relative_transform, dtype=np.float64).copy(),
                    score=float(score),
                    source=str(source),
                )
            )
            seen_signatures.add(signature)

    def _refine_global_transforms(self, global_transforms, progress_cb=None):
        mode = str(getattr(cfg.state.ssm, "bundle_adjustment_mode", "off") or "off").strip().lower()
        self.last_bundle_adjustment_diagnostics = BundleAdjustmentDiagnostics(
            mode=mode if mode else "off",
            edge_count=0,
            adjusted_panels=0,
            mean_translation_shift_px=0.0,
            max_translation_shift_px=0.0,
            iterations=0,
            status="disabled" if mode == "off" else "pending",
        )
        if mode == "off":
            return global_transforms
        try:
            refined, diagnostics = refine_global_transforms(global_transforms, self._bundle_candidates, mode=mode)
        except Exception:
            logger.debug("Bundle refinement failed; keeping original transforms", exc_info=True)
            self.last_bundle_adjustment_diagnostics = BundleAdjustmentDiagnostics(
                mode=mode,
                edge_count=0,
                adjusted_panels=0,
                mean_translation_shift_px=0.0,
                max_translation_shift_px=0.0,
                iterations=0,
                status="error",
            )
            return global_transforms
        self.last_bundle_adjustment_diagnostics = diagnostics
        if diagnostics.status == "ok":
            emit_msg(
                f"Bundle refinement applied: mode={mode}, edges={diagnostics.edge_count}, adjusted={diagnostics.adjusted_panels}, mean_shift={diagnostics.mean_translation_shift_px:.2f}px, max_shift={diagnostics.max_translation_shift_px:.2f}px",
                "debug",
                progress_cb,
            )
        else:
            emit_msg(f"Bundle refinement skipped: mode={mode}, status={diagnostics.status}", "debug", progress_cb)
        return refined

    # -------------------------
    # RELATION
    # -------------------------

    def _scan_relation(self, prev_slot, cur_slot):
        prev_row, prev_col = self._slot_row_col(prev_slot)
        curr_row, curr_col = self._slot_row_col(cur_slot)

        delta_row = curr_row - prev_row
        delta_col = curr_col - prev_col

        if delta_row == 0 and delta_col == 1: return "left"
        if delta_row == 0 and delta_col == -1: return "right"
        if delta_row == 1 and delta_col == 0: return "top"
        if delta_row == -1 and delta_col == 0: return "bottom"

        return None

    # -------------------------
    # MATCH + SAFE FALLBACK
    # -------------------------

    def _pair_global(self, ref_idx, cur_idx, relation, global_h,
                     features, images, cancel_cb, progress_cb=None,
                     return_source=False):
        enable_grid_phase_fallback = cfg.state.ssm.enable_grid_phase_fallback  
        grid_phase_fallback_min_response = cfg.state.ssm.grid_guide.grid_phase_fallback_min_response  # 0.02
        grid_phase_fallback_max_shift_ratio = cfg.state.ssm.grid_guide.grid_phase_fallback_max_shift_ratio  # 0.9

        self.check_cancel(cancel_cb)

        if global_h[ref_idx] is None:
            return None

        ref_keypoints, ref_descriptors, ref_scale = features[ref_idx]
        cur_keypoints, cur_descriptors, cur_scale = features[cur_idx]

        local_transform, inlier_mask, feature_metrics = self._match_feature_pair_with_metrics(
            ref_keypoints, ref_descriptors, ref_scale,
            cur_keypoints, cur_descriptors, cur_scale,
            ref_image=images[ref_idx],
            current_image=images[cur_idx],
        )
        weak_feature_evidence = self._is_weak_feature_evidence(feature_metrics)

        if weak_feature_evidence:
            emit_msg(
                f"Weak-texture pair: ref={ref_idx}, cur={cur_idx}, relation={relation}, "
                f"kp=({feature_metrics['kp_ref']},{feature_metrics['kp_curr']}), matches={feature_metrics['match_count']}",
                "debug",
                progress_cb,
            )

        def _try_phase_candidate():
            if not enable_grid_phase_fallback:
                return None

            phase_local_transform, response = self._estimate_translation_fallback(
                images[ref_idx], images[cur_idx],
                min_response=grid_phase_fallback_min_response,
                max_shift_ratio=grid_phase_fallback_max_shift_ratio,
            )
            if phase_local_transform is None:
                return None

            if not self._is_plausible_neighbor_shift(phase_local_transform, images[ref_idx], relation):
                shift_x = float(phase_local_transform[0, 2])
                shift_y = float(phase_local_transform[1, 2])
                emit_msg(
                    f"Rejected phase fallback by grid guard: ref={ref_idx}, cur={cur_idx}, relation={relation}, tx={shift_x:.1f}, ty={shift_y:.1f}, response={float(response):.3f}",
                    "debug",
                    progress_cb,
                )
                return None

            score = int(round(response * 1000.0))
            return global_h[ref_idx] @ phase_local_transform, score, "phase", {"ref_idx": ref_idx}

        # Prefer phase early only when the pair looks weak-texture and the
        # feature path did not already produce a usable transform.
        # Use a sentinel so we can tell "not yet tried" apart from "tried, got None".
        _not_tried = object()
        phase_candidate = _not_tried
        if weak_feature_evidence and local_transform is None:
            phase_candidate = _try_phase_candidate()
            if phase_candidate is not None:
                emit_msg(
                    f"Phase fallback preferred for weak-texture pair after feature failure: ref={ref_idx}, cur={cur_idx}, relation={relation}",
                    "debug",
                    progress_cb,
                )
                return phase_candidate if return_source else phase_candidate[:2]

        if local_transform is not None:
            if not self._is_plausible_neighbor_shift(local_transform, images[ref_idx], relation):
                shift_x = float(local_transform[0, 2])
                shift_y = float(local_transform[1, 2])
                emit_msg(
                    f"Rejected feature match by grid guard: ref={ref_idx}, cur={cur_idx}, relation={relation}, tx={shift_x:.1f}, ty={shift_y:.1f}",
                    "debug",
                    progress_cb,
                )
                nominal_candidate = self._try_nominal_fallback(ref_idx, relation, images, global_h)
                if nominal_candidate is None:
                    return None
                return nominal_candidate + ("nominal",) if return_source else nominal_candidate

            score = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
            feature_candidate = (
                global_h[ref_idx] @ local_transform,
                score,
                "feature",
                {
                    "ref_idx": ref_idx,
                    "diagnostic_label": str(feature_metrics.get("diagnostic_label", "")),
                    "match_count": int(feature_metrics.get("match_count", 0) or 0),
                    "inlier_count": int(feature_metrics.get("inlier_count", 0) or 0),
                    "inlier_ratio": float(feature_metrics.get("inlier_ratio", 0.0) or 0.0),
                    "subpixel_refinement_applied": bool(feature_metrics.get("subpixel_refinement_applied", False)),
                    "subpixel_refinement_dx": float(feature_metrics.get("subpixel_refinement_dx", 0.0)),
                    "subpixel_refinement_dy": float(feature_metrics.get("subpixel_refinement_dy", 0.0)),
                    "subpixel_refinement_response": float(feature_metrics.get("subpixel_refinement_response", 0.0)),
                },
            )
            return feature_candidate if return_source else feature_candidate[:2]

        if phase_candidate is _not_tried:
            phase_candidate = _try_phase_candidate()
        if phase_candidate is not None:
            return phase_candidate if return_source else phase_candidate[:2]

        nominal_candidate = self._try_nominal_fallback(ref_idx, relation, images, global_h)
        if nominal_candidate is None:
            return None
        return nominal_candidate + ("nominal",) if return_source else nominal_candidate

    # -------------------------
    # WARP
    # -------------------------

    def _warp_and_blend_panels(self, canvas_system, images, names, n,
                               global_h, placement_score,
                               cancel_cb, progress_cb):

        for i in range(1, n): #1, n correct
            if global_h[i] is None:
                continue

            emit_msg(f"{first_and_last_part(names[i], DISPLAY_NAME_SIZE)}", "panel", progress_cb)
            if progress_cb:
                try:
                    progress_cb(i, n)
                except Exception as e:
                    logger.debug(f"Progress callback failed: {e}")

            self._warp_and_blend_single_image(
                canvas_system,
                images[i],
                global_h[i],
                names[i],
                i,
                placement_score=placement_score[i],
                min_blend_score=float(cfg.state.ssm.grid_guide.grid_min_blend_score),
                max_allowed_size=GRID_MODE_MAX_WARP_DIM,
                max_allowed_pixels=GRID_MODE_MAX_WARP_PIXELS,
                cancel_cb=cancel_cb,
                progress_cb=progress_cb
            )

    # -------------------------
    # SCAN ORDER
    # -------------------------

    def _build_scan_order(self, n):
        scan_order = cfg.state.ssm.scan_order
        alternating = bool(cfg.state.ssm.alternating == "yes")
        start_corner = cfg.state.ssm.start_corner

        rows = int(np.ceil(n / float(self.grid_cols)))
        top_first = start_corner.startswith("top")
        left_first = start_corner.endswith("left")
        row_base = list(range(rows)) if top_first else list(range(rows - 1, -1, -1))
        col_base = list(range(self.grid_cols)) if left_first else list(range(self.grid_cols - 1, -1, -1))

        order = []

        if scan_order == "column-wise":
            for idx, col in enumerate(col_base):
                row_order = row_base if (not alternating or idx % 2 == 0) else list(reversed(row_base))
                for row in row_order:
                    slot = row * self.grid_cols + col
                    if 0 <= slot < n:
                        order.append(slot)
        else:
            for idx, row in enumerate(row_base):
                col_order = col_base if (not alternating or idx % 2 == 0) else list(reversed(col_base))
                for col in col_order:
                    slot = row * self.grid_cols + col
                    if 0 <= slot < n:
                        order.append(slot)

        return order

    ## added
    def _is_plausible_neighbor_shift(self, h_local, ref_img, relation):
        enable_grid_shift_guard = cfg.state.ssm.enable_grid_shift_guard 
        grid_expected_shift_tolerance = cfg.state.ssm.grid_guide.grid_expected_shift_tolerance # 0.30
        grid_orthogonal_shift_tolerance = cfg.state.ssm.grid_guide.grid_orthogonal_shift_tolerance # 0.20
        if not enable_grid_shift_guard or relation is None:
            return True

        shift_x = float(h_local[0, 2])
        shift_y = float(h_local[1, 2])
        # h, w = ref_img.shape[:2]

        # Expected neighbor step from nominal overlap.
        step_x, step_y = self._get_overlap_step(ref_img)
        step_x = max(1.0, step_x)
        step_y = max(1.0, step_y)

        tol = float(np.clip(grid_expected_shift_tolerance, 0.05, 0.80))
        ortho_tol = float(np.clip(grid_orthogonal_shift_tolerance, 0.02, 0.80))

        if relation == "left":
            # Current tile is to the right of the reference tile.
            return (
                (shift_x > 0.0) and
                (step_x * (1.0 - tol) <= shift_x <= step_x * (1.0 + tol)) and
                (abs(shift_y) <= step_y * ortho_tol)
            )
        if relation == "right":
            # Current tile is to the left of the reference tile.
            return (
                (shift_x < 0.0) and
                (-step_x * (1.0 + tol) <= shift_x <= -step_x * (1.0 - tol)) and
                (abs(shift_y) <= step_y * ortho_tol)
            )
        if relation == "top":
            # Current tile is below the reference tile.
            return (
                (shift_y > 0.0) and
                (step_y * (1.0 - tol) <= shift_y <= step_y * (1.0 + tol)) and
                (abs(shift_x) <= step_x * ortho_tol)
            )
        if relation == "bottom":
            # Current tile is above the reference tile.
            return (
                (shift_y < 0.0) and
                (-step_y * (1.0 + tol) <= shift_y <= -step_y * (1.0 - tol)) and
                (abs(shift_x) <= step_x * ortho_tol)
            )
        return False
