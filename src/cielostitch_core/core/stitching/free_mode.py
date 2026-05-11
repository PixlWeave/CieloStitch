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
import logging
from ...config.config import cfg
from ...config.constants import AUTO_MODE_MAX_WARP_DIM
from ...core.mosaic_canvas import MosaicCanvas
from ...utils.message import emit_msg
from ...core.stitching.base_mode import BaseStitchMode

logger = logging.getLogger(__name__)

class FreeMode(BaseStitchMode):

    def __init__(self, profile, show_pixel_stat=False):
        super().__init__(profile, show_pixel_stat=show_pixel_stat)

    def stitch(self, image_items, progress_cb=None, cancel_cb=None):
        """Auto stitching mode: matches each image to the best anchor among all previously placed images."""
        # Phase 1: Common initialization (validation, progress counters, feature detection)
        names, images, image_count, features, start_time = self._init_stitch_session(
            image_items, cancel_cb=cancel_cb, progress_cb=progress_cb
        )

        # Phase 2: Initialize canvas and placed references
        first_name, first_image = image_items[0]
        canvas_system = MosaicCanvas(
            first_image,
            enable_overlap_diagnostics=bool(getattr(cfg.prefs, "retain_overlap_diagnostics", True)),
            enable_seam_diagnostics=bool(getattr(cfg.prefs, "retain_seam_diagnostics", True)),
        )

        emit_msg(f"Session stitch mode: {cfg.state.stitch_mode}", "debug", progress_cb)
        emit_msg(
            f"Rotation lock: {'on' if bool(getattr(self.profile, 'lock_rotation', False)) else 'off'}",
            "debug",
            progress_cb,
        )
        emit_msg(
            f"Candidate debug details: {'enabled' if bool(getattr(cfg.prefs, 'enable_candidate_debug', False)) else 'disabled'}",
            "debug",
            progress_cb,
        )
        placed_refs = self._init_random_references(first_image, first_name, features[0])
        counts = getattr(self, "_placement_source_counts", None)
        if counts is not None:
            counts["anchor"] = counts.get("anchor", 0) + 1
        self._record_panel_source(0, "anchor")

        # Phase 3: Iteratively match and place images
        global_transform = np.eye(3, dtype=np.float64)
        skipped = self._match_and_place_images(
            canvas_system, image_items, features, placed_refs, global_transform,
            cancel_cb, progress_cb
        )

        # Phase 4: Finalize
        emit_msg(f"Translation: {global_transform[0, 2]:.2f}, {global_transform[1, 2]:.2f}", "debug", progress_cb)
        self._sync_random_counters(len(image_items), skipped)
        return self._finalize_stitch(canvas_system, start_time, progress_cb=progress_cb, heal_seams=False)

    def _init_random_references(self, first_image, first_name, first_features):
        """Initialize list of placed reference images with the first image."""
        return [self._create_anchor_reference(first_name, first_image, first_features)]

    def _try_place_image(self, canvas_system, image_items, features, placed_refs, global_transform,
                         image_index, image_count, cancel_cb, progress_cb, skip_color="yellow"):
        """Attempt to place a single image against the current placed references."""
        self.check_cancel(cancel_cb)
        current_name, current_image = image_items[image_index]
        current_keypoints, current_descriptors, current_scale = features[image_index]

        emit_msg(
            f"{current_name}: kp_prev = {len(placed_refs[-1]['kp'])}, kp_curr = {len(current_keypoints)}",
            "panel",
            progress_cb,
        )

        best_match = self._find_best_match_or_fallback(
            current_keypoints, current_descriptors, current_scale, current_name, current_image,
            placed_refs, image_count, current_image.shape[:2],
            cancel_cb, progress_cb, skip_color=skip_color
        )
        if best_match is None:
            self._record_panel_source(image_index, "skipped")
            return False

        success = self._process_matched_image(
            canvas_system, current_name, current_image,
            best_match, global_transform, image_index, cancel_cb, progress_cb
        )
        if not success:
            self._record_panel_source(image_index, "skipped")
            self._record_panel_evidence(image_index, None)
            return False

        placed_refs.append(self._create_reference(
            name=current_name,
            image=current_image,
            kp=current_keypoints,
            des=current_descriptors,
            scale=current_scale,
            global_h=global_transform.copy()
        ))
        return True

    def _match_and_place_images(self, canvas_system, image_items, features, placed_refs, global_transform, cancel_cb, progress_cb):
        """Match each remaining image to the best placed reference and blend it.

        Makes one extra retry pass for panels skipped before their neighbors were placed.
        """
        skipped = 0
        panels_placed = 0
        image_count = len(image_items)
        retry_indices = []

        # Image index 0 is the anchor, so placement starts at index 1.
        for image_index in range(1, image_count):
            success = self._try_place_image(
                canvas_system, image_items, features, placed_refs, global_transform,
                image_index, image_count, cancel_cb, progress_cb
            )
            if not success:
                skipped += 1
                retry_indices.append(image_index)
                # Update progress bar even for skipped
                panels_placed += 1
                if progress_cb:
                    try:
                        progress_cb(panels_placed, image_count - 1)
                    except Exception as e:
                        logger.debug(f"Progress callback failed: {e}")
                continue

            # Update progress after a successful placement.
            panels_placed += 1
            if progress_cb:
                try:
                    progress_cb(panels_placed, image_count - 1)
                except Exception as e:
                    logger.debug(f"Progress callback failed: {e}")

        if retry_indices and len(placed_refs) > 1:
            emit_msg(
                f"Retrying {len(retry_indices)} skipped panel(s) after initial pass",
                "debug",
                progress_cb,
            )
            remaining_retries = []
            retry_successes = 0
            for image_index in retry_indices:
                success = self._try_place_image(
                    canvas_system, image_items, features, placed_refs, global_transform,
                    image_index, image_count, cancel_cb, progress_cb, skip_color="red"
                )
                if success:
                    skipped -= 1
                    retry_successes += 1
                    continue
                remaining_retries.append(image_index)

            emit_msg(
                f"Retry pass placed {retry_successes} panel(s)",
                "debug",
                progress_cb,
            )

            if remaining_retries:
                emit_msg(
                    f"Retry pass could not place {len(remaining_retries)} panel(s)",
                    "debug",
                    progress_cb,
                )

        return skipped

    def _find_best_match_or_fallback(self, current_keypoints, current_descriptors, current_scale, current_name, current_image,
                                     placed_refs, total_panels, current_shape,
                                     cancel_cb, progress_cb, skip_color="yellow"):
        """Find the best matching reference via staged or full search, then fallback if needed."""
        staged_mode = cfg.state.ssm.staged_matching_mode or "auto"
        refs_per_stage = int(cfg.state.ssm.refs_per_stage or 6)
        enable_staged = staged_mode != "disabled"
        allow_rotation = not bool(getattr(self.profile, "lock_rotation", False))

        phase_fallback_references = placed_refs

        if enable_staged:
            if staged_mode == "manual":
                refs_per_stage = max(4, min(8, refs_per_stage))
            else:
                refs_per_stage = self._compute_auto_refs_per_stage(
                    total_panels, current_shape[0], current_shape[1]
                )
            phase_fallback_references = placed_refs[-max(1, min(len(placed_refs), refs_per_stage)):]

            best_match, best_match_count, feature_metrics = self._find_best_match_staged(
                current_keypoints, current_descriptors, current_scale, current_name, current_image, placed_refs,
                refs_per_stage, cancel_cb, progress_cb
            )
        else:
            best_match, best_match_count, feature_metrics = self._find_best_match_among_references_with_evidence(
                current_keypoints, current_descriptors, current_scale, current_name, current_image, placed_refs,
                allow_rotation=allow_rotation, cancel_cb=cancel_cb, progress_cb=progress_cb
            )

        enable_grid_phase_fallback = cfg.state.ssm.enable_grid_phase_fallback # False
        weak_feature_evidence = self._is_weak_feature_evidence(feature_metrics)

        if weak_feature_evidence:
            emit_msg(
                f"Weak-texture auto pair set: image={current_name}, kp_curr={feature_metrics['kp_curr']}, matches={feature_metrics['match_count']}, label={feature_metrics.get('diagnostic_label', 'unknown')}",
                "debug",
                progress_cb,
            )

        if best_match is not None:
            best_match["ref_global_h"] = placed_refs[best_match["ref_index"]]["global_h"]

        if best_match is None and enable_grid_phase_fallback and weak_feature_evidence:
            best_match = self._try_phase_fallback(phase_fallback_references, current_image, cancel_cb)

        if best_match is None:
            inlier_count = int(feature_metrics.get("inlier_count", 0))
            match_count = int(feature_metrics.get("match_count", 0))
            best_ref_name = feature_metrics.get("best_ref_name") or "none"
            candidate_debug = feature_metrics.get("candidate_debug") or []
            min_inliers = int(getattr(self.matcher, "min_inliers", 0) or 0)
            min_ratio = float(getattr(self.matcher, "min_inlier_ratio", 0.0) or 0.0)
            inlier_ratio = (float(inlier_count) / float(match_count)) if match_count > 0 else 0.0
            emit_msg(
                f"Skipping image {os.path.basename(current_name)} "
                f"(best_ref={os.path.basename(best_ref_name)}, matches={match_count}, "
                f"inliers={inlier_count}, inlier_ratio={inlier_ratio:.3f}, "
                f"required_inliers>={min_inliers}, required_ratio>={min_ratio:.3f}, "
                f"label={feature_metrics.get('diagnostic_label', 'unknown')}, "
                f"stage={feature_metrics.get('failure_stage', 'unknown')})",
                skip_color, progress_cb
            )
            if candidate_debug:
                candidate_summary = ", ".join(
                    f"{os.path.basename(str(candidate.get('ref_name') or 'none'))}:"
                    f"m={int(candidate.get('match_count', 0))}/"
                    f"i={int(candidate.get('inlier_count', 0))}/"
                    f"ok={'y' if candidate.get('transform_ok', False) else 'n'}"
                    for candidate in candidate_debug
                )
                emit_msg(
                    f"Auto candidate ranking for {os.path.basename(current_name)}: {candidate_summary}",
                    "debug",
                    progress_cb,
                )
        return best_match

    def _find_best_match_staged(self, current_keypoints, current_descriptors, current_scale, current_name, current_image, placed_refs,
                                refs_per_stage, cancel_cb, progress_cb):
        """Try progressively larger reference windows until a match is found."""
        best_match_count = 0
        candidate_count = max(1, min(len(placed_refs), refs_per_stage))
        best_feature_metrics = {
            "kp_ref": 0,
            "kp_curr": len(current_keypoints) if current_keypoints is not None else 0,
            "min_kp": 0,
            "match_count": 0,
            "inlier_count": 0,
            "has_descriptors": bool(current_descriptors is not None),
        }

        while True:
            candidate_references = placed_refs[-candidate_count:]
            best_match, match_count, feature_metrics = self._find_best_match_among_references_with_evidence(
                current_keypoints, current_descriptors, current_scale, current_name, current_image, candidate_references,
                allow_rotation=not bool(getattr(self.profile, "lock_rotation", False)), cancel_cb=cancel_cb, progress_cb=progress_cb
            )
            best_match_count = max(best_match_count, match_count)
            if int(feature_metrics.get("match_count", 0)) >= int(best_feature_metrics.get("match_count", 0)):
                best_feature_metrics = feature_metrics
            if best_match is not None:
                best_match["ref_index"] = len(placed_refs) - candidate_count + best_match["ref_index"]
                emit_msg(
                    f"{current_name}: staged match succeeded with {candidate_count} refs",
                    "panel", progress_cb
                )
                return best_match, best_match_count, best_feature_metrics

            if candidate_count >= len(placed_refs):
                return None, best_match_count, best_feature_metrics

            candidate_count = min(len(placed_refs), candidate_count * 2)

    def _find_best_match_among_references_with_evidence(self, current_keypoints, current_descriptors, current_scale,
                                                        current_name, current_image, references, allow_rotation=True, cancel_cb=None,
                                                        progress_cb=None):
        """Find the best match and retain evidence from the strongest feature pair.

        This keeps auto mode conservative: phase fallback is only considered when
        no feature transform succeeds and the best observed feature evidence still
        looks weak.
        """
        best_match = None
        best_match_count = 0
        best_feature_metrics = {
            "kp_ref": 0,
            "kp_curr": len(current_keypoints) if current_keypoints is not None else 0,
            "min_kp": 0,
            "match_count": 0,
            "inlier_count": 0,
            "has_descriptors": bool(current_descriptors is not None),
            "best_ref_name": None,
            "candidate_debug": [],
        }
        collect_candidate_debug = bool(getattr(cfg.prefs, "enable_candidate_debug", False))

        for ref_index, reference in enumerate(references):
            if cancel_cb is not None:
                self.check_cancel(cancel_cb)

            local_transform, inlier_mask, feature_metrics = self._match_feature_pair_with_metrics(
                reference["kp"], reference["des"], reference["scale"],
                current_keypoints, current_descriptors, current_scale,
                allow_rotation=allow_rotation,
                ref_image=reference.get("image"),
                current_image=current_image,
            )
            match_count = int(feature_metrics.get("match_count", 0))
            inlier_count = int(feature_metrics.get("inlier_count", 0))
            best_match_count = max(best_match_count, match_count)

            candidate_debug = []
            if collect_candidate_debug:
                candidate_debug = list(best_feature_metrics.get("candidate_debug") or [])
                candidate_debug.append({
                    "ref_name": reference["name"],
                    "match_count": match_count,
                    "inlier_count": inlier_count,
                    "transform_ok": bool(local_transform is not None and inlier_mask is not None),
                })
                candidate_debug.sort(
                    key=lambda candidate: (
                        int(candidate.get("match_count", 0)),
                        int(candidate.get("inlier_count", 0)),
                        1 if bool(candidate.get("transform_ok", False)) else 0,
                    ),
                    reverse=True,
                )
                candidate_debug = candidate_debug[:3]

            if match_count >= int(best_feature_metrics.get("match_count", 0)):
                best_feature_metrics = feature_metrics
                best_feature_metrics["best_ref_name"] = reference["name"]
                best_feature_metrics["candidate_debug"] = candidate_debug if collect_candidate_debug else []
            else:
                if collect_candidate_debug:
                    best_feature_metrics["candidate_debug"] = candidate_debug

            if local_transform is None or inlier_mask is None:
                continue

            score = int(np.sum(inlier_mask))
            if best_match is None or score > best_match["score"]:
                best_match = {
                    "score": score,
                    "ref_name": reference["name"],
                    "h_local": local_transform,
                    "ref_index": ref_index,
                    "diagnostic_label": str(feature_metrics.get("diagnostic_label", "")),
                    "match_count": int(feature_metrics.get("match_count", 0) or 0),
                    "inlier_count": int(feature_metrics.get("inlier_count", 0) or 0),
                    "inlier_ratio": float(feature_metrics.get("inlier_ratio", 0.0) or 0.0),
                    "subpixel_refinement_applied": bool(feature_metrics.get("subpixel_refinement_applied", False)),
                    "subpixel_refinement_dx": float(feature_metrics.get("subpixel_refinement_dx", 0.0)),
                    "subpixel_refinement_dy": float(feature_metrics.get("subpixel_refinement_dy", 0.0)),
                    "subpixel_refinement_response": float(feature_metrics.get("subpixel_refinement_response", 0.0)),
                }

        return best_match, best_match_count, best_feature_metrics

    @staticmethod
    def _compute_auto_refs_per_stage(total_panels, panel_h, panel_w):
        """Auto-determine refs per stage based on panel count and image size."""
        mp = (panel_h * panel_w) / 1_000_000.0
        complexity = float(total_panels) * mp

        # More panels and larger panels => tighter staged matching
        if complexity < 120.0:
            return 6
        if complexity < 360.0:
            return 5
        if complexity < 900.0:
            return 4
        return 4

    def _try_phase_fallback(self, placed_refs, current_image, cancel_cb):
        """Try phase correlation as fallback when feature matching fails."""
        grid_phase_fallback_min_response = cfg.state.ssm.grid_guide.grid_phase_fallback_min_response # 0.03
        grid_phase_fallback_max_shift_ratio = cfg.state.ssm.grid_guide.grid_phase_fallback_max_shift_ratio # 0.85
        best_phase_candidate = None
        for reference in reversed(placed_refs):
            self.check_cancel(cancel_cb)
            local_transform, response = self._estimate_translation_fallback(
                reference["image"], current_image,
                min_response = grid_phase_fallback_min_response,
                max_shift_ratio = grid_phase_fallback_max_shift_ratio,
            )
            if local_transform is None:
                continue

            if best_phase_candidate is None or response > best_phase_candidate["response"]:
                best_phase_candidate = {
                    "response": response,
                    "ref_name": reference["name"],
                    "h_local": local_transform,
                    "ref_global_h": reference["global_h"]
                }

        if best_phase_candidate is not None:
            return {
                "score": int(round(best_phase_candidate["response"] * 1000.0)), #type: ignore
                "ref_name": best_phase_candidate["ref_name"],
                "h_local": best_phase_candidate["h_local"],
                "ref_global_h": best_phase_candidate["ref_global_h"],
                "fallback": True
            }
        return None

    def _process_matched_image(self, canvas_system, current_name, current_image,
                               best_match, global_transform, image_index, cancel_cb, progress_cb):
        """Compute global transform and warp/blend matched image onto canvas. Returns True if successful."""
        local_transform = best_match["h_local"]
        # Save previous state in case operation fails
        previous_global_transform = global_transform.copy()
        global_transform[:] = best_match["ref_global_h"] @ local_transform

        if best_match.get("fallback", False):
            emit_msg(f"{current_name}: phase-fallback to {best_match['ref_name']} (score={best_match['score']})", "panel", progress_cb)
        else:
            emit_msg(f"{current_name}: matched to {best_match['ref_name']} with inliers={best_match['score']}", "panel", progress_cb)

        # Use shared warp/blend logic with FreeMode's bounds validation.
        success = self._warp_and_blend_single_image(
            canvas_system, current_image, global_transform, current_name, image_index,
            max_allowed_size=AUTO_MODE_MAX_WARP_DIM,
            cancel_cb=cancel_cb, progress_cb=progress_cb
        )
        if success:
            counts = getattr(self, "_placement_source_counts", None)
            if counts is not None:
                source_key = "phase" if best_match.get("fallback", False) else "feature"
                counts[source_key] = counts.get(source_key, 0) + 1
            self._record_panel_source(image_index, "phase" if best_match.get("fallback", False) else "feature")
            self._record_panel_evidence(
                image_index,
                None if best_match.get("fallback", False) else {
                    "ref_idx": int(best_match.get("ref_index", -1)),
                    "diagnostic_label": str(best_match.get("diagnostic_label", "")),
                    "match_count": int(best_match.get("match_count", 0) or 0),
                    "inlier_count": int(best_match.get("inlier_count", 0) or 0),
                    "inlier_ratio": float(best_match.get("inlier_ratio", 0.0) or 0.0),
                    "subpixel_refinement_applied": bool(best_match.get("subpixel_refinement_applied", False)),
                },
            )
        if success and best_match.get("subpixel_refinement_applied", False):
            self._record_subpixel_refinement_applied(image_index)
            emit_msg(
                f"{current_name}: subpixel alignment to {best_match['ref_name']} dx={float(best_match.get('subpixel_refinement_dx', 0.0)):.2f}, dy={float(best_match.get('subpixel_refinement_dy', 0.0)):.2f}, response={float(best_match.get('subpixel_refinement_response', 0.0)):.3f}",
                "panel",
                progress_cb,
            )
        # Restore previous state if operation failed
        if not success:
            global_transform[:] = previous_global_transform
        return success
