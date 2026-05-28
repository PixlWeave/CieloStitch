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
from cielostitch_core.config.config import cfg
from cielostitch_core.core.bundle import BundleAdjustmentDiagnostics, BundleAdjustmentEdge, refine_global_transforms
from cielostitch_core.config.constants import AUTO_MODE_MAX_WARP_DIM, AUTO_MODE_MAX_WARP_PIXELS
from cielostitch_core.core.apap import APAPEstimator, APAPRegistrationResult
from cielostitch_core.core.mosaic_canvas import MosaicCanvas
from cielostitch_core.utils.message import emit_msg
from cielostitch_core.stitching.base_mode import BaseStitchMode
from cielostitch_core.utils.strings import first_and_last_part
from cielostitch_app.config.ui_constants import DISPLAY_NAME_SIZE

logger = logging.getLogger(__name__)

class FreeMode(BaseStitchMode):

    @staticmethod
    def _prefer_candidate_over_incumbent(candidate, incumbent, transform_mode: str) -> bool:
        if incumbent is None:
            return True

        candidate_score = int(candidate.get("score", 0) or 0)
        incumbent_score = int(incumbent.get("score", 0) or 0)
        if candidate_score > incumbent_score + 1:
            return True
        if incumbent_score > candidate_score + 1:
            return False

        normalized_mode = str(transform_mode or "affine").strip().lower()
        if normalized_mode == "homography":
            candidate_pixels = int(candidate.get("warp_pixels", 0) or 0)
            incumbent_pixels = int(incumbent.get("warp_pixels", 0) or 0)
            if candidate_pixels != incumbent_pixels:
                return candidate_pixels < incumbent_pixels

        candidate_ratio = float(candidate.get("inlier_ratio", 0.0) or 0.0)
        incumbent_ratio = float(incumbent.get("inlier_ratio", 0.0) or 0.0)
        if candidate_ratio != incumbent_ratio:
            return candidate_ratio > incumbent_ratio

        return int(candidate.get("ref_index", -1) or -1) >= int(incumbent.get("ref_index", -1) or -1)

    def __init__(self, profile, show_pixel_stat=False):
        super().__init__(profile, show_pixel_stat=show_pixel_stat)
        self.last_bundle_adjustment_diagnostics = None
        self._bundle_candidates = []
        self._freeform_global_transforms = []
        self._freeform_placed_panels = {}

    def stitch(self, image_items, progress_cb=None, cancel_cb=None):
        """Auto stitching mode: matches each image to the best anchor among all previously placed images."""
        # Phase 1: Common initialization (validation, progress counters, feature detection)
        names, images, image_count, features, start_time = self._init_stitch_session(
            image_items, cancel_cb=cancel_cb, progress_cb=progress_cb
        )
        working_items = list(zip(names, images))

        # Phase 2: Initialize canvas and placed references
        first_name, first_image = working_items[0]
        canvas_system = MosaicCanvas(
            first_image,
            enable_overlap_diagnostics=bool(getattr(cfg.prefs, "retain_overlap_diagnostics", True)),
            enable_seam_diagnostics=bool(getattr(cfg.prefs, "retain_seam_diagnostics", True)),
        )

        # emit_msg(f"Session stitch mode: {cfg.state.stitch_mode}", "debug", progress_cb)
        emit_msg(
            f"Candidate debug details: {'enabled' if bool(getattr(cfg.prefs, 'enable_candidate_debug', False)) else 'disabled'}",
            "debug",
            progress_cb,
        )
        placed_refs = self._init_random_references(first_image, first_name, features[0])
        placed_refs[0]["image_index"] = 0
        self._bundle_candidates = [[] for _ in range(image_count)]
        self._freeform_global_transforms = [None] * image_count
        self._freeform_global_transforms[0] = np.eye(3, dtype=np.float64)
        self._freeform_placed_panels = {
            0: {
                "name": first_name,
                "image": first_image,
                "global_h": np.eye(3, dtype=np.float64),
            }
        }
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
        counts = getattr(self, "_placement_source_counts", None)
        if counts is not None:
            counts["anchor"] = counts.get("anchor", 0) + 1
        self._record_panel_source(0, "anchor")

        # Phase 3: Iteratively match and place images
        global_transform = np.eye(3, dtype=np.float64)
        skipped = self._match_and_place_images(
            canvas_system, working_items, features, placed_refs, global_transform,
            cancel_cb, progress_cb
        )
        canvas_system = self._rebuild_canvas_with_bundle_adjustment(canvas_system, progress_cb=progress_cb, cancel_cb=cancel_cb)

        # Phase 4: Finalize
        emit_msg(f"Translation: {global_transform[0, 2]:.2f}, {global_transform[1, 2]:.2f}", "debug", progress_cb)
        self._sync_random_counters(len(working_items), skipped)
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

        fl_current_name = first_and_last_part(current_name, DISPLAY_NAME_SIZE)

        emit_msg(
            f"{fl_current_name}: kp_prev = {len(placed_refs[-1]['kp'])}, kp_curr = {len(current_keypoints)}",
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
        placed_refs[-1]["image_index"] = int(image_index)
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

        phase_fallback_references = placed_refs
        fl_current_name = first_and_last_part(current_name, DISPLAY_NAME_SIZE)
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
                cancel_cb=cancel_cb, progress_cb=progress_cb
            )

        enable_grid_phase_fallback = self._is_phase_fallback_enabled()
        weak_feature_evidence = self._is_weak_feature_evidence(feature_metrics)

        if weak_feature_evidence:
            emit_msg(
                f"Weak-texture auto pair set: image={fl_current_name}, kp_curr={feature_metrics['kp_curr']}, matches={feature_metrics['match_count']}, label={feature_metrics.get('diagnostic_label', 'unknown')}",
                "debug",
                progress_cb,
            )

        if best_match is not None:
            best_match["ref_global_h"] = placed_refs[best_match["ref_index"]]["global_h"]
            best_match["ref_image_index"] = int(placed_refs[best_match["ref_index"]].get("image_index", best_match["ref_index"]))

        if best_match is None and enable_grid_phase_fallback and weak_feature_evidence:
            best_match = self._try_phase_fallback(phase_fallback_references, current_image, cancel_cb)

        geometry_rejected = bool(feature_metrics.get("geometry_rejected_any", False))
        if best_match is None and enable_grid_phase_fallback and geometry_rejected:
            emit_msg(
                f"{fl_current_name}: all feature candidates rejected by warp limits; trying phase fallback",
                "debug",
                progress_cb,
            )
            best_match = self._try_phase_fallback(phase_fallback_references, current_image, cancel_cb)

        geometry_stage_failed = (
                str(feature_metrics.get("failure_stage") or "").strip().lower() == "geometry"
                and int(feature_metrics.get("match_count", 0) or 0) > 0 >= int(feature_metrics.get("inlier_count", 0) or 0)
        )
        if best_match is None and enable_grid_phase_fallback and geometry_stage_failed:
            emit_msg(
                f"{fl_current_name}: feature solve failed at geometry stage; trying phase fallback",
                "debug",
                progress_cb,
            )
            best_match = self._try_phase_fallback(phase_fallback_references, current_image, cancel_cb)

        if best_match is None:
            inlier_count = int(feature_metrics.get("inlier_count", 0))
            match_count = int(feature_metrics.get("match_count", 0))
            best_ref_name = feature_metrics.get("best_ref_name") or "none"
            candidate_debug = feature_metrics.get("candidate_debug") or []
            geometry_rejected = bool(feature_metrics.get("geometry_rejected_any", False))
            min_inliers = int(getattr(self.matcher, "min_inliers", 0) or 0)
            min_ratio = float(getattr(self.matcher, "min_inlier_ratio", 0.0) or 0.0)
            inlier_ratio = (float(inlier_count) / float(match_count)) if match_count > 0 else 0.0
            fl_current_name = first_and_last_part(current_name, DISPLAY_NAME_SIZE)
            fl_best_ref_name = first_and_last_part(os.path.basename(best_ref_name), DISPLAY_NAME_SIZE)
            emit_msg(
                f"Skipping image {fl_current_name} "
                f"(best_ref={fl_best_ref_name}, matches={match_count}, "
                f"inliers={inlier_count}, inlier_ratio={inlier_ratio:.3f}, "
                f"required_inliers>={min_inliers}, required_ratio>={min_ratio:.3f}, "
                f"label={feature_metrics.get('diagnostic_label', 'unknown')}, "
                f"stage={feature_metrics.get('failure_stage', 'unknown')}, "
                f"geometry_rejected={'yes' if geometry_rejected else 'no'})",
                skip_color, progress_cb
            )
            if candidate_debug:
                candidate_summary = ", ".join(
                    f"{first_and_last_part(os.path.basename(str(candidate.get('ref_name') or 'none')),DISPLAY_NAME_SIZE)}:"
                    f"m={int(candidate.get('match_count', 0))}/"
                    f"i={int(candidate.get('inlier_count', 0))}/"
                    f"ok={'y' if candidate.get('transform_ok', False) else 'n'}/"
                    f"geo={'y' if candidate.get('geometry_ok', True) else 'n'}"
                    for candidate in candidate_debug
                )
                emit_msg(
                    f"Auto candidate ranking for {fl_current_name}: {candidate_summary}",
                    "debug",
                    progress_cb,
                )
        return best_match

    @staticmethod
    def _is_phase_fallback_enabled() -> bool:
        """Return whether phase fallback is enabled for FreeMode."""
        try:
            return bool(getattr(cfg.state.ssm, "enable_grid_phase_fallback", False))
        except Exception:
            return False

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
                cancel_cb=cancel_cb, progress_cb=progress_cb)
            best_match_count = max(best_match_count, match_count)
            if int(feature_metrics.get("match_count", 0)) >= int(best_feature_metrics.get("match_count", 0)):
                best_feature_metrics = feature_metrics
            if best_match is not None:
                best_match["ref_index"] = len(placed_refs) - candidate_count + best_match["ref_index"]
                fl_current_name = first_and_last_part(current_name, DISPLAY_NAME_SIZE)
                emit_msg(
                    f"{fl_current_name}: staged match succeeded with {candidate_count} refs",
                    "panel", progress_cb
                )
                return best_match, best_match_count, best_feature_metrics

            if candidate_count >= len(placed_refs):
                return None, best_match_count, best_feature_metrics

            candidate_count = min(len(placed_refs), candidate_count * 2)

    def _find_best_match_among_references_with_evidence(self, current_keypoints, current_descriptors, current_scale,
                                                        current_name, current_image, references, cancel_cb=None,
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
            "geometry_rejected_any": False,
        }
        collect_candidate_debug = bool(getattr(cfg.prefs, "enable_candidate_debug", False))
        transform_mode = str(getattr(self.profile, "transform_mode", "affine") or "affine").strip().lower()

        for ref_index, reference in enumerate(references):
            if cancel_cb is not None:
                self.check_cancel(cancel_cb)

            local_transform, inlier_mask, feature_metrics = self._match_feature_pair_with_metrics(
                reference["kp"], reference["des"], reference["scale"],
                current_keypoints, current_descriptors, current_scale,
                ref_image=reference.get("image"),
                current_image=current_image,
            )
            match_count = int(feature_metrics.get("match_count", 0))
            inlier_count = int(feature_metrics.get("inlier_count", 0))
            best_match_count = max(best_match_count, match_count)

            if match_count >= int(best_feature_metrics.get("match_count", 0)):
                best_feature_metrics = feature_metrics
                best_feature_metrics["best_ref_name"] = reference["name"]
                if collect_candidate_debug:
                    best_feature_metrics["candidate_debug"] = list(best_feature_metrics.get("candidate_debug") or [])
            else:
                pass

            if local_transform is None or inlier_mask is None:
                if collect_candidate_debug:
                    candidate_debug = list(best_feature_metrics.get("candidate_debug") or [])
                    candidate_debug.append({
                        "ref_name": reference["name"],
                        "match_count": match_count,
                        "inlier_count": inlier_count,
                        "transform_ok": False,
                        "geometry_ok": True,
                    })
                    candidate_debug.sort(
                        key=lambda candidate: (
                            int(candidate.get("match_count", 0)),
                            int(candidate.get("inlier_count", 0)),
                            1 if bool(candidate.get("transform_ok", False)) else 0,
                            1 if bool(candidate.get("geometry_ok", False)) else 0,
                        ),
                        reverse=True,
                    )
                    best_feature_metrics["candidate_debug"] = candidate_debug[:3]
                continue

            # Guard against selecting high-inlier candidates whose composed global
            # transform would immediately violate FreeMode warp safety limits.
            geometry_ok = True
            warp_w = 0
            warp_h = 0
            warp_pixels = 0
            try:
                candidate_global_h = reference["global_h"] @ local_transform
                bmin_x, bmin_y, bmax_x, bmax_y = MosaicCanvas.compute_warp_bounds(current_image, candidate_global_h)
                candidate_apap_registration = feature_metrics.get("apap_registration")
                if isinstance(candidate_apap_registration, APAPRegistrationResult):
                    apap_estimator = getattr(self.matcher, "apap_estimator", None)
                    if apap_estimator is None:
                        apap_estimator = APAPEstimator()
                    candidate_apap_registration = apap_estimator.compose_registration(
                        candidate_apap_registration,
                        reference["global_h"],
                    )
                    bmin_x, bmin_y, bmax_x, bmax_y = self._expand_bounds_with_apap_registration(
                        current_image,
                        (bmin_x, bmin_y, bmax_x, bmax_y),
                        candidate_apap_registration,
                    )
                warp_w = int(np.ceil(abs(float(bmax_x) - float(bmin_x))))
                warp_h = int(np.ceil(abs(float(bmax_y) - float(bmin_y))))
                warp_pixels = int(warp_w * warp_h)
                if warp_w > int(AUTO_MODE_MAX_WARP_DIM) or warp_h > int(AUTO_MODE_MAX_WARP_DIM):
                    geometry_ok = False
                elif warp_pixels > int(AUTO_MODE_MAX_WARP_PIXELS):
                    geometry_ok = False
            except Exception:
                geometry_ok = False

            if collect_candidate_debug:
                candidate_debug = list(best_feature_metrics.get("candidate_debug") or [])
                candidate_debug.append({
                    "ref_name": reference["name"],
                    "match_count": match_count,
                    "inlier_count": inlier_count,
                    "transform_ok": True,
                    "geometry_ok": bool(geometry_ok),
                    "warp_w": int(warp_w),
                    "warp_h": int(warp_h),
                })
                candidate_debug.sort(
                    key=lambda candidate: (
                        int(candidate.get("match_count", 0)),
                        int(candidate.get("inlier_count", 0)),
                        1 if bool(candidate.get("transform_ok", False)) else 0,
                        1 if bool(candidate.get("geometry_ok", False)) else 0,
                    ),
                    reverse=True,
                )
                best_feature_metrics["candidate_debug"] = candidate_debug[:3]

            if not geometry_ok:
                best_feature_metrics["geometry_rejected_any"] = True
                continue

            score = int(np.sum(inlier_mask))
            candidate = {
                "score": score,
                "ref_name": reference["name"],
                "h_local": local_transform,
                "ref_index": ref_index,
                "apap_registration": feature_metrics.get("apap_registration"),
                "diagnostic_label": str(feature_metrics.get("diagnostic_label", "")),
                "match_count": int(feature_metrics.get("match_count", 0) or 0),
                "inlier_count": int(feature_metrics.get("inlier_count", 0) or 0),
                "inlier_ratio": float(feature_metrics.get("inlier_ratio", 0.0) or 0.0),
                "subpixel_refinement_applied": bool(feature_metrics.get("subpixel_refinement_applied", False)),
                "subpixel_refinement_dx": float(feature_metrics.get("subpixel_refinement_dx", 0.0)),
                "subpixel_refinement_dy": float(feature_metrics.get("subpixel_refinement_dy", 0.0)),
                "subpixel_refinement_response": float(feature_metrics.get("subpixel_refinement_response", 0.0)),
                "warp_pixels": int(warp_pixels),
                "warp_w": int(warp_w),
                "warp_h": int(warp_h),
            }
            if self._prefer_candidate_over_incumbent(candidate, best_match, transform_mode):
                best_match = candidate

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
        profile_name = str(getattr(cfg.state, "profile", "") or "").strip().lower()
        if profile_name == "panorama":
            # Panorama sweeps are prone to false translation locks; require stronger phase confidence.
            grid_phase_fallback_min_response = max(float(grid_phase_fallback_min_response), 0.03)
            min_overlap_ratio = 0.12
        else:
            min_overlap_ratio = 0.06
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

            # Reject phase-only candidates that imply too little geometric overlap.
            try:
                ref_h, ref_w = reference["image"].shape[:2]
                cur_h, cur_w = current_image.shape[:2]
                dx = abs(float(local_transform[0, 2]))
                dy = abs(float(local_transform[1, 2]))
                overlap_w = float(min(ref_w, cur_w)) - dx
                overlap_h = float(min(ref_h, cur_h)) - dy
                if overlap_w <= 0.0 or overlap_h <= 0.0:
                    continue
                overlap_area = overlap_w * overlap_h
                min_panel_area = float(min(ref_h * ref_w, cur_h * cur_w))
                overlap_ratio = overlap_area / max(min_panel_area, 1.0)
                if overlap_ratio < float(min_overlap_ratio):
                    continue
            except Exception:
                continue

            if best_phase_candidate is None or response > best_phase_candidate["response"]:
                best_phase_candidate = {
                    "response": response,
                    "ref_name": reference["name"],
                    "h_local": local_transform,
                    "ref_global_h": reference["global_h"],
                    "ref_image_index": int(reference.get("image_index", 0)),
                    "overlap_ratio": float(overlap_ratio),
                }

        if best_phase_candidate is not None:
            return {
                "score": int(round(best_phase_candidate["response"] * 1000.0)), #type: ignore
                "ref_name": best_phase_candidate["ref_name"],
                "h_local": best_phase_candidate["h_local"],
                "ref_global_h": best_phase_candidate["ref_global_h"],
                "ref_image_index": int(best_phase_candidate.get("ref_image_index", 0)),
                "phase_overlap_ratio": float(best_phase_candidate.get("overlap_ratio", 0.0)),
                "fallback": True
            }
        return None

    @staticmethod
    def _is_affine_like(transform) -> bool:
        try:
            mat = np.asarray(transform, dtype=np.float64)
        except Exception:
            return False
        if mat.shape != (3, 3) or not np.isfinite(mat).all():
            return False
        return bool(np.allclose(mat[2], np.array([0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6))

    def _record_bundle_edge(self, best_match, image_index: int) -> None:
        try:
            if best_match.get("apap_registration") is not None:
                return
            relative_transform = np.asarray(best_match.get("h_local"), dtype=np.float64)
            if not self._is_affine_like(relative_transform):
                return
            ref_image_index = int(best_match.get("ref_image_index", -1))
            current_index = int(image_index)
            if ref_image_index < 0 or current_index <= 0 or current_index >= len(self._bundle_candidates):
                return
            self._bundle_candidates[current_index].append(
                BundleAdjustmentEdge(
                    ref_index=ref_image_index,
                    image_index=current_index,
                    relative_transform=relative_transform.copy(),
                    score=float(best_match.get("score", 0) or 0.0),
                    source="phase" if best_match.get("fallback", False) else "feature",
                )
            )
        except Exception:
            logger.debug("Failed to record free-mode bundle edge", exc_info=True)

    def _rebuild_canvas_with_bundle_adjustment(self, canvas_system, progress_cb=None, cancel_cb=None):
        mode = str(getattr(cfg.state.ssm, "bundle_adjustment_mode", "off") or "off").strip().lower()
        if mode == "off":
            return canvas_system

        global_transforms = list(getattr(self, "_freeform_global_transforms", []) or [])
        if not global_transforms:
            self.last_bundle_adjustment_diagnostics = BundleAdjustmentDiagnostics(
                mode=mode,
                edge_count=0,
                adjusted_panels=0,
                mean_translation_shift_px=0.0,
                max_translation_shift_px=0.0,
                iterations=0,
                status="no_edges",
            )
            emit_msg(f"Bundle refinement skipped: mode={mode}, status=no_edges", "debug", progress_cb)
            return canvas_system

        try:
            refined, diagnostics = refine_global_transforms(global_transforms, self._bundle_candidates, mode=mode)
        except Exception:
            logger.debug("Free-mode bundle refinement failed; keeping original canvas", exc_info=True)
            self.last_bundle_adjustment_diagnostics = BundleAdjustmentDiagnostics(
                mode=mode,
                edge_count=0,
                adjusted_panels=0,
                mean_translation_shift_px=0.0,
                max_translation_shift_px=0.0,
                iterations=0,
                status="error",
            )
            return canvas_system

        self.last_bundle_adjustment_diagnostics = diagnostics
        if diagnostics.status != "ok":
            emit_msg(f"Bundle refinement skipped: mode={mode}, status={diagnostics.status}", "debug", progress_cb)
            return canvas_system

        placed_panels = dict(getattr(self, "_freeform_placed_panels", {}) or {})
        anchor_panel = placed_panels.get(0)
        if not anchor_panel:
            return canvas_system

        rebuilt_canvas = MosaicCanvas(
            anchor_panel["image"],
            enable_overlap_diagnostics=bool(getattr(cfg.prefs, "retain_overlap_diagnostics", True)),
            enable_seam_diagnostics=bool(getattr(cfg.prefs, "retain_seam_diagnostics", True)),
        )

        for image_index in sorted(idx for idx in placed_panels.keys() if int(idx) != 0):
            self.check_cancel(cancel_cb)
            panel = placed_panels.get(image_index)
            refined_transform = refined[image_index] if image_index < len(refined) else None
            if panel is None or refined_transform is None:
                continue
            success = self._warp_and_blend_single_image(
                rebuilt_canvas,
                panel["image"],
                np.asarray(refined_transform, dtype=np.float64),
                panel["name"],
                int(image_index),
                max_allowed_size=AUTO_MODE_MAX_WARP_DIM,
                max_allowed_pixels=AUTO_MODE_MAX_WARP_PIXELS,
                diagnostic_context={
                    "ref_name": "bundle",
                    "score": None,
                    "fallback": False,
                },
                cancel_cb=cancel_cb,
                progress_cb=progress_cb,
            )
            if not success:
                logger.debug("Free-mode bundle canvas rebuild failed for panel %s; keeping original canvas", image_index)
                self.last_bundle_adjustment_diagnostics = BundleAdjustmentDiagnostics(
                    mode=mode,
                    edge_count=int(diagnostics.edge_count),
                    adjusted_panels=0,
                    mean_translation_shift_px=0.0,
                    max_translation_shift_px=0.0,
                    iterations=int(diagnostics.iterations),
                    status="error",
                )
                return canvas_system
            self._freeform_global_transforms[image_index] = np.asarray(refined_transform, dtype=np.float64).copy()
            panel["global_h"] = self._freeform_global_transforms[image_index]

        emit_msg(
            f"Bundle refinement applied: mode={mode}, edges={diagnostics.edge_count}, adjusted={diagnostics.adjusted_panels}, mean_shift={diagnostics.mean_translation_shift_px:.2f}px, max_shift={diagnostics.max_translation_shift_px:.2f}px",
            "debug",
            progress_cb,
        )
        return rebuilt_canvas

    def _process_matched_image(self, canvas_system, current_name, current_image,
                               best_match, global_transform, image_index, cancel_cb, progress_cb):
        """Compute global transform and warp/blend matched image onto canvas. Returns True if successful."""
        local_transform = best_match["h_local"]
        # Save previous state in case operation fails
        previous_global_transform = global_transform.copy()
        global_transform[:] = best_match["ref_global_h"] @ local_transform

        fl_current_name = first_and_last_part(current_name, DISPLAY_NAME_SIZE)
        fl_ref_name = first_and_last_part(best_match["ref_name"], DISPLAY_NAME_SIZE)
        if best_match.get("fallback", False):
            emit_msg(
                f"{fl_current_name}): phase-fallback to "
                f"{fl_ref_name} (score={best_match['score']}, overlap={float(best_match.get('phase_overlap_ratio', 0.0)):.3f})",
                "panel",
                progress_cb,
            )
        else:
            emit_msg(
                f"{fl_current_name}: matched to "
                f"{fl_ref_name} with inliers={best_match['score']}", "panel", progress_cb)

        placement_score = None
        min_blend_score = None
        if best_match.get("fallback", False):
            # Phase fallback is translation-only; without a confidence gate it can
            # produce heavy double-edges in large overlap regions.
            placement_score = float(best_match.get("score", 0) or 0)
            min_blend_score = 120.0

        # Use shared warp/blend logic with FreeMode's bounds validation.
        apap_registration = best_match.get("apap_registration")
        if isinstance(apap_registration, APAPRegistrationResult):
            apap_estimator = getattr(self.matcher, "apap_estimator", None)
            if apap_estimator is None:
                apap_estimator = APAPEstimator()
            apap_registration = apap_estimator.compose_registration(apap_registration, best_match["ref_global_h"])

        success = self._warp_and_blend_single_image(
            canvas_system, current_image, global_transform, current_name, image_index,
            placement_score=placement_score,
            min_blend_score=min_blend_score,
            max_allowed_size=AUTO_MODE_MAX_WARP_DIM,
            max_allowed_pixels=AUTO_MODE_MAX_WARP_PIXELS,
            diagnostic_context={
                "ref_name": best_match.get("ref_name"),
                "score": best_match.get("score"),
                "fallback": bool(best_match.get("fallback", False)),
                "apap_registration": apap_registration,
            },
            cancel_cb=cancel_cb, progress_cb=progress_cb
        )
        if success:
            counts = getattr(self, "_placement_source_counts", None)
            if counts is not None:
                source_key = "phase" if best_match.get("fallback", False) else "feature"
                counts[source_key] = counts.get(source_key, 0) + 1
            transform_store = getattr(self, "_freeform_global_transforms", None)
            if isinstance(transform_store, list) and 0 <= int(image_index) < len(transform_store):
                transform_store[int(image_index)] = global_transform.copy()
            panel_store = getattr(self, "_freeform_placed_panels", None)
            if isinstance(panel_store, dict):
                panel_store[int(image_index)] = {
                    "name": current_name,
                    "image": current_image,
                    "global_h": global_transform.copy(),
                }
            self._record_bundle_edge(best_match, image_index)
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
                f"{fl_current_name}: subpixel alignment to {fl_ref_name} dx={float(best_match.get('subpixel_refinement_dx', 0.0)):.2f}, dy={float(best_match.get('subpixel_refinement_dy', 0.0)):.2f}, response={float(best_match.get('subpixel_refinement_response', 0.0)):.3f}",
                "panel",
                progress_cb,
            )
        # Restore previous state if operation failed
        if not success:
            global_transform[:] = previous_global_transform
        return success
