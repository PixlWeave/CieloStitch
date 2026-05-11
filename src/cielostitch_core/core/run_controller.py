# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Qt, Slot

from .stitching.worker import StitchWorker

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

class RunController(QObject):
    """
    Owns the lifecycle of a StitchWorker running in a background QThread.

    Expects to be parented to the main window, which provides access to the
    status bar and other UI components. The controller ensures proper thread/worker
    cleanup and exposes a simple start/cancel API.
    """

    started = Signal()
    finished = Signal(str, object, object)  # output_path, output_img, float_mosaic
    failed = Signal(str)
    cancelled = Signal()
    threadFinished = Signal()

    log = Signal((str, str, object), (str, str), (str,)) # type: ignore


    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._window = parent  # Reference to main window for GUI access
        self._thread: Optional[QThread] = None
        self._worker: Optional[StitchWorker] = None
        # self.last_overlap_mask = None
        self.last_seam_heatmap = None
        self.last_canvas_system = None
        self.last_panel_source_by_index = None
        self.last_panel_evidence_by_index = None

    def is_running(self) -> bool:
        stitch_running = self._worker is not None
        return stitch_running

    def start(self, sorted_panel_paths: list, alpha_policy: str) -> None:
        if self._worker is not None:
            # Already running; ignore.
            return

        # Wait for previous thread to fully finish to avoid OpenCV corruption
        if self._thread is not None:
            try:
                if not self._thread.wait(3000):  # Wait up to 3 seconds for thread cleanup
                    logger.warning("Stitch worker thread did not finish cleanup within 3 seconds")
            except Exception as e:
                logger.warning(f"Error waiting for stitch thread: {e}")

        # Clear large cached arrays from previous run before starting new one
        # This prevents memory accumulation across multiple stitches
        try:
            self.last_canvas_system = None
            # self.last_overlap_mask = None
            self.last_seam_heatmap = None
            self.last_panel_source_by_index = None
            self.last_panel_evidence_by_index = None
        except Exception:
            logger.debug("Error clearing cached arrays before new run", exc_info=True)

        # Reset OpenCV thread count using workload-aware defaults.
        try:
            cpu = max(1, int(os.cpu_count() or 1))
            # Use an approximate physical-core budget and cap by panel count.
            physical_est = max(1, (3 * cpu + 4) // 5)  # ceil(cpu * 0.6)
            panel_cap = max(1, min(len(sorted_panel_paths), 16))
            cv_threads = max(1, min(physical_est, panel_cap))
            cv2.setNumThreads(cv_threads)
        except Exception:
            logger.debug("Error resetting OpenCV thread count", exc_info=True)

        worker = StitchWorker(sorted_panel_paths=sorted_panel_paths, alpha_policy=alpha_policy)
        thread = QThread(self)
        worker.moveToThread(thread)

        # Wire signals
        thread.started.connect(worker.run)

        # Forward each overload explicitly to preserve argument counts (use queued connection for thread safety)
        worker.log[str].connect(self.log[str], Qt.QueuedConnection)
        worker.log[str, str].connect(self.log[str, str], Qt.QueuedConnection)
        worker.log[str, str, object].connect(self.log[str, str, object], Qt.QueuedConnection)

        # Connect progress signals to status bar (thread-safe via Qt signals with queued connection)
        worker.progress_update.connect(self._on_progress_update, Qt.QueuedConnection)
        worker.placement_update.connect(self._on_placement_update, Qt.QueuedConnection)
        worker.progress_finished.connect(self._on_progress_finished, Qt.QueuedConnection)

        # Capture mask data before emitting finished signal
        # _on_finished will emit self.finished after capturing data
        worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        worker.failed.connect(self.failed, Qt.QueuedConnection)
        worker.images_read.connect(self._on_images_read, Qt.QueuedConnection)
        worker.cancelled.connect(self.cancelled, Qt.QueuedConnection)
        worker.done.connect(thread.quit)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._worker = worker
        self._thread = thread
        try:
            if self._window and hasattr(self._window, 'status_bar') and self._window.status_bar is not None:
                self._window.status_bar.begin_placement_progress()
        except Exception:
            logger.debug("Error initializing placement status", exc_info=True)
        self.started.emit()
        thread.start()

    def cancel(self) -> None:
        cancelled_any = False

        if self._worker is not None:
            try:
                self._worker.request_cancel()
                cancelled_any = True
            except Exception:
                logger.debug("Error requesting cancel on worker", exc_info=True)

        if not cancelled_any:
            return

    # Internal
    @Slot(int, int)
    def _on_progress_update(self, current: int, total: int) -> None:
        """Handle progress updates from worker thread."""
        try:
            if self._window and hasattr(self._window, 'status_bar') and self._window.status_bar is not None:
                self._window.status_bar.set_progress(current, total)
        except Exception:
            logger.debug("Error updating progress bar", exc_info=True)

    @Slot(int)
    def _on_placement_update(self, placed: int) -> None:
        """Handle successful-placement updates for right-side status text."""
        try:
            if self._window and hasattr(self._window, 'status_bar') and self._window.status_bar is not None:
                self._window.status_bar.update_placement_progress(placed)
        except Exception:
            logger.debug("Error updating placement status", exc_info=True)
    @Slot(int)
    def _on_images_read(self, _input_bit_depth: int) -> None:
        """Handle images read from worker thread."""
        try:
            if self._window and hasattr(self._window, 'status_bar') and self._window.status_bar is not None:
                self._window.status_bar.update_loaded()
        except Exception:
            logger.debug("Error updating status bar on images read", exc_info=True)

    @Slot()
    def _on_progress_finished(self) -> None:
        """Handle progress completion."""
        try:
            if self._window and hasattr(self._window, 'status_bar') and self._window.status_bar is not None:
                self._window.status_bar.finish_progress()
        except Exception:
            logger.debug("Error finishing progress bar", exc_info=True)

    def _on_thread_finished(self):
        self._worker = None
        self._thread = None
        # Clear OpenCV's internal states after stitching completes
        try:
            # Force release of any OpenCV memory
            cv2.getTickCount()  # Prime OpenCV
            # Reset OpenCL states if available
            try:
                cv2.ocl.setUseOpenCL(cv2.ocl.useOpenCL())  # Toggle to reset states
            except Exception:
                logger.debug("Error resetting OpenCL state", exc_info=True)
        except Exception:
            logger.debug("Error during post-thread OpenCV cleanup", exc_info=True)
        self.threadFinished.emit()

    def _on_finished(self, output_path: str, output_img: object, float_mosaic: object):
        # Capture overlap mask from the worker before forwarding the signal
        # try:
        #     self.last_overlap_mask = getattr(self._worker, "last_overlap_mask", None)
        # except Exception:
        #     self.last_overlap_mask = None
        # Capture seam heatmap from the worker
        try:
            self.last_seam_heatmap = getattr(self._worker, "last_seam_heatmap", None)
        except Exception:
            logger.debug("Error capturing last_seam_heatmap from worker", exc_info=True)
            self.last_seam_heatmap = None
        # Capture canvas system (contains coverage_count for overlap regions)
        try:
            self.last_canvas_system = getattr(self._worker, "last_canvas_system", None)
        except Exception:
            logger.debug("Error capturing last_canvas_system from worker", exc_info=True)
            self.last_canvas_system = None
        try:
            self.last_panel_source_by_index = getattr(self._worker, "last_panel_source_by_index", None)
        except Exception:
            logger.debug("Error capturing last_panel_source_by_index from worker", exc_info=True)
            self.last_panel_source_by_index = None
        try:
            self.last_panel_evidence_by_index = getattr(self._worker, "last_panel_evidence_by_index", None)
        except Exception:
            logger.debug("Error capturing last_panel_evidence_by_index from worker", exc_info=True)
            self.last_panel_evidence_by_index = None
        self.finished.emit(output_path, output_img, float_mosaic)
