# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QFont, QGuiApplication

# thread_unsafe - Functions that MUST be called from the Qt main/GUI thread only

BADGE_FONT_POINT_SIZE = 8 # 10
BADGE_WIDTH_PER_CHAR = 8 # 10
BADGE_MIN_WIDTH = 24 # 30
BADGE_MAX_HEIGHT = 24 # 30
BADGE_MARGIN = 2 # 4
BADGE_HORIZONTAL_PADDING = 5 # 18
BADGE_CORNER_RADIUS = 4 # 8


def _badge_width_for_text(text: str) -> int:
    return max(BADGE_MIN_WIDTH, BADGE_HORIZONTAL_PADDING + (BADGE_WIDTH_PER_CHAR * len(text)))


def _assert_gui_thread():
    """Raise RuntimeError if not called from Qt main thread.

    Raises:
        RuntimeError: If called from a thread other than the GUI thread.
    """
    gui_app = QGuiApplication.instance()
    if gui_app is None:
        raise RuntimeError("No QGuiApplication instance found")

    current_thread = QThread.currentThread()
    main_thread = gui_app.thread()

    if current_thread != main_thread:
        raise RuntimeError(
            f"This function must be called from the GUI thread, "
            f"not '{current_thread.objectName()}'. "
            f"QPixmap and QPainter are not thread-safe."
        )


def composite_overlay(pix: QPixmap, overlay_rgba: np.ndarray) -> QPixmap:
    """Composite an RGBA overlay over a base QPixmap and return the composited pixmap.

    ⚠️  GUI THREAD ONLY: This function must be called from the Qt main thread.
    QPixmap and QPainter operations are not thread-safe.

    Args:
        pix: Base QPixmap to composite over.
        overlay_rgba: RGBA uint8 array (H x W x 4, C-contiguous) to draw on top.

    Returns:
        QPixmap: New pixmap with overlay composited on top.

    Raises:
        RuntimeError: If called from a non-GUI thread.
    """
    _assert_gui_thread()
    out = QPixmap(pix)
    p = QPainter(out)
    p.drawImage(0, 0, qimage_from_rgba(overlay_rgba))
    p.end()
    return out


def with_order_badge(pix: QPixmap, order_value: int) -> QPixmap:
    """Draw an order badge (number in top-left corner) on a QPixmap.

    ⚠️  GUI THREAD ONLY: This function must be called from the Qt main thread.
    QPixmap and QPainter operations are not thread-safe.

    Args:
        pix: Base QPixmap to add badge to.
        order_value: Integer value to display in the badge.

    Returns:
        QPixmap: New pixmap with order badge added (unchanged if input is null).

    Raises:
        RuntimeError: If called from a non-GUI thread.
    """
    return with_text_badge(pix, str(order_value))


def with_text_badge(pix: QPixmap, badge_text: str, bg_color: str | QColor | None = None) -> QPixmap:
    """Draw a text badge (top-left corner) on a QPixmap.

    ⚠️  GUI THREAD ONLY: This function must be called from the Qt main thread.
    QPixmap and QPainter operations are not thread-safe.
    """
    _assert_gui_thread()
    if pix.isNull():
        return pix

    text = str(badge_text or "").strip()
    if not text:
        return pix

    out = QPixmap(pix)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)

    badge_w = _badge_width_for_text(text)
    badge_h = BADGE_MAX_HEIGHT
    x, y = BADGE_MARGIN, BADGE_MARGIN
    p.setPen(Qt.NoPen)
    badge_fill = QColor(bg_color) if bg_color is not None else QColor(0, 0, 0, 175)
    if not badge_fill.isValid():
        badge_fill = QColor(0, 0, 0, 175)
    elif badge_fill.alpha() == 255:
        badge_fill.setAlpha(215)
    p.setBrush(badge_fill)
    p.drawRoundedRect(x, y, badge_w, badge_h, BADGE_CORNER_RADIUS, BADGE_CORNER_RADIUS)

    p.setPen(QColor("#ffffff"))
    f = QFont()
    f.setBold(True)
    f.setPointSize(BADGE_FONT_POINT_SIZE)
    p.setFont(f)
    p.drawText(x, y, badge_w, badge_h, Qt.AlignCenter, text)
    p.end()
    return out


def with_text_badge_at(
    pix: QPixmap,
    badge_text: str,
    *,
    corner: str = "top-left",
    bg_color: str | QColor | None = None,
) -> QPixmap:
    """Draw a text badge at a specific corner on a QPixmap."""
    _assert_gui_thread()
    if pix.isNull():
        return pix

    text = str(badge_text or "").strip()
    if not text:
        return pix

    out = QPixmap(pix)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)

    badge_w = _badge_width_for_text(text)
    badge_h = BADGE_MAX_HEIGHT
    margin = BADGE_MARGIN
    x = margin
    y = margin
    if corner == "top-right":
        x = max(margin, out.width() - badge_w - margin)
    elif corner == "bottom-left":
        y = max(margin, out.height() - badge_h - margin)
    elif corner == "bottom-right":
        x = max(margin, out.width() - badge_w - margin)
        y = max(margin, out.height() - badge_h - margin)

    p.setPen(Qt.NoPen)
    badge_fill = QColor(bg_color) if bg_color is not None else QColor(0, 0, 0, 175)
    if not badge_fill.isValid():
        badge_fill = QColor(0, 0, 0, 175)
    elif badge_fill.alpha() == 255:
        badge_fill.setAlpha(215)
    p.setBrush(badge_fill)
    p.drawRoundedRect(x, y, badge_w, badge_h, BADGE_CORNER_RADIUS, BADGE_CORNER_RADIUS)

    p.setPen(QColor("#ffffff"))
    f = QFont()
    f.setBold(True)
    f.setPointSize(BADGE_FONT_POINT_SIZE)
    p.setFont(f)
    p.drawText(x, y, badge_w, badge_h, Qt.AlignCenter, text)
    p.end()
    return out


def qimage_from_rgba(rgba: np.ndarray) -> QImage:
    """Create a QImage from an RGBA uint8 array (H x W x 4, C-contiguous).

    This function is thread-safe since QImage is thread-safe.

    Args:
        rgba: RGBA uint8 array with shape (H, W, 4), C-contiguous.

    Returns:
        QImage: Thread-safe QImage copy.
    """
    return QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
