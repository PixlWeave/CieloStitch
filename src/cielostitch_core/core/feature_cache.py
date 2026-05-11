# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

"""
Feature detection cache module.

The cache stores feature detection results (keypoints, descriptors) to avoid
redundant SIFT computations on repeated images across multiple stitching runs.

The cache object is created lazily only when caching is enabled for a run.
"""

from __future__ import annotations

import os
import sys
import threading
import logging
from collections import OrderedDict
from typing import Any, Tuple

import numpy as np
from ..config.constants import FEATURE_CACHE_CAPACITY


logger = logging.getLogger(__name__)


def _file_fingerprint(path: str) -> Tuple[str, int, float]:
    try:
        st = os.stat(path)
        return os.path.abspath(path), int(st.st_size), float(st.st_mtime)
    except Exception:
        return os.path.abspath(path), -1, -1.0


def _image_quick_hash(img: np.ndarray) -> Tuple[int, Tuple[int, ...]]:
    try:
        # Very fast, weak checksum sufficient for cache key alongside size/shape
        arr = img
        if not isinstance(arr, np.ndarray):
            return 0, (0,)
        # Use a stride-based subsample to keep this O(N/64)
        # For small arrays, use all elements to avoid false hits
        flat = arr.ravel()
        if len(flat) <= 64:
            sample = flat.astype(np.uint64, copy=False)
        else:
            sample = flat[::64].astype(np.uint64, copy=False)
        checksum = int(sample.sum() % (1 << 61))
        return checksum, tuple(arr.shape)
    except Exception:
        return 0, (0,)


def _detector_fingerprint(detector: Any) -> Tuple[float, int | None, float]:
    # Extract the parameters that affect SIFT detection from our FeatureDetector
    down = float(detector.downscale_factor)
    maxf = detector.max_features
    sens = float(detector.feature_sensitivity)
    return down, maxf if (isinstance(maxf, int) or maxf is None) else None, sens


class _LRU:
    def __init__(self, capacity: int = FEATURE_CACHE_CAPACITY):
        self.capacity = max(8, int(capacity))
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def put_and_get_stats(self, key: str, value: Any) -> tuple[Any | None, Any | None]:
        """Put a value and return (replaced_value, evicted_value) for statistics tracking.

        Returns:
            (replaced_value, evicted_value) - both None if no stats-relevant event occurred
        """
        with self._lock:
            replaced_value = self._store.get(key)
            evicted_value = None
            self._store[key] = value
            self._store.move_to_end(key)
            if len(self._store) > self.capacity:
                _evicted_key, evicted_value = self._store.popitem(last=False)
            return replaced_value, evicted_value

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()




class FeatureCache:
    """Thread-safe in-memory cache for feature detection results.

    Stores tuples: (keypoints, descriptors, scale)
    """

    def __init__(self, capacity: int = FEATURE_CACHE_CAPACITY):
        self._lru = _LRU(capacity)
        self._stats_lock = threading.Lock()
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict[str, int]:
        return {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "entries": 0,
            "descriptor_bytes": 0,
            "keypoints": 0,
            "keypoint_bytes": 0,
        }

    @staticmethod
    def _estimate_value_bytes(value: Any) -> tuple[int, int, int]:
        try:
            keypoints, descriptors, _scale = value
        except Exception:
            return 0, 0, 0

        descriptor_bytes = 0
        if isinstance(descriptors, np.ndarray):
            descriptor_bytes = int(descriptors.nbytes)

        keypoint_count = len(keypoints) if keypoints is not None else 0
        keypoint_bytes = 0
        if keypoints is not None:
            try:
                keypoint_bytes += int(sys.getsizeof(keypoints))
            except Exception:
                pass
            try:
                keypoint_bytes += sum(int(sys.getsizeof(kp)) for kp in keypoints)
            except Exception:
                pass

        return descriptor_bytes, int(keypoint_count), keypoint_bytes

    def _record_request(self, hit: bool) -> None:
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["hits"] += 1 if hit else 0
            self._stats["misses"] += 0 if hit else 1

    def _record_store(self, value: Any, evicted_value: Any | None) -> None:
        desc_bytes, kp_count, kp_bytes = self._estimate_value_bytes(value)
        ev_desc_bytes = 0
        ev_kp_count = 0
        ev_kp_bytes = 0
        if evicted_value is not None:
            ev_desc_bytes, ev_kp_count, ev_kp_bytes = self._estimate_value_bytes(evicted_value)
        with self._stats_lock:
            self._stats["stores"] += 1
            self._stats["entries"] += 1
            self._stats["descriptor_bytes"] += desc_bytes
            self._stats["keypoints"] += kp_count
            self._stats["keypoint_bytes"] += kp_bytes
            if evicted_value is not None:
                self._stats["evictions"] += 1
                self._stats["entries"] = max(0, self._stats["entries"] - 1)
                self._stats["descriptor_bytes"] = max(0, self._stats["descriptor_bytes"] - ev_desc_bytes)
                self._stats["keypoints"] = max(0, self._stats["keypoints"] - ev_kp_count)
                self._stats["keypoint_bytes"] = max(0, self._stats["keypoint_bytes"] - ev_kp_bytes)

    def _record_replace(self, old_value: Any, new_value: Any) -> None:
        old_desc_bytes, old_kp_count, old_kp_bytes = self._estimate_value_bytes(old_value)
        new_desc_bytes, new_kp_count, new_kp_bytes = self._estimate_value_bytes(new_value)
        with self._stats_lock:
            self._stats["descriptor_bytes"] += new_desc_bytes - old_desc_bytes
            self._stats["keypoints"] += new_kp_count - old_kp_count
            self._stats["keypoint_bytes"] += new_kp_bytes - old_kp_bytes

    def snapshot_stats(self, reset_counters: bool = False) -> dict[str, int | float]:
        with self._stats_lock:
            stats = dict(self._stats)
            if reset_counters:
                self._stats["requests"] = 0
                self._stats["hits"] = 0
                self._stats["misses"] = 0
                self._stats["stores"] = 0
                self._stats["evictions"] = 0
        requests = int(stats.get("requests", 0))
        hits = int(stats.get("hits", 0))
        stats["hit_rate"] = int(100.0 * hits / requests) if requests > 0 else 0
        stats["memory_bytes"] = int(stats.get("descriptor_bytes", 0)) + int(stats.get("keypoint_bytes", 0))
        return stats

    @staticmethod
    def format_stats_line(stats: dict[str, int | float], prefix: str = "Feature cache") -> str:
        memory_mb = float(stats.get("memory_bytes", 0)) / (1024.0 * 1024.0)
        desc_mb = float(stats.get("descriptor_bytes", 0)) / (1024.0 * 1024.0)
        kp_mb = float(stats.get("keypoint_bytes", 0)) / (1024.0 * 1024.0)
        return (
            f"{prefix}: req={int(stats.get('requests', 0))}, hit={int(stats.get('hits', 0))}, "
            f"miss={int(stats.get('misses', 0))}, hit%={float(stats.get('hit_rate', 0.0)):.0f}, "
            f"entries={int(stats.get('entries', 0))}, kp={int(stats.get('keypoints', 0))}, "
            f"mem={memory_mb:.1f} MB (des={desc_mb:.1f}, kp={kp_mb:.1f})"
        )

    @staticmethod
    def _make_key(source_id: str | None, image: np.ndarray, detector: Any) -> str:
        parts = ["v1", repr(_detector_fingerprint(detector))]  # bump to invalidate across releases if structure changes
        if source_id and isinstance(source_id, str) and len(source_id) > 0:
            fp = _file_fingerprint(source_id)
            parts.append(repr(fp))
        else:
            parts.append(repr(_image_quick_hash(image)))
        return "|".join(parts)

    def detect(self, detector: Any, image: np.ndarray, source_id: str | None, enabled: bool):
        """Run `detector.detect(image)` using cache when `enabled` is True.

        Returns a tuple: (keypoints, descriptors, scale)
        """
        if not enabled:
            return detector.detect(image)

        key = self._make_key(source_id, image, detector)
        hit = self._lru.get(key)
        if hit is not None:
            self._record_request(hit=True)
            return hit

        self._record_request(hit=False)

        out = detector.detect(image)
        try:
            replaced_value, evicted_value = self._lru.put_and_get_stats(key, out)
            if replaced_value is not None:
                self._record_replace(replaced_value, out)
            else:
                self._record_store(out, evicted_value)
        except Exception as exc:
            # Never fail the pipeline due to caching
            logger.warning("Feature cache store failed; continuing without cache update: %s", exc)
        return out

    def clear(self):
        """Clear all cached entries to prevent corruption across runs."""
        try:
            self._lru.clear()
            with self._stats_lock:
                self._stats = self._new_stats()
        except Exception as exc:
            logger.warning("Feature cache clear failed; cache state may be stale: %s", exc)


_global_feature_cache: FeatureCache | None = None
_global_feature_cache_lock = threading.Lock()


def get_global_feature_cache(create: bool = True) -> FeatureCache | None:
    global _global_feature_cache
    with _global_feature_cache_lock:
        if _global_feature_cache is None and create:
            _global_feature_cache = FeatureCache(capacity=FEATURE_CACHE_CAPACITY)
        return _global_feature_cache


def clear_global_feature_cache() -> None:
    cache = get_global_feature_cache(create=False)
    if cache is not None:
        cache.clear()


def get_feature_cache_stats(reset_counters: bool = False) -> dict[str, int | float]:
    cache = get_global_feature_cache(create=False)
    if cache is None:
        return {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "entries": 0,
            "descriptor_bytes": 0,
            "keypoints": 0,
            "keypoint_bytes": 0,
            "hit_rate": 0.0,
            "memory_bytes": 0,
        }
    return cache.snapshot_stats(reset_counters=reset_counters)


def format_feature_cache_stats(stats: dict[str, int | float], prefix: str = "Feature cache") -> str:
    return FeatureCache.format_stats_line(stats, prefix=prefix)


def detect_with_cache(detector: Any, image: np.ndarray, source_id: str | None, enabled: bool):
    if not enabled:
        return detector.detect(image)
    cache = get_global_feature_cache(create=True)
    return cache.detect(detector, image, source_id, True)


__all__ = [
    "clear_global_feature_cache",
    "detect_with_cache",
    "format_feature_cache_stats",
    "get_feature_cache_stats",
    "get_global_feature_cache",
]
