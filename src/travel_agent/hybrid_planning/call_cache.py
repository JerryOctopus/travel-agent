"""Thread-safe, request-scoped deduplication for bounded hybrid calls."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _CacheEntry:
    ready: threading.Event = field(default_factory=threading.Event)
    value: Any = None


class HybridCallCache:
    """Coordinate identical calls across planner/reviewer worker snapshots."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], _CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_entries = 256

    def reserve(self, scope: str, module: str, fingerprint: str) -> tuple[str, Any]:
        key = (scope or "standalone", module, fingerprint)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= self._max_entries:
                    oldest_ready = next(
                        (old_key for old_key, old in self._entries.items() if old.ready.is_set()),
                        None,
                    )
                    if oldest_ready is not None:
                        self._entries.pop(oldest_ready, None)
                self._entries[key] = _CacheEntry()
                return "reserved", None
            if entry.ready.is_set():
                return "cache_hit", entry.value
            return "in_progress", entry

    def wait(self, entry: Any, timeout_seconds: float) -> tuple[bool, Any]:
        if not isinstance(entry, _CacheEntry):
            return False, None
        if not entry.ready.wait(max(0.0, timeout_seconds)):
            return False, None
        return True, entry.value

    def publish(self, scope: str, module: str, fingerprint: str, value: Any) -> None:
        key = (scope or "standalone", module, fingerprint)
        with self._lock:
            entry = self._entries.setdefault(key, _CacheEntry())
            entry.value = value
            entry.ready.set()
