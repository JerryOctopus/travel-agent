"""agent_trace：多 Agent 架构的统一执行轨迹（Step 2 并发安全实现）。

并发契约（严格执行）：

- ``append`` / ``append_many`` 在内部 ``RLock`` 保护下原子完成，
  并发追加不丢记录；
- **禁止**通过"读取 latest → 修改 → 再 put"完成并发追加——轨迹在内存中
  增量累积，仅在回合边界由单线程调用 ``flush_to_store`` 一次性落 artifact；
- 每条记录携带 request_id / task_id / agent，并发场景下可精确关联。

Step 5 将用它统一替换 layer_trace；Step 2 只提供线程安全的轨迹设施与测试。
"""

from __future__ import annotations

import threading
import time
from typing import Any


class AgentTraceLog:
    """线程安全的 agent_trace 累积器。"""

    KIND = "agent_trace"

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def append(
        self,
        kind: str,
        *,
        agent: str = "",
        task_id: str = "",
        status: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """原子追加一条轨迹记录并返回该记录。"""
        entry = {
            "request_id": self.request_id,
            "task_id": task_id,
            "agent": agent,
            "kind": kind,
            "status": status,
            "detail": dict(detail or {}),
            "created_at": time.time(),
        }
        with self._lock:
            self._entries.append(entry)
        return entry

    def append_many(self, entries: list[dict[str, Any]]) -> None:
        """批量原子追加（一次锁内完成，避免中途被读到半截状态）。"""
        with self._lock:
            self._entries.extend(entries)

    def snapshot(self) -> list[dict[str, Any]]:
        """返回按追加顺序的浅拷贝列表。"""
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def flush_to_store(self, store: Any) -> str:
        """回合边界（单线程）一次性落 artifact；返回 artifact_id。

        不做"读 latest → 合并 → put"：同一 request 的轨迹整体覆盖写入，
        携带 request_id 元数据便于并发回合区分。
        """
        return store.put(
            self.KIND,
            {"request_id": self.request_id, "items": self.snapshot()},
            request_id=self.request_id,
            agent="engine",
        )
